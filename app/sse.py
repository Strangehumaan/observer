"""Incremental SSE parsing and accumulation of a chat completion stream."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

MAX_EVENT_BYTES = 16 * 1024 * 1024


def loads_object(data: bytes | str) -> dict[str, Any] | None:
    """Parse JSON and return it only if it is an object."""
    try:
        value = json.loads(data)
    except (ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def normalize_error(error: Any) -> dict[str, Any]:
    return error if isinstance(error, dict) else {"message": str(error)}


class SSEParser:
    """Turns arbitrary byte chunks into the data payload of each complete event.

    Lines end in LF or CRLF. Bytes stay buffered until a whole line has arrived, so
    events and multi-byte characters may be split across chunks.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._data: list[bytes] = []

    def feed(self, chunk: bytes) -> list[str]:
        # The buffer never holds a complete line between calls, so only new bytes are searched.
        search_from = len(self._buffer)
        self._buffer += chunk
        events: list[str] = []
        line_start = 0
        while (line_end := self._buffer.find(b"\n", search_from)) != -1:
            self._handle_line(bytes(self._buffer[line_start:line_end]).removesuffix(b"\r"), events)
            line_start = search_from = line_end + 1
        del self._buffer[:line_start]
        if len(self._buffer) > MAX_EVENT_BYTES:
            raise ValueError("SSE line exceeds the parser's size limit")
        return events

    def _handle_line(self, line: bytes, events: list[str]) -> None:
        if not line:
            if self._data:
                events.append(b"\n".join(self._data).decode("utf-8", "replace"))
                self._data = []
            return
        if line.startswith(b":"):
            return
        field, _, value = line.partition(b":")
        if field == b"data":
            self._data.append(value.removeprefix(b" "))


@dataclass
class CompletionSummary:
    response_model: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    output_text: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    error: dict[str, Any] | None = None


class StreamAccumulator:
    """Rebuilds the completion from a copy of the stream."""

    def __init__(self, started: float, clock: Callable[[], float] = time.perf_counter) -> None:
        self.ttft_ms: float | None = None
        self._started = started
        self._clock = clock
        self._parser = SSEParser()
        self._done = False
        self._model: str | None = None
        self._finish_reason: str | None = None
        self._usage: dict[str, Any] | None = None
        self._error: dict[str, Any] | None = None
        self._text: list[str] = []
        self._tool_calls: dict[int, dict[str, Any]] = {}

    def feed(self, chunk: bytes) -> None:
        if self._done:
            return
        for data in self._parser.feed(chunk):
            if data.strip() == "[DONE]":
                self._done = True
                return
            event = loads_object(data)
            if event is not None:
                self._apply_event(event)

    def summary(self) -> CompletionSummary:
        tool_calls = [self._tool_calls[index] for index in sorted(self._tool_calls)]
        return CompletionSummary(
            response_model=self._model,
            finish_reason=self._finish_reason,
            usage=self._usage,
            output_text="".join(self._text) or None,
            tool_calls=tool_calls or None,
            error=self._error,
        )

    def _apply_event(self, event: dict[str, Any]) -> None:
        if isinstance(event.get("model"), str):
            self._model = event["model"]
        if isinstance(event.get("usage"), dict):
            self._usage = event["usage"]
        if event.get("error"):
            self._error = normalize_error(event["error"])
        choices = event.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if isinstance(choice, dict) and choice.get("index", 0) == 0:
                self._apply_choice(choice)
                return

    def _apply_choice(self, choice: dict[str, Any]) -> None:
        reason = choice.get("finish_reason")
        if isinstance(reason, str):
            # OpenRouter's usage chunk repeats the finish reason, so the first one stands.
            if self._finish_reason is None or reason == "error":
                self._finish_reason = reason
            if reason == "error" and self._error is None:
                self._error = normalize_error(choice.get("error") or "finish_reason: error")

        delta = choice.get("delta")
        if not isinstance(delta, dict):
            return
        content = delta.get("content")
        if isinstance(content, str) and content:
            self._text.append(content)
            self._mark_first_output()
        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    self._apply_tool_call(tool_call)

    def _apply_tool_call(self, delta: dict[str, Any]) -> None:
        index = delta.get("index", 0)
        call = self._tool_calls.setdefault(
            index, {"id": None, "type": None, "function": {"name": None, "arguments": ""}}
        )
        function = delta.get("function")
        if not isinstance(function, dict):
            function = {}
        if delta.get("id"):
            call["id"] = delta["id"]
        if delta.get("type"):
            call["type"] = delta["type"]
        if function.get("name"):
            call["function"]["name"] = function["name"]
        if isinstance(function.get("arguments"), str):
            call["function"]["arguments"] += function["arguments"]
        if delta.get("id") or function.get("name") or function.get("arguments"):
            self._mark_first_output()

    def _mark_first_output(self) -> None:
        if self.ttft_ms is None:
            self.ttft_ms = (self._clock() - self._started) * 1000
