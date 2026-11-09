"""Fake upstream and ASGI driver. Nothing here touches the network."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI

from app.db import Filters

CHAT_PATH = "/v1/chat/completions"
SECRET = "sk-or-secret-value"
USAGE = {
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_tokens": 15,
    "cost": 0.00123,
    "prompt_tokens_details": {"cached_tokens": 3},
    "completion_tokens_details": {"reasoning_tokens": 2},
}
STREAM_BODY = b'{"model":"test/model","stream":true,"messages":[{"role":"user","content":"hi"}]}'
PLAIN_BODY = b'{"model":"test/model","messages":[{"role":"user","content":"hi"}]}'
SSE_HEADERS = {"content-type": "text/event-stream"}
DONE = b"data: [DONE]\n\n"


def sse(payload: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n"


def delta(**fields: Any) -> dict[str, Any]:
    return {"model": "test/model-1", "choices": [{"index": 0, "delta": fields, "finish_reason": None}]}


def usage_chunk(finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "model": "test/model-1",
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        "usage": USAGE,
    }


def completion(content: str = "hello there") -> dict[str, Any]:
    return {
        "id": "gen-1",
        "model": "test/model-1",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": USAGE,
    }


class ScriptedStream(httpx.AsyncByteStream):
    """An upstream body delivered in exact chunks, optionally hanging or failing at the end."""

    def __init__(
        self, chunks: list[bytes], hang: bool = False, error: Exception | None = None
    ) -> None:
        self.chunks = chunks
        self.hang = hang
        self.error = error
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error
        if self.hang:
            await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed = True


class UnreadBodyTransport(httpx.MockTransport):
    """A mock whose responses are unread streams, as they are from a real network transport."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await super().handle_async_request(request)
        if not response.is_stream_consumed:
            return response
        return httpx.Response(
            response.status_code, headers=response.headers, stream=ScriptedStream([response.content])
        )


def stream_upstream(stream: ScriptedStream, headers: dict[str, str] | None = None):
    """A fake upstream handler returning `stream` as SSE."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={**SSE_HEADERS, **(headers or {})}, stream=stream)

    return handler


@dataclass
class AsgiResult:
    status: int | None = None
    headers: dict[str, str] = field(default_factory=dict)
    chunks: list[bytes] = field(default_factory=list)

    @property
    def body(self) -> bytes:
        return b"".join(self.chunks)


@dataclass
class Harness:
    app: FastAPI
    client: httpx.AsyncClient

    async def records(self) -> list[dict[str, Any]]:
        """Wait for queued records to be written, then return every stored row."""
        await self.app.state.recorder.queue.join()
        return await self.app.state.database.list_requests(Filters(), 1000, 0)

    async def stream(
        self,
        body: bytes = STREAM_BODY,
        headers: list[tuple[bytes, bytes]] | None = None,
        disconnect_after_chunks: int | None = None,
    ) -> AsgiResult:
        """Drive the ASGI app directly, keeping chunk boundaries and allowing a disconnect."""
        result = AsgiResult()
        disconnected = asyncio.Event()
        request_delivered = False

        async def receive() -> dict[str, Any]:
            nonlocal request_delivered
            if not request_delivered:
                request_delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                result.status = message["status"]
                result.headers = {k.decode(): v.decode() for k, v in message["headers"]}
                return
            if message["body"]:
                result.chunks.append(message["body"])
            if not message.get("more_body", False):
                disconnected.set()  # Servers report a disconnect once the response is complete.
            elif disconnect_after_chunks is not None and len(result.chunks) >= disconnect_after_chunks:
                disconnected.set()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": CHAT_PATH,
            "raw_path": CHAT_PATH.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-type", b"application/json"), *(headers or [])],
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 8000),
        }
        await self.app(scope, receive, send)
        return result
