"""Upstream forwarding for the OpenAI-compatible routes."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

import httpx
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import Receive, Scope, Send

from app.config import Settings
from app.db import RequestRecord
from app.recorder import Recorder
from app.sse import CompletionSummary, StreamAccumulator, loads_object, normalize_error

logger = logging.getLogger(__name__)

RawHeaders = list[tuple[bytes, bytes]]

HOP_BY_HOP = frozenset(
    {
        b"connection",
        b"keep-alive",
        b"proxy-authenticate",
        b"proxy-authorization",
        b"te",
        b"trailer",
        b"transfer-encoding",
        b"upgrade",
    }
)
REQUEST_HEADERS_TO_DROP = HOP_BY_HOP | {
    b"host",
    b"content-length",
    b"accept-encoding",
    b"x-obs-session-id",
}
# Uvicorn adds its own Date and Server, so forwarding them would duplicate both.
RESPONSE_HEADERS_TO_DROP = HOP_BY_HOP | {
    b"content-length",
    b"date",
    b"server",
    b"x-obs-request-id",
}
STREAM_HEADERS_TO_REPLACE = {b"content-type", b"cache-control", b"x-accel-buffering"}
REQUEST_ID_HEADER = b"x-obs-request-id"
ERROR_FIELDS_SAFE_WITHOUT_CONTENT = ("code", "type")
MAX_STORED_ERROR_BODY = 2000


def filter_headers(raw: Iterable[tuple[bytes, bytes]], drop: frozenset[bytes]) -> RawHeaders:
    """Lowercase header names and remove `drop` plus anything named by Connection."""
    headers = [(name.lower(), value) for name, value in raw]
    listed = {
        token.strip().lower()
        for name, value in headers
        if name == b"connection"
        for token in value.split(b",")
    }
    return [(name, value) for name, value in headers if name not in drop and name not in listed]


def upstream_headers(raw: Iterable[tuple[bytes, bytes]], api_key: str) -> RawHeaders:
    headers = filter_headers(raw, REQUEST_HEADERS_TO_DROP)
    headers.append((b"accept-encoding", b"identity"))
    if api_key and not any(name == b"authorization" for name, _ in headers):
        headers.append((b"authorization", f"Bearer {api_key}".encode()))
    return headers


def upstream_url(settings: Settings, path: str, query: bytes) -> str:
    url = f"{settings.openrouter_base_url.rstrip('/')}/{path}"
    return f"{url}?{query.decode('latin-1')}" if query else url


def gateway_error_body(exc: Exception) -> bytes:
    message = f"Could not reach OpenRouter: {type(exc).__name__}"
    return json.dumps({"error": {"code": 502, "message": message}}).encode()


def summarize_body(body: bytes, status_code: int) -> CompletionSummary:
    """Extract log fields from a non-streaming response body."""
    payload = loads_object(body)
    summary = CompletionSummary()
    if payload is not None:
        if isinstance(payload.get("model"), str):
            summary.response_model = payload["model"]
        if isinstance(payload.get("usage"), dict):
            summary.usage = payload["usage"]
        if payload.get("error"):
            summary.error = normalize_error(payload["error"])
        choices = payload.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            _summarize_choice(choices[0], summary)
    if summary.error is None and status_code >= 400:
        text = body.decode("utf-8", "replace")[:MAX_STORED_ERROR_BODY]
        summary.error = {"code": status_code, "message": text}
    return summary


def _summarize_choice(choice: dict[str, Any], summary: CompletionSummary) -> None:
    if isinstance(choice.get("finish_reason"), str):
        summary.finish_reason = choice["finish_reason"]
    if summary.error is None and (choice.get("error") or summary.finish_reason == "error"):
        summary.error = normalize_error(choice.get("error") or "finish_reason: error")
    message = choice.get("message")
    if isinstance(message, dict):
        if isinstance(message.get("content"), str):
            summary.output_text = message["content"] or None
        if isinstance(message.get("tool_calls"), list):
            summary.tool_calls = message["tool_calls"] or None


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _nested_int(usage: dict[str, Any], parent: str, key: str) -> int | None:
    details = usage.get(parent)
    return _int_or_none(details.get(key)) if isinstance(details, dict) else None


def _cost(usage: dict[str, Any]) -> float | None:
    value = usage.get("cost")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _without_content(error: dict[str, Any]) -> dict[str, Any]:
    return {key: error[key] for key in ERROR_FIELDS_SAFE_WITHOUT_CONTENT if key in error}


async def _wait_for_disconnect(receive: Receive) -> None:
    while (await receive())["type"] != "http.disconnect":
        pass


class ChatCompletionExchange(Response):
    """Forwards one chat completion and writes exactly one log record.

    The whole exchange runs inside the ASGI response so a disconnect is seen at any
    point, including while waiting for the upstream's first byte. `StreamingResponse`
    can't guarantee this: a disconnect during a send leaves its generator suspended,
    so the upstream stays open and the generator's `finally` doesn't run.
    """

    def __init__(
        self,
        request: Request,
        body: bytes,
        client: httpx.AsyncClient,
        settings: Settings,
        recorder: Recorder,
    ) -> None:
        super().__init__()
        self._monotonic_start = time.perf_counter()
        self._created_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        self._request_id = str(uuid.uuid4())
        self._body = body
        self._client = client
        self._settings = settings
        self._recorder = recorder
        self._url = upstream_url(settings, "chat/completions", request.scope.get("query_string", b""))
        self._headers = upstream_headers(request.headers.raw, settings.openrouter_api_key)
        self._session_id = request.headers.get("x-obs-session-id")
        self._request = loads_object(body) or {}

        self._http_status: int | None = None
        self._generation_id: str | None = None
        self._summary = CompletionSummary()
        self._accumulator: StreamAccumulator | None = None
        self._saw_sse_response = False
        self._parser_failed = False
        self._failure: dict[str, Any] | None = None
        self._proxy_failed = False
        self._client_gone = False
        self._response_started = False
        self._body_complete = False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        relay = asyncio.create_task(self._relay(send))
        disconnect = asyncio.create_task(_wait_for_disconnect(receive))
        try:
            await asyncio.wait({relay, disconnect}, return_when=asyncio.FIRST_COMPLETED)
            # Servers also report a disconnect once the last byte is sent, so that isn't a client abort.
            if disconnect.done() and not self._body_complete:
                self._client_gone = True
                relay.cancel()
            await asyncio.wait({relay})
        finally:
            disconnect.cancel()
            relay.cancel()
            try:
                await asyncio.gather(relay, disconnect, return_exceptions=True)
            finally:
                self._record()
        if not relay.cancelled() and relay.exception() is not None:
            # Dropping the connection tells the client this stream was truncated.
            raise relay.exception()

    async def _relay(self, send: Send) -> None:
        try:
            outbound = self._client.build_request(
                "POST", self._url, content=self._body, headers=self._headers
            )
            upstream = await self._client.send(outbound, stream=True)
        except Exception as exc:
            logger.warning("Upstream request failed: %s", type(exc).__name__)
            await self._send_gateway_error(send, exc)
            return

        try:
            self._http_status = upstream.status_code
            self._generation_id = upstream.headers.get("x-generation-id")
            content_type = upstream.headers.get("content-type", "")
            if content_type.lower().startswith("text/event-stream"):
                await self._relay_stream(upstream, send)
            else:
                await self._relay_body(upstream, send)
        except OSError:
            self._client_gone = True
        except Exception as exc:
            if not self._response_started:
                logger.warning("Upstream response failed: %s", type(exc).__name__)
                await self._send_gateway_error(send, exc)
                return
            self._proxy_failed = not isinstance(exc, httpx.HTTPError)
            self._failure = {"message": f"Stream interrupted: {type(exc).__name__}"}
            raise
        finally:
            await self._close(upstream)

    async def _relay_stream(self, upstream: httpx.Response, send: Send) -> None:
        self._saw_sse_response = True
        self._accumulator = StreamAccumulator(self._monotonic_start)
        headers = filter_headers(
            upstream.headers.raw, RESPONSE_HEADERS_TO_DROP | STREAM_HEADERS_TO_REPLACE
        )
        headers += [
            (b"content-type", b"text/event-stream"),
            (b"cache-control", b"no-cache"),
            (b"x-accel-buffering", b"no"),
        ]
        await self._start_response(send, upstream.status_code, headers)
        async for chunk in upstream.aiter_raw():
            # Parsed before sending, so a disconnect during the send still logs this chunk.
            self._observe(chunk)
            await send({"type": "http.response.body", "body": chunk, "more_body": True})
        self._body_complete = True
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def _relay_body(self, upstream: httpx.Response, send: Send) -> None:
        body = b"".join([chunk async for chunk in upstream.aiter_raw()])
        self._summary = summarize_body(body, upstream.status_code)
        headers = filter_headers(upstream.headers.raw, RESPONSE_HEADERS_TO_DROP)
        headers.append((b"content-length", str(len(body)).encode()))
        await self._send_complete(send, upstream.status_code, headers, body)

    async def _send_gateway_error(self, send: Send, exc: Exception) -> None:
        body = gateway_error_body(exc)
        self._proxy_failed = True
        self._http_status = 502
        self._failure = loads_object(body)["error"]
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ]
        await self._send_complete(send, 502, headers, body)

    async def _send_complete(
        self, send: Send, status: int, headers: RawHeaders, body: bytes
    ) -> None:
        await self._start_response(send, status, headers)
        self._body_complete = True
        await send({"type": "http.response.body", "body": body, "more_body": False})

    async def _start_response(self, send: Send, status: int, headers: RawHeaders) -> None:
        self._response_started = True
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [*headers, (REQUEST_ID_HEADER, self._request_id.encode())],
            }
        )

    def _observe(self, chunk: bytes) -> None:
        if self._parser_failed or self._accumulator is None:
            return
        try:
            self._accumulator.feed(chunk)
        except Exception:
            self._parser_failed = True
            logger.exception("Stream parsing stopped for request %s", self._request_id)

    @staticmethod
    async def _close(upstream: httpx.Response) -> None:
        try:
            await upstream.aclose()
        except Exception:
            logger.exception("Could not close the upstream response")

    def _record(self) -> None:
        try:
            self._recorder.submit(self._build_record())
        except Exception:
            logger.exception("Could not record request %s", self._request_id)

    def _build_record(self) -> RequestRecord:
        summary = self._accumulator.summary() if self._accumulator else self._summary
        error = self._failure or summary.error
        usage = summary.usage or {}
        log_content = self._settings.log_content
        model = self._request.get("model")
        record = RequestRecord(
            id=self._request_id,
            created_at=self._created_at,
            session_id=self._session_id,
            generation_id=self._generation_id,
            requested_model=model if isinstance(model, str) else None,
            response_model=summary.response_model,
            is_stream=int(self._request.get("stream") is True or self._saw_sse_response),
            http_status=self._http_status,
            status=self._status(error),
            finish_reason=summary.finish_reason,
            prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
            completion_tokens=_int_or_none(usage.get("completion_tokens")),
            total_tokens=_int_or_none(usage.get("total_tokens")),
            cached_tokens=_nested_int(usage, "prompt_tokens_details", "cached_tokens"),
            reasoning_tokens=_nested_int(usage, "completion_tokens_details", "reasoning_tokens"),
            cost=_cost(usage),
            latency_ms=(time.perf_counter() - self._monotonic_start) * 1000,
            ttft_ms=self._accumulator.ttft_ms if self._accumulator else None,
        )
        if error is not None:
            stored_error = error if log_content else _without_content(error)
            record.error_json = json.dumps(stored_error, ensure_ascii=False)
        if log_content:
            record.request_json = self._body.decode("utf-8", "replace")
            record.output_text = summary.output_text
            if summary.tool_calls:
                record.tool_calls_json = json.dumps(summary.tool_calls, ensure_ascii=False)
        return record

    def _status(self, error: dict[str, Any] | None) -> str:
        if self._proxy_failed:
            return "proxy_error"
        http_failed = self._http_status is not None and not 200 <= self._http_status < 300
        if error is not None or http_failed:
            return "upstream_error"
        return "client_disconnected" if self._client_gone else "success"


async def forward_models(
    request: Request, client: httpx.AsyncClient, settings: Settings
) -> Response:
    """Pass `GET /models` through to OpenRouter. These calls are not logged."""
    request_id = str(uuid.uuid4()).encode()
    try:
        outbound = client.build_request(
            "GET",
            upstream_url(settings, "models", request.scope.get("query_string", b"")),
            headers=upstream_headers(request.headers.raw, settings.openrouter_api_key),
        )
        upstream = await client.send(outbound, stream=True)
        try:
            body = b"".join([chunk async for chunk in upstream.aiter_raw()])
        finally:
            await upstream.aclose()
    except Exception as exc:
        logger.warning("Upstream models request failed: %s", type(exc).__name__)
        response = Response(gateway_error_body(exc), 502, media_type="application/json")
        response.raw_headers.append((REQUEST_ID_HEADER, request_id))
        return response

    response = Response(body, upstream.status_code)
    response.raw_headers = [
        *filter_headers(upstream.headers.raw, RESPONSE_HEADERS_TO_DROP),
        (b"content-length", str(len(body)).encode()),
        (REQUEST_ID_HEADER, request_id),
    ]
    return response
