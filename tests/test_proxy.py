from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app import sse as sse_module
from app.db import RequestRecord
from tests.helpers import (
    CHAT_PATH,
    DONE,
    PLAIN_BODY,
    SECRET,
    STREAM_BODY,
    USAGE,
    ScriptedStream,
    completion,
    delta,
    sse,
    stream_upstream,
    usage_chunk,
)

AUTH = {"Authorization": f"Bearer {SECRET}"}
MIDSTREAM_ERROR = {
    "error": {"code": 502, "message": "Provider disconnected"},
    "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}],
}


def json_upstream(payload: dict, status: int = 200, headers: dict | None = None):
    return lambda _: httpx.Response(status, json=payload, headers=headers)


# --- Brief test 1: non-streaming success -----------------------------------------------


async def test_non_streaming_success_records_usage_and_cost(proxy):
    upstream = json_upstream(completion(), headers={"X-Generation-Id": "gen-abc"})
    async with proxy(upstream) as h:
        response = await h.client.post(
            CHAT_PATH, content=PLAIN_BODY, headers={"X-Obs-Session-Id": "session-1"}
        )
        [row] = await h.records()

    assert response.status_code == 200
    assert response.json() == completion()
    assert response.headers["x-generation-id"] == "gen-abc"
    assert response.headers["x-obs-request-id"] == row["id"]
    assert row["status"] == "success"
    assert row["http_status"] == 200
    assert row["is_stream"] == 0
    assert row["requested_model"] == "test/model"
    assert row["response_model"] == "test/model-1"
    assert row["finish_reason"] == "stop"
    assert row["session_id"] == "session-1"
    assert row["generation_id"] == "gen-abc"
    assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (10, 5, 15)
    assert (row["cached_tokens"], row["reasoning_tokens"]) == (3, 2)
    assert row["cost"] == pytest.approx(0.00123)
    assert row["output_text"] == "hello there"
    assert row["ttft_ms"] is None
    assert row["latency_ms"] > 0


# --- Brief test 2: request body and headers ---------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        b'{ "z":1.0e0,   "model":"x",\n"stream":false,"a":"\\u00e9" }',
        b"this is not json at all",
    ],
)
async def test_request_body_reaches_upstream_byte_for_byte(proxy, body):
    received = []

    def upstream(request: httpx.Request) -> httpx.Response:
        received.append(request.content)
        return httpx.Response(200, json=completion())

    async with proxy(upstream) as h:
        response = await h.client.post(CHAT_PATH, content=body)
        [row] = await h.records()

    assert received == [body]
    assert response.status_code == 200
    assert row["request_json"] == body.decode()


async def test_upstream_request_headers_are_filtered(proxy):
    seen = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        seen["url"] = str(request.url)
        return httpx.Response(200, json=completion())

    headers = {
        "X-Obs-Session-Id": "session-1",
        "Accept-Encoding": "gzip",
        "HTTP-Referer": "https://example.test",
        "Keep-Alive": "timeout=5",
    }
    async with proxy(upstream) as h:
        await h.client.post(CHAT_PATH, content=PLAIN_BODY, headers=headers)

    assert seen["url"] == "http://upstream.test/api/v1/chat/completions"
    assert seen["accept-encoding"] == "identity"
    assert seen["http-referer"] == "https://example.test"
    assert seen["host"] == "upstream.test"
    assert "x-obs-session-id" not in seen
    assert "keep-alive" not in seen


# --- Brief test 3: Authorization ---------------------------------------------------------


async def test_authorization_is_forwarded_but_never_stored(proxy):
    seen = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        return httpx.Response(200, json=completion())

    async with proxy(upstream) as h:
        await h.client.post(CHAT_PATH, content=PLAIN_BODY, headers=AUTH)
        [row] = await h.records()

    assert seen == [f"Bearer {SECRET}"]
    assert SECRET not in json.dumps(row)


async def test_configured_key_is_used_only_without_client_authorization(proxy):
    seen = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json=completion())

    async with proxy(upstream, openrouter_api_key="server-key") as h:
        await h.client.post(CHAT_PATH, content=PLAIN_BODY)
        await h.client.post(CHAT_PATH, content=PLAIN_BODY, headers=AUTH)

    assert seen == ["Bearer server-key", f"Bearer {SECRET}"]


# --- Brief test 4: streaming bytes --------------------------------------------------------


async def test_streaming_bytes_reach_the_client_unchanged(proxy):
    # Deliberately awkward chunk boundaries, a comment, CRLF and non-ASCII bytes.
    whole = sse(delta(content="héllo"))
    chunks = [
        b": OPENROUTER PROCESSING\n\n",
        whole[:9],
        whole[9:] + sse(delta(content=" world")),
        b"data: not-json\r\n\r\n",
        sse(usage_chunk()),
        DONE,
    ]
    upstream = stream_upstream(
        ScriptedStream(chunks),
        headers={"content-length": "999", "cache-control": "public", "X-Generation-Id": "gen-s"},
    )
    async with proxy(upstream) as h:
        result = await h.stream()
        [row] = await h.records()

    assert result.chunks == chunks
    assert result.status == 200
    assert result.headers["content-type"] == "text/event-stream"
    assert result.headers["cache-control"] == "no-cache"
    assert result.headers["x-accel-buffering"] == "no"
    assert result.headers["x-generation-id"] == "gen-s"
    assert result.headers["x-obs-request-id"] == row["id"]
    assert "content-length" not in result.headers
    assert row["status"] == "success"
    assert row["is_stream"] == 1
    assert row["output_text"] == "héllo world"
    assert row["generation_id"] == "gen-s"
    assert row["ttft_ms"] is not None


# --- Brief tests 5 and 6 through the proxy ----------------------------------------------


async def test_split_and_combined_events_are_logged_through_the_proxy(proxy):
    first = sse(delta(content="a"))
    chunks = [first[:5], first[5:] + sse(delta(content="b")) + sse(delta(content="c")), DONE]
    async with proxy(stream_upstream(ScriptedStream(chunks))) as h:
        await h.stream()
        [row] = await h.records()

    assert row["output_text"] == "abc"


# --- Brief test 7 through the proxy --------------------------------------------------------


async def test_comment_only_stream_leaves_ttft_unset(proxy):
    chunks = [b": OPENROUTER PROCESSING\n\n", DONE]
    async with proxy(stream_upstream(ScriptedStream(chunks))) as h:
        await h.stream()
        [row] = await h.records()

    assert row["ttft_ms"] is None
    assert row["output_text"] is None


# --- Brief test 8 through the proxy --------------------------------------------------------


async def test_usage_chunk_records_usage_and_does_not_change_output(proxy):
    chunks = [sse(delta(content="final answer")), sse(usage_chunk()), DONE]
    async with proxy(stream_upstream(ScriptedStream(chunks))) as h:
        await h.stream()
        [row] = await h.records()

    assert row["output_text"] == "final answer"
    assert row["finish_reason"] == "stop"
    assert row["total_tokens"] == USAGE["total_tokens"]
    assert row["cost"] == pytest.approx(USAGE["cost"])


# --- Brief test 9 ----------------------------------------------------------------------


async def test_mid_stream_error_is_recorded_as_upstream_error(proxy):
    chunks = [sse(delta(content="partial")), sse(MIDSTREAM_ERROR), DONE]
    async with proxy(stream_upstream(ScriptedStream(chunks))) as h:
        result = await h.stream()
        [row] = await h.records()

    assert result.status == 200
    assert result.chunks == chunks
    assert row["http_status"] == 200
    assert row["status"] == "upstream_error"
    assert row["finish_reason"] == "error"
    assert row["output_text"] == "partial"
    assert json.loads(row["error_json"]) == MIDSTREAM_ERROR["error"]


async def test_lost_upstream_connection_mid_stream_is_upstream_error(proxy):
    stream = ScriptedStream([sse(delta(content="partial"))], error=httpx.ReadError("reset"))
    async with proxy(stream_upstream(stream)) as h:
        with pytest.raises(httpx.ReadError):
            await h.stream()  # The exception makes the server drop the client connection.
        [row] = await h.records()

    assert stream.closed
    assert row["status"] == "upstream_error"
    assert row["output_text"] == "partial"
    assert "ReadError" in row["error_json"]


# --- Brief test 10 ---------------------------------------------------------------------


@pytest.mark.parametrize("body", [PLAIN_BODY, STREAM_BODY])
async def test_upstream_429_is_returned_unchanged_and_logged(proxy, body):
    payload = b'{"error": {"code": 429, "message": "Rate limited"}}'

    def upstream(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            content=payload,
            headers={"content-type": "application/json", "retry-after": "7"},
        )

    async with proxy(upstream) as h:
        response = await h.client.post(CHAT_PATH, content=body)
        [row] = await h.records()

    assert response.status_code == 429
    assert response.content == payload
    assert response.headers["retry-after"] == "7"
    assert response.headers["x-obs-request-id"] == row["id"]
    assert row["http_status"] == 429
    assert row["status"] == "upstream_error"
    assert json.loads(row["error_json"]) == {"code": 429, "message": "Rate limited"}


# --- Brief test 11 ---------------------------------------------------------------------


async def test_unreachable_upstream_returns_502(proxy):
    def upstream(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    async with proxy(upstream) as h:
        response = await h.client.post(CHAT_PATH, content=PLAIN_BODY)
        [row] = await h.records()

    assert response.status_code == 502
    assert response.json()["error"]["code"] == 502
    assert response.json()["error"]["message"]
    assert response.headers["x-obs-request-id"] == row["id"]
    assert row["http_status"] == 502
    assert row["status"] == "proxy_error"
    assert row["requested_model"] == "test/model"


# --- Brief test 12 ---------------------------------------------------------------------


async def test_client_disconnect_closes_upstream_and_stores_partial_output(proxy):
    stream = ScriptedStream([sse(delta(content="partial "))], hang=True)
    async with proxy(stream_upstream(stream)) as h:
        result = await h.stream(disconnect_after_chunks=1)
        rows = await h.records()

    assert result.chunks == [sse(delta(content="partial "))]
    assert stream.closed
    assert len(rows) == 1
    assert rows[0]["status"] == "client_disconnected"
    assert rows[0]["output_text"] == "partial "


async def test_completed_stream_is_not_reported_as_a_disconnect(proxy):
    stream = ScriptedStream([sse(delta(content="whole")), DONE])
    async with proxy(stream_upstream(stream)) as h:
        await h.stream()
        rows = await h.records()

    assert stream.closed
    assert [row["status"] for row in rows] == ["success"]


# --- Brief test 13 through the proxy -------------------------------------------------------


async def test_tool_call_deltas_are_stored_joined(proxy):
    start = {"index": 0, "id": "call_1", "type": "function", "function": {"name": "f", "arguments": '{"a"'}}
    rest = {"index": 0, "function": {"arguments": ":1}"}}
    chunks = [sse(delta(tool_calls=[start])), sse(delta(tool_calls=[rest])), DONE]
    async with proxy(stream_upstream(ScriptedStream(chunks))) as h:
        await h.stream()
        [row] = await h.records()

    assert json.loads(row["tool_calls_json"]) == [
        {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": '{"a":1}'}}
    ]
    assert row["ttft_ms"] is not None


# --- Brief test 14 ---------------------------------------------------------------------


async def test_database_failure_does_not_break_the_response(proxy, monkeypatch):
    async with proxy(json_upstream(completion())) as h:
        async def fail(_: RequestRecord) -> None:
            raise RuntimeError("database is broken")

        monkeypatch.setattr(h.app.state.database, "insert", fail)
        response = await h.client.post(CHAT_PATH, content=PLAIN_BODY)
        await h.app.state.recorder.queue.join()

    assert response.status_code == 200
    assert response.json() == completion()


async def test_recorder_failure_does_not_break_the_stream(proxy, monkeypatch):
    chunks = [sse(delta(content="ok")), DONE]
    async with proxy(stream_upstream(ScriptedStream(chunks))) as h:
        def fail(_: RequestRecord) -> None:
            raise RuntimeError("recorder is broken")

        monkeypatch.setattr(h.app.state.recorder, "submit", fail)
        result = await h.stream()

    assert result.chunks == chunks


async def test_parser_failure_does_not_interrupt_forwarding(proxy, monkeypatch):
    def fail(*_: object) -> None:
        raise RuntimeError("parser is broken")

    monkeypatch.setattr(sse_module.StreamAccumulator, "feed", fail)
    chunks = [sse(delta(content="one")), sse(delta(content="two")), DONE]
    async with proxy(stream_upstream(ScriptedStream(chunks))) as h:
        result = await h.stream()
        [row] = await h.records()

    assert result.chunks == chunks
    assert row["status"] == "success"


async def test_unopenable_database_does_not_break_the_response(proxy, tmp_path):
    (tmp_path / "test.db").mkdir()  # A directory where the database file should be.
    async with proxy(json_upstream(completion())) as h:
        response = await h.client.post(CHAT_PATH, content=PLAIN_BODY)
        await h.app.state.recorder.queue.join()
        health = await h.client.get("/health")
        dashboard = await h.client.get("/")

    assert response.status_code == 200
    assert health.json()["database"] == "unavailable"
    assert dashboard.status_code == 503


# --- Brief test 15 ---------------------------------------------------------------------


async def test_log_content_false_stores_metrics_but_no_content(proxy):
    async with proxy(json_upstream(completion("very private output")), log_content=False) as h:
        await h.client.post(CHAT_PATH, content=PLAIN_BODY)
        [row] = await h.records()

    assert row["total_tokens"] == 15
    assert row["cost"] == pytest.approx(0.00123)
    assert row["request_json"] is None
    assert row["output_text"] is None
    assert row["tool_calls_json"] is None


async def test_log_content_false_streaming_and_error_message_omitted(proxy):
    error = {"error": {"code": 400, "type": "bad", "message": "echoes the private prompt"}}
    chunks = [sse(delta(content="secret stream")), sse(error), DONE]
    async with proxy(stream_upstream(ScriptedStream(chunks)), log_content=False) as h:
        await h.stream()
        [row] = await h.records()

    assert row["status"] == "upstream_error"
    assert row["output_text"] is None
    assert json.loads(row["error_json"]) == {"code": 400, "type": "bad"}
    assert "private" not in json.dumps(row)


# --- Models, health and recording -----------------------------------------------------


async def test_models_are_passed_through(proxy):
    seen = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(200, json={"data": [{"id": "a/b"}]})

    async with proxy(upstream) as h:
        response = await h.client.get("/v1/models?supported=tools", headers=AUTH)
        rows = await h.records()

    assert seen["url"] == "http://upstream.test/api/v1/models?supported=tools"
    assert seen["authorization"] == f"Bearer {SECRET}"
    assert response.json() == {"data": [{"id": "a/b"}]}
    assert "x-obs-request-id" in response.headers
    assert rows == []


async def test_unreachable_upstream_for_models_returns_502(proxy):
    def upstream(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    async with proxy(upstream) as h:
        response = await h.client.get("/v1/models")

    assert response.status_code == 502
    assert response.json()["error"]["code"] == 502


async def test_full_queue_drops_records_and_health_reports_it(proxy, monkeypatch):
    def full(_: RequestRecord) -> None:
        raise asyncio.QueueFull

    async with proxy(json_upstream(completion())) as h:
        assert h.app.state.recorder.queue.maxsize == 10_000
        monkeypatch.setattr(h.app.state.recorder.queue, "put_nowait", full)
        response = await h.client.post(CHAT_PATH, content=PLAIN_BODY)
        health = await h.client.get("/health")

    assert response.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["dropped_records"] == 1
