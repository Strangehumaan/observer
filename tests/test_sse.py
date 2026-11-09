from __future__ import annotations

import pytest

from app.sse import SSEParser, StreamAccumulator
from tests.helpers import DONE, USAGE, delta, sse, usage_chunk


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def accumulator() -> tuple[StreamAccumulator, FakeClock]:
    clock = FakeClock()
    return StreamAccumulator(started=0.0, clock=clock), clock


def test_event_split_across_two_chunks_parses_correctly():
    whole = sse(delta(content="héllo"))
    split_inside_multibyte = whole.index("é".encode()) + 1
    acc, _ = accumulator()

    acc.feed(whole[:split_inside_multibyte])
    assert acc.summary().output_text is None
    acc.feed(whole[split_inside_multibyte:])

    assert acc.summary().output_text == "héllo"


def test_multiple_events_in_one_chunk_parse_correctly():
    acc, _ = accumulator()

    acc.feed(sse(delta(content="one ")) + sse(delta(content="two ")) + sse(delta(content="three")))

    assert acc.summary().output_text == "one two three"


@pytest.mark.parametrize("separator", [b"\n\n", b"\r\n\r\n"])
def test_both_event_separators_are_supported(separator):
    eol = separator[: len(separator) // 2]
    stream = eol.join([b"data: first"]) + separator + b"data: second" + separator

    assert SSEParser().feed(stream) == ["first", "second"]


def test_crlf_separator_split_between_chunks():
    parser = SSEParser()

    assert parser.feed(b"data: first\r") == []
    assert parser.feed(b"\n\r") == []
    assert parser.feed(b"\n") == ["first"]


def test_comment_is_ignored_and_does_not_set_ttft():
    acc, clock = accumulator()

    clock.now = 1.0
    acc.feed(b": OPENROUTER PROCESSING\n\n")
    assert acc.ttft_ms is None

    clock.now = 2.5
    acc.feed(sse(delta(content="hi")))
    assert acc.ttft_ms == 2500.0
    assert acc.summary().output_text == "hi"


def test_empty_content_delta_does_not_set_ttft():
    acc, _ = accumulator()

    acc.feed(sse(delta(role="assistant", content="")))

    assert acc.ttft_ms is None


def test_usage_chunk_with_empty_delta_records_usage_without_changing_output():
    acc, _ = accumulator()

    acc.feed(sse({"choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}]}))
    acc.feed(sse(usage_chunk()))
    acc.feed(DONE)

    summary = acc.summary()
    assert summary.usage == USAGE
    assert summary.output_text == "done"
    assert summary.finish_reason == "stop"


def test_parsing_stops_at_done():
    acc, _ = accumulator()

    acc.feed(sse(delta(content="kept")) + DONE + sse(delta(content="ignored")))

    assert acc.summary().output_text == "kept"


def test_tool_call_argument_deltas_are_joined_by_index():
    acc, clock = accumulator()
    clock.now = 0.25

    first = {"index": 0, "id": "call_a", "type": "function", "function": {"name": "search", "arguments": ""}}
    second = {"index": 1, "id": "call_b", "type": "function", "function": {"name": "fetch", "arguments": '{"u'}}
    acc.feed(sse(delta(tool_calls=[first])))
    acc.feed(sse(delta(tool_calls=[second])))
    acc.feed(sse(delta(tool_calls=[{"index": 0, "function": {"arguments": '{"q":'}}])))
    acc.feed(sse(delta(tool_calls=[{"index": 1, "function": {"arguments": 'rl":1}'}}])))
    acc.feed(sse(delta(tool_calls=[{"index": 0, "function": {"arguments": '"x"}'}}])))

    assert acc.summary().tool_calls == [
        {"id": "call_a", "type": "function", "function": {"name": "search", "arguments": '{"q":"x"}'}},
        {"id": "call_b", "type": "function", "function": {"name": "fetch", "arguments": '{"url":1}'}},
    ]
    assert acc.ttft_ms == 250.0


def test_malformed_event_is_skipped():
    acc, _ = accumulator()

    acc.feed(b"data: {not json\n\n" + sse(delta(content="ok")))

    assert acc.summary().output_text == "ok"
