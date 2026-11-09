from __future__ import annotations

import json

import httpx

from tests.helpers import CHAT_PATH, PLAIN_BODY, completion

PAYLOAD = "<script>alert(1)</script>"
ESCAPED = "&lt;script&gt;alert(1)&lt;/script&gt;"


def body_for(model: str = "test/model", content: str = "hi") -> bytes:
    return json.dumps({"model": model, "messages": [{"role": "user", "content": content}]}).encode()


def ok(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=completion())


async def test_dashboard_escapes_script_in_logged_content(proxy):
    def upstream(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion(PAYLOAD))

    async with proxy(upstream) as h:
        await h.client.post(
            CHAT_PATH,
            content=body_for(model=PAYLOAD, content=PAYLOAD),
            headers={"X-Obs-Session-Id": PAYLOAD},
        )
        [row] = await h.records()
        index = await h.client.get("/")
        detail = await h.client.get(f"/requests/{row['id']}")

    for page in (index.text, detail.text):
        assert PAYLOAD not in page
        assert ESCAPED in page


async def test_dashboard_escapes_unparseable_request_and_upstream_error(proxy):
    def upstream(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=PAYLOAD.encode())

    async with proxy(upstream) as h:
        await h.client.post(CHAT_PATH, content=PAYLOAD.encode())
        [row] = await h.records()
        detail = await h.client.get(f"/requests/{row['id']}")

    assert PAYLOAD not in detail.text
    assert ESCAPED in detail.text


async def test_index_shows_summary_and_filters(proxy):
    async with proxy(ok) as h:
        await h.client.post(CHAT_PATH, content=body_for("a/one"), headers={"X-Obs-Session-Id": "s1"})
        await h.client.post(CHAT_PATH, content=body_for("b/two"), headers={"X-Obs-Session-Id": "s2"})
        await h.records()
        everything = await h.client.get("/")
        filtered = await h.client.get("/", params={"model": "a/one", "status": "success", "session_id": "s1"})
        empty = await h.client.get("/", params={"session_id": "missing"})

    assert ">2</dd>" in everything.text
    assert "$0.002460" in everything.text
    assert ">30</dd>" in everything.text
    assert ">1</dd>" in filtered.text
    assert "$0.001230" in filtered.text
    assert "<td>a/one</td>" in filtered.text
    assert "<td>b/two</td>" not in filtered.text
    assert "No requests found" in empty.text


async def test_index_paginates_newest_first_at_fifty_rows(proxy):
    async with proxy(ok) as h:
        for number in range(51):
            await h.client.post(CHAT_PATH, content=body_for(f"model/{number:02d}"))
        rows = await h.records()
        first = await h.client.get("/")
        second = await h.client.get("/", params={"page": 2})

    assert len(rows) == 51
    assert first.text.count('href="/requests/') == 50
    assert "<td>model/50</td>" in first.text
    assert "<td>model/00</td>" not in first.text
    assert second.text.count('href="/requests/') == 1
    assert "<td>model/00</td>" in second.text
    assert "Page 2 of 2" in second.text


async def test_detail_shows_metadata_messages_and_output(proxy):
    async with proxy(ok) as h:
        await h.client.post(CHAT_PATH, content=PLAIN_BODY)
        [row] = await h.records()
        detail = await h.client.get(f"/requests/{row['id']}")

    assert detail.status_code == 200
    assert row["id"] in detail.text
    assert "hello there" in detail.text
    assert "<pre>hi</pre>" in detail.text


async def test_detail_returns_404_for_unknown_id(proxy):
    async with proxy(ok) as h:
        response = await h.client.get("/requests/does-not-exist")

    assert response.status_code == 404
