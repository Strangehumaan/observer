"""FastAPI app, lifespan and routes."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from http.cookiejar import DefaultCookiePolicy
from math import ceil
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import urlencode

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from jinja2 import Environment, FileSystemLoader

from app.config import Settings, load_settings
from app.db import STATUSES, Database, DatabaseUnavailable, Filters
from app.proxy import ChatCompletionExchange, forward_models
from app.recorder import Recorder

logger = logging.getLogger(__name__)

PAGE_SIZE = 50
UPSTREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)
METADATA_FIELDS = (
    "id",
    "created_at",
    "session_id",
    "generation_id",
    "requested_model",
    "response_model",
    "is_stream",
    "http_status",
    "status",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_tokens",
    "reasoning_tokens",
    "cost",
    "latency_ms",
    "ttft_ms",
)

# Autoescaping is what makes logged prompts and outputs safe to render.
templates = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "templates"), autoescape=True
)
templates.filters["cost"] = lambda value: "—" if value is None else f"${value:.6f}"
templates.filters["ms"] = lambda value: "—" if value is None else f"{value:,.0f} ms"
templates.filters["count"] = lambda value: "—" if value is None else f"{value:,}"


def _render(name: str, **context: Any) -> HTMLResponse:
    return HTMLResponse(templates.get_template(name).render(**context))


def _pretty(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)


def _pretty_json(text: str | None) -> str | None:
    if text is None:
        return None
    try:
        return _pretty(json.loads(text))
    except ValueError:
        return text


def _messages(request_json: str | None) -> list[dict[str, str | None]] | None:
    """Split a logged request into display messages, or None if it has no message list."""
    try:
        parsed = json.loads(request_json) if request_json else None
    except ValueError:
        return None
    messages = parsed.get("messages") if isinstance(parsed, dict) else None
    if not isinstance(messages, list):
        return None
    display = []
    for message in messages:
        if not isinstance(message, dict):
            display.append({"role": "unknown", "content": _pretty(message), "extra": None})
            continue
        extra = {key: value for key, value in message.items() if key not in ("role", "content")}
        content = message.get("content")
        display.append(
            {
                "role": str(message.get("role", "unknown")),
                "content": None if content is None else _pretty(content),
                "extra": _pretty(extra) if extra else None,
            }
        )
    return display


def _page_url(filters: Filters, page: int) -> str:
    query = {
        "model": filters.model,
        "status": filters.status,
        "session_id": filters.session_id,
        "page": page,
    }
    return "/?" + urlencode({key: value for key, value in query.items() if value})


def create_app(
    settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None
) -> FastAPI:
    resolved = settings or load_settings()
    database = Database(resolved.database_path)
    recorder = Recorder(database)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        client = httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT, transport=transport)
        # The client is shared, so it must not carry one caller's cookies to the next.
        client.cookies.jar.set_policy(DefaultCookiePolicy(allowed_domains=[]))
        app.state.http = client
        try:
            await database.connect()
        except Exception:
            logger.exception("Could not open %s; requests will not be logged", resolved.database_path)
        recorder.start()
        try:
            yield
        finally:
            await recorder.stop()
            await database.close()
            await client.aclose()

    app = FastAPI(
        title="OpenRouter Observability Proxy",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = resolved
    app.state.database = database
    app.state.recorder = recorder

    @app.exception_handler(DatabaseUnavailable)
    async def database_unavailable(_: Request, __: DatabaseUnavailable) -> HTMLResponse:
        return HTMLResponse("Log database is unavailable. Check the server logs.", 503)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        body = await request.body()
        return ChatCompletionExchange(request, body, app.state.http, resolved, recorder)

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        return await forward_models(request, app.state.http, resolved)

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "database": "ok" if database.available else "unavailable",
                "queued_records": recorder.queue.qsize(),
                "dropped_records": recorder.dropped,
            }
        )

    @app.get("/", response_class=HTMLResponse)
    async def index(
        model: str | None = None,
        status: str | None = None,
        session_id: str | None = None,
        page: int = Query(1, ge=1),
    ) -> HTMLResponse:
        filters = Filters(model=model or None, status=status or None, session_id=session_id or None)
        summary = await database.summary(filters)
        rows = await database.list_requests(filters, PAGE_SIZE, (page - 1) * PAGE_SIZE)
        page_count = max(1, ceil(summary["request_count"] / PAGE_SIZE))
        return _render(
            "index.html",
            summary=summary,
            rows=rows,
            filters=filters,
            models=await database.list_models(),
            statuses=STATUSES,
            page=page,
            page_count=page_count,
            previous_url=_page_url(filters, page - 1) if page > 1 else None,
            next_url=_page_url(filters, page + 1) if page < page_count else None,
        )

    @app.get("/requests/{request_id}", response_class=HTMLResponse)
    async def detail(request_id: str) -> HTMLResponse:
        row = await database.get_request(request_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Request not found")
        return _render(
            "detail.html",
            row=row,
            metadata=[(field, row[field]) for field in METADATA_FIELDS],
            messages=_messages(row["request_json"]),
            raw_request=row["request_json"],
            output_text=row["output_text"],
            tool_calls=_pretty_json(row["tool_calls_json"]),
            error=_pretty_json(row["error_json"]),
        )

    return app


app = create_app()


def main() -> None:
    settings = app.state.settings
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
