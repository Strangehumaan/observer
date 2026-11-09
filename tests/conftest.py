from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable

import httpx
import pytest

from app.config import Settings
from app.main import create_app
from tests.helpers import Harness, UnreadBodyTransport


@pytest.fixture
def proxy(tmp_path) -> Callable[..., Any]:
    """Start the app against a fake upstream handler, with a temporary database."""

    @asynccontextmanager
    async def start(handler: Callable[..., Any], **overrides: Any) -> AsyncIterator[Harness]:
        settings = Settings(
            openrouter_base_url="http://upstream.test/api/v1",
            database_path=str(tmp_path / "test.db"),
            **overrides,
        )
        app = create_app(settings, transport=UnreadBodyTransport(handler))
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://proxy.test") as client:
                yield Harness(app, client)

    return start
