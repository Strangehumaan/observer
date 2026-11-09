"""SQLite schema and queries."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    session_id TEXT,
    generation_id TEXT,
    requested_model TEXT,
    response_model TEXT,
    is_stream INTEGER NOT NULL,
    http_status INTEGER,
    status TEXT NOT NULL,
    finish_reason TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    cached_tokens INTEGER,
    reasoning_tokens INTEGER,
    cost REAL,
    latency_ms REAL NOT NULL,
    ttft_ms REAL,
    request_json TEXT,
    output_text TEXT,
    tool_calls_json TEXT,
    error_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_created_at ON requests (created_at);
CREATE INDEX IF NOT EXISTS idx_requests_session_id ON requests (session_id);
CREATE INDEX IF NOT EXISTS idx_requests_requested_model ON requests (requested_model);
"""

STATUSES = ("success", "upstream_error", "client_disconnected", "proxy_error")
ERROR_STATUSES = ("upstream_error", "proxy_error")


class DatabaseUnavailable(Exception):
    """The SQLite file could not be opened."""


@dataclass
class RequestRecord:
    id: str
    created_at: str
    session_id: str | None = None
    generation_id: str | None = None
    requested_model: str | None = None
    response_model: str | None = None
    is_stream: int = 0
    http_status: int | None = None
    status: str = "success"
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost: float | None = None
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    request_json: str | None = None
    output_text: str | None = None
    tool_calls_json: str | None = None
    error_json: str | None = None


_COLUMNS = [field.name for field in fields(RequestRecord)]
_INSERT = (
    f"INSERT INTO requests ({', '.join(_COLUMNS)}) "
    f"VALUES ({', '.join(':' + column for column in _COLUMNS)})"
)


@dataclass(frozen=True)
class Filters:
    model: str | None = None
    status: str | None = None
    session_id: str | None = None

    def where(self) -> tuple[str, list[str]]:
        clauses: list[str] = []
        params: list[str] = []
        for column, value in (
            ("requested_model", self.model),
            ("status", self.status),
            ("session_id", self.session_id),
        ):
            if value:
                clauses.append(f"{column} = ?")
                params.append(value)
        return (" WHERE " + " AND ".join(clauses) if clauses else ""), params


class Database:
    def __init__(self, path: str) -> None:
        self._path = path
        self._connection: aiosqlite.Connection | None = None

    @property
    def available(self) -> bool:
        return self._connection is not None

    async def connect(self) -> None:
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        connection = await aiosqlite.connect(self._path)
        try:
            connection.row_factory = aiosqlite.Row
            await connection.execute("PRAGMA journal_mode=WAL")
            await connection.execute("PRAGMA synchronous=NORMAL")
            await connection.executescript(SCHEMA)
            await connection.commit()
        except BaseException:
            await connection.close()
            raise
        self._connection = connection

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    def _require_connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise DatabaseUnavailable
        return self._connection

    async def insert(self, record: RequestRecord) -> None:
        connection = self._require_connection()
        await connection.execute(_INSERT, asdict(record))
        await connection.commit()

    async def summary(self, filters: Filters) -> dict[str, Any]:
        where, params = filters.where()
        placeholders = ", ".join("?" for _ in ERROR_STATUSES)
        query = (
            "SELECT COUNT(*) AS request_count, "
            "COALESCE(SUM(cost), 0) AS total_cost, "
            "COALESCE(SUM(total_tokens), 0) AS total_tokens, "
            f"COALESCE(SUM(status IN ({placeholders})), 0) AS error_count, "
            f"AVG(latency_ms) AS average_latency_ms FROM requests{where}"
        )
        cursor = await self._require_connection().execute(query, [*ERROR_STATUSES, *params])
        row = await cursor.fetchone()
        await cursor.close()
        return dict(row)

    async def list_requests(self, filters: Filters, limit: int, offset: int) -> list[dict[str, Any]]:
        where, params = filters.where()
        query = f"SELECT * FROM requests{where} ORDER BY created_at DESC, id LIMIT ? OFFSET ?"
        cursor = await self._require_connection().execute(query, [*params, limit, offset])
        rows = await cursor.fetchall()
        await cursor.close()
        return [dict(row) for row in rows]

    async def get_request(self, request_id: str) -> dict[str, Any] | None:
        cursor = await self._require_connection().execute(
            "SELECT * FROM requests WHERE id = ?", [request_id]
        )
        row = await cursor.fetchone()
        await cursor.close()
        return dict(row) if row is not None else None

    async def list_models(self) -> list[str]:
        cursor = await self._require_connection().execute(
            "SELECT DISTINCT requested_model FROM requests "
            "WHERE requested_model IS NOT NULL ORDER BY requested_model"
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [row[0] for row in rows]
