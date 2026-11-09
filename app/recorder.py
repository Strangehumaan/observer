"""Background log writer, so proxied requests never wait for the database."""

from __future__ import annotations

import asyncio
import logging

from app.db import Database, RequestRecord

logger = logging.getLogger(__name__)

MAX_QUEUE_SIZE = 10_000
DRAIN_TIMEOUT_SECONDS = 5.0


class Recorder:
    def __init__(self, database: Database, max_queue_size: int = MAX_QUEUE_SIZE) -> None:
        self._database = database
        self.queue: asyncio.Queue[RequestRecord] = asyncio.Queue(maxsize=max_queue_size)
        self.dropped = 0
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._write_records())

    def submit(self, record: RequestRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except asyncio.QueueFull:
            self.dropped += 1

    async def stop(self, timeout: float = DRAIN_TIMEOUT_SECONDS) -> None:
        if self._task is None:
            return
        try:
            await asyncio.wait_for(self.queue.join(), timeout)
        except TimeoutError:
            logger.warning("Shutdown drain timed out with %d records unwritten", self.queue.qsize())
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)

    async def _write_records(self) -> None:
        while True:
            record = await self.queue.get()
            try:
                await self._database.insert(record)
            except Exception:
                logger.exception("Could not write log record %s", record.id)
            finally:
                self.queue.task_done()
