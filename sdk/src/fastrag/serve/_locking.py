"""Session-level Postgres advisory locks serializing generations per chat."""

from __future__ import annotations

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from fastrag.serve._errors import ApiError

_LOCK_SQL = text("SELECT pg_try_advisory_lock(hashtextextended(:chat_id, 0))")
_UNLOCK_SQL = text("SELECT pg_advisory_unlock(hashtextextended(:chat_id, 0))")
_PING_SQL = text("SELECT 1")


async def acquire_chat_lock(engine: AsyncEngine, chat_id: str) -> AsyncConnection:
    """Acquire a session-level advisory lock serializing generations per chat.

    Must be acquired before the prompt is persisted and before the SSE
    stream opens — lock contention is a plain HTTP ``409``, not a mid-stream
    failure. Callers must release via :func:`release_chat_lock` in a
    ``finally``, once the generation this lock guards has actually finished
    — never ``pg_advisory_xact_lock``. Uses a dedicated connection in
    autocommit mode, held for the lock's lifetime, so no long-lived
    idle-in-transaction connection sits open while a generation runs.

    The lock holds a bare ``AsyncEngine`` connection, opened from the
    ``ChatStore``'s engine — the app owns no database engine of its own; the
    store's engine is the only one in the process.
    """
    connection = await engine.connect()
    connection = await connection.execution_options(isolation_level="AUTOCOMMIT")

    result = await connection.execute(_LOCK_SQL, {"chat_id": chat_id})
    acquired = result.scalar()

    if not acquired:
        await connection.close()
        raise ApiError(409, "CONFLICT", "A generation is already running for this chat")

    return connection


class LockKeepAlive:
    """Ping a lock's connection every ``interval`` seconds until stopped.

    The connection sits idle for as long as the generation it guards runs. A
    proxy or firewall between the app and Postgres may close an idle
    connection sooner than ``generation_timeout_s``, which drops the session
    the lock lives on and fails the unlock at the end.
    """

    def __init__(self, connection: AsyncConnection, interval: float) -> None:
        self._stopped = asyncio.Event()
        self._task = asyncio.ensure_future(self._ping(connection, interval))

    async def _ping(self, connection: AsyncConnection, interval: float) -> None:
        while not self._stopped.is_set():
            try:
                async with asyncio.timeout(interval):
                    await self._stopped.wait()
            except TimeoutError:
                await connection.execute(_PING_SQL)

    async def stop(self) -> None:
        """Stop pinging, letting a ping in flight finish so the connection is free.

        Raises whatever failed a ping.
        """
        self._stopped.set()
        await self._task


async def release_chat_lock(
    connection: AsyncConnection,
    chat_id: str,
    *,
    keepalive: LockKeepAlive | None = None,
) -> None:
    try:
        if keepalive is not None:
            await keepalive.stop()
        await connection.execute(_UNLOCK_SQL, {"chat_id": chat_id})
    finally:
        await connection.close()
