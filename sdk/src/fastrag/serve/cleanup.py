"""Expired-row cleanup, callable from an external scheduler."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

_DELETE_SESSIONS = text(
    "DELETE FROM sessions WHERE id IN "
    "(SELECT id FROM sessions WHERE expires_at < now() LIMIT :batch_size)"
)
_DELETE_CHATS = text(
    "DELETE FROM chats WHERE id IN "
    "(SELECT id FROM chats WHERE expires_at < now() LIMIT :batch_size)"
)


@dataclass(frozen=True)
class CleanupReport:
    """How many expired rows were deleted."""

    sessions: int
    chats: int


async def cleanup_expired(
    engine: AsyncEngine,
    *,
    target: str = "all",
    batch_size: int = 1000,
) -> CleanupReport:
    """Delete expired sessions and chats.

    Deleting a session cascades to its chats and messages; chats are swept
    separately because a chat can go idle while its session stays active.

    Deletes in bounded batches — one transaction per batch — for the same
    reason a single unbounded ``DELETE`` would be wrong: cascading through
    chats and messages in one very large transaction would hold locks and
    bloat WAL for as long as it ran. Sessions are swept before chats,
    mirroring the cascade.

    This function carries the deletion logic but never runs it itself —
    nothing in the request path calls it. It is meant to be invoked on a
    daily cadence by an external scheduler (cron, a Kubernetes ``CronJob``,
    ...); a short script under cron calls it.

    Args:
        engine: The async engine to run the deletes on. The caller typically
            passes a :class:`~fastrag.query.chat.postgres.PostgresChatStore`'s
            ``engine``. The SQL targets the plain table names
            (``sessions``/``chats``) on the engine's search path — a store
            pointed at a non-default schema requires a schema-qualified
            variant (see this task's result file).
        target: ``"sessions"``, ``"chats"``, or ``"all"``.
        batch_size: Rows deleted per transaction.

    Returns:
        A :class:`CleanupReport` of rows deleted per table.
    """
    if target not in ("sessions", "chats", "all"):
        raise ValueError("target must be one of: sessions, chats, all")

    deleted_sessions = 0
    deleted_chats = 0

    if target in ("sessions", "all"):
        deleted_sessions = await _delete_expired_batches(
            engine, _DELETE_SESSIONS, batch_size
        )
    if target in ("chats", "all"):
        deleted_chats = await _delete_expired_batches(engine, _DELETE_CHATS, batch_size)

    return CleanupReport(sessions=deleted_sessions, chats=deleted_chats)


async def _delete_expired_batches(engine: AsyncEngine, sql, batch_size: int) -> int:
    """Delete in bounded batches, one transaction per batch."""
    total = 0
    while True:
        async with engine.begin() as connection:
            result = await connection.execute(sql, {"batch_size": batch_size})
            deleted = result.rowcount

        total += deleted
        if deleted == 0:
            break

    return total
