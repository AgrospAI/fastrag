"""ULID-cursor pagination shared by :meth:`~fastrag.query.chat.base.ChatStore.list_chats` and :meth:`~fastrag.query.chat.base.ChatStore.list_messages`.

Keyset-paginates with a Core ``AsyncConnection``
statement — this store has no ORM layer (see ``fastrag.query.chat.postgres``'s
module docstring). One deliberate behaviour choice: a page is returned
newest-first (highest id first) and is **not** reversed into chronological
order before being handed back. ``before`` still narrows to strictly older
rows, so paging still walks backward through time page by page; only the
order *within* one page differs. A caller that wants chronological order for
a chat model builds it via :func:`~fastrag.query.chat.base.to_llm_history`
instead, which is where that reordering belongs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import ColumnElement, Row, Select
from sqlalchemy.ext.asyncio import AsyncConnection


async def fetch_cursor_page(
    connection: AsyncConnection,
    statement: Select[Any],
    id_column: ColumnElement[str],
    before: str | None,
    limit: int,
) -> tuple[Sequence[Row[Any]], bool]:
    """Fetch one ULID-cursor page of ``statement``'s rows, newest-first.

    Args:
        connection: An open connection to execute ``statement`` on.
        statement: A ``SELECT`` already filtered to the desired rows (e.g.
            one chat's messages), not yet ordered or limited.
        id_column: The ULID primary-key column to page on.
        before: When given, only rows with ``id_column < before`` are
            considered.
        limit: Maximum rows to return.

    Returns:
        This page's rows (at most ``limit``, newest-first) and whether an
        older row exists beyond them.
    """
    if before:
        statement = statement.where(id_column < before)

    statement = statement.order_by(id_column.desc()).limit(limit + 1)

    result = await connection.execute(statement)
    rows = result.all()

    has_more = len(rows) > limit
    return rows[:limit], has_more
