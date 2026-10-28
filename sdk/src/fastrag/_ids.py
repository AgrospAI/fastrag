"""ULID minting and timestamp derivation shared by the chat store and serve.

Mints the ULIDs :class:`~fastrag.query.chat.postgres.PostgresChatStore` uses
as session, chat, and message row ids — the same ids ``fastrag.serve`` mints
for a generation's stored assistant message and, for the session id
specifically, reuses as the bearer/observer credential — and derives
``created_at`` timestamps from them. Private to ``fastrag`` — the one
underscore-prefixed top-level module that package-private code may share
across packages; nothing here is public API and it is not documented in the
reference.
"""

from __future__ import annotations

from datetime import datetime

from ulid import ULID


def new_id() -> str:
    """Mint a new ULID for a session, chat, or message row."""
    return str(ULID())


def created_at_from_id(id: str) -> datetime:
    """Derive a row's ``created_at`` from its ULID.

    ``ORDER BY id`` and ``ORDER BY created_at`` must always agree, so the
    timestamp is read out of the ULID rather than taken from ``now()``.
    """
    return ULID.from_str(id).datetime
