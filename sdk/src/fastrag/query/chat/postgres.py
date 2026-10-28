"""PostgreSQL-backed :class:`~fastrag.query.chat.base.ChatStore`.

The schema is self-managed: ``__aenter__`` creates the target Postgres schema,
its three tables, and its three ``ENUM`` types if they are absent — the same
pattern :class:`~fastrag.ingestion.cache.postgres.PostgresCache` established
for the ingestion cache. Nothing here touches Alembic or any shared
declarative ``Base``.

Every driver failure is translated into a
:class:`~fastrag.exceptions.ChatStoreError` (or its transient subclass),
including a failure to *establish* the connection at all (bad password,
unreachable host, DNS) — closed in
``PostgresChatStore.__aenter__``/``_connection``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, Self

import asyncpg
from sqlalchemy import Row, delete, insert, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine
from sqlalchemy.schema import CreateSchema

from fastrag._expiry import new_expiry, should_bump_expiry
from fastrag._ids import created_at_from_id, new_id
from fastrag._sql import Transaction, is_transient, make_engine
from fastrag.exceptions import ChatStoreError, TransientChatStoreError
from fastrag.query.chat._cursor import fetch_cursor_page
from fastrag.query.chat.base import (
    DEFAULT_LIMIT,
    Chat,
    ChatMessage,
    ChatStore,
    ErrorCode,
    FinishReason,
    MessageRole,
    Page,
    Session,
)
from fastrag.query.chat.schema import DEFAULT_SCHEMA, ChatSchema, build_schema


class PostgresChatStore(ChatStore):
    """Persist sessions, chats, and messages on a PostgreSQL server.

    .. code-block:: python

        async with PostgresChatStore(database_url=DATABASE_URL) as chat_store:
            session = await chat_store.create_session(ttl_days=30)

    Args:
        database_url: ``postgresql://user:password@host:port/database``.
            Accepted as a plain ``postgresql://`` URL; the asyncpg driver is
            selected internally.
        schema: The Postgres schema this store's tables live in. Created if
            absent. Defaults to a dedicated schema so the store never
            collides with application tables.
        pool_size: Connections held by this store's engine.
    """

    def __init__(
        self,
        *,
        database_url: str,
        schema: str = DEFAULT_SCHEMA,
        pool_size: int = 5,
    ) -> None:
        """Store configuration only; no I/O happens until ``__aenter__``."""
        self._url = database_url
        self._schema_name = schema
        self._pool_size = pool_size
        self._tables: ChatSchema = build_schema(schema)
        self._engine: AsyncEngine | None = None

    @property
    def engine(self) -> AsyncEngine:
        """The store's underlying async engine.

        The serving layer (``fastrag.serve``) uses this engine — not one it
        owns — for two Postgres-specific things that need a raw engine rather
        than a ``ChatStore`` method: the per-chat advisory lock that
        serializes generations, and the ``/healthz`` connection check.
        Requires the store to have been entered (``__aenter__``), else it
        raises :class:`fastrag.exceptions.ChatStoreError`. Not part of the
        :class:`~fastrag.query.chat.base.ChatStore` contract — a non-Postgres
        store may not offer one (in which case those two serving features are
        unavailable).
        """
        if self._engine is None:
            raise ChatStoreError(
                "engine is only available after PostgresChatStore has been entered"
            )
        return self._engine

    async def __aenter__(self) -> Self:
        """Open the engine and ensure the schema, tables, and enums exist.

        Idempotent: entering again — on this instance, or on a fresh
        instance pointed at the same database — finds everything already
        there and does nothing further.
        """
        if self._engine is None:
            self._engine = make_engine(self._url, pool_size=self._pool_size)

        try:
            async with self._engine.begin() as connection:
                await connection.execute(
                    CreateSchema(self._schema_name, if_not_exists=True)
                )
                await connection.run_sync(
                    self._tables.metadata.create_all, checkfirst=True
                )
        except SQLAlchemyError as error:
            raise _as_chat_store_error(error) from error
        except (OSError, asyncpg.exceptions.PostgresError) as error:
            # Failures during the physical connection (auth, DNS, refused,
            # timeout) surface directly from asyncpg rather than through
            # SQLAlchemy's statement-error path, so they bypass the
            # SQLAlchemyError branch above and need translating here too.
            raise _as_chat_store_error_from_driver(error) from error

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        """Dispose of the engine and its pooled connections."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None

    # -- sessions ----------------------------------------------------------

    async def create_session(
        self,
        *,
        ttl_days: int,
        ip_address: str | None = None,
        country: str | None = None,
    ) -> Session:
        """See :meth:`~fastrag.query.chat.base.ChatStore.create_session`."""
        id = new_id()
        row = {
            "id": id,
            "created_at": created_at_from_id(id),
            "expires_at": new_expiry(ttl_days),
            "ip_address": ip_address,
            "country": country,
        }
        async with self._connection() as connection:
            await connection.execute(insert(self._tables.sessions).values(**row))
        return Session(**row)

    async def get_session(self, session_id: str) -> Session | None:
        """See :meth:`~fastrag.query.chat.base.ChatStore.get_session`."""
        tables = self._tables
        statement = select(tables.sessions).where(
            tables.sessions.c.id == session_id,
            tables.sessions.c.expires_at > datetime.now(UTC),
        )
        async with self._connection() as connection:
            row = (await connection.execute(statement)).first()
        return _session_from_row(row) if row is not None else None

    async def touch_session(
        self, session_id: str, *, ttl_days: int, min_bump_days: int
    ) -> Session | None:
        """See :meth:`~fastrag.query.chat.base.ChatStore.touch_session`."""
        tables = self._tables
        now = datetime.now(UTC)
        async with self._connection() as connection:
            row = (
                await connection.execute(
                    select(tables.sessions).where(
                        tables.sessions.c.id == session_id,
                        tables.sessions.c.expires_at > now,
                    )
                )
            ).first()
            if row is None:
                return None

            session = _session_from_row(row)
            candidate = new_expiry(ttl_days, now)
            if not should_bump_expiry(session.expires_at, candidate, min_bump_days):
                return session

            updated = (
                await connection.execute(
                    update(tables.sessions)
                    .where(tables.sessions.c.id == session_id)
                    .values(expires_at=candidate)
                    .returning(*tables.sessions.c)
                )
            ).first()
            assert updated is not None
            return _session_from_row(updated)

    # -- chats ---------------------------------------------------------

    async def create_chat(self, *, session_id: str, ttl_days: int) -> Chat:
        """See :meth:`~fastrag.query.chat.base.ChatStore.create_chat`."""
        id = new_id()
        row = {
            "id": id,
            "session_id": session_id,
            "created_at": created_at_from_id(id),
            "expires_at": new_expiry(ttl_days),
        }
        async with self._connection() as connection:
            await connection.execute(insert(self._tables.chats).values(**row))
        return Chat(**row)

    async def get_chat(self, chat_id: str, *, session_id: str) -> Chat | None:
        """See :meth:`~fastrag.query.chat.base.ChatStore.get_chat`."""
        tables = self._tables
        statement = select(tables.chats).where(
            tables.chats.c.id == chat_id, tables.chats.c.session_id == session_id
        )
        async with self._connection() as connection:
            row = (await connection.execute(statement)).first()
        return _chat_from_row(row) if row is not None else None

    async def list_chats(
        self,
        session_id: str,
        *,
        before: str | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> Page[Chat]:
        """See :meth:`~fastrag.query.chat.base.ChatStore.list_chats`."""
        tables = self._tables
        statement = select(tables.chats).where(tables.chats.c.session_id == session_id)
        async with self._connection() as connection:
            rows, has_more = await fetch_cursor_page(
                connection, statement, tables.chats.c.id, before, limit
            )
        return Page(items=[_chat_from_row(row) for row in rows], has_more=has_more)

    async def delete_chat(self, chat_id: str, *, session_id: str) -> bool:
        """See :meth:`~fastrag.query.chat.base.ChatStore.delete_chat`."""
        tables = self._tables
        statement = (
            delete(tables.chats)
            .where(
                tables.chats.c.id == chat_id, tables.chats.c.session_id == session_id
            )
            .returning(tables.chats.c.id)
        )
        async with self._connection() as connection:
            row = (await connection.execute(statement)).first()
        return row is not None

    # -- messages ------------------------------------------------------

    async def append_message(
        self,
        chat_id: str,
        *,
        role: MessageRole,
        content: str | None = None,
        sources: list[dict[str, Any]] | None = None,
        finish_reason: FinishReason | None = None,
        error_code: ErrorCode | None = None,
        error_message: str | None = None,
    ) -> ChatMessage:
        """See :meth:`~fastrag.query.chat.base.ChatStore.append_message`."""
        id = new_id()
        row = {
            "id": id,
            "chat_id": chat_id,
            "created_at": created_at_from_id(id),
            "role": role,
            "content": content,
            "sources": sources,
            "finish_reason": finish_reason,
            "error_code": error_code,
            "error_message": error_message,
        }
        async with self._connection() as connection:
            await connection.execute(insert(self._tables.messages).values(**row))
        return ChatMessage(**row)

    async def list_messages(
        self,
        chat_id: str,
        *,
        before: str | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> Page[ChatMessage]:
        """See :meth:`~fastrag.query.chat.base.ChatStore.list_messages`."""
        tables = self._tables
        statement = select(tables.messages).where(tables.messages.c.chat_id == chat_id)
        async with self._connection() as connection:
            rows, has_more = await fetch_cursor_page(
                connection, statement, tables.messages.c.id, before, limit
            )
        return Page(items=[_message_from_row(row) for row in rows], has_more=has_more)

    async def update_last_assistant_message(
        self,
        chat_id: str,
        *,
        content: str | None,
        sources: list[dict[str, Any]] | None = None,
        finish_reason: FinishReason,
        error_code: ErrorCode | None = None,
        error_message: str | None = None,
    ) -> ChatMessage | None:
        """See :meth:`~fastrag.query.chat.base.ChatStore.update_last_assistant_message`."""
        tables = self._tables
        target_id = (
            select(tables.messages.c.id)
            .where(
                tables.messages.c.chat_id == chat_id,
                tables.messages.c.role == "assistant",
            )
            .order_by(tables.messages.c.id.desc())
            .limit(1)
            .scalar_subquery()
        )
        statement = (
            update(tables.messages)
            .where(tables.messages.c.id == target_id)
            .values(
                content=content,
                sources=sources,
                finish_reason=finish_reason,
                error_code=error_code,
                error_message=error_message,
            )
            .returning(*tables.messages.c)
        )
        async with self._connection() as connection:
            row = (await connection.execute(statement)).first()
        return _message_from_row(row) if row is not None else None

    # -- shared statement helpers ----------------------------------------

    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[AsyncConnection]:
        """Yield a transactional connection, with every driver failure translated.

        ``Transaction`` already translates a ``SQLAlchemyError`` raised once
        a connection exists (a malformed statement, a constraint violation).
        The one gap it leaves is a failure during the *physical* connection
        itself — auth, DNS, refused, timeout — which asyncpg raises
        directly, never wrapped as a ``SQLAlchemyError``. This adds that
        second, narrower translation around the same block — including
        whatever the caller's own body does with the yielded connection —
        so no driver exception escapes either path.
        """
        if self._engine is None:
            raise ChatStoreError(
                "PostgresChatStore must be entered with 'async with' before use"
            )
        try:
            async with Transaction(self._engine, _as_chat_store_error) as connection:
                yield connection
        except (OSError, asyncpg.exceptions.PostgresError) as error:
            raise _as_chat_store_error_from_driver(error) from error


def _session_from_row(row: Row) -> Session:
    mapping = row._mapping
    ip_address = mapping["ip_address"]
    return Session(
        id=mapping["id"],
        created_at=mapping["created_at"],
        expires_at=mapping["expires_at"],
        # asyncpg's INET codec returns an ipaddress.IPv4Address/IPv6Address,
        # not the str a caller wrote — round-trip back to str so a caller
        # never has to know that.
        ip_address=str(ip_address) if ip_address is not None else None,
        country=mapping["country"],
    )


def _chat_from_row(row: Row) -> Chat:
    mapping = row._mapping
    return Chat(
        id=mapping["id"],
        session_id=mapping["session_id"],
        created_at=mapping["created_at"],
        expires_at=mapping["expires_at"],
    )


def _message_from_row(row: Row) -> ChatMessage:
    mapping = row._mapping
    sources = mapping["sources"]
    return ChatMessage(
        id=mapping["id"],
        chat_id=mapping["chat_id"],
        created_at=mapping["created_at"],
        role=mapping["role"],
        content=mapping["content"],
        sources=list(sources) if sources is not None else None,
        finish_reason=mapping["finish_reason"],
        error_code=mapping["error_code"],
        error_message=mapping["error_message"],
    )


def _as_chat_store_error(error: SQLAlchemyError) -> ChatStoreError:
    """Mark connection-level failures transient and everything else terminal."""
    transient = is_transient(error)
    message = f"chat store operation failed: {error}"
    return TransientChatStoreError(message) if transient else ChatStoreError(message)


def _as_chat_store_error_from_driver(
    error: OSError | asyncpg.exceptions.PostgresError,
) -> ChatStoreError:
    """Translate a raw driver failure that never became a SQLAlchemyError.

    A failure establishing the physical connection — refused, DNS, timeout,
    a dropped socket — surfaces as a plain ``OSError`` or an
    ``asyncpg.exceptions.PostgresConnectionError``, both worth retrying. Bad
    credentials or a missing database (``InvalidPasswordError``,
    ``InvalidCatalogNameError``) are ``PostgresError`` but not connection
    errors, and retrying them fails identically every time.
    """
    transient = isinstance(error, OSError | asyncpg.exceptions.PostgresConnectionError)
    message = f"chat store operation failed: {error}"
    return TransientChatStoreError(message) if transient else ChatStoreError(message)
