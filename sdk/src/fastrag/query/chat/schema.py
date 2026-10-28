"""The tables behind :class:`~fastrag.query.chat.postgres.PostgresChatStore`.

The same columns, the same ``CHECK`` constraints, the same ``message_role`` /
``finish_reason`` / ``error_code`` enumerations, the same indexes as the
original serving schema — but as a
self-managed schema rather than an Alembic-managed ORM ``Base``:
``PostgresChatStore`` creates its schema, its three tables, and its three
Postgres ``ENUM`` types on ``__aenter__`` if they are absent (the same
pattern ``fastrag.ingestion.cache.schema`` already established for the
ingestion cache). It therefore declares its own ``MetaData`` rather than
attaching to any shared declarative ``Base`` — no Alembic, no shared
``Base``, per this task's own constraints.

Every :class:`~fastrag.query.chat.postgres.PostgresChatStore` instance calls
:func:`build_schema` with its own target schema name at construction time
(pure Python object construction, no I/O), so the same table shapes can live
in as many Postgres schemas as the caller wants — this is what lets the test
suite give every test its own isolated schema.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    MetaData,
    String,
    Table,
    Text,
)
from sqlalchemy.dialects.postgresql import ENUM, INET, JSONB

from fastrag.query.chat.base import ERROR_CODES, FINISH_REASONS, MESSAGE_ROLES

DEFAULT_SCHEMA = "fastrag_chat"
"""Schema name used when a :class:`~fastrag.query.chat.postgres.PostgresChatStore` caller does not pick one."""

# Deterministic constraint/index names, the same convention
# ``fastrag.ingestion.cache.schema.NAMING_CONVENTION``. Duplicated rather
# than imported: this schema must stand on its own, the same way the
# ingestion cache's does.
# Index relations already live in their table schema; omitting the schema
# keeps all current names below PostgreSQL's 63-byte identifier limit.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


@dataclass(frozen=True, slots=True)
class ChatSchema:
    """The three chat tables, bound to one target Postgres schema."""

    metadata: MetaData
    sessions: Table
    chats: Table
    messages: Table


def build_schema(schema: str = DEFAULT_SCHEMA) -> ChatSchema:
    """Build the chat table definitions targeting Postgres schema ``schema``.

    Pure metadata construction: no connection is opened and no statement is
    sent. ``PostgresChatStore.__aenter__`` is what actually creates
    ``schema``, these tables, and the three ``ENUM`` types in the database,
    if absent.
    """
    metadata = MetaData(schema=schema, naming_convention=NAMING_CONVENTION)

    message_role = ENUM(*MESSAGE_ROLES, name="message_role", metadata=metadata)
    finish_reason = ENUM(*FINISH_REASONS, name="finish_reason", metadata=metadata)
    error_code = ENUM(*ERROR_CODES, name="error_code", metadata=metadata)

    sessions = Table(
        "sessions",
        metadata,
        Column("id", String(26), primary_key=True),  # ULID
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("expires_at", DateTime(timezone=True), nullable=False, index=True),
        Column("ip_address", INET, nullable=True),
        Column("country", String(2), nullable=True),
    )

    chats = Table(
        "chats",
        metadata,
        Column("id", String(26), primary_key=True),  # ULID
        Column(
            "session_id",
            String(26),
            ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("expires_at", DateTime(timezone=True), nullable=False, index=True),
        Index("idx_chats_session_id_id", "session_id", "id"),
    )

    messages = Table(
        "messages",
        metadata,
        Column("id", String(26), primary_key=True),  # ULID
        Column(
            "chat_id",
            String(26),
            ForeignKey("chats.id", ondelete="CASCADE"),
            nullable=False,
        ),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("role", message_role, nullable=False),
        Column("content", Text, nullable=True),
        # none_as_null: a Python None must bind to SQL NULL, not the JSON
        # literal `null` — otherwise `sources IS NULL` is false and the
        # shape CHECK below (jsonb_array_length) fails on that scalar.
        Column("sources", JSONB(none_as_null=True), nullable=True),
        Column("finish_reason", finish_reason, nullable=True),
        Column("error_code", error_code, nullable=True),
        Column("error_message", Text, nullable=True),
        Index("idx_messages_chat_id_id", "chat_id", "id"),
        CheckConstraint("length(content) <= 20000", name="content_length_check"),
        CheckConstraint(
            "sources IS NULL OR ("
            "jsonb_array_length(sources) <= 20 "
            "AND octet_length(sources::text) <= 10000"
            ")",
            name="sources_shape_check",
        ),
        CheckConstraint(
            "role <> 'assistant' OR finish_reason IS NOT NULL",
            name="finish_reason_required_for_assistant",
        ),
        # A CHECK passes when it evaluates to NULL, so both constraints below
        # compare the nullable finish_reason with IS [NOT] DISTINCT FROM to
        # force a definite true/false. Written as `finish_reason = 'error'`,
        # they would silently accept any row where finish_reason IS NULL.
        CheckConstraint(
            "content IS NOT NULL OR finish_reason IS NOT DISTINCT FROM 'error'",
            name="content_required_unless_error",
        ),
        CheckConstraint(
            "CASE WHEN finish_reason IS NOT DISTINCT FROM 'error' "
            "THEN error_code IS NOT NULL "
            "ELSE error_code IS NULL AND error_message IS NULL END",
            name="error_fields_match_finish_reason",
        ),
        CheckConstraint(
            "length(error_message) <= 2000", name="error_message_length_check"
        ),
    )

    return ChatSchema(
        metadata=metadata, sessions=sessions, chats=chats, messages=messages
    )
