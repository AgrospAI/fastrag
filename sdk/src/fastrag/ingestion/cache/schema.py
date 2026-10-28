"""The ORM models behind :class:`~fastrag.ingestion.cache.postgres.PostgresCache`.

This schema is owned by the cache itself, not by a repository-wide Alembic
setup: ``PostgresCache`` creates its schema and tables on ``__aenter__`` if
they are absent (see ``docs/specification/API.md``, "Databases schemas are
created automatically"). It therefore declares its own :class:`CacheBase`
rather than attaching to a shared declarative base — doing the latter would
make ``alembic revision --autogenerate`` try to manage these tables too,
which is exactly what must not happen.

The models carry no Postgres schema. Each ``PostgresCache`` engine maps them
onto its own target schema through ``schema_translate_map``, so the same
models serve as many Postgres schemas as the caller wants.

Sources and documents share one three-table layout:

* A **bucket** (:class:`SourceBucket`, :class:`DocumentBucket`) is one cache
  id: the output of one step configuration. A bucket with no snapshots is a
  known-empty result, distinct from a miss.
* A **snapshot** (:class:`SourceSnapshot`, :class:`DocumentSnapshot`) is one
  item of a bucket's output, with its full payload.
* A **record** (:class:`SourceRecord`, :class:`DocumentRecord`) is the
  logical identity shared by every snapshot of the same id. Its
  ``last_accessed`` is the garbage-collection clock for sources.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, LargeBinary, MetaData, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DEFAULT_SCHEMA = "fastrag_cache"
"""Schema name used when a :class:`~fastrag.ingestion.cache.postgres.PostgresCache` caller does not pick one."""

# Deterministic constraint/index names, conventional naming. Duplicated rather
# than imported: this schema must stand on its own.
# Index relations already live in their table schema; omitting the schema
# keeps all current names below PostgreSQL's 63-byte identifier limit.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class CacheBase(DeclarativeBase):
    """Declarative base owning the cache tables, and only them."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {
        str: Text,
        bytes: LargeBinary,
        datetime: DateTime(timezone=True),
        dict[str, Any]: JSONB,
        list[float]: ARRAY(Float),
    }


# -- sources ---------------------------------------------------------------


class SourceBucket(CacheBase):
    """One source step configuration's cache id."""

    __tablename__ = "source_cache"

    id: Mapped[str] = mapped_column(primary_key=True)


class SourceRecord(CacheBase):
    """A source's logical identity and garbage-collection clock.

    The payload columns mirror the latest snapshot written for this id; reads
    serve payloads from :class:`SourceSnapshot`, never from here.
    """

    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(primary_key=True)
    uri: Mapped[str]
    content: Mapped[bytes]
    content_encoding: Mapped[str | None]
    content_type: Mapped[str]
    content_hash: Mapped[str]
    etag: Mapped[str | None]
    last_modified: Mapped[datetime | None]
    # ``metadata`` is reserved on declarative classes.
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata")
    last_validated: Mapped[datetime]
    last_accessed: Mapped[datetime] = mapped_column(index=True)


class SourceSnapshot(CacheBase):
    """A source as one configuration rendered it: bytes and validators."""

    __tablename__ = "source_cache_entries"

    bucket_id: Mapped[str] = mapped_column(
        "source_cache_id",
        ForeignKey(SourceBucket.id, ondelete="CASCADE"),
        primary_key=True,
    )
    source_id: Mapped[str] = mapped_column(
        ForeignKey(SourceRecord.id, ondelete="CASCADE"),
        primary_key=True,
        index=True,
    )
    uri: Mapped[str]
    content: Mapped[bytes]
    content_encoding: Mapped[str | None]
    content_type: Mapped[str]
    content_hash: Mapped[str]
    etag: Mapped[str | None]
    last_modified: Mapped[datetime | None]
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata")
    last_validated: Mapped[datetime]


# -- documents -------------------------------------------------------------


class DocumentBucket(CacheBase):
    """One transformation step configuration's cache id."""

    __tablename__ = "document_cache"

    id: Mapped[str] = mapped_column(primary_key=True)


class DocumentRecord(CacheBase):
    """A document's logical identity, shared by every stage that emits it.

    The payload columns are legacy: they stay blank, because one logical id
    has a different payload per stage (an embedder keeps its input id but
    adds a vector). Payloads live on :class:`DocumentSnapshot`.
    """

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(primary_key=True)
    source_id: Mapped[str] = mapped_column(index=True)
    step_config_hash: Mapped[str]
    content: Mapped[str]
    embedding: Mapped[list[float] | None]
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata")
    last_accessed: Mapped[datetime] = mapped_column(index=True)


class DocumentSnapshot(CacheBase):
    """A document as one stage produced it.

    ``payload`` holds the compressed text and metadata; the vector and the
    header fields live in their own columns so index-only reads skip it.
    """

    __tablename__ = "document_cache_entries"

    bucket_id: Mapped[str] = mapped_column(
        "document_cache_id",
        ForeignKey(DocumentBucket.id, ondelete="CASCADE"),
        primary_key=True,
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey(DocumentRecord.id, ondelete="CASCADE"),
        primary_key=True,
        index=True,
    )
    payload: Mapped[bytes]
    header_metadata: Mapped[dict[str, Any]]
    step_config_hash: Mapped[str]
    embedding: Mapped[list[float] | None]
    source_id: Mapped[str] = mapped_column(index=True)
    last_accessed: Mapped[datetime] = mapped_column(index=True)
