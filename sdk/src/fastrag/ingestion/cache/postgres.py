"""PostgreSQL-backed :class:`~fastrag.ingestion.cache.base.CacheStore`.

The schema is self-managed: ``__aenter__`` creates the target Postgres schema
and its tables if they are absent (``docs/specification/API.md``: "Databases
schemas are created automatically if they don't exist ... The user can
customize the target database schema"). The cache owns its own schema and
Alembic history is untouched by it.

Statements are written against the ORM models in
:mod:`fastrag.ingestion.cache.schema`; the engine's ``schema_translate_map``
places them in the configured schema.
"""

from __future__ import annotations

import asyncio
import math
import zlib
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Self

import asyncpg
from sqlalchemy import (
    ColumnElement,
    Executable,
    Row,
    Select,
    bindparam,
    delete,
    exists,
    select,
    text,
    tuple_,
    update,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine
from sqlalchemy.orm import InstrumentedAttribute
from sqlalchemy.schema import CreateIndex, CreateSchema

from fastrag._sql import Transaction, is_transient, make_engine
from fastrag.exceptions import CacheError, TransientCacheError
from fastrag.ingestion.cache._batching import ReadBatcher
from fastrag.ingestion.cache._payloads import (
    deduplicate,
    document_payload,
    pack_payload,
    unpack_payload,
)
from fastrag.ingestion.cache.base import CacheStore, _DocumentHeader, _SourceHeader
from fastrag.ingestion.cache.schema import (
    DEFAULT_SCHEMA,
    CacheBase,
    DocumentBucket,
    DocumentRecord,
    DocumentSnapshot,
    SourceBucket,
    SourceRecord,
    SourceSnapshot,
)
from fastrag.ingestion.models import Source
from fastrag.models import Document

type _Bucket = type[SourceBucket] | type[DocumentBucket]
type _Snapshot = type[SourceSnapshot] | type[DocumentSnapshot]

# Snapshot columns read back into a ``Source``. Headers skip the bytes.
_SOURCE_HEADER_COLUMNS = (
    SourceSnapshot.source_id.label("id"),
    SourceSnapshot.uri,
    SourceSnapshot.content_type,
    SourceSnapshot.content_hash,
    SourceSnapshot.etag,
    SourceSnapshot.last_modified,
    SourceSnapshot.metadata_.label("metadata"),
    SourceSnapshot.last_validated,
)
_SOURCE_COLUMNS = (
    *_SOURCE_HEADER_COLUMNS,
    SourceSnapshot.content,
    SourceSnapshot.content_encoding,
)
# Snapshot columns read back into a ``Document``. Headers skip text and vector.
_DOCUMENT_HEADER_COLUMNS = (
    DocumentSnapshot.document_id.label("id"),
    DocumentSnapshot.source_id,
    DocumentSnapshot.step_config_hash,
    DocumentSnapshot.header_metadata,
)
_DOCUMENT_COLUMNS = (
    DocumentSnapshot.document_id.label("id"),
    DocumentSnapshot.payload,
    DocumentSnapshot.embedding,
)


class PostgresCache(CacheStore):
    """Cache pipeline sources and documents on a PostgreSQL server.

    .. code-block:: python

        async with PostgresCache(url=DATABASE_URL) as cache:
            sources = SitemapCrawler(url=SITEMAP_URL, cache=cache)

    Args:
        url: ``postgresql://user:password@host:port/database``. Accepted as a
            plain ``postgresql://`` URL; the asyncpg driver is selected
            internally.
        schema: The Postgres schema this cache's tables live in. Created if
            absent. Defaults to a dedicated schema so the cache never collides
            with application tables.
        pool_size: Connections held by this cache's engine.
        read_batch_size: Maximum buckets per bulk read (default 128).
            Concurrent single reads are coalesced up to this limit.
        read_batch_delay: Seconds to collect concurrent reads (default 0.001).
            Zero yields once to the event loop; size 1 disables grouping.
    """

    def __init__(
        self,
        *,
        url: str,
        schema: str = DEFAULT_SCHEMA,
        pool_size: int = 5,
        read_batch_size: int = 128,
        read_batch_delay: float = 0.001,
    ) -> None:
        """Store configuration only; no I/O happens until ``__aenter__``."""
        if read_batch_size < 1 or read_batch_size > 10000:
            raise ValueError("read_batch_size must be between 1 and 10000")
        if read_batch_delay < 0 or not math.isfinite(read_batch_delay):
            raise ValueError("read_batch_delay must be finite and non-negative")
        self._read_batch_size = read_batch_size
        self._source_reads = ReadBatcher(
            self.get_source_entries_many, read_batch_size, read_batch_delay
        )
        self._document_reads = ReadBatcher(
            self.get_document_entries_many, read_batch_size, read_batch_delay
        )
        self._url = url
        self._schema_name = schema
        self._pool_size = pool_size
        self._engine: AsyncEngine | None = None

    async def __aenter__(self) -> Self:
        """Open the engine and ensure the schema and tables exist.

        Idempotent: entering again — on this instance, or on a fresh
        instance pointed at the same database — finds the schema and tables
        already there. Missing indexes are installed on existing tables.
        """
        if self._engine is None:
            self._engine = make_engine(
                self._url, pool_size=self._pool_size, schema=self._schema_name
            )

        try:
            async with self._engine.connect() as connection:
                connection = await connection.execution_options(
                    isolation_level="AUTOCOMMIT"
                )
                if await self._schema_ready(connection):
                    return self
            async with self._engine.begin() as connection:
                await self._create_or_upgrade_schema(connection)
        except SQLAlchemyError as error:
            raise _as_cache_error(error) from error
        except (OSError, asyncpg.exceptions.PostgresError) as error:
            # Failures during the physical connection (auth, DNS, refused,
            # timeout) surface directly from asyncpg rather than through
            # SQLAlchemy's statement-error path, so they bypass the
            # SQLAlchemyError branch above and need translating here too.
            raise _as_cache_error_from_driver(error) from error

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        """Dispose of the engine and its pooled connections."""
        await asyncio.gather(self._source_reads.close(), self._document_reads.close())
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None

    # -- sources ------------------------------------------------------------

    async def get_source_entries(self, source_cache_id: str) -> list[Source] | None:
        """See :meth:`~fastrag.ingestion.cache.base.CacheStore.get_source_entries`."""
        return await self._source_reads.get(source_cache_id)

    async def get_source_entries_many(
        self, source_cache_ids: Sequence[str]
    ) -> dict[str, list[Source] | None]:
        """Fetch source buckets in bounded batches, one SQL statement per batch."""
        buckets = await self._read_source_buckets(source_cache_ids, _SOURCE_COLUMNS)
        return {
            bucket_id: None if rows is None else [_source_from_row(row) for row in rows]
            for bucket_id, rows in buckets.items()
        }

    async def get_source_headers_many(
        self, cache_ids: Sequence[str]
    ) -> dict[str, list[Source] | None]:
        """Retrieve only identities, validators and routing metadata for indexing."""
        buckets = await self._read_source_buckets(cache_ids, _SOURCE_HEADER_COLUMNS)
        return {
            bucket_id: None
            if rows is None
            else [
                _SourceHeader(
                    id=row.id,
                    uri=row.uri,
                    content=b"",
                    content_type=row.content_type,
                    content_hash=row.content_hash,
                    etag=row.etag,
                    last_modified=row.last_modified,
                    metadata=dict(row.metadata),
                    last_validated=row.last_validated,
                    cached=True,
                    _store=self,
                    _bucket=bucket_id,
                )
                for row in rows
            ]
            for bucket_id, rows in buckets.items()
        }

    async def get_bucket_sources(
        self, source_cache_id: str, source_ids: Sequence[str]
    ) -> list[Source]:
        """See :meth:`~fastrag.ingestion.cache.base.CacheStore.get_bucket_sources`."""
        if not source_ids:
            return []
        statement = select(*_SOURCE_COLUMNS).where(
            SourceSnapshot.bucket_id == source_cache_id,
            SourceSnapshot.source_id.in_(source_ids),
        )
        async with self._connection() as connection:
            rows = (await connection.execute(statement)).all()
        return [_source_from_row(row) for row in rows]

    async def put_source_entries(
        self, source_cache_id: str, sources: Sequence[Source]
    ) -> None:
        """See :meth:`~fastrag.ingestion.cache.base.CacheStore.put_source_entries`."""
        now = datetime.now(UTC)
        sources = deduplicate(sources)
        payloads = [_source_payload(source, now) for source in sources]

        async with self._connection() as connection:
            await _ensure_bucket(connection, SourceBucket, source_cache_id)
            await _upsert(
                connection,
                SourceRecord,
                [
                    {"id": source.id, **payload, "last_accessed": now}
                    for source, payload in zip(sources, payloads, strict=True)
                ],
                update=[
                    SourceRecord.uri,
                    SourceRecord.content,
                    SourceRecord.content_type,
                    SourceRecord.content_encoding,
                    SourceRecord.content_hash,
                    SourceRecord.etag,
                    SourceRecord.last_modified,
                    SourceRecord.metadata_,
                    SourceRecord.last_validated,
                    SourceRecord.last_accessed,
                ],
            )
            await _replace_bucket(
                connection,
                SourceSnapshot,
                source_cache_id,
                [
                    {"bucket_id": source_cache_id, "source_id": source.id, **payload}
                    for source, payload in zip(sources, payloads, strict=True)
                ],
            )

    # -- documents ----------------------------------------------------------

    async def get_document_entries(
        self, document_cache_id: str
    ) -> list[Document] | None:
        """See :meth:`~fastrag.ingestion.cache.base.CacheStore.get_document_entries`."""
        return await self._document_reads.get(document_cache_id)

    async def get_document_entries_many(
        self, document_cache_ids: Sequence[str]
    ) -> dict[str, list[Document] | None]:
        """Fetch document buckets in bounded batches, one SQL statement per batch."""
        buckets = await self._read_document_buckets(
            document_cache_ids, _DOCUMENT_COLUMNS
        )
        return {
            bucket_id: None
            if rows is None
            else [
                Document(
                    **unpack_payload(row.payload),
                    embedding=None if row.embedding is None else list(row.embedding),
                    cached=True,
                )
                for row in rows
            ]
            for bucket_id, rows in buckets.items()
        }

    async def get_document_headers_many(
        self, cache_ids: Sequence[str]
    ) -> dict[str, list[Document] | None]:
        """Retrieve snapshot identifiers and routing metadata without large payloads."""
        buckets = await self._read_document_buckets(cache_ids, _DOCUMENT_HEADER_COLUMNS)
        return {
            bucket_id: None
            if rows is None
            else [
                _DocumentHeader(
                    id=row.id,
                    source_id=row.source_id,
                    content="",
                    step_config_hash=row.step_config_hash,
                    metadata=dict(row.header_metadata),
                    cached=True,
                    _store=self,
                    _bucket=bucket_id,
                )
                for row in rows
            ]
            for bucket_id, rows in buckets.items()
        }

    async def put_document_entries(
        self, document_cache_id: str, documents: Sequence[Document]
    ) -> None:
        """See :meth:`~fastrag.ingestion.cache.base.CacheStore.put_document_entries`."""
        now = datetime.now(UTC)
        documents = deduplicate(documents)

        async with self._connection() as connection:
            await _ensure_bucket(connection, DocumentBucket, document_cache_id)
            # Records only register the logical id; payloads live on snapshots.
            await _upsert(
                connection,
                DocumentRecord,
                [
                    {
                        "id": document.id,
                        "source_id": document.source_id,
                        "step_config_hash": "",
                        "content": "",
                        "embedding": None,
                        "metadata_": {},
                        "last_accessed": now,
                    }
                    for document in documents
                ],
                update=[DocumentRecord.source_id, DocumentRecord.last_accessed],
            )
            await _replace_bucket(
                connection,
                DocumentSnapshot,
                document_cache_id,
                [
                    {
                        "bucket_id": document_cache_id,
                        "document_id": document.id,
                        "payload": pack_payload(document_payload(document)),
                        "header_metadata": document.metadata,
                        "step_config_hash": document.step_config_hash,
                        "embedding": document.embedding,
                        "source_id": document.source_id,
                        "last_accessed": now,
                    }
                    for document in documents
                ],
            )

    # -- administration -----------------------------------------------------

    async def clear_sources(self) -> int:
        """See :meth:`~fastrag.ingestion.cache.base.CacheStore.clear_sources`."""
        async with self._connection() as connection:
            return await _clear(connection, SourceRecord, SourceBucket)

    async def clear_documents(self) -> int:
        """See :meth:`~fastrag.ingestion.cache.base.CacheStore.clear_documents`."""
        async with self._connection() as connection:
            return await _clear(connection, DocumentRecord, DocumentBucket)

    async def clear(self) -> int:
        """See :meth:`~fastrag.ingestion.cache.base.CacheStore.clear`."""
        async with self._connection() as connection:
            sources_removed = await _clear(connection, SourceRecord, SourceBucket)
            documents_removed = await _clear(connection, DocumentRecord, DocumentBucket)
        return sources_removed + documents_removed

    async def delete_sources(self, source_ids: Sequence[str]) -> int:
        """See :meth:`~fastrag.ingestion.cache.base.CacheStore.delete_sources`."""
        if not source_ids:
            return 0
        async with self._connection() as connection:
            return await _delete_sources(connection, SourceRecord.id.in_(source_ids))

    async def delete_documents(self, source_ids: Sequence[str]) -> int:
        """See :meth:`~fastrag.ingestion.cache.base.CacheStore.delete_documents`."""
        if not source_ids:
            return 0
        async with self._connection() as connection:
            removed = await _delete_document_snapshots(
                connection, DocumentSnapshot.source_id.in_(source_ids)
            )
            return removed + await _delete_orphan_documents(
                connection, DocumentRecord.source_id.in_(source_ids)
            )

    async def collect_garbage(self, max_age: timedelta) -> int:
        """See :meth:`~fastrag.ingestion.cache.base.CacheStore.collect_garbage`."""
        if max_age <= timedelta(0):
            raise ValueError(f"max_age must be positive, got {max_age!r}")

        cutoff = datetime.now(UTC) - max_age
        async with self._connection() as connection:
            sources_removed = await _delete_sources(
                connection, SourceRecord.last_accessed < cutoff
            )
            documents_removed = await _delete_document_snapshots(
                connection, DocumentSnapshot.last_accessed < cutoff
            )
            # Replacements can orphan old logical document ids without leaving
            # a snapshot. Collect those too, but never delete a referenced id.
            documents_removed += await _delete_orphan_documents(
                connection, DocumentRecord.last_accessed < cutoff
            )
        return sources_removed + documents_removed

    # -- reads --------------------------------------------------------------

    async def _read_source_buckets(
        self, bucket_ids: Sequence[str], columns: Sequence[Any]
    ) -> dict[str, list[Row] | None]:
        """Read source snapshots by bucket, refreshing their records' GC clock."""

        def statement(batch: list[str]) -> Select:
            touched = SourceRecord.id.in_(
                select(SourceSnapshot.source_id).where(
                    SourceSnapshot.bucket_id.in_(batch)
                )
            )
            return (
                select(SourceBucket.id.label("bucket_id"), *columns)
                .select_from(SourceBucket)
                .outerjoin(SourceSnapshot, SourceSnapshot.bucket_id == SourceBucket.id)
                .where(SourceBucket.id.in_(batch))
                .order_by(SourceBucket.id, SourceSnapshot.source_id)
                .add_cte(_touch(SourceRecord, touched, SourceRecord.id))
            )

        return await self._read_buckets(bucket_ids, statement)

    async def _read_document_buckets(
        self, bucket_ids: Sequence[str], columns: Sequence[Any]
    ) -> dict[str, list[Row] | None]:
        """Read document snapshots by bucket, refreshing their GC clock."""

        def statement(batch: list[str]) -> Select:
            touched = DocumentSnapshot.bucket_id.in_(batch)
            return (
                select(DocumentBucket.id.label("bucket_id"), *columns)
                .select_from(DocumentBucket)
                .outerjoin(
                    DocumentSnapshot, DocumentSnapshot.bucket_id == DocumentBucket.id
                )
                .where(DocumentBucket.id.in_(batch))
                .order_by(DocumentBucket.id, DocumentSnapshot.document_id)
                .add_cte(
                    _touch(
                        DocumentSnapshot,
                        touched,
                        DocumentSnapshot.bucket_id,
                        DocumentSnapshot.document_id,
                    )
                )
            )

        return await self._read_buckets(bucket_ids, statement)

    async def _read_buckets(
        self,
        bucket_ids: Sequence[str],
        statement: Callable[[list[str]], Executable],
    ) -> dict[str, list[Row] | None]:
        """Group rows by ``bucket_id``, telling misses from known-empty buckets.

        ``statement`` outer-joins buckets to their snapshots, so an empty
        bucket yields one row whose ``id`` is ``NULL``, and a missing bucket
        yields none at all.
        """
        unique_ids = list(dict.fromkeys(bucket_ids))
        buckets: dict[str, list[Row] | None] = dict.fromkeys(unique_ids)
        if not unique_ids:
            return buckets
        async with self._connection() as connection:
            for offset in range(0, len(unique_ids), self._read_batch_size):
                batch = unique_ids[offset : offset + self._read_batch_size]
                for row in (await connection.execute(statement(batch))).all():
                    rows = buckets[row.bucket_id]
                    if rows is None:
                        rows = buckets[row.bucket_id] = []
                    if row.id is not None:
                        rows.append(row)
        return buckets

    # -- schema management --------------------------------------------------

    async def _schema_ready(self, connection: AsyncConnection) -> bool:
        """Check tables, indexes and upgrades in one startup round trip."""
        return (await self._inspect_schema(connection)).is_current

    async def _inspect_schema(self, connection: AsyncConnection) -> _SchemaState:
        rows = (
            await connection.execute(_SCHEMA_STATE_SQL, {"schema": self._schema_name})
        ).all()
        return _SchemaState(
            columns={
                (row.table_name, row.column_name): row
                for row in rows
                if not row.indexed
            },
            indexed=frozenset(
                (row.table_name, row.column_name) for row in rows if row.indexed
            ),
        )

    async def _create_or_upgrade_schema(self, connection: AsyncConnection) -> None:
        """Create what is missing and upgrade legacy layouts, under a schema lock."""
        await connection.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:schema))"),
            {"schema": self._schema_name},
        )
        await connection.execute(CreateSchema(self._schema_name, if_not_exists=True))
        await connection.run_sync(CacheBase.metadata.create_all, checkfirst=True)
        state = await self._inspect_schema(connection)
        await self._upgrade_document_snapshots(connection, state)
        await self._upgrade_source_encoding(connection, state)
        await self._upgrade_source_snapshots(connection, state)
        for index, column in _INDEXES.items():
            if column not in state.indexed:
                # The schema lock serializes creation; a conflicting unrelated
                # relation must raise, not hide a missing index.
                await connection.execute(CreateIndex(index))

    def _quoted_schema(self, connection: AsyncConnection) -> str:
        return connection.dialect.identifier_preparer.quote_schema(self._schema_name)

    async def _upgrade_document_snapshots(
        self, connection: AsyncConnection, state: _SchemaState
    ) -> None:
        """Upgrade stage snapshots, preserving only unambiguous legacy vectors."""
        schema = self._quoted_schema(connection)
        snapshots = f"{schema}.document_cache_entries"
        payload = state.column(DocumentSnapshot, "payload")
        has_headers = state.has(DocumentSnapshot, "header_metadata") and state.has(
            DocumentSnapshot, "step_config_hash"
        )
        has_source = state.has(DocumentSnapshot, "source_id")
        has_access = state.has(DocumentSnapshot, "last_accessed")

        if not has_headers:
            await connection.execute(
                text(
                    f"ALTER TABLE {snapshots} "
                    "ADD COLUMN IF NOT EXISTS header_metadata JSONB NOT NULL DEFAULT '{}'::jsonb, "
                    "ADD COLUMN IF NOT EXISTS step_config_hash TEXT NOT NULL DEFAULT ''"
                )
            )
        if not has_source:
            await connection.execute(
                text(f"ALTER TABLE {snapshots} ADD COLUMN source_id TEXT")
            )
        if not has_access:
            await connection.execute(
                text(
                    f"ALTER TABLE {snapshots} "
                    "ADD COLUMN last_accessed TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP"
                )
            )

        if payload is None:
            # Oldest layout: payloads lived on the shared logical record.
            await connection.execute(
                text(
                    f"ALTER TABLE {snapshots} "
                    "ADD COLUMN payload BYTEA, ADD COLUMN embedding DOUBLE PRECISION[]"
                )
            )
            # A shared record cannot say which embedder produced its vector:
            # keep only buckets whose vector belongs to exactly one embedder.
            await connection.execute(
                text(f"""
                DELETE FROM {schema}.document_cache c
                WHERE c.id NOT LIKE '%embedder:%' OR EXISTS (
                    SELECT 1 FROM {snapshots} e
                    WHERE e.document_cache_id = c.id AND e.document_id IN (
                        SELECT document_id FROM {snapshots}
                        WHERE document_cache_id LIKE '%embedder:%'
                        GROUP BY document_id HAVING count(*) > 1
                    )
                )
            """)
            )
            await connection.execute(
                text(f"""
                UPDATE {snapshots} e
                SET payload = convert_to(jsonb_build_object('id', d.id, 'source_id', d.source_id,
                    'content', d.content, 'step_config_hash', d.step_config_hash,
                    'metadata', d.metadata)::text, 'UTF8'), embedding = d.embedding
                FROM {schema}.documents d WHERE e.document_id = d.id
            """)
            )
            await connection.execute(
                text(f"ALTER TABLE {snapshots} ALTER COLUMN payload SET NOT NULL")
            )
            await connection.execute(update(DocumentRecord).values(embedding=None))
        elif payload.udt_name != "bytea":
            # Intermediate layout: uncompressed JSONB payloads.
            await connection.execute(
                text(
                    f"ALTER TABLE {snapshots} "
                    "ALTER COLUMN payload TYPE BYTEA USING convert_to(payload::text, 'UTF8')"
                )
            )
        elif has_source and has_access and has_headers:
            return

        # Recompress payloads and fill the header columns from them. Only the
        # small payloads cross the network, never the vectors. The surrounding
        # transaction makes the upgrade atomic.
        query = select(
            DocumentSnapshot.bucket_id,
            DocumentSnapshot.document_id,
            DocumentSnapshot.payload,
        )
        async for rows in _pages(
            connection, query, DocumentSnapshot.bucket_id, DocumentSnapshot.document_id
        ):
            upgraded = []
            for bucket_id, document_id, payload in rows:
                decoded = unpack_payload(payload)
                upgraded.append(
                    {
                        "bucket_id": bucket_id,
                        "document_id": document_id,
                        "payload": pack_payload(decoded),
                        "source_id": decoded["source_id"],
                        "header_metadata": decoded["metadata"],
                        "step_config_hash": decoded["step_config_hash"],
                        "last_accessed": datetime.now(UTC),
                    }
                )
            statement = insert(DocumentSnapshot).values(upgraded)
            await connection.execute(
                statement.on_conflict_do_update(
                    index_elements=[
                        DocumentSnapshot.bucket_id,
                        DocumentSnapshot.document_id,
                    ],
                    set_=_excluded(
                        statement,
                        [
                            DocumentSnapshot.payload,
                            DocumentSnapshot.source_id,
                            DocumentSnapshot.header_metadata,
                            DocumentSnapshot.step_config_hash,
                        ],
                    ),
                )
            )
        await connection.execute(
            text(f"ALTER TABLE {snapshots} ALTER COLUMN source_id SET NOT NULL")
        )
        await connection.execute(
            update(DocumentRecord).values(
                content="", metadata_={}, step_config_hash="", embedding=None
            )
        )

    async def _upgrade_source_encoding(
        self, connection: AsyncConnection, state: _SchemaState
    ) -> None:
        """Compress legacy source bytes once, without changing content hashes."""
        if state.has(SourceRecord, "content_encoding"):
            return
        schema = self._quoted_schema(connection)
        await connection.execute(
            text(f"ALTER TABLE {schema}.sources ADD COLUMN content_encoding TEXT")
        )
        # Bind names must differ from column names in an executemany UPDATE.
        statement = (
            update(SourceRecord)
            .where(SourceRecord.id == bindparam("source_key"))
            .values(content=bindparam("compressed"), content_encoding="zlib")
        )
        query = select(SourceRecord.id, SourceRecord.content)
        async for rows in _pages(connection, query, SourceRecord.id):
            await connection.execute(
                statement,
                [
                    {
                        "source_key": row.id,
                        "compressed": zlib.compress(bytes(row.content)),
                    }
                    for row in rows
                ],
            )

    async def _upgrade_source_snapshots(
        self, connection: AsyncConnection, state: _SchemaState
    ) -> None:
        """Give each configuration its own source bytes and validators.

        Legacy buckets shared mutable URL rows, so their renderer provenance
        cannot be recovered. Invalidate those buckets to force a correct fetch.
        """
        if state.has(SourceSnapshot, "content"):
            return
        await connection.execute(delete(SourceBucket))
        schema = self._quoted_schema(connection)
        await connection.execute(
            text(f"""ALTER TABLE {schema}.source_cache_entries
                ADD COLUMN uri TEXT NOT NULL,
                ADD COLUMN content BYTEA NOT NULL,
                ADD COLUMN content_encoding TEXT,
                ADD COLUMN content_type TEXT NOT NULL,
                ADD COLUMN content_hash TEXT NOT NULL,
                ADD COLUMN etag TEXT,
                ADD COLUMN last_modified TIMESTAMP WITH TIME ZONE,
                ADD COLUMN metadata JSONB NOT NULL,
                ADD COLUMN last_validated TIMESTAMP WITH TIME ZONE NOT NULL
            """)
        )

    # -- connection ---------------------------------------------------------

    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[AsyncConnection]:
        """A transactional connection, with every driver failure translated.

        ``Transaction`` already translates a ``SQLAlchemyError`` raised once a
        connection exists (a malformed statement, a constraint violation). The
        one gap it leaves is a failure during the *physical* connection itself
        — auth, DNS, refused, timeout — which asyncpg raises directly, never
        wrapped as a ``SQLAlchemyError``. This adds that second, narrower
        translation around the same block so no driver exception escapes
        either path.
        """
        if self._engine is None:
            raise CacheError(
                "PostgresCache must be entered with 'async with' before use"
            )
        try:
            async with Transaction(self._engine, _as_cache_error) as connection:
                yield connection
        except (OSError, asyncpg.exceptions.PostgresError) as error:
            raise _as_cache_error_from_driver(error) from error


# -- schema state -------------------------------------------------------------

# Coverage, not names, preserves legacy indexes without duplicating them.
# Cache indexes are single-column, full-table B-trees. Partial/expression or
# invalid indexes cannot satisfy the startup/upgrade contract.
_INDEX_COVERAGE_SQL = """
    SELECT t.relname AS table_name, a.attname AS column_name
    FROM pg_catalog.pg_index i
    JOIN pg_catalog.pg_class t ON t.oid = i.indrelid
    JOIN pg_catalog.pg_namespace n ON n.oid = t.relnamespace
    JOIN pg_catalog.pg_class ix ON ix.oid = i.indexrelid
    JOIN pg_catalog.pg_am am ON am.oid = ix.relam
    JOIN pg_catalog.pg_attribute a ON a.attrelid = t.oid
        AND a.attnum = i.indkey[0]
    WHERE n.nspname = :schema AND i.indisvalid AND i.indisready
        AND i.indnkeyatts = 1 AND i.indpred IS NULL AND i.indexprs IS NULL
        AND am.amname = 'btree'
"""

# Every existing column, then every index-covered column, of the schema.
_SCHEMA_STATE_SQL = text(f"""
    SELECT table_name::text, column_name::text, udt_name::text,
        is_nullable = 'YES' AS nullable, false AS indexed
    FROM information_schema.columns WHERE table_schema = :schema
    UNION ALL
    SELECT table_name::text, column_name::text, NULL, NULL, true
    FROM ({_INDEX_COVERAGE_SQL}) covered
""")

# Each single-column cache index, with the ``(table, column)`` it covers.
_INDEXES = {
    index: (table.name, next(iter(index.columns)).name)
    for table in CacheBase.metadata.sorted_tables
    for index in table.indexes
}
_EXPECTED_COLUMNS = frozenset(
    (table.name, column.name)
    for table in CacheBase.metadata.sorted_tables
    for column in table.columns
)


@dataclass(frozen=True, slots=True)
class _SchemaState:
    """The cache tables as they exist in the database right now."""

    columns: dict[tuple[str, str], Row]
    indexed: frozenset[tuple[str, str]]

    def column(self, model: type[CacheBase], name: str) -> Row | None:
        return self.columns.get((model.__tablename__, name))

    def has(self, model: type[CacheBase], name: str) -> bool:
        return self.column(model, name) is not None

    @property
    def is_current(self) -> bool:
        """Every table, column and index exists, and every upgrade has run."""
        if not _EXPECTED_COLUMNS <= self.columns.keys():
            return False
        if not set(_INDEXES.values()) <= self.indexed:
            return False
        payload = self.column(DocumentSnapshot, "payload")
        document_source = self.column(DocumentSnapshot, "source_id")
        source_content = self.column(SourceSnapshot, "content")
        return (
            payload is not None
            and payload.udt_name == "bytea"
            and document_source is not None
            and not document_source.nullable
            and source_content is not None
            and not source_content.nullable
        )


# -- statement helpers --------------------------------------------------------

#: Reads refresh ``last_accessed`` (the garbage-collection clock) at most
#: this often per row: a warm run then reads without writing.
_TOUCH_INTERVAL = timedelta(hours=1)


def _touch(
    model: type[SourceRecord] | type[DocumentSnapshot],
    selection: ColumnElement[bool],
    *key: InstrumentedAttribute[Any],
):
    """A data-modifying CTE refreshing ``last_accessed`` without ever waiting.

    ``key`` is ``model``'s primary key, and ``selection`` picks its rows. Reads used to ``UPDATE ... RETURNING`` the rows they
    returned, so two concurrent reads of overlapping buckets each waited on
    the other's row locks, in arbitrary order — a deadlock that failed the
    read. Rows another transaction holds are skipped (their timestamp is
    being refreshed or rewritten anyway), as are rows refreshed within
    ``_TOUCH_INTERVAL``. Postgres runs a data-modifying CTE even when the
    query never reads it.
    """
    now = datetime.now(UTC)
    stale = (
        select(*key)
        .where(selection, model.last_accessed < now - _TOUCH_INTERVAL)
        .with_for_update(skip_locked=True)
    )
    return (
        update(model)
        .where(tuple_(*key).in_(stale))
        .values(last_accessed=now)
        .cte("touch")
    )


async def _pages(
    connection: AsyncConnection, query: Select, *key: InstrumentedAttribute[Any]
) -> AsyncIterator[Sequence[Row]]:
    """Yield ``query`` 1000 rows at a time, paging on ``key`` (its first columns)."""
    after: tuple[Any, ...] | None = None
    while True:
        page = query.order_by(*key).limit(1000)
        if after is not None:
            page = page.where(tuple_(*key) > after)
        rows = (await connection.execute(page)).all()
        if not rows:
            return
        yield rows
        after = tuple(rows[-1])[: len(key)]


def _excluded(statement, columns: Iterable[InstrumentedAttribute[Any]]) -> dict:
    """``SET column = EXCLUDED.column`` for each of ``columns`` in an upsert."""
    return {column: statement.excluded[column.name] for column in columns}


async def _ensure_bucket(
    connection: AsyncConnection, model: _Bucket, bucket_id: str
) -> None:
    await connection.execute(
        insert(model)
        .values(id=bucket_id)
        .on_conflict_do_nothing(index_elements=[model.id])
    )


async def _upsert(
    connection: AsyncConnection,
    model: type[SourceRecord] | type[DocumentRecord],
    rows: list[dict[str, Any]],
    *,
    update: Sequence[InstrumentedAttribute[Any]],
) -> None:
    """Insert ``rows`` by ``id``, overwriting ``update`` on existing records."""
    for batch in _row_batches(rows):
        statement = insert(model).values(batch)
        await connection.execute(
            statement.on_conflict_do_update(
                index_elements=[model.id], set_=_excluded(statement, update)
            )
        )


async def _replace_bucket(
    connection: AsyncConnection,
    model: _Snapshot,
    bucket_id: str,
    rows: list[dict[str, Any]],
) -> None:
    """Make ``rows`` the bucket's whole snapshot set."""
    await connection.execute(delete(model).where(model.bucket_id == bucket_id))
    for batch in _row_batches(rows):
        await connection.execute(insert(model).values(batch))


async def _clear(
    connection: AsyncConnection,
    record: type[SourceRecord] | type[DocumentRecord],
    bucket: _Bucket,
) -> int:
    """Unconditionally empty ``record`` and every bucket.

    Every bucket becomes empty by definition, so no per-bucket "is this now
    empty" check is needed the way ``_delete_empty_buckets`` needs one. Two
    statements regardless of row count.
    """
    deleted = (await connection.scalars(delete(record).returning(record.id))).all()
    await connection.execute(delete(bucket))
    return len(deleted)


async def _delete_empty_buckets(
    connection: AsyncConnection,
    bucket: _Bucket,
    snapshot: _Snapshot,
    bucket_ids: Iterable[str],
) -> None:
    """Delete those of ``bucket_ids`` that no longer hold any snapshot.

    Only buckets that just lost a snapshot are candidates: a bucket that was
    already empty is a legitimate known-empty result and is kept.
    """
    bucket_ids = list(bucket_ids)
    if bucket_ids:
        await connection.execute(
            delete(bucket).where(
                bucket.id.in_(bucket_ids),
                ~exists().where(snapshot.bucket_id == bucket.id),
            )
        )


async def _delete_sources(
    connection: AsyncConnection, condition: ColumnElement[bool]
) -> int:
    """Delete source records matching ``condition``, and buckets left empty.

    ``ON DELETE CASCADE`` removes the records' snapshots; what cascade cannot
    do is notice that a bucket now has none left and should stop being
    served as a known-empty ``[]``. Three statements, independent of how many
    rows match.
    """
    affected_buckets = (
        await connection.scalars(
            select(SourceSnapshot.bucket_id.distinct())
            .select_from(SourceSnapshot)
            .join(SourceRecord, SourceRecord.id == SourceSnapshot.source_id)
            .where(condition)
        )
    ).all()
    deleted = (
        await connection.scalars(
            delete(SourceRecord).where(condition).returning(SourceRecord.id)
        )
    ).all()
    await _delete_empty_buckets(
        connection, SourceBucket, SourceSnapshot, affected_buckets
    )
    return len(deleted)


async def _delete_document_snapshots(
    connection: AsyncConnection, condition: ColumnElement[bool]
) -> int:
    """Delete matching snapshots without invalidating other stages or sources.

    Returns the number of logical documents left without any snapshot, and
    therefore deleted too.
    """
    removed = (
        await connection.execute(
            delete(DocumentSnapshot)
            .where(condition)
            .returning(DocumentSnapshot.bucket_id, DocumentSnapshot.document_id)
        )
    ).all()
    if not removed:
        return 0
    await _delete_empty_buckets(
        connection,
        DocumentBucket,
        DocumentSnapshot,
        {bucket_id for bucket_id, _ in removed},
    )
    return await _delete_orphan_documents(
        connection,
        DocumentRecord.id.in_({document_id for _, document_id in removed}),
    )


async def _delete_orphan_documents(
    connection: AsyncConnection, condition: ColumnElement[bool]
) -> int:
    """Delete matching document records that no snapshot references."""
    deleted = (
        await connection.scalars(
            delete(DocumentRecord)
            .where(
                condition,
                ~exists().where(DocumentSnapshot.document_id == DocumentRecord.id),
            )
            .returning(DocumentRecord.id)
        )
    ).all()
    return len(deleted)


def _row_batches(rows: list[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
    """Keep multi-row statements under asyncpg's 32767 parameter limit."""
    if rows:
        size = min(1000, 30000 // len(rows[0]))
        for offset in range(0, len(rows), size):
            yield rows[offset : offset + size]


# -- conversions --------------------------------------------------------------


def _source_payload(source: Source, now: datetime) -> dict[str, Any]:
    """The columns a source record and its snapshot share, bytes compressed."""
    return {
        "uri": source.uri,
        "content": zlib.compress(source.content),
        "content_encoding": "zlib",
        "content_type": source.content_type,
        "content_hash": source.content_hash,
        "etag": source.etag,
        "last_modified": source.last_modified,
        "metadata_": source.metadata,
        # Being written at all means the source was just validated.
        "last_validated": now
        if source.last_validated is None
        else source.last_validated,
    }


def _source_from_row(row: Row) -> Source:
    content = bytes(row.content)
    return Source(
        id=row.id,
        uri=row.uri,
        content=zlib.decompress(content) if row.content_encoding == "zlib" else content,
        content_type=row.content_type,
        content_hash=row.content_hash,
        etag=row.etag,
        last_modified=row.last_modified,
        metadata=dict(row.metadata) if row.metadata else {},
        last_validated=row.last_validated,
        cached=True,
    )


# -- errors -------------------------------------------------------------------


def _as_cache_error(error: SQLAlchemyError) -> CacheError:
    """Mark connection-level failures transient and everything else terminal."""
    transient = is_transient(error)
    message = f"cache operation failed: {error}"
    return TransientCacheError(message) if transient else CacheError(message)


def _as_cache_error_from_driver(
    error: OSError | asyncpg.exceptions.PostgresError,
) -> CacheError:
    """Translate a raw driver failure that never became a SQLAlchemyError.

    A failure establishing the physical connection — refused, DNS, timeout,
    a dropped socket — surfaces as a plain ``OSError`` or an
    ``asyncpg.exceptions.PostgresConnectionError``, both worth retrying. Bad
    credentials or a missing database (``InvalidPasswordError``,
    ``InvalidCatalogNameError``) are ``PostgresError`` but not connection
    errors, and retrying them fails identically every time.
    """
    transient = isinstance(error, OSError | asyncpg.exceptions.PostgresConnectionError)
    message = f"cache operation failed: {error}"
    return TransientCacheError(message) if transient else CacheError(message)
