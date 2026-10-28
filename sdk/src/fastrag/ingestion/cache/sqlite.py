"""SQLite-backed :class:`~fastrag.ingestion.cache.base.CacheStore`.

One file and no server, for a caller who does not want to run Postgres.
Built on the standard-library :mod:`sqlite3` module, so it adds no
dependency.

The layout mirrors :mod:`fastrag.ingestion.cache.schema`: a **bucket** per
cache id, a **snapshot** per item of a bucket's output, and a **record** per
logical id carrying the garbage-collection clock. ``__aenter__`` creates the
tables if they are absent.

``sqlite3`` is blocking, so every operation runs as one transaction on a
dedicated worker thread that owns the connection. Operations from one
instance are therefore serialized; separate processes may share the file,
since the database runs in WAL mode and waits out a writer's lock.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import struct
import zlib
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Self

from fastrag.exceptions import CacheError, TransientCacheError
from fastrag.ingestion.cache._payloads import (
    deduplicate,
    document_payload,
    pack_payload,
    unpack_payload,
)
from fastrag.ingestion.cache.base import CacheStore, _DocumentHeader, _SourceHeader
from fastrag.ingestion.models import Source
from fastrag.models import Document

_SCHEMA = """
CREATE TABLE IF NOT EXISTS source_cache (id TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    last_accessed REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_sources_last_accessed ON sources (last_accessed);
CREATE TABLE IF NOT EXISTS source_cache_entries (
    source_cache_id TEXT NOT NULL REFERENCES source_cache (id) ON DELETE CASCADE,
    source_id TEXT NOT NULL REFERENCES sources (id) ON DELETE CASCADE,
    uri TEXT NOT NULL,
    content BLOB NOT NULL,
    content_type TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    etag TEXT,
    last_modified TEXT,
    metadata TEXT NOT NULL,
    last_validated TEXT NOT NULL,
    PRIMARY KEY (source_cache_id, source_id)
);
CREATE INDEX IF NOT EXISTS ix_source_cache_entries_source_id
    ON source_cache_entries (source_id);

CREATE TABLE IF NOT EXISTS document_cache (id TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    last_accessed REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_documents_source_id ON documents (source_id);
CREATE INDEX IF NOT EXISTS ix_documents_last_accessed ON documents (last_accessed);
CREATE TABLE IF NOT EXISTS document_cache_entries (
    document_cache_id TEXT NOT NULL REFERENCES document_cache (id) ON DELETE CASCADE,
    document_id TEXT NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    payload BLOB NOT NULL,
    header_metadata TEXT NOT NULL,
    step_config_hash TEXT NOT NULL,
    embedding BLOB,
    source_id TEXT NOT NULL,
    last_accessed REAL NOT NULL,
    PRIMARY KEY (document_cache_id, document_id)
);
CREATE INDEX IF NOT EXISTS ix_document_cache_entries_document_id
    ON document_cache_entries (document_id);
CREATE INDEX IF NOT EXISTS ix_document_cache_entries_source_id
    ON document_cache_entries (source_id);
CREATE INDEX IF NOT EXISTS ix_document_cache_entries_last_accessed
    ON document_cache_entries (last_accessed);
"""


@dataclass(frozen=True, slots=True)
class _Tables:
    """The bucket, record and snapshot tables of one entry kind."""

    bucket: str
    record: str
    entries: str
    bucket_key: str
    item_key: str


_SOURCES = _Tables(
    "source_cache", "sources", "source_cache_entries", "source_cache_id", "source_id"
)
_DOCUMENTS = _Tables(
    "document_cache",
    "documents",
    "document_cache_entries",
    "document_cache_id",
    "document_id",
)

# Snapshot columns read back into a ``Source``. Headers skip the bytes.
_SOURCE_HEADER_COLUMNS = (
    "e.uri, e.content_type, e.content_hash, e.etag, e.last_modified, "
    "e.metadata, e.last_validated"
)
_SOURCE_COLUMNS = f"{_SOURCE_HEADER_COLUMNS}, e.content"
# Snapshot columns read back into a ``Document``. Headers skip text and vector.
_DOCUMENT_HEADER_COLUMNS = "e.source_id, e.step_config_hash, e.header_metadata"
_DOCUMENT_COLUMNS = "e.payload, e.embedding"

#: Reads refresh ``last_accessed`` (the garbage-collection clock) at most
#: this often per row: a warm run then reads without writing.
_TOUCH_INTERVAL = timedelta(hours=1)

#: Ids bound per ``IN (...)`` list, well under SQLite's variable limit.
_BATCH_SIZE = 500

type _Rows = dict[str, list[sqlite3.Row] | None]


class SQLiteCache(CacheStore):
    """Cache pipeline sources and documents in a local SQLite file.

    .. code-block:: python

        async with SQLiteCache(path=".fastrag/cache.sqlite") as cache:
            sources = SitemapCrawler(url=SITEMAP_URL, cache=cache)

    Args:
        path: The database file. Created, along with missing parent
            directories, on ``__aenter__``. ``":memory:"`` keeps the cache in
            memory for the lifetime of the ``async with`` block.
        timeout: Seconds to wait for another process's write lock before an
            operation fails with
            :class:`~fastrag.exceptions.TransientCacheError`.
    """

    def __init__(self, *, path: str | os.PathLike[str], timeout: float = 30.0) -> None:
        """Store configuration only; no I/O happens until ``__aenter__``."""
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        self._path = os.fspath(path)
        self._timeout = timeout
        self._executor: ThreadPoolExecutor | None = None
        self._connection: sqlite3.Connection | None = None

    async def __aenter__(self) -> Self:
        """Open the database file and ensure the tables exist.

        Idempotent: entering again — on this instance, or on a fresh
        instance pointed at the same file — finds the tables already there.
        """
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="fastrag-sqlite-cache"
            )
        try:
            await asyncio.get_running_loop().run_in_executor(self._executor, self._open)
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        """Close the connection and stop its worker thread."""
        if self._executor is None:
            return
        executor, self._executor = self._executor, None
        await asyncio.get_running_loop().run_in_executor(executor, self._close)
        executor.shutdown(wait=False)

    # -- sources ------------------------------------------------------------

    async def get_source_entries(self, source_cache_id: str) -> list[Source] | None:
        """See :meth:`~fastrag.ingestion.cache.base.CacheStore.get_source_entries`."""
        return (await self.get_source_entries_many([source_cache_id]))[source_cache_id]

    async def get_source_entries_many(
        self, source_cache_ids: Sequence[str]
    ) -> dict[str, list[Source] | None]:
        """Fetch source buckets in one transaction."""
        return await self._read_buckets(
            _SOURCES, _SOURCE_COLUMNS, source_cache_ids, lambda _, row: _source(row)
        )

    async def get_source_headers_many(
        self, cache_ids: Sequence[str]
    ) -> dict[str, list[Source] | None]:
        """Retrieve only identities, validators and routing metadata for indexing."""
        return await self._read_buckets(
            _SOURCES,
            _SOURCE_HEADER_COLUMNS,
            cache_ids,
            lambda bucket_id, row: _SourceHeader(
                **_source_header_fields(row),
                content=b"",
                cached=True,
                _store=self,
                _bucket=bucket_id,
            ),
        )

    async def get_bucket_sources(
        self, source_cache_id: str, source_ids: Sequence[str]
    ) -> list[Source]:
        """See :meth:`~fastrag.ingestion.cache.base.CacheStore.get_bucket_sources`."""
        if not source_ids:
            return []

        def read(connection: sqlite3.Connection) -> list[Source]:
            sources = []
            for batch in _batches(list(dict.fromkeys(source_ids))):
                rows = connection.execute(
                    f"SELECT e.source_id AS id, {_SOURCE_COLUMNS} "
                    "FROM source_cache_entries e "
                    f"WHERE e.source_cache_id = ? AND e.source_id IN ({_marks(batch)})",
                    [source_cache_id, *batch],
                )
                sources.extend(_source(row) for row in rows)
            return sources

        return await self._run(read)

    async def put_source_entries(
        self, source_cache_id: str, sources: Sequence[Source]
    ) -> None:
        """See :meth:`~fastrag.ingestion.cache.base.CacheStore.put_source_entries`."""
        sources = deduplicate(sources)

        def write(connection: sqlite3.Connection) -> None:
            now = datetime.now(UTC)
            _ensure_bucket(connection, _SOURCES, source_cache_id)
            # An upsert, never ``INSERT OR REPLACE``: replacing a record would
            # cascade into every other bucket's snapshot of it.
            connection.executemany(
                "INSERT INTO sources (id, last_accessed) VALUES (?, ?) "
                "ON CONFLICT (id) DO UPDATE SET last_accessed = excluded.last_accessed",
                [(source.id, now.timestamp()) for source in sources],
            )
            _clear_bucket(connection, _SOURCES, source_cache_id)
            connection.executemany(
                "INSERT INTO source_cache_entries (source_cache_id, source_id, uri, "
                "content, content_type, content_hash, etag, last_modified, metadata, "
                "last_validated) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        source_cache_id,
                        source.id,
                        source.uri,
                        zlib.compress(source.content),
                        source.content_type,
                        source.content_hash,
                        source.etag,
                        _encode_time(source.last_modified),
                        _encode_json(source.metadata),
                        # Being written at all means the source was just validated.
                        _encode_time(source.last_validated or now),
                    )
                    for source in sources
                ],
            )

        await self._run(write)

    # -- documents ----------------------------------------------------------

    async def get_document_entries(
        self, document_cache_id: str
    ) -> list[Document] | None:
        """See :meth:`~fastrag.ingestion.cache.base.CacheStore.get_document_entries`."""
        return (await self.get_document_entries_many([document_cache_id]))[
            document_cache_id
        ]

    async def get_document_entries_many(
        self, document_cache_ids: Sequence[str]
    ) -> dict[str, list[Document] | None]:
        """Fetch document buckets in one transaction."""
        return await self._read_buckets(
            _DOCUMENTS,
            _DOCUMENT_COLUMNS,
            document_cache_ids,
            lambda _, row: Document(
                **unpack_payload(row["payload"]),
                embedding=_decode_vector(row["embedding"]),
                cached=True,
            ),
        )

    async def get_document_headers_many(
        self, cache_ids: Sequence[str]
    ) -> dict[str, list[Document] | None]:
        """Retrieve snapshot identifiers and routing metadata without large payloads."""
        return await self._read_buckets(
            _DOCUMENTS,
            _DOCUMENT_HEADER_COLUMNS,
            cache_ids,
            lambda bucket_id, row: _DocumentHeader(
                id=row["id"],
                source_id=row["source_id"],
                content="",
                step_config_hash=row["step_config_hash"],
                metadata=json.loads(row["header_metadata"]),
                cached=True,
                _store=self,
                _bucket=bucket_id,
            ),
        )

    async def put_document_entries(
        self, document_cache_id: str, documents: Sequence[Document]
    ) -> None:
        """See :meth:`~fastrag.ingestion.cache.base.CacheStore.put_document_entries`."""
        documents = deduplicate(documents)

        def write(connection: sqlite3.Connection) -> None:
            now = datetime.now(UTC).timestamp()
            _ensure_bucket(connection, _DOCUMENTS, document_cache_id)
            # Records only register the logical id; payloads live on snapshots.
            connection.executemany(
                "INSERT INTO documents (id, source_id, last_accessed) VALUES (?, ?, ?) "
                "ON CONFLICT (id) DO UPDATE SET source_id = excluded.source_id, "
                "last_accessed = excluded.last_accessed",
                [(document.id, document.source_id, now) for document in documents],
            )
            _clear_bucket(connection, _DOCUMENTS, document_cache_id)
            connection.executemany(
                "INSERT INTO document_cache_entries (document_cache_id, document_id, "
                "payload, header_metadata, step_config_hash, embedding, source_id, "
                "last_accessed) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        document_cache_id,
                        document.id,
                        pack_payload(document_payload(document)),
                        _encode_json(document.metadata),
                        document.step_config_hash,
                        _encode_vector(document.embedding),
                        document.source_id,
                        now,
                    )
                    for document in documents
                ],
            )

        await self._run(write)

    # -- administration -----------------------------------------------------

    async def clear_sources(self) -> int:
        """See :meth:`~fastrag.ingestion.cache.base.CacheStore.clear_sources`."""
        return await self._run(lambda connection: _clear(connection, _SOURCES))

    async def clear_documents(self) -> int:
        """See :meth:`~fastrag.ingestion.cache.base.CacheStore.clear_documents`."""
        return await self._run(lambda connection: _clear(connection, _DOCUMENTS))

    async def clear(self) -> int:
        """See :meth:`~fastrag.ingestion.cache.base.CacheStore.clear`."""
        return await self._run(
            lambda connection: (
                _clear(connection, _SOURCES) + _clear(connection, _DOCUMENTS)
            )
        )

    async def delete_sources(self, source_ids: Sequence[str]) -> int:
        """See :meth:`~fastrag.ingestion.cache.base.CacheStore.delete_sources`."""
        if not source_ids:
            return 0

        def delete(connection: sqlite3.Connection) -> int:
            return sum(
                _delete_sources(connection, f"id IN ({_marks(batch)})", batch)
                for batch in _batches(list(dict.fromkeys(source_ids)))
            )

        return await self._run(delete)

    async def delete_documents(self, source_ids: Sequence[str]) -> int:
        """See :meth:`~fastrag.ingestion.cache.base.CacheStore.delete_documents`."""
        if not source_ids:
            return 0

        def delete(connection: sqlite3.Connection) -> int:
            removed = 0
            for batch in _batches(list(dict.fromkeys(source_ids))):
                # Snapshots and records both carry ``source_id``.
                condition = f"source_id IN ({_marks(batch)})"
                removed += _delete_document_snapshots(connection, condition, batch)
                removed += _delete_orphan_documents(connection, condition, batch)
            return removed

        return await self._run(delete)

    async def collect_garbage(self, max_age: timedelta) -> int:
        """See :meth:`~fastrag.ingestion.cache.base.CacheStore.collect_garbage`."""
        if max_age <= timedelta(0):
            raise ValueError(f"max_age must be positive, got {max_age!r}")

        def collect(connection: sqlite3.Connection) -> int:
            cutoff = [(datetime.now(UTC) - max_age).timestamp()]
            # Every table swept here keys its clock on ``last_accessed``.
            condition = "last_accessed < ?"
            removed = _delete_sources(connection, condition, cutoff)
            removed += _delete_document_snapshots(connection, condition, cutoff)
            # Replacements can orphan old logical document ids without leaving
            # a snapshot. Collect those too, but never delete a referenced id.
            return removed + _delete_orphan_documents(connection, condition, cutoff)

        return await self._run(collect)

    # -- reads --------------------------------------------------------------

    async def _read_buckets[T](
        self,
        tables: _Tables,
        columns: str,
        bucket_ids: Sequence[str],
        convert: Callable[[str, sqlite3.Row], T],
    ) -> dict[str, list[T] | None]:
        """Read and convert buckets, telling misses from known-empty buckets."""
        unique_ids = list(dict.fromkeys(bucket_ids))
        if not unique_ids:
            return {}

        def read(connection: sqlite3.Connection) -> dict[str, list[T] | None]:
            rows = _select_buckets(connection, tables, columns, unique_ids)
            return {
                bucket_id: None
                if entries is None
                else [convert(bucket_id, row) for row in entries]
                for bucket_id, entries in rows.items()
            }

        return await self._run(read)

    # -- connection ---------------------------------------------------------

    def _open(self) -> None:
        """Connect on the worker thread and create missing tables."""
        if self._connection is not None:
            return
        try:
            if self._path != ":memory:":
                Path(self._path).parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                self._path, timeout=self._timeout, isolation_level=None
            )
        except (OSError, sqlite3.Error) as error:
            raise _as_cache_error(error) from error
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            # WAL lets other processes read while this one writes.
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(_SCHEMA)
        except sqlite3.Error as error:
            connection.close()
            raise _as_cache_error(error) from error
        self._connection = connection

    def _close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    async def _run[T](self, operation: Callable[[sqlite3.Connection], T]) -> T:
        """Run ``operation`` as one transaction on the connection's thread."""
        if self._executor is None:
            raise CacheError("SQLiteCache must be entered with 'async with' before use")
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, self._transaction, operation
        )

    def _transaction[T](self, operation: Callable[[sqlite3.Connection], T]) -> T:
        connection = self._connection
        assert connection is not None
        try:
            # Reads write too (the access clock): take the write lock up front
            # rather than failing to upgrade a read lock mid-transaction.
            connection.execute("BEGIN IMMEDIATE")
            try:
                result = operation(connection)
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")
        except sqlite3.Error as error:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise _as_cache_error(error) from error
        return result


# -- statement helpers --------------------------------------------------------


def _marks(values: Sequence[Any]) -> str:
    return ", ".join("?" * len(values))


def _batches(values: list[str]) -> Iterator[list[str]]:
    for offset in range(0, len(values), _BATCH_SIZE):
        yield values[offset : offset + _BATCH_SIZE]


def _select_buckets(
    connection: sqlite3.Connection,
    tables: _Tables,
    columns: str,
    bucket_ids: list[str],
) -> _Rows:
    """Group snapshot rows by bucket, refreshing their GC clock.

    Buckets are outer-joined to their snapshots, so an empty bucket yields
    one row whose ``id`` is ``NULL``, and a missing bucket yields none.
    """
    buckets: _Rows = dict.fromkeys(bucket_ids)
    for batch in _batches(bucket_ids):
        rows = connection.execute(
            f"SELECT b.id AS bucket_id, e.{tables.item_key} AS id, {columns} "
            f"FROM {tables.bucket} b LEFT JOIN {tables.entries} e "
            f"ON e.{tables.bucket_key} = b.id "
            f"WHERE b.id IN ({_marks(batch)}) ORDER BY b.id, e.{tables.item_key}",
            batch,
        )
        for row in rows:
            entries = buckets[row["bucket_id"]]
            if entries is None:
                entries = buckets[row["bucket_id"]] = []
            if row["id"] is not None:
                entries.append(row)
        _touch(connection, tables, batch)
    return buckets


def _touch(connection: sqlite3.Connection, tables: _Tables, bucket_ids: list[str]):
    """Refresh ``last_accessed`` on what ``bucket_ids`` hold, if it is stale.

    Sources keep their clock on the shared record, documents on the snapshot
    — the same split :class:`~fastrag.ingestion.cache.postgres.PostgresCache`
    uses.
    """
    now = datetime.now(UTC)
    clock = [now.timestamp(), (now - _TOUCH_INTERVAL).timestamp(), *bucket_ids]
    if tables is _SOURCES:
        connection.execute(
            "UPDATE sources SET last_accessed = ? WHERE last_accessed < ? AND id IN "
            "(SELECT source_id FROM source_cache_entries "
            f"WHERE source_cache_id IN ({_marks(bucket_ids)}))",
            clock,
        )
    else:
        connection.execute(
            "UPDATE document_cache_entries SET last_accessed = ? "
            f"WHERE last_accessed < ? AND document_cache_id IN ({_marks(bucket_ids)})",
            clock,
        )


def _ensure_bucket(connection: sqlite3.Connection, tables: _Tables, bucket_id: str):
    connection.execute(
        f"INSERT INTO {tables.bucket} (id) VALUES (?) ON CONFLICT (id) DO NOTHING",
        [bucket_id],
    )


def _clear_bucket(connection: sqlite3.Connection, tables: _Tables, bucket_id: str):
    connection.execute(
        f"DELETE FROM {tables.entries} WHERE {tables.bucket_key} = ?", [bucket_id]
    )


def _clear(connection: sqlite3.Connection, tables: _Tables) -> int:
    """Unconditionally empty ``tables``' records and every bucket."""
    removed = connection.execute(f"DELETE FROM {tables.record}").rowcount
    connection.execute(f"DELETE FROM {tables.bucket}")
    return removed


def _delete_empty_buckets(
    connection: sqlite3.Connection, tables: _Tables, bucket_ids: Iterable[str]
) -> None:
    """Delete those of ``bucket_ids`` that no longer hold any snapshot.

    Only buckets that just lost a snapshot are candidates: a bucket that was
    already empty is a legitimate known-empty result and is kept.
    """
    for batch in _batches(list(bucket_ids)):
        connection.execute(
            f"DELETE FROM {tables.bucket} WHERE id IN ({_marks(batch)}) "
            f"AND NOT EXISTS (SELECT 1 FROM {tables.entries} e "
            f"WHERE e.{tables.bucket_key} = {tables.bucket}.id)",
            batch,
        )


def _delete_sources(
    connection: sqlite3.Connection, condition: str, parameters: Sequence[Any]
) -> int:
    """Delete source records matching ``condition``, and buckets left empty.

    ``ON DELETE CASCADE`` removes the records' snapshots; what cascade cannot
    do is notice that a bucket now has none left and should stop being
    served as a known-empty ``[]``.
    """
    affected_buckets = [
        bucket_id
        for (bucket_id,) in connection.execute(
            "SELECT DISTINCT source_cache_id FROM source_cache_entries "
            f"WHERE source_id IN (SELECT id FROM sources WHERE {condition})",
            parameters,
        )
    ]
    removed = connection.execute(
        f"DELETE FROM sources WHERE {condition}", parameters
    ).rowcount
    _delete_empty_buckets(connection, _SOURCES, affected_buckets)
    return removed


def _delete_document_snapshots(
    connection: sqlite3.Connection, condition: str, parameters: Sequence[Any]
) -> int:
    """Delete matching snapshots without invalidating other stages or sources.

    Returns the number of logical documents left without any snapshot, and
    therefore deleted too.
    """
    removed = connection.execute(
        "SELECT document_cache_id, document_id FROM document_cache_entries "
        f"WHERE {condition}",
        parameters,
    ).fetchall()
    if not removed:
        return 0
    connection.execute(
        f"DELETE FROM document_cache_entries WHERE {condition}", parameters
    )
    _delete_empty_buckets(
        connection, _DOCUMENTS, {bucket_id for bucket_id, _ in removed}
    )
    document_ids = list({document_id for _, document_id in removed})
    return sum(
        _delete_orphan_documents(connection, f"id IN ({_marks(batch)})", batch)
        for batch in _batches(document_ids)
    )


def _delete_orphan_documents(
    connection: sqlite3.Connection, condition: str, parameters: Sequence[Any]
) -> int:
    """Delete matching document records that no snapshot references."""
    return connection.execute(
        f"DELETE FROM documents WHERE ({condition}) AND NOT EXISTS "
        "(SELECT 1 FROM document_cache_entries e WHERE e.document_id = documents.id)",
        parameters,
    ).rowcount


# -- conversions --------------------------------------------------------------


def _encode_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False)


def _encode_time(value: datetime | None) -> str | None:
    """ISO 8601 keeps the offset, so aware and naive times both round-trip."""
    return None if value is None else value.isoformat()


def _decode_time(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


def _encode_vector(vector: list[float] | None) -> bytes | None:
    """Pack a vector as little-endian doubles: exact, and compact next to JSON."""
    return None if vector is None else struct.pack(f"<{len(vector)}d", *vector)


def _decode_vector(packed: bytes | None) -> list[float] | None:
    if packed is None:
        return None
    return list(struct.unpack(f"<{len(packed) // 8}d", packed))


def _source_header_fields(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "uri": row["uri"],
        "content_type": row["content_type"],
        "content_hash": row["content_hash"],
        "etag": row["etag"],
        "last_modified": _decode_time(row["last_modified"]),
        "metadata": json.loads(row["metadata"]),
        "last_validated": _decode_time(row["last_validated"]),
    }


def _source(row: sqlite3.Row) -> Source:
    return Source(
        **_source_header_fields(row),
        content=zlib.decompress(row["content"]),
        cached=True,
    )


# -- errors -------------------------------------------------------------------

#: Another connection holds the lock: retrying once it is released succeeds.
_TRANSIENT_ERROR_CODES = frozenset({sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED})


def _as_cache_error(error: OSError | sqlite3.Error) -> CacheError:
    """Mark lock contention transient and everything else terminal."""
    code = getattr(error, "sqlite_errorcode", None)
    # Extended result codes carry the primary code in their low byte.
    transient = code is not None and code & 0xFF in _TRANSIENT_ERROR_CODES
    message = f"cache operation failed: {error}"
    return TransientCacheError(message) if transient else CacheError(message)
