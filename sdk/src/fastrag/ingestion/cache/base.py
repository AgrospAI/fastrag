"""The pluggable cache contract every step-side cache backend implements.

Following ``docs/specification/caching.md``: a step is handed a
``CacheStore`` or nothing at all, so caching stays explicit and per-step
rather than a hidden global.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timedelta
from typing import Any, Protocol, Self

from fastrag.ingestion.models import Source
from fastrag.exceptions import TransientCacheError
from fastrag.models import Document


class SourceFreshnessValidator(Protocol):
    """Protocol for checking if an upstream resource has changed."""

    async def is_source_modified(
        self,
        source_id: str,
        cached_etag: str | None,
        cached_last_modified: datetime | None,
        cached_content_hash: str,
    ) -> bool:
        """Validates whether a source entry is stale against the upstream server.

        Returns:
            True if modified, False if cached version remains fresh.
        """
        ...


class CacheStore(AbstractAsyncContextManager["CacheStore"], ABC):
    """Pluggable abstract cache backend with distinct lifecycle support."""

    async def __aenter__(self) -> Self:
        """Open cache resources."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        """Release cache resources."""
        return None

    @abstractmethod
    async def get_source_entries(self, source_cache_id: str) -> list[Source] | None:
        """Fetches cached sources associated with a step run.

        Args:
            source_cache_id: Unique hash of the source step configuration and target.

        Returns:
            The cached sources, ``[]`` if the id is known but holds no
            entries, or ``None`` on a cache miss. Each returned ``Source``
            carries its persisted ``last_validated``, so a caller can judge
            freshness without any per-process state.
        """
        ...

    @abstractmethod
    async def put_source_entries(
        self, source_cache_id: str, sources: Sequence[Source]
    ) -> None:
        """Persists fetched sources and updates source reference tables.

        Idempotent: writing the same ``source_cache_id`` again replaces its
        entry set rather than appending to it.

        Args:
            source_cache_id: Unique hash of the source step configuration.
            sources: Sequence of Source objects to persist. Each source's
                ``last_validated`` is written verbatim when set; a source
                with ``last_validated=None`` is stamped with the current time,
                since being written at all means it was just validated.
        """
        ...

    @abstractmethod
    async def get_document_entries(
        self, document_cache_id: str
    ) -> list[Document] | None:
        """Fetches cached documents associated with a transformation step.

        Args:
            document_cache_id: Hash composed of document ID and step configuration.

        Returns:
            The cached documents, ``[]`` if the id is known but holds no
            entries, or ``None`` on a cache miss.
        """
        ...

    async def get_source_entries_many(
        self, source_cache_ids: Sequence[str]
    ) -> dict[str, list[Source] | None]:
        """Read multiple buckets, including misses and known-empty results.

        Backends may override this with a bulk query. The default preserves
        compatibility with backends implementing only single-bucket reads.
        Duplicate ids are read once; an empty input performs no I/O.
        """
        return {
            cache_id: await self.get_source_entries(cache_id)
            for cache_id in dict.fromkeys(source_cache_ids)
        }

    async def get_document_entries_many(
        self, document_cache_ids: Sequence[str]
    ) -> dict[str, list[Document] | None]:
        """Read document buckets with the same semantics as single reads.

        The default uses single-bucket reads for existing custom backends.
        Duplicate ids are read once; an empty input performs no I/O.
        """
        return {
            cache_id: await self.get_document_entries(cache_id)
            for cache_id in dict.fromkeys(document_cache_ids)
        }

    @abstractmethod
    async def put_document_entries(
        self, document_cache_id: str, documents: Sequence[Document]
    ) -> None:
        """Persists transformed documents and updates document cache references.

        Idempotent: writing the same ``document_cache_id`` again replaces its
        entry set rather than appending to it.

        Args:
            document_cache_id: Hash composed of document ID and step configuration.
            documents: Transformed Document objects to persist.
        """
        ...

    async def get_source_headers_many(
        self, cache_ids: Sequence[str]
    ) -> dict[str, list[Source] | None]:
        """Read source identities and validators, with lazy content when supported.

        The fallback returns complete sources for existing custom backends.
        """
        return await self.get_source_entries_many(cache_ids)

    async def get_bucket_sources(
        self, source_cache_id: str, source_ids: Sequence[str]
    ) -> list[Source]:
        """Read only ``source_ids`` (with content) from one cached bucket.

        The fallback loads the whole bucket; backends should override it with
        a targeted query so hydrating one page of a large crawl bucket does
        not transfer every sibling page.
        """
        wanted = set(source_ids)
        return [
            entry
            for entry in await self.get_source_entries(source_cache_id) or []
            if entry.id in wanted
        ]

    async def get_document_headers_many(
        self, cache_ids: Sequence[str]
    ) -> dict[str, list[Document] | None]:
        """Read cached identities and routing metadata without text or vectors.

        Optimized backends return lazy headers; the fallback returns complete
        documents. Ordinary cache getters always return complete outputs.
        """
        return await self.get_document_entries_many(cache_ids)

    @abstractmethod
    async def clear_sources(self) -> int:
        """Removes every cached source and every source cache id.

        Cache administration only: emits nothing, retries nothing, and never
        touches a vector store.

        Returns:
            The number of source rows removed.
        """
        ...

    @abstractmethod
    async def clear_documents(self) -> int:
        """Removes every cached document and every document cache id.

        Cache administration only: emits nothing, retries nothing, and never
        touches a vector store.

        Returns:
            The number of document rows removed.
        """
        ...

    @abstractmethod
    async def clear(self) -> int:
        """Removes everything this store holds: sources, documents, and both cache id tables.

        Cache administration only: emits nothing, retries nothing, and never
        touches a vector store.

        Returns:
            The total number of source and document rows removed.
        """
        ...

    @abstractmethod
    async def delete_sources(self, source_ids: Sequence[str]) -> int:
        """Removes the named cached sources.

        Any source cache id left referencing none of its entries after this
        deletion is removed too, so a later ``get_source_entries`` on that id
        returns ``None`` (a miss) rather than ``[]`` (a known-empty result).

        Idempotent and tolerant: a ``source_ids`` id that is not cached is
        silently skipped rather than raising. An empty ``source_ids`` is a
        no-op — no statement is sent.

        Args:
            source_ids: The source ids to remove.

        Returns:
            The number of source rows actually removed.
        """
        ...

    @abstractmethod
    async def delete_documents(self, source_ids: Sequence[str]) -> int:
        """Removes every cached document whose ``source_id`` is one of `source_ids`.

        Any document cache id left referencing none of its entries after this
        deletion is removed too, so a later ``get_document_entries`` on that
        id returns ``None`` rather than ``[]``.

        Idempotent and tolerant: a ``source_ids`` id with no cached documents
        is silently skipped rather than raising. An empty ``source_ids`` is a
        no-op — no statement is sent.

        Args:
            source_ids: Remove every cached document belonging to one of
                these source ids.

        Returns:
            The number of document rows actually removed.
        """
        ...

    @abstractmethod
    async def collect_garbage(self, max_age: timedelta) -> int:
        """Removes every source and document whose ``last_accessed`` predates ``now - max_age``.

        An age-based sweep keyed on the column every read already touches.
        Any source or document cache id left
        referencing none of its entries after the sweep is removed too, with
        the same empty-bucket guarantee as ``delete_sources``/
        ``delete_documents``.

        Args:
            max_age: How old an entry may be, by ``last_accessed``, before it
                is collected. Must be positive.

        Returns:
            The total number of source and document rows removed.

        Raises:
            ValueError: If ``max_age`` is zero or negative.
        """
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class _SourceHeader(Source):
    """Private index-only projection, hydrated before any content transformation."""

    _store: CacheStore = field(repr=False, compare=False)
    _bucket: str = field(repr=False, compare=False)

    async def materialize(self) -> Source:
        """Retrieve the complete source if downstream work needs its bytes."""
        entries = await self._store.get_bucket_sources(self._bucket, [self.id])
        for item in entries:
            if item.id == self.id:
                return replace(item, metadata={**item.metadata, **self.metadata})
        raise TransientCacheError("source cache changed while materializing a header")


@dataclass(frozen=True, slots=True, kw_only=True)
class _DocumentHeader(Document):
    """Private index-only projection retaining the owning snapshot reference."""

    _store: CacheStore = field(repr=False, compare=False)
    _bucket: str = field(repr=False, compare=False)

    async def materialize(self) -> Document:
        """Retrieve the exact stage snapshot before a transform cache miss."""
        entries = await self._store.get_document_entries(self._bucket)
        for item in entries or []:
            if item.id == self.id:
                return item
        raise TransientCacheError("document cache changed while materializing a header")


def _public_item(item: Any) -> Any:
    """Remove private lazy origins from public events and sink results."""
    model = Source if isinstance(item, _SourceHeader) else Document
    if isinstance(item, (_SourceHeader, _DocumentHeader)):
        return model(
            **{column.name: getattr(item, column.name) for column in fields(model)}
        )
    return item
