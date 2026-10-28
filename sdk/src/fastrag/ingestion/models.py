"""Value types produced by the ingestion pipeline's source-fetching stage.

A :class:`Source` is raw fetched content, produced by a fetcher and consumed
by a parser. What a ``Source`` becomes once parsed — :class:`~fastrag.models.Document`
— is shared by every later stage (parsing, chunking, embedding, indexing) as
well as by retrieval, querying, serving and benchmarking, so it lives at the
top level, in :mod:`fastrag.models`, rather than here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class Source:
    """Raw content fetched from somewhere, plus enough metadata to place it.

    A ``Source`` is produced by a fetcher and carries the original bytes
    together with enough information to identify, validate, cache, and later
    transform it into one or more :class:`~fastrag.models.Document` instances.
    Sources are independently cacheable, and their freshness may be validated
    against the originating system using ETags, modification timestamps,
    content hashes, or equivalent mechanisms.

    Attributes:
        id: Unique identifier for this source.
        uri: Where the content came from.
        content: The raw fetched bytes.
        content_type: The source's MIME media type.
        content_hash: Deterministic hash of ``content``, used as its address.
        etag: HTTP validator for conditional refetching, when the origin
            supports it.
        last_modified: Origin-reported modification time, when available.
        metadata: Extended, source-specific metadata.
        last_validated: When this source was last confirmed fresh against its
            origin (or, for one that has never been revalidated, first
            fetched). Persisted: round-tripped through the cache so that
            ``SourceStep.cache_validation_interval`` means the same thing
            across process restarts, not just within one. ``None`` on a
            ``Source`` that has never been through a cache write, e.g. one
            just produced by ``evaluate()`` and not yet cached.
        cached: Whether this instance was served from the cache rather than
            freshly fetched. Runtime-only: never persisted.
    """

    id: str
    uri: str
    content: bytes
    content_type: str
    content_hash: str
    etag: str | None = None
    last_modified: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    last_validated: datetime | None = None
    cached: bool = False
