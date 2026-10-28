"""The pluggable step cache.

The :class:`~fastrag.ingestion.cache.base.CacheStore` contract, its PostgreSQL and SQLite implementations, and the
identity helpers every caching step builds cache ids with.

Caching is explicit and per-step (``docs/specification/API.md``: "Cache is
explicit in each step ... it is better to be explicit about it"): a step is
handed a :class:`~fastrag.ingestion.cache.base.CacheStore` or ``cache=False``, never a hidden global.
"""

from __future__ import annotations

from fastrag.ingestion.cache.ids import document_cache_id, source_cache_id
from fastrag.ingestion.cache.postgres import PostgresCache
from fastrag.ingestion.cache.base import CacheStore, SourceFreshnessValidator
from fastrag.ingestion.cache.sqlite import SQLiteCache

__all__ = [
    "CacheStore",
    "PostgresCache",
    "SQLiteCache",
    "SourceFreshnessValidator",
    "document_cache_id",
    "source_cache_id",
]
