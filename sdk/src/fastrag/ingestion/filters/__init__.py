"""``fastrag.ingestion.filters`` — composable predicates over sources and documents.

Re-exports :class:`~fastrag.ingestion.filters.base.Filter`, the composable
predicate base, and its three concrete filters,
:class:`~fastrag.ingestion.filters.mediatype.MediaTypeFilter`,
:class:`~fastrag.ingestion.filters.url.URLFilter`, and
:class:`~fastrag.ingestion.filters.contentformat.ContentFormatFilter`.
"""

from __future__ import annotations

from fastrag.ingestion.filters.base import Filter
from fastrag.ingestion.filters.contentformat import ContentFormatFilter
from fastrag.ingestion.filters.mediatype import MediaTypeFilter
from fastrag.ingestion.filters.url import URLFilter

__all__ = ["ContentFormatFilter", "Filter", "MediaTypeFilter", "URLFilter"]
