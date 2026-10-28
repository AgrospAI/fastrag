"""``MediaTypeFilter``: filters sources or documents by MIME media type."""

from __future__ import annotations

import mimetypes
from collections.abc import Sequence

from fastrag.ingestion.filters.base import Filter
from fastrag.ingestion.models import Source
from fastrag.models import Document


class MediaTypeFilter(Filter[Source | Document]):
    """Filters sources or documents by MIME media type."""

    def __init__(
        self,
        include_types: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
    ) -> None:
        """Initialize MediaTypeFilter.

        Args:
            include_types: Allowed MIME types (e.g. ``["text/html"]``).
            exclude_types: Disallowed MIME types.
        """
        self._include_types = tuple(include_types) if include_types else ()
        self._exclude_types = tuple(exclude_types) if exclude_types else ()

    @property
    def header_safe(self) -> bool:
        """This predicate uses identity/metadata only."""
        return True

    def matches(self, item: Source | Document) -> bool:
        """Return True when ``item``'s content type passes the filter.

        Reads ``Source.content_type`` directly, and ``Document`` from
        ``metadata["content_type"]``, comparing the bare media type (no
        ``charset`` or other parameters). A source sent as
        ``application/octet-stream`` is matched by the type its URI's
        extension implies, when it implies one. A ``Document`` with no
        recorded content type matches only when ``include_types`` is empty.
        """
        if isinstance(item, Source):
            content_type = item.content_type
        else:
            content_type = item.metadata.get("content_type")
        if content_type is None:
            return not self._include_types
        content_type = content_type.split(";", 1)[0].strip().lower()
        if content_type == "application/octet-stream" and isinstance(item, Source):
            content_type = mimetypes.guess_type(item.uri)[0] or content_type
        if self._exclude_types and content_type in self._exclude_types:
            return False
        if self._include_types:
            return content_type in self._include_types
        return True
