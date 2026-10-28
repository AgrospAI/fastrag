"""``URLFilter``: filters sources or documents by URI regex patterns."""

from __future__ import annotations

import re
from collections.abc import Sequence

from fastrag.ingestion.filters.base import Filter
from fastrag.ingestion.models import Source
from fastrag.models import Document


class URLFilter(Filter[Source | Document]):
    """Filters sources or documents by URI regex patterns."""

    def __init__(
        self,
        include_patterns: Sequence[str] | None = None,
        exclude_patterns: Sequence[str] | None = None,
    ) -> None:
        """Initialize URLFilter.

        Args:
            include_patterns: Regex patterns an ID must match.
            exclude_patterns: Regex patterns an ID must not match.
        """
        self._include_patterns = tuple(include_patterns) if include_patterns else ()
        self._exclude_patterns = tuple(exclude_patterns) if exclude_patterns else ()

    @property
    def header_safe(self) -> bool:
        """This predicate uses identity/metadata only."""
        return True

    def matches(self, item: Source | Document) -> bool:
        """Return True when ``item``'s URI passes the filter.

        Reads ``Source.uri`` directly, and a ``Document``'s
        ``metadata["source_uri"]``, falling back to ``source_id`` when absent.
        """
        uri = (
            item.uri
            if isinstance(item, Source)
            else item.metadata.get("source_uri", item.source_id)
        )
        if self._exclude_patterns and any(
            re.search(pattern, uri) for pattern in self._exclude_patterns
        ):
            return False
        if self._include_patterns:
            return any(re.search(pattern, uri) for pattern in self._include_patterns)
        return True
