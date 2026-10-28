"""``ContentFormatFilter``: filters documents by their content format."""

from __future__ import annotations

from collections.abc import Sequence

from fastrag.ingestion.filters.base import Filter
from fastrag.models import ContentFormat, Document


class ContentFormatFilter(Filter[Document]):
    """Filters documents by ``metadata["content_format"]``.

    Unlike :class:`~fastrag.ingestion.filters.mediatype.MediaTypeFilter`, a
    document carrying no recorded format is not treated as unmatchable — it
    is treated as :attr:`~fastrag.models.ContentFormat.TEXT`, since every
    document that existed before ``content_format`` was introduced is plain
    text by definition.
    """

    def __init__(
        self,
        include_formats: Sequence[ContentFormat | str] | None = None,
        exclude_formats: Sequence[ContentFormat | str] | None = None,
    ) -> None:
        """Initialize ContentFormatFilter.

        Args:
            include_formats: Allowed content formats (e.g.
                ``[ContentFormat.MARKDOWN]``).
            exclude_formats: Disallowed content formats.
        """
        self._include_formats = tuple(include_formats) if include_formats else ()
        self._exclude_formats = tuple(exclude_formats) if exclude_formats else ()

    @property
    def header_safe(self) -> bool:
        """This predicate uses identity/metadata only."""
        return True

    def matches(self, item: Document) -> bool:
        """Return True when ``item``'s content format passes the filter.

        Reads ``item.metadata["content_format"]``, defaulting to
        :attr:`~fastrag.models.ContentFormat.TEXT` when the key is absent.
        """
        content_format = item.metadata.get("content_format", ContentFormat.TEXT)
        if self._exclude_formats and content_format in self._exclude_formats:
            return False
        if self._include_formats:
            return content_format in self._include_formats
        return True
