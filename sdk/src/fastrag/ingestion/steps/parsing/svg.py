"""Searchable SVG labels without requiring a vision model."""

from collections.abc import AsyncIterator, Mapping
from typing import Any

from fastrag.ingestion.models import Source
from fastrag.ingestion.steps.base import ExecutionContext
from fastrag.ingestion.steps.parsing.html import HtmlParser
from fastrag.models import Document


class SvgParser(HtmlParser):
    """Extract titles, descriptions, text and foreignObject labels from SVG."""

    accepted_media_types = frozenset({"image/svg+xml"})

    def config(self) -> Mapping[str, Any]:
        """Return label extraction settings for cache identity."""
        return {**super().config(), "svg": True}

    async def transform(
        self, item: Source, context: ExecutionContext
    ) -> AsyncIterator[Document]:
        """Use the HTML content pipeline to preserve meaningful SVG labels."""
        self.validate_input(item)
        async for document in super().transform(item, context):
            yield document
