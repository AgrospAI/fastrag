"""``ChunkerRouter``: dispatch each document to the chunker registered for it."""

from __future__ import annotations

from collections.abc import Iterator

from fastrag.ingestion.steps.base import RoutedStep
from fastrag.ingestion.steps.chunking.base import Chunker
from fastrag.models import Document


class ChunkerRouter(RoutedStep[Document], Chunker):
    """Route each document to the first matching ``.route`` chunker, or ``.default``.

    Holds no chunking logic of its own: a document matching no route with no
    default registered is skipped with an ``ItemFailed``.

    .. code-block:: python

        from fastrag.ingestion.filters import ContentFormatFilter
        from fastrag.ingestion.steps.chunking import (
            ChunkerRouter,
            DoclingChunker,
            RecursiveCharacterTextSplitter,
        )

        chunker = (
            ChunkerRouter(cache=cache)
            .route(ContentFormatFilter(include_formats=["docling"]), DoclingChunker())
            .default(RecursiveCharacterTextSplitter(chunk_size=500))
        )
    """

    routing_requires_content = False

    async def validate_output(self, item, produced, context) -> None:
        """Forward resolved output validation to the selected child chunker."""
        target = self._select(item)
        if isinstance(target, Chunker):
            await target.validate_output(item, produced, context)

    def split(self, document: Document) -> Iterator[str]:
        """Never called: ``RoutedStep.transform`` shadows ``Chunker.transform``.

        Defined only because ``Chunker`` declares ``split`` abstract.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "ChunkerRouter routes to a child's transform(); split() is never called"
        )
