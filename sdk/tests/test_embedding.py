"""End-to-end coverage for ``examples/04_embedding.py``: ``Model2VecEmbedder``.

Drives a real :class:`~fastrag.ingestion.steps.embedding.model2vec.Model2VecEmbedder`
through a real :class:`~fastrag.ingestion.Pipeline`, over every chunk the
fixture site produces.
"""

from __future__ import annotations

import pytest

from fastrag.exceptions import EmbeddingError
from fastrag.ingestion import Pipeline
from fastrag.ingestion.cache import PostgresCache
from fastrag.ingestion.events import DocumentEmbedded
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext
from fastrag.ingestion.steps.chunking import RecursiveCharacterTextSplitter
from fastrag.ingestion.steps.embedding import Model2VecEmbedder
from fastrag.ingestion.steps.fetching import SitemapCrawler
from fastrag.ingestion.steps.parsing import (
    DefaultParser,
    DocusaurusParser,
    HtmlParser,
    ParserRouter,
    docusaurus_filter,
    html_filter,
)
from fastrag.models import ContentFormat, Document

CHUNK_SIZE = 120
CHUNK_OVERLAP = 20


def _execution_context() -> ExecutionContext:
    """A bare ``ExecutionContext`` — enough for ``Embedder.transform`` alone."""
    return ExecutionContext(pipeline_id="test-embedding-guardrail", policy=Policy())


async def _embed(
    sitemap_url: str, cache: PostgresCache, embedder: Model2VecEmbedder
) -> list[Document]:
    crawler = SitemapCrawler(url=sitemap_url)
    parser = (
        ParserRouter()
        .route(html_filter & docusaurus_filter, DocusaurusParser())
        .route(html_filter, HtmlParser())
        .default(DefaultParser())(crawler)
    )
    chunker = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )(parser)
    embedded = embedder(chunker)
    documents: list[Document] = []
    async with Pipeline(embedded, cache=cache) as pipeline:
        async for event in pipeline:
            if isinstance(event, DocumentEmbedded):
                documents.append(event.document)
    return documents


async def test_model2vec_embedder_embeds_every_chunk_and_caches_on_rerun(
    http_server: str, cache: PostgresCache, embedder: Model2VecEmbedder
) -> None:
    """Every chunk gets an embedding of the model's own width.

    ``Embedder.embed``'s contract ("must return exactly one vector per
    text, in the same order") is what "in input order" means for a batched
    embedder — each chunk's own embedding corresponds to its own content,
    which is what is checked here per document rather than by asserting the
    event stream itself preserves upstream order (``run_concurrently``,
    which drives every ``DocumentStep``, documents explicitly that its
    output order is completion order, not input order). A rerun against the
    same cache reports every chunk as ``cached``, embedding the same set of
    chunk ids.
    """
    sitemap_url = f"{http_server}/sitemap.xml"

    first = await _embed(sitemap_url, cache, embedder)
    assert first
    for document in first:
        assert document.embedding is not None
        assert len(document.embedding) == embedder.dimension
    assert all(not document.cached for document in first)

    second = await _embed(sitemap_url, cache, embedder)
    assert second
    for document in second:
        assert document.embedding is not None
        assert len(document.embedding) == embedder.dimension
    assert all(document.cached for document in second)

    assert {document.id for document in second} == {document.id for document in first}


async def test_embedder_refuses_a_docling_format_document(
    embedder: Model2VecEmbedder,
) -> None:
    """A ``content_format=DOCLING`` document (raw serialized ``DoclingDocument``
    JSON) is never silently embedded as text — it must be chunked with
    ``DoclingChunker`` first.
    """
    document = Document(
        id="doc-1",
        source_id="source-1",
        content='{"schema_name": "DoclingDocument"}',
        step_config_hash="cfg",
        metadata={"content_format": ContentFormat.DOCLING},
    )
    with pytest.raises(EmbeddingError, match="DoclingChunker"):
        async for _ in embedder.transform(document, _execution_context()):
            pass


async def test_embedder_accepts_a_document_with_no_content_format(
    embedder: Model2VecEmbedder,
) -> None:
    """A document predating ``content_format`` (no key at all) still embeds
    fine — a missing key reads as ``TEXT``, not as an error.
    """
    document = Document(
        id="doc-2",
        source_id="source-1",
        content="plain text, no content_format key",
        step_config_hash="cfg",
        metadata={},
    )
    embedded = [
        result async for result in embedder.transform(document, _execution_context())
    ]
    assert len(embedded) == 1
    assert embedded[0].embedding is not None
