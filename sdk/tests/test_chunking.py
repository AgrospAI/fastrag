"""End-to-end coverage for ``fastrag.ingestion.steps.chunking``: every concrete Chunker.

``RecursiveCharacterTextSplitter`` is driven through a real ``Pipeline`` over
the fixture site's one long page. ``DoclingChunker`` is driven both as a pure
unit (``chunker.split(document)``, no ``Pipeline`` needed) and end to end
through ``ChunkerRouter`` routing, a real XLSX, and ``VectorRetriever`` -- task
012's two Docling ingestion approaches: Markdown mode
(``DoclingParser(output_format="markdown")``) feeds whatever text chunker the
caller already uses, completely unmodified; native Docling mode
(``DoclingParser(output_format="docling")``) chunks the real
``DoclingDocument`` structure with ``DoclingChunker``, carrying
``headings``/``page_no`` into chunk metadata. Both XLSX routes run against a
real, locally generated workbook (via ``openpyxl``) -- no network, no model
download.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.line_chunker import LineBasedTokenChunker
from helpers import WordTokenizer
from openpyxl import Workbook

from fastrag.exceptions import ChunkingError
from fastrag.ingestion import Pipeline
from fastrag.ingestion.cache import PostgresCache
from fastrag.ingestion.events import DocumentChunked, DocumentEmbedded, DocumentSaved
from fastrag.ingestion.filters import ContentFormatFilter
from fastrag.ingestion.steps.chunking import (
    ChunkerRouter,
    DoclingChunker,
    RecursiveCharacterTextSplitter,
)
from fastrag.ingestion.steps.embedding import Model2VecEmbedder
from fastrag.ingestion.steps.fetching import LocalFileFetcher, SitemapCrawler
from fastrag.ingestion.steps.parsing import (
    DefaultParser,
    DoclingParser,
    DocusaurusParser,
    HtmlParser,
    ParserRouter,
    docusaurus_filter,
    html_filter,
    office_filter,
)
from fastrag.models import ContentFormat, Document
from fastrag.retrieval import VectorRetriever
from fastrag.vector_store import MilvusVectorStore

CHUNK_SIZE = 120
CHUNK_OVERLAP = 20
#: The cell value the Docling-route tests query for.
CELL_VALUE = "Alice"


# -- RecursiveCharacterTextSplitter ----------------------------------------------------------


async def _chunk(sitemap_url: str, cache: PostgresCache) -> list[DocumentChunked]:
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
    events: list[DocumentChunked] = []
    async with Pipeline(chunker, cache=cache) as pipeline:
        async for event in pipeline:
            if isinstance(event, DocumentChunked):
                events.append(event)
    return events


async def test_recursive_splitter_chunks_the_long_page_and_caches_on_rerun(
    http_server: str, cache: PostgresCache
) -> None:
    """The long page splits into several chunks, each correctly identified.

    Every chunk carries its parent's id and a 0-based ``chunk_index``, no
    chunk is whitespace-only, and a rerun against the same cache reports
    every chunk as ``cached``.
    """
    sitemap_url = f"{http_server}/sitemap.xml"

    first = await _chunk(sitemap_url, cache)
    by_uri = {event.document.metadata["source_uri"]: event for event in first}
    long_event = by_uri[f"{http_server}/long.html"]
    chunks = long_event.chunks
    assert len(chunks) > 1

    chunk_indices = [chunk.metadata["chunk_index"] for chunk in chunks]
    assert chunk_indices[0] == 0
    assert all(a < b for a, b in zip(chunk_indices, chunk_indices[1:], strict=False))
    assert all(
        chunk.metadata["parent_id"] == long_event.document.id for chunk in chunks
    )

    all_chunks = [chunk for event in first for chunk in event.chunks]
    assert all_chunks
    assert all(chunk.content.strip() for chunk in all_chunks)  # none whitespace-only
    assert all(not chunk.cached for chunk in all_chunks)

    second = await _chunk(sitemap_url, cache)
    second_chunks = [chunk for event in second for chunk in event.chunks]
    assert second_chunks
    assert all(chunk.cached for chunk in second_chunks)


# -- DoclingChunker: pure-unit contract ----------------------------------------------------------


def _docling_document_json() -> str:
    """A real, small ``DoclingDocument``, serialized -- same shape
    ``test_parsing.py``'s ``_xlsx_source`` builds, converted directly rather
    than via ``DoclingParser`` (``DoclingChunker.split`` is synchronous, so
    nothing here needs an ``ExecutionContext``).
    """
    workbook = Workbook()
    sheet = workbook.active
    sheet["A1"] = "Name"
    sheet["B1"] = "Score"
    sheet["A2"] = "Alice"
    sheet["B2"] = 42

    buffer = io.BytesIO()
    workbook.save(buffer)
    buffer.seek(0)

    stream = DocumentStream(name="chunking-test.xlsx", stream=buffer)
    document = DocumentConverter().convert(stream).document
    return document.model_dump_json()


def _docling_document() -> Document:
    return Document(
        id="parent-doc",
        source_id="source-1",
        content=_docling_document_json(),
        step_config_hash="cfg",
        metadata={"content_format": ContentFormat.DOCLING},
    )


def test_docling_chunker_unit_contract() -> None:
    """``DoclingChunker.split`` as a pure unit: the default ``HierarchicalChunker``
    route (each yielded piece is ``(text, metadata)``, carrying ``headings``/
    ``page_no``); its refusal of a document that is not Docling-formatted,
    whether that is a different declared format or none at all; and wiring
    in a caller-constructed ``HybridChunker``/``LineBasedTokenChunker``.
    """
    default_pieces = list(DoclingChunker().split(_docling_document()))
    assert default_pieces
    text, metadata = default_pieces[0]
    assert "Alice" in text
    assert metadata["headings"] == []  # a bare spreadsheet declares no headings
    assert metadata["page_no"] == 1

    markdown_document = Document(
        id="doc-1",
        source_id="source-1",
        content="# Just markdown",
        step_config_hash="cfg",
        metadata={"content_format": ContentFormat.MARKDOWN},
    )
    with pytest.raises(ChunkingError, match="DOCLING"):
        list(DoclingChunker().split(markdown_document))

    no_format_document = Document(
        id="doc-2",
        source_id="source-1",
        content="plain text",
        step_config_hash="cfg",
        metadata={},
    )
    with pytest.raises(ChunkingError, match="DOCLING"):
        list(DoclingChunker().split(no_format_document))

    hybrid_pieces = list(
        DoclingChunker(chunker=HybridChunker(tokenizer=WordTokenizer())).split(
            _docling_document()
        )
    )
    assert hybrid_pieces and "Alice" in hybrid_pieces[0][0]

    line_pieces = list(
        DoclingChunker(chunker=LineBasedTokenChunker(tokenizer=WordTokenizer())).split(
            _docling_document()
        )
    )
    assert line_pieces and "Alice" in line_pieces[0][0]


# -- DoclingChunker/DoclingParser: end to end over a real XLSX ----------------------------------------------------------


def _write_xlsx(path: Path) -> None:
    """Write a small, real two-cell XLSX workbook to ``path``."""
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "People"
    sheet["A1"] = "Name"
    sheet["B1"] = "Score"
    sheet["A2"] = CELL_VALUE
    sheet["B2"] = 42
    workbook.save(path)


async def test_docling_chunker_end_to_end_over_a_real_xlsx_via_vector_retrieval(
    tmp_path: Path,
    cache: PostgresCache,
    vector_store: MilvusVectorStore,
    embedder: Model2VecEmbedder,
) -> None:
    """The native-Docling route: XLSX bytes to a retrieved, headings-intact chunk.

    ``LocalFileFetcher`` -> ``ParserRouter`` (routed to
    ``DoclingParser(output_format="docling")``) -> ``ChunkerRouter`` (routed to
    ``DoclingChunker``) -> ``Model2VecEmbedder`` -> ``MilvusVectorStore`` ->
    ``VectorRetriever``, all under one real ``Pipeline`` so lifecycle events
    fire. Querying for the cell value returns a chunk carrying it, with
    ``headings`` (an empty list -- a bare spreadsheet declares none) and
    ``page_no`` both present. A rerun against the same cache and collection
    reports every document as ``cached``, with the same chunk ids both times.
    """
    xlsx_path = tmp_path / "people.xlsx"
    _write_xlsx(xlsx_path)

    async def _index() -> list[Document]:
        fetcher = LocalFileFetcher(patterns=[str(xlsx_path)])
        parser = ParserRouter(cache=cache).route(
            office_filter, DoclingParser(output_format="docling")
        )(fetcher)
        chunker = (
            ChunkerRouter(cache=cache)
            .route(
                ContentFormatFilter(include_formats=[ContentFormat.DOCLING]),
                DoclingChunker(),
            )
            .default(RecursiveCharacterTextSplitter(chunk_size=500))
        )(parser)
        embedded = embedder(chunker)
        saved_step = vector_store.save(embedded)
        saved: list[Document] = []
        async with Pipeline(saved_step, cache=cache) as pipeline:
            async for event in pipeline:
                if isinstance(event, DocumentSaved):
                    saved.append(event.document)
        return saved

    first = await _index()
    assert first
    assert all(
        document.metadata["content_format"] == ContentFormat.TEXT for document in first
    )
    assert all("headings" in document.metadata for document in first)
    assert all(not document.cached for document in first)

    retriever = VectorRetriever(vector_store=vector_store, embedder=embedder)
    results = await retriever.retrieve(CELL_VALUE, k=3)
    match = next(
        (document for document in results if CELL_VALUE in document.content), None
    )
    assert match is not None, (
        f"no retrieved chunk contains {CELL_VALUE!r}: "
        f"{[document.content for document in results]}"
    )
    assert match.metadata["headings"] == []
    assert match.metadata["page_no"] == 1
    assert match.metadata["content_format"] == ContentFormat.TEXT
    assert match.metadata["parent_id"]
    assert isinstance(match.metadata["chunk_index"], int)

    second = await _index()
    assert second
    assert all(document.cached for document in second)
    assert {document.id for document in second} == {document.id for document in first}


async def test_docling_parser_markdown_mode_feeds_the_existing_recursive_splitter(
    tmp_path: Path, cache: PostgresCache, embedder: Model2VecEmbedder
) -> None:
    """The Markdown route: ``DoclingParser(output_format="markdown")`` + the
    existing splitter, completely unmodified.

    No ``DoclingChunker`` involved at all -- proves a Markdown-mode Docling
    document keeps working with an ordinary text chunker, producing chunks
    that are actually embeddable: each carries ``content_format =
    MARKDOWN``, the cell value survives into chunk text, and a real
    ``Model2VecEmbedder`` call succeeds on every one of them.
    """
    xlsx_path = tmp_path / "people.xlsx"
    _write_xlsx(xlsx_path)

    fetcher = LocalFileFetcher(patterns=[str(xlsx_path)])
    parser = ParserRouter(cache=cache).route(office_filter, DoclingParser())(fetcher)
    chunker = RecursiveCharacterTextSplitter(chunk_size=500, cache=cache)(parser)
    embedded = embedder(chunker)

    documents: list[Document] = []
    async with Pipeline(embedded, cache=cache) as pipeline:
        async for event in pipeline:
            if isinstance(event, DocumentEmbedded):
                documents.append(event.document)

    assert documents
    assert all(
        document.metadata["content_format"] == ContentFormat.MARKDOWN
        for document in documents
    )
    assert any(CELL_VALUE in document.content for document in documents)
    assert all(document.embedding is not None for document in documents)
    assert all(len(document.embedding) == embedder.dimension for document in documents)
