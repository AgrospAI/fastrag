"""End-to-end coverage for ``fastrag.ingestion.steps.parsing``: ``ParserRouter``
routing and the concrete ``Parser`` implementations.

Drives a real ``ParserRouter`` (HTML to ``HtmlParser``, everything else to
``DefaultParser``) through a real ``Pipeline``, over the fixture
site's mix of HTML pages and one plain-text file. The ``DoclingParser``
tests below drive it directly (``parser.transform(source, context)``, no
``Pipeline``/``http_server``/``cache`` fixtures needed) over an XLSX built
in-test with ``openpyxl`` -- a real dependency here, since it is what
Docling's own XLSX backend depends on. No network, no model download.
``DocusaurusParser``/``RoutedStep`` dispatch is covered in ``test_pipeline.py``,
alongside the other routing/caching-identity contract tests.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    ConvertPipelineOptions,
    PictureDescriptionApiOptions,
)
from docling.document_converter import ExcelFormatOption
from docling_core.types.doc.document import DoclingDocument
from openpyxl import Workbook
from pydantic import AnyUrl

from fastrag.exceptions import ParseError
from fastrag.ingestion import Pipeline
from fastrag.ingestion.cache import PostgresCache
from fastrag.ingestion.events import DocumentParsed
from fastrag.ingestion.models import Source
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext
from fastrag.ingestion.steps.fetching import SitemapCrawler
from fastrag.ingestion.steps.parsing import (
    DefaultParser,
    DoclingParser,
    DocusaurusParser,
    HtmlParser,
    ParserRouter,
    docusaurus_filter,
    html_filter,
)
from fastrag.models import ContentFormat, Document

NOTES_PATH = Path(__file__).parent / "fixtures" / "site" / "notes.txt"


async def _parse(sitemap_url: str, cache: PostgresCache) -> list[Document]:
    crawler = SitemapCrawler(url=sitemap_url)
    parser = (
        ParserRouter()
        .route(html_filter & docusaurus_filter, DocusaurusParser())
        .route(html_filter, HtmlParser())
        .default(DefaultParser())(crawler)
    )
    documents: list[Document] = []
    async with Pipeline(parser, cache=cache) as pipeline:
        async for event in pipeline:
            if isinstance(event, DocumentParsed):
                documents.append(event.document)
    return documents


async def test_parser_router_routes_by_content_type_and_caches_on_rerun(
    http_server: str, cache: PostgresCache
) -> None:
    """HTML routes to ``HtmlParser``, the text file to ``DefaultParser``.

    Every document carries ``source_uri``/``content_type`` metadata, and a
    rerun against the same cache reports every document as ``cached``.
    Each parser also stamps ``content_format`` per its own declared default:
    ``DefaultParser`` -> ``TEXT``, ``HtmlParser`` -> ``MARKDOWN``.
    """
    sitemap_url = f"{http_server}/sitemap.xml"

    first = await _parse(sitemap_url, cache)
    assert len(first) == 7
    by_uri = {document.metadata["source_uri"]: document for document in first}

    text_document = by_uri[f"{http_server}/notes.txt"]
    assert text_document.metadata["content_type"] == "text/plain"
    # DefaultParser only decodes bytes to text; it never touches markup, so
    # the raw fixture text survives byte-for-byte (proving DefaultParser,
    # not HtmlParser, handled it).
    assert text_document.content == NOTES_PATH.read_text().strip()
    assert text_document.metadata["content_format"] == ContentFormat.TEXT

    html_document = by_uri[f"{http_server}/index.html"]
    assert html_document.metadata["content_type"] == "text/html"
    # HtmlParser converts to Markdown: a source "<h1>Fixture Site</h1>"
    # becomes a leading "# Fixture Site", which only appears if the HTML
    # route (not DefaultParser's raw decode) ran.
    assert "# Fixture Site" in html_document.content
    assert "<h1>" not in html_document.content
    assert html_document.metadata["content_format"] == ContentFormat.MARKDOWN

    assert all(not document.cached for document in first)

    second = await _parse(sitemap_url, cache)
    assert len(second) == 7
    assert all(document.cached for document in second)


def _xlsx_source(*, uri: str = "file:///tmp/data.xlsx") -> Source:
    """A ``Source`` wrapping a small, freshly built two-cell XLSX workbook."""
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    sheet["A1"] = "Name"
    sheet["B1"] = "Score"
    sheet["A2"] = "Alice"
    sheet["B2"] = 42

    buffer = io.BytesIO()
    workbook.save(buffer)
    content = buffer.getvalue()

    return Source(
        id="xlsx-source",
        uri=uri,
        content=content,
        content_type=(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
        content_hash="xlsx-fixture-hash",
    )


def _execution_context() -> ExecutionContext:
    """A bare ``ExecutionContext`` -- enough for ``Parser.transform`` alone."""
    return ExecutionContext(pipeline_id="test-docling-parser", policy=Policy())


async def _transform(parser: DoclingParser, source: Source) -> list[Document]:
    return [
        document async for document in parser.transform(source, _execution_context())
    ]


async def test_docling_parser_markdown_and_docling_modes_round_trip() -> None:
    """Markdown mode's table holds the worksheet's own values; ``docling``
    mode's serialized JSON round-trips to exactly the same Markdown --
    proof the JSON in ``Document.content`` is the same document, not a
    lossy stand-in.

    Also covers the metadata every mode carries: ``docling_format`` names
    the matched ``InputFormat``, ``page_count`` counts the one worksheet,
    and everything is JSON-serializable (as it must be to survive a JSONB
    cache column or Milvus JSON metadata).
    """
    source = _xlsx_source()

    markdown_document = (await _transform(DoclingParser(), source))[0]
    assert "Alice" in markdown_document.content
    assert "42" in markdown_document.content
    assert markdown_document.metadata["content_format"] == ContentFormat.MARKDOWN
    assert markdown_document.metadata["docling_format"] == "xlsx"
    assert markdown_document.metadata["page_count"] == 1
    json.dumps(markdown_document.metadata)

    docling_document = (
        await _transform(DoclingParser(output_format="docling"), source)
    )[0]
    assert docling_document.metadata["content_format"] == ContentFormat.DOCLING
    assert docling_document.metadata["docling_format"] == "xlsx"
    assert docling_document.metadata["page_count"] == 1
    json.dumps(docling_document.metadata)

    restored = DoclingDocument.model_validate_json(docling_document.content)
    assert restored.export_to_markdown().strip() == markdown_document.content


async def test_docling_parser_raises_parse_error_naming_the_source_on_corrupt_input() -> (
    None
):
    """A source Docling refuses raises ``ParseError`` naming the source's URI.

    Never a bare Docling exception -- a caller catching
    ``fastrag.exceptions.SourceError`` must catch this too.
    """
    content = b"this is not a real xlsx workbook"
    source = Source(
        id="corrupt-xlsx-source",
        uri="file:///tmp/corrupt.xlsx",
        content=content,
        content_type="application/octet-stream",
        content_hash="corrupt-fixture-hash",
    )

    with pytest.raises(ParseError, match=re.escape(source.uri)):
        await _transform(DoclingParser(), source)


def test_docling_parser_format_options_change_the_cache_identity() -> None:
    """Two parsers differing only in ``format_options`` never share a cache entry.

    ``format_options`` reaches Docling pipeline options that change what is
    extracted from a given file -- picture description being the motivating
    one -- so a parser that describes images must not be served the
    placeholder-only text an earlier, plainer parser cached from the same
    bytes.
    """
    described = DoclingParser(
        format_options={
            InputFormat.XLSX: ExcelFormatOption(
                pipeline_options=ConvertPipelineOptions(
                    do_picture_description=True,
                    picture_description_options=PictureDescriptionApiOptions(
                        url=AnyUrl("https://vision.example/v1/chat/completions"),
                    ),
                    enable_remote_services=True,
                )
            )
        }
    )
    plain = DoclingParser()

    assert plain.config()["format_options"] is None
    assert "xlsx" in described.config()["format_options"]
    assert described.config_hash != plain.config_hash


def test_identical_parsed_text_keeps_independent_source_identity():
    from fastrag.ingestion._hashing import config_hash
    from fastrag.ingestion.steps.parsing import DefaultParser

    parser = DefaultParser()
    a = Source(
        id="source-a",
        uri="https://example.test/a",
        content=b"Same text",
        content_type="text/plain",
        content_hash="same-content",
    )
    b = Source(
        id="source-b",
        uri="https://example.test/b",
        content=b"Same text",
        content_type="text/plain",
        content_hash="same-content",
    )
    first = parser.build_document(a, "Same text")
    second = parser.build_document(b, "Same text")
    assert first.id != second.id
    assert parser.build_document(a, "Same text").id == first.id
    assert parser.build_document(a, "Changed text").id != first.id
    assert first.source_id == a.id and second.source_id == b.id
    assert first.metadata["source_uri"] == a.uri
    assert second.metadata["source_uri"] == b.uri
    assert parser.config_hash != config_hash(type(parser).__name__, parser.config())
