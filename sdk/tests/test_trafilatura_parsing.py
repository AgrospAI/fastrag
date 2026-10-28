"""Coverage for ``TrafilaturaParser``: heuristic main-content extraction.

Drives the parser directly (``parser.transform(source, context)``) over an
in-test article page. No network: Trafilatura only reads the supplied bytes.
"""

from __future__ import annotations

import json

import pytest

from fastrag.exceptions import ConfigurationError, ParseError
from fastrag.ingestion.models import Source
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext
from fastrag.ingestion.steps.parsing import TrafilaturaParser
from fastrag.models import ContentFormat, Document

PAGE_URI = "https://example.test/blog/2024/soil.html"

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Soil Moisture Sensing</title>
  <meta name="author" content="Ada Farmer">
  <meta name="description" content="How capacitive probes measure soil water.">
  <meta property="article:published_time" content="2024-03-05T10:00:00Z">
</head>
<body>
  <div class="topbar"><a href="/">Home</a> <a href="/about">About</a></div>
  <div class="sidebar"><ul><li><a href="/a">Related A</a></li>
  <li><a href="/b">Related B</a></li></ul></div>
  <article>
    <h1>Soil Moisture Sensing</h1>
    <p>Capacitive probes estimate volumetric water content by measuring the
    dielectric permittivity of the surrounding soil, which rises sharply as
    the pores fill with water after irrigation or rainfall.</p>
    <h2>Calibration</h2>
    <p>Every soil type needs its own curve, as described in the
    <a href="calibration.html">calibration guide</a>, because clay and sand
    hold water at very different tensions for the same reading.</p>
    <table>
      <tr><th>Soil</th><th>Field capacity</th></tr>
      <tr><td>Sand</td><td>10%</td></tr>
      <tr><td>Clay</td><td>40%</td></tr>
    </table>
    <p>Readings drift with temperature and salinity, so a field deployment
    logs both alongside the raw count and corrects for them afterwards.</p>
  </article>
  <div class="footer-links"><a href="/privacy">Privacy</a>
  <a href="/terms">Terms</a></div>
</body>
</html>
"""


def _source(content: bytes, *, content_type: str = "text/html") -> Source:
    return Source(
        id="trafilatura-source",
        uri=PAGE_URI,
        content=content,
        content_type=content_type,
        content_hash="trafilatura-fixture-hash",
    )


async def _transform(parser: TrafilaturaParser, source: Source) -> list[Document]:
    context = ExecutionContext(pipeline_id="test-trafilatura", policy=Policy())
    return [document async for document in parser.transform(source, context)]


async def test_extracts_article_as_markdown_with_metadata() -> None:
    """The article survives as Markdown; the chrome around it does not."""
    [document] = await _transform(TrafilaturaParser(), _source(PAGE.encode()))

    assert "## Calibration" in document.content
    assert "dielectric permittivity" in document.content
    assert "| Sand | 10% |" in document.content
    # Chrome built from anonymous divs, which HtmlParser's selectors miss.
    assert "Related A" not in document.content
    assert "Privacy" not in document.content
    # Relative to the page's own directory, not the host root.
    assert (
        "[calibration guide](https://example.test/blog/2024/calibration.html)"
        in document.content
    )

    assert document.metadata["content_format"] == ContentFormat.MARKDOWN
    assert document.metadata["source_uri"] == PAGE_URI
    assert document.metadata["title"] == "Soil Moisture Sensing"
    assert document.metadata["author"] == "Ada Farmer"
    assert document.metadata["date"] == "2024-03-05"
    assert document.metadata["description"].startswith("How capacitive probes")
    json.dumps(document.metadata)


async def test_options_change_output_and_cache_identity() -> None:
    """Dropping links and tables changes both the text and ``config_hash``."""
    default = TrafilaturaParser()
    plain = TrafilaturaParser(include_links=False, include_tables=False)
    [document] = await _transform(plain, _source(PAGE.encode()))

    assert "calibration guide" in document.content
    assert "](https://" not in document.content
    assert "| Sand |" not in document.content
    assert plain.config_hash != default.config_hash


async def test_blank_and_contentless_pages_yield_nothing() -> None:
    """Neither empty bytes nor a page with no extractable text is a document."""
    parser = TrafilaturaParser()
    assert await _transform(parser, _source(b"   ")) == []
    assert await _transform(parser, _source(b"<html><body></body></html>")) == []


async def test_rejects_non_html_sources_and_conflicting_options() -> None:
    """A PDF is refused up front; precision and recall cannot both be favored."""
    with pytest.raises(ParseError, match="application/pdf"):
        await _transform(
            TrafilaturaParser(), _source(b"%PDF-1.7", content_type="application/pdf")
        )
    with pytest.raises(ConfigurationError, match="mutually exclusive"):
        TrafilaturaParser(favor_precision=True, favor_recall=True)
