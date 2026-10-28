"""Coverage for ``fastrag.ingestion``'s own execution contract: the ``Pipeline``
event stream, ``Policy``, ``RoutedStep`` dispatch and per-child caching,
filter composition, ``ConsoleLogger``, and the ``CacheStore`` contract
(``PostgresCache``) every step reads and writes through.

Caching is explicit and per-step (see ``fastrag.ingestion.cache``'s module
docstring): ``PostgresCache``'s own deletion/GC surface, its default
inherit-from-pipeline behaviour, and its persisted validation timestamp all
live here alongside the pipeline mechanics they support, since none of them
are specific to any one fetcher/parser/chunker.
"""

from __future__ import annotations

import asyncio
import functools
import http.server
import io
import re
import threading
from collections import Counter
from collections.abc import AsyncIterator, Iterator, Mapping
from datetime import timedelta
from typing import Any
from uuid import uuid4

import asyncpg
import pytest

#: Imported, not re-declared: the ``cache_validation`` test below needs a
#: brand-new ``PostgresCache`` instance per crawl (no shared Python state), so
#: it cannot go through the shared ``cache`` fixture -- but it must still point
#: at whatever Postgres ``conftest.py`` resolved from ``FASTRAG_DATABASE_URL``,
#: which is ``postgres:5432`` under ``task e2e`` and ``localhost:5433`` under
#: ``task test``.
from conftest import DATABASE_URL

from fastrag.ingestion import Pipeline, Policy
from fastrag.ingestion.cache import PostgresCache, document_cache_id
from fastrag.ingestion.events import (
    DocumentChunked,
    DocumentEmbedded,
    DocumentFetched,
    DocumentParsed,
    DocumentSaved,
    ItemFailed,
    PipelineEvent,
)
from fastrag.ingestion.filters import ContentFormatFilter, MediaTypeFilter, URLFilter
from fastrag.ingestion.logger import ConsoleLogger
from fastrag.ingestion.models import Source
from fastrag.ingestion.steps.base import ExecutionContext, SourceStep
from fastrag.ingestion.steps.chunking import (
    ChunkerRouter,
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from fastrag.ingestion.steps.embedding import Model2VecEmbedder
from fastrag.ingestion.steps.fetching import SitemapCrawler, WebCrawler
from fastrag.ingestion.steps.parsing import (
    DefaultParser,
    DocusaurusParser,
    HtmlParser,
    ParserRouter,
    docusaurus_filter,
    html_filter,
)
from fastrag.ingestion.steps.parsing.base import Parser
from fastrag.ingestion.steps.parsing.docusaurus import FALLBACK_METADATA_KEY
from fastrag.models import ContentFormat, Document
from fastrag.vector_store import MilvusVectorStore

# -- Pipeline event stream ----------------------------------------------------------


async def _run_full_graph(
    start_url: str,
    cache: PostgresCache,
    vector_store: MilvusVectorStore,
    embedder: Model2VecEmbedder,
) -> list[PipelineEvent]:
    crawler = WebCrawler(url=start_url, max_depth=1)
    parser = (
        ParserRouter()
        .route(html_filter & docusaurus_filter, DocusaurusParser())
        .route(html_filter, HtmlParser())
        .default(DefaultParser())(crawler)
    )
    chunker = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)(parser)
    embedded = embedder(chunker)
    saved = vector_store.save(embedded)
    events: list[PipelineEvent] = []
    async with Pipeline(saved, cache=cache) as pipeline:
        async for event in pipeline:
            events.append(event)
    return events


async def test_pipeline_event_stream_carries_every_item_and_survives_a_dead_link(
    http_server: str,
    cache: PostgresCache,
    vector_store: MilvusVectorStore,
    embedder: Model2VecEmbedder,
) -> None:
    """Every domain event carries the real item, correctly marked ``cached``.

    ``index.html`` links to ``page-a.html``, ``page-b.html``, and the dead
    ``missing.html`` -- with ``max_depth=1``, ``WebCrawler`` fetches all
    three at the next level, so exactly one page fails. Under the default
    ``policy.on_error == "skip"``, that single failure must not stop the
    crawl, nor anything downstream of it. A second run against the same
    (now warm) cache must report ``cached`` on every domain event at every
    stage, and must not re-attempt the dead link at all -- the whole crawl
    is one cache bucket, keyed by the start URL.
    """
    start_url = f"{http_server}/index.html"

    cold = await _run_full_graph(start_url, cache, vector_store, embedder)

    fetched = [e for e in cold if isinstance(e, DocumentFetched)]
    parsed = [e for e in cold if isinstance(e, DocumentParsed)]
    chunked = [e for e in cold if isinstance(e, DocumentChunked)]
    embedded_events = [e for e in cold if isinstance(e, DocumentEmbedded)]
    saved = [e for e in cold if isinstance(e, DocumentSaved)]
    failed = [e for e in cold if isinstance(e, ItemFailed)]

    fetched_uris = {e.source.uri for e in fetched}
    assert any(u.endswith("index.html") for u in fetched_uris)
    assert any(u.endswith("page-a.html") for u in fetched_uris)
    assert any(u.endswith("page-b.html") for u in fetched_uris)
    assert not any(u.endswith("missing.html") for u in fetched_uris)
    assert not any(e.source.cached for e in fetched)
    assert not any(e.document.cached for e in parsed)
    assert chunked and all(not chunk.cached for e in chunked for chunk in e.chunks)
    assert embedded_events and not any(e.document.cached for e in embedded_events)
    assert saved and not any(e.document.cached for e in saved)

    assert len(failed) == 1, f"expected exactly one ItemFailed, got {failed}"
    failure = failed[0]
    assert failure.step == "WebCrawler"
    assert "missing.html" in failure.uri
    assert failure.source_id, "the failing URL must be recorded as the source_id"
    assert failure.source_id == failure.uri
    # The run reached every downstream stage despite the failure -- it did
    # not abort.
    assert len(saved) >= len(fetched_uris)

    warm = await _run_full_graph(start_url, cache, vector_store, embedder)
    warm_fetched = [e for e in warm if isinstance(e, DocumentFetched)]
    warm_parsed = [e for e in warm if isinstance(e, DocumentParsed)]
    warm_chunked = [e for e in warm if isinstance(e, DocumentChunked)]
    warm_embedded = [e for e in warm if isinstance(e, DocumentEmbedded)]
    warm_saved = [e for e in warm if isinstance(e, DocumentSaved)]
    warm_failed = [e for e in warm if isinstance(e, ItemFailed)]

    assert warm_fetched and all(e.source.cached for e in warm_fetched)
    assert warm_parsed and all(e.document.cached for e in warm_parsed)
    assert warm_chunked and all(
        chunk.cached for e in warm_chunked for chunk in e.chunks
    )
    assert warm_embedded and all(e.document.cached for e in warm_embedded)
    assert warm_saved and all(e.document.cached for e in warm_saved)
    assert warm_failed == [], "a fully cached rerun must not retry the dead link"


# -- Policy governance ----------------------------------------------------------

CONTENT = b"one source, one slow parse"


class _OneSource(SourceStep):
    """Yields exactly one in-memory ``Source``, doing no I/O."""

    def config(self) -> Mapping[str, Any]:
        return {}

    async def evaluate(self, context: ExecutionContext) -> AsyncIterator[Source]:
        yield Source(
            id="slow-source",
            uri="memory:///slow.txt",
            content=CONTENT,
            content_type="text/plain",
            content_hash="slow-source-hash",
        )


class _SlowParser(Parser):
    """A parser whose one item takes ``delay`` seconds to produce."""

    def __init__(self, *, delay: float, policy: Policy | None = None) -> None:
        super().__init__(cache=False, policy=policy)
        self.delay = delay

    def config(self) -> Mapping[str, Any]:
        return {"delay": self.delay}

    async def transform(
        self, item: Source, context: ExecutionContext
    ) -> AsyncIterator[Document]:
        await asyncio.sleep(self.delay)
        yield self.build_document(item, item.content.decode())


async def _events(root: Any) -> list[PipelineEvent]:
    async with Pipeline(root) as pipeline:
        return [event async for event in pipeline]


async def test_policy_governs_routed_children_and_reports_timeouts() -> None:
    """A routed child's own ``policy`` beats the router's for the child's
    own work, and an expired timeout reports a non-empty, ``TimeoutError``
    message.

    The retry/timeout wrap lives in the evaluator driving the *router*, one
    level above the child that does the work, so without
    ``RoutedStep._policy_for_item`` the child's ``policy`` would be silently
    ignored -- the exact shape of ``ParserRouter(...).default(DoclingParser(
    policy=Policy(timeout=600)))``, where a conversion that legitimately
    takes minutes gets cancelled by the router's 30-second default.
    ``asyncio.timeout`` raises an argument-less ``TimeoutError``, which a
    consumer formatting the failure as ``f"{error}"`` would otherwise render
    as an empty string -- the type must stay ``TimeoutError`` (callers catch
    it), but it must carry a message naming the timeout that expired.
    Nothing here touches the network, Postgres, or Milvus -- every step
    declares ``cache=False``, and the "work" is an ``asyncio.sleep``.
    """
    governed_router = ParserRouter(cache=False, policy=Policy(timeout=0.05, retries=0))
    governed_router.default(
        _SlowParser(delay=0.3, policy=Policy(timeout=5.0, retries=0))
    )
    governed_events = await _events(governed_router(_OneSource(cache=False)))

    assert not [e for e in governed_events if isinstance(e, ItemFailed)]
    parsed = [e for e in governed_events if isinstance(e, DocumentParsed)]
    assert len(parsed) == 1
    assert parsed[0].document.content == CONTENT.decode()

    expiring_router = ParserRouter(cache=False, policy=Policy(timeout=0.05, retries=0))
    expiring_router.default(_SlowParser(delay=1.0))
    expiring_events = await _events(expiring_router(_OneSource(cache=False)))

    failures = [e for e in expiring_events if isinstance(e, ItemFailed)]
    assert len(failures) == 1
    error = failures[0].error
    assert isinstance(error, TimeoutError)
    assert str(error), "a timed-out item must not be reported with an empty message"
    assert "0.05" in str(error)


# -- RoutedStep dispatch, per-child caching, and filter composition -----------------


_docs_filter = URLFilter(include_patterns=[r"docs\.html$"])


def _build_routed_parser(
    *, strip_selectors: tuple[str, ...] | None = None
) -> ParserRouter:
    docusaurus = (
        DocusaurusParser(strip_selectors=strip_selectors)
        if strip_selectors is not None
        else DocusaurusParser()
    )
    return ParserRouter().route(_docs_filter, docusaurus).default(DefaultParser())


def _build_routed_chunker() -> ChunkerRouter:
    return (
        ChunkerRouter()
        .route(_docs_filter, CharacterTextSplitter(chunk_size=100, chunk_overlap=20))
        .default(RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20))
    )


async def _fetch_and_parse(
    sitemap_url: str,
    cache: PostgresCache,
    *,
    strip_selectors: tuple[str, ...] | None = None,
):
    crawler = SitemapCrawler(url=sitemap_url)
    parser = _build_routed_parser(strip_selectors=strip_selectors)(crawler)
    chunker = _build_routed_chunker()(parser)
    fetched = {}
    parsed = {}
    async with Pipeline(chunker, cache=cache) as pipeline:
        async for event in pipeline:
            if isinstance(event, DocumentFetched):
                fetched[event.source.uri] = event.source
            elif isinstance(event, DocumentParsed):
                parsed[event.document.metadata["source_uri"]] = event.document
    return fetched, parsed


def _content_format_document(
    content_format: ContentFormat | None, content_type: str | None
) -> Document:
    metadata: dict[str, object] = {}
    if content_format is not None:
        metadata["content_format"] = content_format
    if content_type is not None:
        metadata["content_type"] = content_type
    return Document(
        id=content_type or "none", source_id="s", content="x", metadata=metadata
    )


async def test_routed_dispatch_caches_under_the_matched_child_and_filters_compose(
    http_server: str, cache: PostgresCache
) -> None:
    """Routing correctness, per-child cache buckets, per-route invalidation,
    and (as a pure unit, no fixtures) ``ContentFormatFilter``'s own matching
    and its ``&``/``|``/``~`` composition with ``MediaTypeFilter``.

    The three routing assertions all follow from ``RoutedStep``/
    ``_dispatch_route``: a routed item is cached under *the matched child's*
    identity (``document_cache_id(child.name, item.id, child.config_hash)``),
    never the router's own -- so routing is correct, caching lands in the
    right bucket, and changing one route's configuration invalidates only
    that route. ``docs.html`` is the only fixture page carrying Docusaurus
    markup, so it is the one page routed away from the ``.default(...)``.
    """
    sitemap_url = f"{http_server}/sitemap.xml"

    # --- cold run: docs.html reaches DocusaurusParser, everything else the default ---
    fetched, parsed = await _fetch_and_parse(sitemap_url, cache)
    docs_uri = next(uri for uri in fetched if uri.endswith("docs.html"))
    other_uris = [uri for uri in fetched if uri != docs_uri]
    assert other_uris, "the fixture site must have pages besides docs.html"

    docs_document = parsed[docs_uri]
    assert FALLBACK_METADATA_KEY in docs_document.metadata, (
        "docs.html must be parsed by DocusaurusParser, the only parser that "
        "sets this key"
    )
    assert docs_document.metadata[FALLBACK_METADATA_KEY] is False, (
        "docs.html carries a real markdown div; DocusaurusParser must find it, "
        "not fall back to parsing the whole page"
    )
    assert docs_document.metadata["content_format"] == ContentFormat.MARKDOWN
    for uri in other_uris:
        assert FALLBACK_METADATA_KEY not in parsed[uri].metadata, (
            f"{uri} must be routed to the default parser, not DocusaurusParser"
        )
        # DefaultParser retains raw source syntax and declares it truthfully.
        expected_format = (
            ContentFormat.HTML
            if fetched[uri].content_type.split(";", 1)[0] == "text/html"
            else ContentFormat.TEXT
        )
        assert parsed[uri].metadata["content_format"] == expected_format
    assert not any(document.cached for document in parsed.values())

    # --- the routed document lives in the matched child's own cache bucket ---
    docs_source_id = fetched[docs_uri].id
    other_source_id = fetched[other_uris[0]].id

    docusaurus_cache_id = document_cache_id(
        "DocusaurusParser", docs_source_id, DocusaurusParser().config_hash
    )
    assert await cache.get_document_entries(docusaurus_cache_id) is not None

    router_cache_id = document_cache_id(
        "ParserRouter", docs_source_id, ParserRouter().config_hash
    )
    assert await cache.get_document_entries(router_cache_id) is None, (
        "a routed document must never be cached under the router's own identity"
    )

    default_cache_id = document_cache_id(
        "DefaultParser", other_source_id, DefaultParser().config_hash
    )
    assert await cache.get_document_entries(default_cache_id) is not None

    # --- warm rerun: every route's own bucket is now a hit ---
    _, warm_parsed = await _fetch_and_parse(sitemap_url, cache)
    assert all(document.cached for document in warm_parsed.values())

    # --- changing the docs.html route's own configuration invalidates only
    # that route's entries; every page still going through the (unchanged)
    # DefaultParser default stays cached ---
    _, reconfigured_parsed = await _fetch_and_parse(
        sitemap_url, cache, strip_selectors=("nav", "footer", "header")
    )
    assert not reconfigured_parsed[docs_uri].cached
    for uri in other_uris:
        assert reconfigured_parsed[uri].cached, (
            f"{uri}'s DefaultParser route is unchanged and must remain cached"
        )

    # --- ContentFormatFilter: matching, missing-key default, composition ---
    markdown_doc = _content_format_document(ContentFormat.MARKDOWN, "text/html")
    text_doc = _content_format_document(ContentFormat.TEXT, "text/plain")
    docling_doc = _content_format_document(ContentFormat.DOCLING, "application/json")
    documents = [markdown_doc, text_doc, docling_doc]

    markdown_filter = ContentFormatFilter(include_formats=[ContentFormat.MARKDOWN])
    assert [
        document for document in documents if markdown_filter.matches(document)
    ] == [markdown_doc]
    exclude_docling_filter = ContentFormatFilter(
        exclude_formats=[ContentFormat.DOCLING]
    )
    assert [
        document for document in documents if exclude_docling_filter.matches(document)
    ] == [markdown_doc, text_doc]

    undeclared_doc = _content_format_document(None, "text/plain")
    assert ContentFormatFilter(include_formats=[ContentFormat.TEXT]).matches(
        undeclared_doc
    ), "a document with no recorded content_format must match as TEXT"
    assert not ContentFormatFilter(include_formats=[ContentFormat.MARKDOWN]).matches(
        undeclared_doc
    )
    assert ContentFormatFilter().matches(undeclared_doc), (
        "no include/exclude list at all must match everything"
    )

    markdown_html_doc = _content_format_document(ContentFormat.MARKDOWN, "text/html")
    markdown_other_doc = _content_format_document(
        ContentFormat.MARKDOWN, "application/xml"
    )
    text_html_doc = _content_format_document(ContentFormat.TEXT, "text/html")

    combined = ContentFormatFilter(
        include_formats=[ContentFormat.MARKDOWN]
    ) & MediaTypeFilter(include_types=["text/html"])
    assert combined.matches(markdown_html_doc)
    assert not combined.matches(markdown_other_doc)
    assert not combined.matches(text_html_doc)

    either = ContentFormatFilter(
        include_formats=[ContentFormat.MARKDOWN]
    ) | MediaTypeFilter(include_types=["text/html"])
    assert either.matches(markdown_other_doc)
    assert either.matches(text_html_doc)

    inverted = ~ContentFormatFilter(include_formats=[ContentFormat.MARKDOWN])
    assert inverted.matches(text_html_doc)
    assert not inverted.matches(markdown_html_doc)


# -- ConsoleLogger ----------------------------------------------------------

#: One line per domain event ``ConsoleLogger`` prints with ``show_domain_events``.
_FETCH_OR_SIMPLE_RE = re.compile(
    r"^\[(fetched|parsed|embedded|saved)\]\s+\S+\s+\S+\s+\((new|cached)\)$", re.M
)
#: ``[chunked]`` lines carry a chunk *count*, and only ever mark "(cached)" --
#: never "(new)".
_CHUNK_RE = re.compile(r"^\[chunked\]\s+(\d+)\s+\S+(\s+\(cached\))?$", re.M)
#: The rendered table's one data row: ``| SitemapCrawler | 7/0/0 | ... |``.
_ROW_RE = re.compile(
    r"^\|\s*(\S+)\s*\|\s*(\d+/\d+/\d+)\s*\|\s*(\d+/\d+/\d+)\s*\|\s*(\d+/\d+/\d+)\s*\|"
    r"\s*(\d+/\d+/\d+)\s*\|\s*(\d+/\d+/\d+)\s*\|$",
    re.M,
)


def _build_logged_pipeline(
    sitemap_url: str,
    cache: PostgresCache,
    vector_store: MilvusVectorStore,
    embedder: Model2VecEmbedder,
    *,
    skip_existing: bool = False,
) -> Pipeline:
    crawler = SitemapCrawler(url=sitemap_url)
    parser = (
        ParserRouter()
        .route(html_filter & docusaurus_filter, DocusaurusParser())
        .route(html_filter, HtmlParser())
        .default(DefaultParser())(crawler)
    )
    chunker = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)(parser)
    embedded = embedder(chunker)
    saved = vector_store.save(embedded, skip_existing=skip_existing)
    return Pipeline(saved, cache=cache)


def _counts(marker_lines: list[str]) -> tuple[int, int]:
    """``(new_count, cached_count)`` from a list of ``"new"``/``"cached"`` markers."""
    return marker_lines.count("new"), marker_lines.count("cached")


def _cell(text: str) -> tuple[int, int, int]:
    new, cached, errors = (int(part) for part in text.split("/"))
    return new, cached, errors


@pytest.mark.parametrize("skip_existing", [False, True])
async def test_console_logger_table_matches_the_pipelines_own_event_stream(
    http_server: str,
    cache: PostgresCache,
    vector_store: MilvusVectorStore,
    embedder: Model2VecEmbedder,
    skip_existing: bool,
) -> None:
    """A single-fetcher run renders one row whose cells match the domain events.

    Piped output (``io.StringIO`` has no live terminal) must carry zero ANSI
    escape sequences -- ``ConsoleLogger`` only ever writes cursor-
    repositioning codes when a live terminal is attached. A second run
    against the same (now warm) cache must flip every cell from ``new`` to
    ``cached``. Checked against the pipeline's own domain-event stream, not
    hardcoded counts (fragile to chunker configuration) and not any private
    ``ConsoleLogger`` state, which the class does not expose.
    """
    sitemap_url = f"{http_server}/sitemap.xml"

    async def _run_and_render() -> str:
        stream = io.StringIO()
        async with _build_logged_pipeline(
            sitemap_url, cache, vector_store, embedder, skip_existing=skip_existing
        ) as pipeline:
            logger = ConsoleLogger(pipeline, show_domain_events=True, stream=stream)
            await logger.run()
        return stream.getvalue()

    cold_text = await _run_and_render()
    headers = [
        cell.strip() for cell in cold_text.splitlines() if cell.startswith("| Source")
    ][0]
    assert [cell.strip() for cell in headers.split("|")[2:-1]] == [
        "Fetching",
        "Parsing",
        "Chunking",
        "Embedding",
        "Indexing",
    ]
    assert "\x1b" not in cold_text, "piped (non-TTY) output must carry no ANSI escapes"

    rows = _ROW_RE.findall(cold_text)
    assert len(rows) == 1, f"expected exactly one row, got {rows}"
    name, fetching, parsing, chunking, embedding, indexing = rows[0]
    assert name == "SitemapCrawler"

    fetch_markers = [
        m[1] for m in _FETCH_OR_SIMPLE_RE.findall(cold_text) if m[0] == "fetched"
    ]
    parse_markers = [
        m[1] for m in _FETCH_OR_SIMPLE_RE.findall(cold_text) if m[0] == "parsed"
    ]
    embed_markers = [
        m[1] for m in _FETCH_OR_SIMPLE_RE.findall(cold_text) if m[0] == "embedded"
    ]
    save_markers = [
        m[1] for m in _FETCH_OR_SIMPLE_RE.findall(cold_text) if m[0] == "saved"
    ]
    chunk_matches = _CHUNK_RE.findall(cold_text)

    assert _cell(fetching) == (*_counts(fetch_markers), 0)
    assert _cell(parsing) == (*_counts(parse_markers), 0)
    assert _cell(embedding) == (*_counts(embed_markers), 0)
    assert _cell(indexing) == (*_counts(save_markers), 0)

    chunk_new = sum(int(count) for count, marker in chunk_matches if not marker)
    chunk_cached = sum(int(count) for count, marker in chunk_matches if marker)
    assert _cell(chunking) == (chunk_new, chunk_cached, 0)
    assert chunk_new > 0, "the fixture site's pages must produce at least one chunk"

    warm_text = await _run_and_render()
    warm_rows = _ROW_RE.findall(warm_text)
    assert len(warm_rows) == 1
    _, w_fetching, w_parsing, w_chunking, w_embedding, w_indexing = warm_rows[0]
    assert _cell(w_fetching) == (0, _cell(fetching)[0] + _cell(fetching)[1], 0)
    assert _cell(w_parsing) == (0, _cell(parsing)[0] + _cell(parsing)[1], 0)
    assert _cell(w_chunking) == (0, chunk_new + chunk_cached, 0)
    assert _cell(w_embedding) == (0, _cell(embedding)[0] + _cell(embedding)[1], 0)
    assert _cell(w_indexing) == (0, _cell(indexing)[0] + _cell(indexing)[1], 0)


@pytest.mark.parametrize(
    ("column_order", "expected"),
    [
        (
            None,
            ["Fetching", "Parsing", "Chunking", "Embedding", "Indexing", "Enrichment"],
        ),
        (
            ["Fetching", "Parsing", "Enrichment", "Chunking", "Embedding", "Indexing"],
            ["Fetching", "Parsing", "Enrichment", "Chunking", "Embedding", "Indexing"],
        ),
        (
            ["Embedding"],
            ["Embedding", "Indexing", "Enrichment", "Chunking", "Parsing", "Fetching"],
        ),
        (
            [],
            ["Indexing", "Enrichment", "Chunking", "Embedding", "Parsing", "Fetching"],
        ),
    ],
)
async def test_console_logger_orders_stages_discovered_during_execution(
    column_order, expected
):
    from fastrag.ingestion.events import StepStarted

    class AnnouncingSource(SourceStep):
        event_name = "Indexing"

        def config(self):
            return {}

        async def evaluate(self, context):
            for label in ["Enrichment", "Chunking", "Embedding", "Parsing", "Fetching"]:
                await context.emit(StepStarted(step=label, event_name=label))
            # This graph exercises stages that start but produce no items.
            for item in []:
                yield item

    stream = io.StringIO()
    async with Pipeline(AnnouncingSource()) as pipeline:
        await ConsoleLogger(pipeline, column_order=column_order, stream=stream).run()
    header = next(
        line for line in stream.getvalue().splitlines() if line.startswith("| Source")
    )
    assert [cell.strip() for cell in header.split("|")[2:-1]] == expected


async def test_console_logger_keeps_embedding_column_when_indexed_items_bypass_embedding():
    from fastrag.ingestion.steps.embedding.base import Embedder
    from fastrag.vector_store import VectorStore

    class TextSource(SourceStep):
        event_name = "Fetching"

        def config(self):
            return {}

        async def evaluate(self, context):
            yield Source(
                id="source",
                uri="memory://source",
                content=b"hello world",
                content_type="text/plain",
                content_hash="hash",
            )

    class TestEmbedder(Embedder):
        calls = 0

        def config(self):
            return {}

        @property
        def dimension(self):
            return 2

        @property
        def model_id(self):
            return "test"

        async def embed(self, texts):
            self.calls += len(texts)
            return [[0.1, 0.2] for text in texts]

    class MemoryStore(VectorStore):
        def __init__(self):
            super().__init__()
            self.present = set()

        async def ensure_ready(self, dimension):
            assert dimension == 2

        async def existing_ids(self, ids):
            return self.present.intersection(ids)

        async def upsert(self, documents):
            assert all(doc.embedding == [0.1, 0.2] for doc in documents)
            self.present.update(doc.id for doc in documents)

        async def search(self, embedding, *, k, filter=None):
            return []

    embedder = TestEmbedder()
    store = MemoryStore()
    indexed = store.save(embedder(DefaultParser()(TextSource())), skip_existing=True)

    async def render():
        stream = io.StringIO()
        async with Pipeline(indexed) as pipeline:
            await ConsoleLogger(pipeline, stream=stream).run()
        lines = stream.getvalue().splitlines()
        header = next(line for line in lines if line.startswith("| Source"))
        row = next(line for line in lines if line.startswith("| TextSource"))
        return dict(
            zip(
                (cell.strip() for cell in header.split("|")[2:-1]),
                (cell.strip() for cell in row.split("|")[2:-1]),
            )
        )

    cold = await render()
    assert list(cold) == ["Fetching", "Parsing", "Embedding", "Indexing"]
    assert cold["Embedding"] == cold["Indexing"] == "1/0/0"
    assert embedder.calls == 1
    warm = await render()
    assert list(warm) == list(cold)
    assert warm["Embedding"] == "0/1/0"
    assert warm["Indexing"] == "0/1/0"
    assert embedder.calls == 1
    from fastrag.ingestion.steps.base import event_sink_scope

    observed = []

    async def observe(event):
        observed.append(event)

    with event_sink_scope(observe):
        results = await indexed.run()
    embedded_events = [e for e in observed if isinstance(e, DocumentEmbedded)]
    saved_events = [e for e in observed if isinstance(e, DocumentSaved)]
    assert len(embedded_events) == len(saved_events) == len(results) == 1
    assert embedded_events[0].document == saved_events[0].document == results[0]
    assert embedded_events[0].document.cached
    assert embedded_events[0].document.embedding is None
    assert embedded_events[0].step == embedder.name
    assert embedded_events[0].event_name == embedder.event_name
    assert observed.index(embedded_events[0]) < observed.index(saved_events[0])
    assert embedder.calls == 1
    store.present.clear()
    assert (await render())["Embedding"] == "1/0/0"
    assert embedder.calls == 2


@pytest.mark.parametrize("skip_existing", [False, True])
@pytest.mark.parametrize("on", ["store", "pipeline"])
async def test_recreate_cache_overwrites_documents_the_store_already_holds(
    skip_existing: bool, on: str
) -> None:
    """A rebuild rewrites indexed documents, e.g. to replace their metadata."""
    from fastrag.ingestion.steps.embedding.base import Embedder
    from fastrag.vector_store import VectorStore

    class TextSource(SourceStep):
        def config(self):
            return {}

        async def evaluate(self, context):
            yield _make_source("source")

    class TestEmbedder(Embedder):
        def config(self):
            return {}

        @property
        def dimension(self):
            return 2

        @property
        def model_id(self):
            return "test"

        async def embed(self, texts):
            return [[0.1, 0.2] for text in texts]

    class MemoryStore(VectorStore):
        def __init__(self, policy=None):
            super().__init__(policy=policy)
            self.upserted: list[str] = []

        async def ensure_ready(self, dimension):
            pass

        async def existing_ids(self, ids):
            return set(self.upserted).intersection(ids)

        async def upsert(self, documents):
            self.upserted.extend(doc.id for doc in documents)

        async def search(self, embedding, *, k, filter=None):
            return []

    recreate = Policy(recreate_cache=True)
    store = MemoryStore(policy=recreate if on == "store" else None)
    indexed = store.save(
        TestEmbedder()(DefaultParser()(TextSource())), skip_existing=skip_existing
    )

    async def run(policy: Policy | None) -> list[Document]:
        async with Pipeline(indexed, policy=policy) as pipeline:
            return [
                event.document
                async for event in pipeline
                if isinstance(event, DocumentSaved)
            ]

    await run(None)
    [saved] = await run(recreate if on == "pipeline" else None)

    assert store.upserted == [saved.id, saved.id]
    assert not saved.cached


# -- CacheStore contract (PostgresCache) ----------------------------------------------------


def _make_source(id_: str) -> Source:
    return Source(
        id=id_,
        uri=f"http://example.invalid/{id_}",
        content=b"hello",
        content_type="text/plain",
        content_hash="deadbeef",
    )


def _make_document(id_: str, source_id: str) -> Document:
    return Document(id=id_, source_id=source_id, content="hello world")


async def _run_notes_pipeline(sitemap_url: str, cache: PostgresCache):
    crawler = SitemapCrawler(url=sitemap_url, include=["*/notes.txt"])
    parser = (
        ParserRouter()
        .route(html_filter & docusaurus_filter, DocusaurusParser())
        .route(html_filter, HtmlParser())
        .default(DefaultParser())(crawler)
    )
    sources: list[Source] = []
    documents: list[Document] = []
    async with Pipeline(parser, cache=cache) as pipeline:
        async for event in pipeline:
            if isinstance(event, DocumentFetched):
                sources.append(event.source)
            elif isinstance(event, DocumentParsed):
                documents.append(event.document)
    return sources, documents


async def test_cache_cleanup_methods_remove_rows_report_counts_and_reset_the_cache(
    http_server: str, cache: PostgresCache
) -> None:
    """All six cleanup methods, against one real, isolated ``PostgresCache``.

    ``delete_sources``/``delete_documents``: remove exactly what they claim,
    and empty a cache id to a miss (``None``), never a known-but-empty hit
    (``[]``) -- the distinction the whole feature exists to get right.
    ``collect_garbage``: removes only entries older than ``max_age``, and
    rejects a non-positive one. ``clear_sources``/``clear_documents``: wipe
    only their own kind. ``clear``: wipes everything, and a real pipeline
    run straight afterward behaves exactly like a cold run.
    """
    # --- delete_sources / delete_documents ---
    await cache.put_source_entries("sc:a", [_make_source("s1"), _make_source("s2")])
    await cache.put_source_entries("sc:b", [_make_source("s3")])
    await cache.put_document_entries(
        "dc:a", [_make_document("d1", "s1"), _make_document("d2", "s2")]
    )
    await cache.put_document_entries("dc:b", [_make_document("d3", "s3")])

    removed = await cache.delete_sources(["s1", "s2"])
    assert removed == 2
    assert await cache.get_source_entries("sc:a") is None
    assert (await cache.get_source_entries("sc:b")) is not None

    removed = await cache.delete_documents(["s3"])
    assert removed == 1
    assert await cache.get_document_entries("dc:b") is None
    assert (await cache.get_document_entries("dc:a")) is not None

    assert await cache.delete_sources([]) == 0
    assert await cache.delete_documents([]) == 0
    assert await cache.delete_sources(["never-existed"]) == 0

    # --- collect_garbage ---
    await cache.clear()
    await cache.put_source_entries("sc:old", [_make_source("so1")])
    await asyncio.sleep(1.1)
    await cache.put_source_entries("sc:fresh", [_make_source("sf1")])

    removed = await cache.collect_garbage(timedelta(seconds=1))
    assert removed == 1
    assert await cache.get_source_entries("sc:old") is None
    assert (await cache.get_source_entries("sc:fresh")) is not None

    with pytest.raises(ValueError):
        await cache.collect_garbage(timedelta(0))
    with pytest.raises(ValueError):
        await cache.collect_garbage(timedelta(seconds=-1))

    # --- clear_sources / clear_documents ---
    await cache.put_source_entries("sc:x", [_make_source("sx1")])
    await cache.put_document_entries("dc:x", [_make_document("dx1", "sx1")])

    removed_sources = await cache.clear_sources()
    assert removed_sources >= 1
    assert await cache.get_source_entries("sc:x") is None
    assert (await cache.get_document_entries("dc:x")) is not None

    removed_documents = await cache.clear_documents()
    assert removed_documents >= 1
    assert await cache.get_document_entries("dc:x") is None

    # --- clear(): a pipeline run right after it behaves like a cold run ---
    sitemap_url = f"{http_server}/sitemap.xml"

    cold_sources, cold_documents = await _run_notes_pipeline(sitemap_url, cache)
    assert not any(source.cached for source in cold_sources)
    assert not any(document.cached for document in cold_documents)

    warm_sources, warm_documents = await _run_notes_pipeline(sitemap_url, cache)
    assert all(source.cached for source in warm_sources)
    assert all(document.cached for document in warm_documents)

    removed = await cache.clear()
    assert removed > 0

    reset_sources, reset_documents = await _run_notes_pipeline(sitemap_url, cache)
    assert not any(source.cached for source in reset_sources), (
        "a pipeline run straight after clear() must behave like a cold run"
    )
    assert not any(document.cached for document in reset_documents)


async def test_default_cache_inherits_while_explicit_false_never_caches(
    http_server: str, cache: PostgresCache
) -> None:
    """``cache=None`` (the default) inherits the pipeline's cache; an
    explicit ``cache=False`` never caches, even though the same pipeline's
    cache is right there and its own upstream is warm.
    """

    async def _run() -> tuple[bool, list[bool]]:
        crawler = SitemapCrawler(
            url=f"{http_server}/sitemap.xml", include=["*/notes.txt"]
        )
        # No `cache=` argument at all -- the default, which must inherit the
        # pipeline's cache.
        parser = DefaultParser()(crawler)
        # Explicit `cache=False` -- must never cache, regardless of what the
        # pipeline around it provides.
        chunker = RecursiveCharacterTextSplitter(
            chunk_size=20, chunk_overlap=5, cache=False
        )(parser)

        parsed_cached: bool | None = None
        chunk_cached: list[bool] = []
        async with Pipeline(chunker, cache=cache) as pipeline:
            async for event in pipeline:
                if isinstance(event, DocumentParsed):
                    parsed_cached = event.document.cached
                elif isinstance(event, DocumentChunked):
                    chunk_cached.extend(chunk.cached for chunk in event.chunks)

        assert parsed_cached is not None
        assert chunk_cached
        return parsed_cached, chunk_cached

    cold_parsed_cached, cold_chunks_cached = await _run()
    assert cold_parsed_cached is False
    assert not any(cold_chunks_cached)

    warm_parsed_cached, warm_chunks_cached = await _run()
    assert warm_parsed_cached is True, (
        "a parser built with no cache= argument must inherit the pipeline's "
        "cache and hit on a rerun"
    )
    assert not any(warm_chunks_cached), (
        "a chunker built with cache=False must never cache, even though the "
        "same pipeline's cache is right there and its own upstream is warm"
    )


_VALIDATION_PAGE = b"<html><body>fixed content</body></html>"
_VALIDATION_ETAG = '"fixed-etag"'
_VALIDATION_SITEMAP = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    "<url><loc>{base}/page.html</loc></url>"
    "</urlset>"
)


class _CountingHandler(http.server.BaseHTTPRequestHandler):
    """Serves a one-page fixture and tallies every request it receives."""

    def __init__(
        self, *args: object, site: "_CountingOrigin", **kwargs: object
    ) -> None:
        self._site = site
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass

    def do_GET(self) -> None:  # noqa: N802
        self._site.counts[f"GET {self.path}"] += 1
        self._respond()

    def do_HEAD(self) -> None:  # noqa: N802
        self._site.counts[f"HEAD {self.path}"] += 1
        self._respond()

    def _respond(self) -> None:
        if self.path == "/sitemap.xml":
            body = _VALIDATION_SITEMAP.format(base=self._site.base_url).encode()
            self._write(body, "application/xml")
        elif self.path == "/page.html":
            self._write(_VALIDATION_PAGE, "text/html", etag=_VALIDATION_ETAG)
        else:
            self.send_response(404)
            self.end_headers()

    def _write(
        self, body: bytes, content_type: str, *, etag: str | None = None
    ) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if etag is not None:
            self.send_header("ETag", etag)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)


class _CountingOrigin:
    """A local HTTP origin that tallies every request it receives, by method+path."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        handler = functools.partial(_CountingHandler, site=self)
        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def counting_origin() -> Iterator[_CountingOrigin]:
    origin = _CountingOrigin()
    try:
        yield origin
    finally:
        origin.close()


async def _crawl_through_a_fresh_cache_instance(
    sitemap_url: str, schema: str, interval: float
) -> list[Source]:
    """Crawl through a brand-new ``PostgresCache`` instance -- no shared Python state."""
    async with PostgresCache(url=DATABASE_URL, schema=schema) as store:
        crawler = SitemapCrawler(url=sitemap_url, cache_validation_interval=interval)
        sources: list[Source] = []
        async with Pipeline(crawler, cache=store) as pipeline:
            async for event in pipeline:
                if isinstance(event, DocumentFetched):
                    sources.append(event.source)
        return sources


async def test_validation_interval_survives_a_fresh_cache_instance(
    counting_origin: _CountingOrigin,
) -> None:
    """A fresh ``PostgresCache`` instance serves a bucket within the interval
    untouched -- ``last_validated`` is persisted, not in-process state, so a
    restarted process still remembers when a bucket was last confirmed
    fresh.

    Three ``SitemapCrawler`` crawls, each through its own brand-new
    ``PostgresCache`` instance pointed at the same schema (the closest a
    single test process can get to "a different process, later"): cold (one
    real ``GET /page.html``); a fresh instance still within a 60s interval
    (served from the persisted timestamp, zero further requests); a fresh
    instance after the interval has elapsed (revalidates with one ``HEAD``,
    but -- the origin's ``ETag`` is unchanged -- does not refetch the body).
    """
    schema = f"fastrag_test_{uuid4().hex}"
    admin = await asyncpg.connect(DATABASE_URL)
    try:
        sitemap_url = f"{counting_origin.base_url}/sitemap.xml"

        first = await _crawl_through_a_fresh_cache_instance(
            sitemap_url, schema, interval=60.0
        )
        assert len(first) == 1
        assert not first[0].cached
        assert counting_origin.counts["GET /page.html"] == 1
        assert counting_origin.counts["HEAD /page.html"] == 0

        second = await _crawl_through_a_fresh_cache_instance(
            sitemap_url, schema, interval=60.0
        )
        assert second[0].cached
        assert counting_origin.counts["GET /page.html"] == 1, (
            "a fresh cache instance within the interval must reach the "
            "fixture origin zero times"
        )
        assert counting_origin.counts["HEAD /page.html"] == 0

        await asyncio.sleep(1.1)
        third = await _crawl_through_a_fresh_cache_instance(
            sitemap_url, schema, interval=1.0
        )
        assert third[0].cached, "an unchanged ETag must still serve the cached body"
        assert counting_origin.counts["HEAD /page.html"] == 1, (
            "past the interval, a fresh instance must revalidate against the origin"
        )
        assert counting_origin.counts["GET /page.html"] == 1, (
            "an unmodified ETag must not trigger a full refetch"
        )
    finally:
        await admin.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await admin.close()
