"""End-to-end coverage for ``fastrag.ingestion.steps.fetching``: every concrete Fetcher.

Drives real fetchers (``SitemapCrawler``, ``FileSource``, ``LocalFileFetcher``,
``WebCrawler``) and their ``Sources`` aggregation through a real
``Pipeline``, against the fixture site (or, where a specific origin
behaviour needs finer control -- a changing body, a redirecting alias -- a
small local HTTP origin built just for that test). ``LocalFileFetcher``'s
own tests use ``tmp_path`` instead of the fixture site: there is no network
involved, so nothing to serve.
"""

from __future__ import annotations

import functools
import http.server
import io
import re
import threading
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pytest

from fastrag.exceptions import FetchError
from fastrag.ingestion import Pipeline
from fastrag.ingestion.cache import PostgresCache
from fastrag.ingestion.events import DocumentFetched, ItemFailed, PipelineEvent
from fastrag.ingestion.logger import ConsoleLogger
from fastrag.ingestion.models import Source
from fastrag.ingestion.steps.chunking import RecursiveCharacterTextSplitter
from fastrag.ingestion.steps.embedding import Model2VecEmbedder
from fastrag.ingestion.steps.fetching import (
    FileSource,
    LocalFileFetcher,
    SitemapCrawler,
    Sources,
    WebCrawler,
)
from fastrag.ingestion.steps.parsing import (
    DefaultParser,
    DocusaurusParser,
    HtmlParser,
    ParserRouter,
    docusaurus_filter,
    html_filter,
)
from fastrag.vector_store import MilvusVectorStore

#: Same fixture tree ``conftest.py``'s ``http_server`` serves, read directly
#: here to assert fetched bytes against the file on disk.
FIXTURE_SITE = Path(__file__).parent / "fixtures" / "site"


# -- SitemapCrawler ----------------------------------------------------------


async def test_sitemap_cache_isolates_renderer_models_and_settings(
    http_server, cache, monkeypatch
):
    from dataclasses import replace

    from fastrag.ingestion.cache import source_cache_id
    from fastrag.ingestion.steps.fetching import DecisionsRenderer

    calls = []

    async def render(renderer, page):
        calls.append((renderer.model, renderer.goal))
        return replace(
            page,
            content=f"{renderer.model}: {renderer.goal}".encode(),
            etag=None,
            last_modified=None,
        )

    monkeypatch.setattr(DecisionsRenderer, "render", render)
    uri = f"{http_server}/index.html"
    args = dict(
        model="first", goal="all content", api_key="secret", base_url=http_server
    )

    async def crawl(renderer=None):
        crawler = SitemapCrawler(
            url=f"{http_server}/sitemap.xml", include=uri, renderer=renderer
        )
        async with Pipeline(crawler, cache=cache) as pipeline:
            sources = [
                event.source
                async for event in pipeline
                if isinstance(event, DocumentFetched)
            ]
        key = source_cache_id(crawler.name, uri, crawler.config_hash)
        return sources[0], key

    original, http_key = await crawl()
    first, first_key = await crawl(DecisionsRenderer(**args))
    second, second_key = await crawl(DecisionsRenderer(**{**args, "model": "second"}))
    settings, settings_key = await crawl(
        DecisionsRenderer(**{**args, "goal": "only reports"})
    )
    assert len({http_key, first_key, second_key, settings_key}) == 4
    assert all(not source.cached for source in [original, first, second, settings])

    for renderer, expected in [
        (None, original),
        (DecisionsRenderer(**{**args, "api_key": "rotated"}), first),
        (DecisionsRenderer(**{**args, "model": "second"}), second),
        (DecisionsRenderer(**{**args, "goal": "only reports"}), settings),
    ]:
        hit, key = await crawl(renderer)
        assert hit.cached and hit.content == expected.content
        assert hit.content_hash == expected.content_hash
        headers = await cache.get_source_headers_many([key])
        header = headers[key][0]
        assert header.content_hash == expected.content_hash
        assert (await header.materialize()).content == expected.content
    assert calls == [
        ("first", "all content"),
        ("second", "all content"),
        ("first", "only reports"),
    ]


async def _crawl(sitemap_url: str, cache: PostgresCache) -> list[Source]:
    crawler = SitemapCrawler(url=sitemap_url)
    sources: list[Source] = []
    async with Pipeline(crawler, cache=cache) as pipeline:
        async for event in pipeline:
            if isinstance(event, DocumentFetched):
                sources.append(event.source)
    return sources


async def test_sitemap_crawler_fetches_every_listed_page_and_caches_on_rerun(
    http_server: str, cache: PostgresCache
) -> None:
    """One ``Source`` per sitemap entry, with the right uris/content types.

    A second crawl against the same (now warm) cache reports every source
    as ``cached``.
    """
    sitemap_url = f"{http_server}/sitemap.xml"
    expected_uris = {
        f"{http_server}/index.html",
        f"{http_server}/page-a.html",
        f"{http_server}/page-b.html",
        f"{http_server}/page-c.html",
        f"{http_server}/long.html",
        f"{http_server}/docs.html",
        f"{http_server}/notes.txt",
    }

    first = await _crawl(sitemap_url, cache)
    assert {source.uri for source in first} == expected_uris
    assert not any(source.cached for source in first)

    content_types = {source.uri: source.content_type for source in first}
    assert content_types[f"{http_server}/notes.txt"] == "text/plain"
    for uri in expected_uris - {f"{http_server}/notes.txt"}:
        assert content_types[uri] == "text/html"

    second = await _crawl(sitemap_url, cache)
    assert {source.uri for source in second} == expected_uris
    assert all(source.cached for source in second)


@pytest.mark.parametrize(
    ("include", "exclude", "expected"),
    [
        ("{base}/page-a.html", None, ["page-a.html"]),
        (["{base}/page-a.html"], None, ["page-a.html"]),
        ("{base}/page-*.html", "{base}/page-b.html", ["page-a.html", "page-c.html"]),
        (["{base}/page-*.html"], ["*/page-b.html"], ["page-a.html", "page-c.html"]),
        ("{base}/missing.html", None, []),
    ],
)
async def test_sitemap_crawler_filters_single_strings_and_sequences(
    http_server: str,
    include: str | list[str],
    exclude: str | list[str] | None,
    expected: list[str],
) -> None:
    def expand(patterns):
        if isinstance(patterns, str):
            return patterns.format(base=http_server)
        return [pattern.format(base=http_server) for pattern in patterns or []]

    sources = await SitemapCrawler(
        url=f"{http_server}/sitemap.xml",
        include=expand(include),
        exclude=expand(exclude),
    ).run()

    assert sorted(source.uri for source in sources) == [
        f"{http_server}/{path}" for path in expected
    ]
    assert all(source.content for source in sources)


# -- FileSource ----------------------------------------------------------


async def _fetch_file(url: str, cache: PostgresCache) -> list[Source]:
    fetcher = FileSource(url=url)
    sources: list[Source] = []
    async with Pipeline(fetcher, cache=cache) as pipeline:
        async for event in pipeline:
            if isinstance(event, DocumentFetched):
                sources.append(event.source)
    return sources


async def test_file_source_fetches_caches_on_rerun_and_404s_on_a_missing_file(
    http_server: str, cache: PostgresCache
) -> None:
    """A bare, uncached fetch yields the right bytes/media type; a rerun
    against the same warm cache reports the source as cached; a missing
    file raises ``FetchError`` naming it.
    """
    url = f"{http_server}/notes.txt"

    bare_sources = await FileSource(url=url).run()
    assert len(bare_sources) == 1
    bare_source = bare_sources[0]
    assert bare_source.uri == url
    assert bare_source.content_type == "text/plain"
    assert bare_source.content == (FIXTURE_SITE / "notes.txt").read_bytes()

    first = await _fetch_file(url, cache)
    assert len(first) == 1
    assert not first[0].cached

    second = await _fetch_file(url, cache)
    assert len(second) == 1
    assert second[0].cached

    with pytest.raises(FetchError, match="missing-file.pdf"):
        await FileSource(url=f"{http_server}/missing-file.pdf").run()


class _MutableFileHandler(http.server.BaseHTTPRequestHandler):
    """Serves one path (``/report.txt``) whose body can be swapped between requests.

    Sends no ``ETag``/``Last-Modified`` header, so
    ``FileSource.is_source_modified`` always falls through to its
    refetch-and-compare-content-hash step -- the ladder's last rung -- the
    same way this module's ``_CountingOrigin`` (in ``test_pipeline.py``)
    exercises the ETag rung instead.
    """

    def __init__(
        self, *args: object, site: "_MutableFileOrigin", **kwargs: object
    ) -> None:
        self._site = site
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass

    def do_GET(self) -> None:  # noqa: N802
        self._respond()

    def do_HEAD(self) -> None:  # noqa: N802
        self._respond()

    def _respond(self) -> None:
        if self.path != "/report.txt":
            self.send_response(404)
            self.end_headers()
            return
        body = self._site.content
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)


class _MutableFileOrigin:
    """A local HTTP origin serving one path whose body can change between requests."""

    def __init__(self) -> None:
        self.content = b"version one"
        handler = functools.partial(_MutableFileHandler, site=self)
        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def mutable_file_origin() -> Iterator[_MutableFileOrigin]:
    origin = _MutableFileOrigin()
    try:
        yield origin
    finally:
        origin.close()


async def test_file_source_revalidates_after_the_origin_bytes_change(
    mutable_file_origin: _MutableFileOrigin, cache: PostgresCache
) -> None:
    """With ``cache_validation_interval=0``, changed bytes are refetched, not served stale."""
    url = f"{mutable_file_origin.base_url}/report.txt"

    async def _fetch() -> list[Source]:
        fetcher = FileSource(url=url, cache_validation_interval=0)
        sources: list[Source] = []
        async with Pipeline(fetcher, cache=cache) as pipeline:
            async for event in pipeline:
                if isinstance(event, DocumentFetched):
                    sources.append(event.source)
        return sources

    first = await _fetch()
    assert first[0].content == b"version one"
    assert not first[0].cached

    mutable_file_origin.content = b"version two"

    second = await _fetch()
    assert second[0].content == b"version two"
    assert not second[0].cached


# -- LocalFileFetcher ----------------------------------------------------------


async def test_local_file_fetcher_pattern_matching(tmp_path: Path) -> None:
    """A single file, a directory expansion, a recursive glob, include/exclude
    filtering, overlapping-pattern dedup, and a pattern matching nothing --
    all through the same ``run()`` surface.
    """
    single_file = tmp_path / "report.md"
    single_file.write_text("# hello")
    single_sources = await LocalFileFetcher(patterns=[str(single_file)]).run()
    assert len(single_sources) == 1
    source = single_sources[0]
    assert source.uri == single_file.resolve().as_uri()
    assert source.id == source.uri
    assert source.content == b"# hello"
    assert source.content_type == "text/markdown"

    directory = tmp_path / "directory"
    directory.mkdir()
    (directory / "a.txt").write_text("a")
    (directory / "b.txt").write_text("b")
    nested = directory / "nested"
    nested.mkdir()
    (nested / "c.txt").write_text("c")
    directory_sources = await LocalFileFetcher(patterns=[str(directory)]).run()
    assert {s.content for s in directory_sources} == {b"a", b"b", b"c"}
    assert len(directory_sources) == 3

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "one.xlsx").write_bytes(b"one")
    sub = docs / "sub"
    sub.mkdir()
    (sub / "two.xlsx").write_bytes(b"two")
    (docs / "ignore.txt").write_text("ignore")
    glob_sources = await LocalFileFetcher(patterns=[str(docs / "**" / "*.xlsx")]).run()
    assert {s.content for s in glob_sources} == {b"one", b"two"}
    assert all(
        s.content_type
        == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        for s in glob_sources
    )

    filtered = tmp_path / "filtered"
    filtered.mkdir()
    (filtered / "keep.txt").write_text("keep")
    (filtered / "drop.txt").write_text("drop")
    (filtered / "keep.log").write_text("log")
    root = filtered.resolve()
    filtered_sources = await LocalFileFetcher(
        patterns=[str(filtered)],
        include=[f"{root}/*.txt"],
        exclude=[f"{root}/drop.*"],
    ).run()
    assert len(filtered_sources) == 1
    assert filtered_sources[0].content == b"keep"

    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()
    shared_file = shared_dir / "shared.txt"
    shared_file.write_text("shared")
    shared_sub = shared_dir / "sub"
    shared_sub.mkdir()
    dotdot_pattern = str(shared_sub / ".." / "shared.txt")
    dedup_sources = await LocalFileFetcher(
        patterns=[str(shared_file), str(shared_dir), dotdot_pattern]
    ).run()
    assert len(dedup_sources) == 1
    assert dedup_sources[0].content == b"shared"
    assert dedup_sources[0].uri == shared_file.resolve().as_uri()

    missing_pattern = str(tmp_path / "does-not-exist" / "*.pdf")
    with pytest.raises(FetchError, match=re.escape(missing_pattern)):
        await LocalFileFetcher(patterns=[missing_pattern]).run()


async def test_local_file_fetcher_caches_on_rerun_and_revalidates_on_change(
    tmp_path: Path, cache: PostgresCache
) -> None:
    """A rerun against a warm cache reports the file as cached; with
    ``cache_validation_interval=0``, changed bytes are refetched instead of
    served stale, and one file's own bucket is untouched by an unrelated
    new sibling appearing under the same directory.
    """

    async def _fetch_local(patterns: list[str]) -> list[Source]:
        fetcher = LocalFileFetcher(patterns=patterns)
        sources: list[Source] = []
        async with Pipeline(fetcher, cache=cache) as pipeline:
            async for event in pipeline:
                if isinstance(event, DocumentFetched):
                    sources.append(event.source)
        return sources

    cached_path = tmp_path / "cached.txt"
    cached_path.write_text("first")
    first = await _fetch_local([str(cached_path)])
    assert not first[0].cached
    assert first[0].content == b"first"
    second = await _fetch_local([str(cached_path)])
    assert second[0].cached
    assert second[0].content == b"first"

    mutable_path = tmp_path / "mutable.txt"
    mutable_path.write_text("short")

    async def _fetch_mutable() -> list[Source]:
        fetcher = LocalFileFetcher(
            patterns=[str(mutable_path)], cache_validation_interval=0
        )
        sources: list[Source] = []
        async with Pipeline(fetcher, cache=cache) as pipeline:
            async for event in pipeline:
                if isinstance(event, DocumentFetched):
                    sources.append(event.source)
        return sources

    mutable_first = await _fetch_mutable()
    assert mutable_first[0].content == b"short"
    assert not mutable_first[0].cached

    mutable_path.write_text("a considerably longer second version")
    mutable_second = await _fetch_mutable()
    assert mutable_second[0].content == b"a considerably longer second version"
    assert not mutable_second[0].cached


# -- Sources aggregation ----------------------------------------------------------

_SITEMAP_ONLY = ["*/long.html", "*/docs.html", "*/notes.txt"]


def _sources_pipeline(
    http_server: str,
    cache: PostgresCache,
    vector_store: MilvusVectorStore,
    embedder: Model2VecEmbedder,
) -> Pipeline:
    sources = Sources(
        sources=[
            SitemapCrawler(url=f"{http_server}/sitemap.xml", include=_SITEMAP_ONLY),
            WebCrawler(url=f"{http_server}/index.html", max_depth=2),
        ]
    )
    parser = (
        ParserRouter()
        .route(html_filter & docusaurus_filter, DocusaurusParser())
        .route(html_filter, HtmlParser())
        .default(DefaultParser())(sources)
    )
    chunker = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)(parser)
    embedded = embedder(chunker)
    saved = vector_store.save(embedded)
    return Pipeline(saved, cache=cache)


async def test_sources_aggregate_attributes_rows_correctly_and_fully_caches_on_rerun(
    http_server: str,
    cache: PostgresCache,
    vector_store: MilvusVectorStore,
    embedder: Model2VecEmbedder,
) -> None:
    """A cold raw run proves per-fetcher fetch counts and the diamond dedup.

    ``index.html`` links to ``page-a.html`` and ``page-b.html``, both of
    which link on to ``page-c.html`` -- the fixture site's diamond. Every
    fetched ``Source`` is re-emitted exactly once more by ``Sources``'s own
    outer lifecycle wrap, so every id appears exactly twice in the raw event
    stream, uniformly -- including ``page-c.html``, despite its two inbound
    links. The two crawlers are pointed at disjoint parts of the fixture
    site so their ``Source.id``\\ s never collide. A rerun, through
    ``ConsoleLogger`` this time, shows both rows fully cached with no
    double-counting.
    """
    events: list[PipelineEvent] = []
    async with _sources_pipeline(
        http_server, cache, vector_store, embedder
    ) as pipeline:
        async for event in pipeline:
            events.append(event)

    fetched = [e for e in events if isinstance(e, DocumentFetched)]
    failed = [e for e in events if isinstance(e, ItemFailed)]

    web_ids = [
        e.source.id
        for e in fetched
        if e.source.metadata.get("source_step") == "WebCrawler"
    ]
    sitemap_ids = [
        e.source.id
        for e in fetched
        if e.source.metadata.get("source_step") == "SitemapCrawler"
    ]

    assert set(Counter(web_ids).values()) == {2}, (
        f"every WebCrawler source id should be re-emitted exactly twice "
        f"(child + aggregate), got {Counter(web_ids)}"
    )
    assert set(Counter(sitemap_ids).values()) == {2}, Counter(sitemap_ids)
    assert not (set(web_ids) & set(sitemap_ids)), "the two crawlers must not overlap"
    assert len(set(web_ids)) == 4  # index, page-a, page-b, page-c -- not missing.html
    assert len(set(sitemap_ids)) == 3  # long, docs, notes
    assert len(failed) == 1
    assert failed[0].step == "WebCrawler"
    assert "missing.html" in failed[0].uri

    stream = io.StringIO()
    async with _sources_pipeline(
        http_server, cache, vector_store, embedder
    ) as pipeline:
        logger = ConsoleLogger(pipeline, stream=stream)
        await logger.run()
    table = stream.getvalue()
    sitemap_line = next(
        line
        for line in table.splitlines()
        if line.strip().startswith("| SitemapCrawler")
    )
    web_line = next(
        line for line in table.splitlines() if line.strip().startswith("| WebCrawler")
    )
    assert sitemap_line.split("|")[2].strip() == "0/3/0"
    assert web_line.split("|")[2].strip() == "0/4/0"


# -- URL/id dedup across fetchers ----------------------------------------------------------

_ALIAS_START = (
    b"<html><body>"
    b'<a href="/page.html">canonical</a>'
    b'<a href="/alias.html">alias</a>'
    b'<a href="/page.html#section">fragment</a>'
    b"</body></html>"
)
_ALIAS_PAGE = b"<html><body>the one page</body></html>"
#: Three spellings of one page: bare, trailing slash, and a fragment. All
#: three normalize to the same id, so they must produce one target.
_ALIAS_SITEMAP_TEMPLATE = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    "<url><loc>{base}/page.html</loc></url>"
    "<url><loc>{base}/page.html/</loc></url>"
    "<url><loc>{base}/page.html#section</loc></url>"
    "</urlset>"
)


class _AliasHandler(http.server.BaseHTTPRequestHandler):
    """Serves one page, one alias redirecting onto it, and a sitemap."""

    def __init__(self, *args: object, base_url_holder: list[str], **kwargs: object):
        self._base_url_holder = base_url_holder
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass

    def do_GET(self) -> None:  # noqa: N802
        self._respond()

    def do_HEAD(self) -> None:  # noqa: N802
        self._respond()

    def _respond(self) -> None:
        if self.path == "/alias.html":
            self.send_response(301)
            self.send_header("Location", "/page.html")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/start.html":
            self._write(_ALIAS_START, "text/html")
        elif self.path == "/page.html":
            self._write(_ALIAS_PAGE, "text/html")
        elif self.path == "/sitemap.xml":
            body = _ALIAS_SITEMAP_TEMPLATE.format(
                base=self._base_url_holder[0]
            ).encode()
            self._write(body, "application/xml")
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def _write(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)


@pytest.fixture
def alias_origin() -> Iterator[str]:
    """A local origin whose ``/alias.html`` redirects onto ``/page.html``."""
    holder: list[str] = [""]
    handler = functools.partial(_AliasHandler, base_url_holder=holder)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    holder[0] = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield holder[0]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


async def _run_to_exhaustion(
    step: SitemapCrawler | WebCrawler, cache: PostgresCache
) -> list[Source]:
    """Run ``step`` to exhaustion, failing loudly on any ``ItemFailed``."""
    sources: list[Source] = []
    events: list[PipelineEvent] = []
    async with Pipeline(step, cache=cache) as pipeline:
        async for event in pipeline:
            events.append(event)
            if isinstance(event, DocumentFetched):
                sources.append(event.source)
    failures = [event for event in events if isinstance(event, ItemFailed)]
    assert not failures, f"unexpected failures: {[f.error for f in failures]}"
    return sources


@pytest.mark.parametrize("excluded", [False, True])
async def test_sitemap_crawler_exact_filters_match_normalized_urls(
    alias_origin: str, excluded: bool
) -> None:
    # The sitemap lists bare, trailing-slash and fragment spellings.
    url = f"{alias_origin}/page.html/"
    sources = await SitemapCrawler(
        url=f"{alias_origin}/sitemap.xml",
        include=url,
        exclude=f"{alias_origin}/page.html" if excluded else None,
    ).run()
    assert [source.uri for source in sources] == (
        [] if excluded else [f"{alias_origin}/page.html"]
    )


async def test_web_and_sitemap_crawlers_collapse_duplicate_spellings_of_one_page(
    alias_origin: str, cache: PostgresCache
) -> None:
    """One page reached under several URLs/spellings is fetched and cached once.

    Without deduplication by resolved id, a crawl yielding the same id twice
    writes two rows in one cache batch, which Postgres refuses outright
    ("ON CONFLICT DO UPDATE command cannot affect row a second time") --
    taking the whole run down. ``WebCrawler`` follows a redirecting alias
    onto an already-crawled page; ``SitemapCrawler`` collapses three
    ``<loc>`` spellings of one page to one target.
    """
    web_sources = await _run_to_exhaustion(
        WebCrawler(url=f"{alias_origin}/start.html", max_depth=1), cache
    )
    web_ids = [source.id for source in web_sources]
    assert sorted(web_ids) == [
        f"{alias_origin}/page.html",
        f"{alias_origin}/start.html",
    ]

    cached = await _run_to_exhaustion(
        WebCrawler(url=f"{alias_origin}/start.html", max_depth=1), cache
    )
    assert {source.id for source in cached} == set(web_ids)
    assert all(source.cached for source in cached)

    sitemap_sources = await _run_to_exhaustion(
        SitemapCrawler(url=f"{alias_origin}/sitemap.xml"), cache
    )
    assert [source.id for source in sitemap_sources] == [f"{alias_origin}/page.html"]


async def test_cache_survives_a_batch_carrying_one_id_twice(
    cache: PostgresCache,
) -> None:
    """A duplicated batch is written as one row, not rejected by Postgres."""
    duplicated = [
        Source(
            id="https://example.invalid/audit",
            uri="https://example.invalid/audit",
            content=b"first",
            content_type="text/html",
            content_hash="hash-first",
        ),
        Source(
            id="https://example.invalid/audit",
            uri="https://example.invalid/audit/",
            content=b"second",
            content_type="text/html",
            content_hash="hash-second",
        ),
    ]

    await cache.put_source_entries("webcrawler:duplicated", duplicated)

    stored = await cache.get_source_entries("webcrawler:duplicated")
    assert stored is not None
    assert [source.uri for source in stored] == ["https://example.invalid/audit/"]


async def test_web_crawler_lends_a_page_site_identity_to_same_origin_sources(
    http_server: str,
) -> None:
    """A page's ``og:site_name``/icon are stored, and lent to the PDF-like file it links."""
    sources = {
        source.uri: source
        for source in await WebCrawler(
            url=f"{http_server}/site-identity.html", max_depth=1
        ).run()
    }
    identity = {
        "site_name": "Fixture Site",
        "favicon_url": f"{http_server}/img/favicon.ico",
    }
    page = sources[f"{http_server}/site-identity.html"]
    notes = sources[f"{http_server}/notes.txt"]
    assert {key: page.metadata[key] for key in identity} == identity
    assert {key: notes.metadata[key] for key in identity} == identity
