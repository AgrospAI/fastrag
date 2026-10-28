"""URL hierarchy, subtree totals, status colors and failure summaries."""

import io

import pytest
from helpers import EventPipeline

from fastrag.ingestion import Pipeline
from fastrag.ingestion.events import DocumentFetched, ItemFailed, StepFailed
from fastrag.ingestion.logger import SourceTreeLogger
from fastrag.ingestion.models import Source
from fastrag.ingestion.steps.fetching import LocalFileFetcher, Sources


def fetched(uri, *, cached=False, source_id=None):
    return DocumentFetched(
        Source(source_id or uri, uri, b"content", "text/plain", "hash", cached=cached)
    )


async def test_hierarchy_unique_urls_subtree_totals_and_order():
    event = fetched("https://example.org/docs/intro", cached=True)
    stream = io.StringIO()
    logger = SourceTreeLogger(
        EventPipeline(
            [
                fetched("https://other.org/page"),
                event,
                event,
                fetched("https://example.org/docs"),
                fetched("https://example.org/docs/intro?language=en"),
                fetched("http://example.org/docs/intro"),
                fetched("https://example.org/docs-other"),
            ]
        ),
        stream=stream,
    )
    assert not stream.getvalue()
    await logger.run()
    assert logger.source_count == 6
    assert stream.getvalue().splitlines() == [
        "example.org 4 1",
        "├── /docs 3 1",
        "│   └── /intro 2 1",
        "└── /docs-other 1",
        "other.org 1",
        "└── /page 1",
        "",
        "5 new 1 cached 0 error",
    ]
    assert "\x1b" not in stream.getvalue()
    logger.pipeline.events = [fetched("https://replacement.org")]
    await logger.run()
    assert logger.source_count == 1
    assert stream.getvalue().splitlines()[-3] == "replacement.org 1"
    assert stream.getvalue().splitlines()[-1] == "1 new 0 cached 0 error"


@pytest.mark.parametrize(
    "depth, visible", [(1, "example.org"), (2, "docs"), (3, "intro")]
)
async def test_depth_preserves_hidden_totals(depth, visible):
    stream = io.StringIO()
    await SourceTreeLogger(
        EventPipeline(
            [
                fetched("https://example.org/docs/intro"),
                fetched("https://example.org/docs/other", cached=True),
            ]
        ),
        stream=stream,
        max_depth=depth,
    ).run()
    lines = stream.getvalue().splitlines()
    assert f"{visible} " in lines[depth - 1]
    assert "example.org 1 1" in lines[0]
    assert ("/other " in stream.getvalue()) == (depth == 3)
    assert ("/docs " in stream.getvalue()) == (depth >= 2)


@pytest.mark.parametrize("depth", [0, -1, True, 1.5, "2"])
def test_invalid_depth(depth):
    with pytest.raises(ValueError, match="max_depth"):
        SourceTreeLogger(EventPipeline([]), max_depth=depth)


@pytest.mark.parametrize("fatal", [False, True])
async def test_failure_lineage_deduplication_and_partial_summary(fatal):
    error = RuntimeError("pipeline failed")
    stream = io.StringIO()
    events = [
        ItemFailed(uri="chunk-1", source_id="s"),
        fetched("https://example.org/docs/intro", cached=True, source_id="s"),
        ItemFailed(uri="chunk-2", source_id="s"),
        ItemFailed(uri="https://example.org/missing"),
        ItemFailed(uri="unknown-document-id"),
        StepFailed("Parser", error),
    ]
    logger = SourceTreeLogger(
        EventPipeline(events, error if fatal else None), stream=stream
    )
    if fatal:
        with pytest.raises(RuntimeError, match="pipeline failed"):
            await logger.run()
    else:
        await logger.run()
    output = stream.getvalue()
    assert logger.source_count == 2
    assert "example.org 2" in output
    assert "/intro 1" in output
    assert "Unattributed item failures: 1" in output
    assert "Failed steps: 1" in output
    assert output.splitlines()[-1] == "0 new 0 cached 2 error"


@pytest.mark.parametrize("color", [None, True, False])
async def test_status_colors(color):
    class Terminal(io.StringIO):
        def isatty(self):
            return True

    stream = Terminal()
    await SourceTreeLogger(
        EventPipeline(
            [
                fetched("https://example.org/new"),
                fetched("https://example.org/cached", cached=True),
                ItemFailed(uri="https://example.org/error"),
            ]
        ),
        stream=stream,
        color=color,
    ).run()
    output = stream.getvalue()
    lines = output.splitlines()
    if color is False:
        assert "\x1b" not in output
        assert lines[0] == "example.org 1 1 1"
        assert lines[-1] == "1 new 1 cached 1 error"
    else:
        assert lines[0] == (
            "example.org \x1b[32m1\x1b[0m \x1b[36m1\x1b[0m \x1b[31m1\x1b[0m"
        )
        assert lines[-1] == (
            "\x1b[32m1\x1b[0m new \x1b[36m1\x1b[0m cached \x1b[31m1\x1b[0m error"
        )


async def test_empty_and_literal_output():
    stream = io.StringIO()
    await SourceTreeLogger(EventPipeline([]), stream=stream).run()
    assert stream.getvalue() == "0 new 0 cached 0 error\n"
    stream = io.StringIO()
    await SourceTreeLogger(
        EventPipeline(
            [fetched("file:///docs/[bold]literal[/bold]"), fetched("http://[invalid")]
        ),
        stream=stream,
    ).run()
    assert "file://" in stream.getvalue()
    assert "[bold]literal[" in stream.getvalue()
    assert "(other)" in stream.getvalue()


async def test_real_sources_aggregate(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("test")
    stream = io.StringIO()
    logger = SourceTreeLogger(
        Pipeline(Sources([LocalFileFetcher(patterns=[str(path)])])),
        stream=stream,
    )
    await logger.run()
    assert logger.source_count == 1
    assert "/source.txt 1" in stream.getvalue()


async def test_source_headers_match_full_tree_and_preserve_freshness(
    cache, http_server, monkeypatch
):
    from unittest.mock import AsyncMock

    from fastrag.ingestion.cache import source_cache_id
    from fastrag.ingestion.steps.fetching import SitemapCrawler

    crawler = SitemapCrawler(url=f"{http_server}/sitemap.xml")
    full = io.StringIO()
    headers = io.StringIO()
    try:
        await SourceTreeLogger(Pipeline(crawler, cache=cache), stream=full).run()
        full.seek(0)
        full.truncate()
        await SourceTreeLogger(Pipeline(crawler, cache=cache), stream=full).run()
        read = cache.get_source_entries_many
        spy = AsyncMock(wraps=read)
        monkeypatch.setattr(cache, "get_source_entries_many", spy)
        logger = SourceTreeLogger(
            Pipeline(crawler, cache=cache, source_headers_only=True), stream=headers
        )
        await logger.run()
        assert headers.getvalue() == full.getvalue()
        assert logger.source_count == 7
        spy.assert_not_awaited()

        # Expired entries still hydrate before validation, preserving bytes.
        key = source_cache_id(
            crawler.name, f"{http_server}/page-a.html", crawler.config_hash
        )
        original = (await read([key]))[key][0]
        hydrate = AsyncMock(wraps=cache.get_bucket_sources)
        monkeypatch.setattr(cache, "get_bucket_sources", hydrate)
        crawler.cache_validation_interval = 0
        await logger.run()
        assert hydrate.await_count > 0
        current = (await read([key]))[key][0]
        assert current.content == original.content
        assert current.content_hash == original.content_hash
    finally:
        await crawler.aclose()


def test_source_headers_reject_transform_graphs():
    from fastrag.ingestion.steps.parsing import DefaultParser

    source = LocalFileFetcher(patterns=[])
    with pytest.raises(ValueError, match="source-only"):
        Pipeline(DefaultParser()(source), source_headers_only=True)


async def test_headers_only_pipeline_reads_cached_transform_graphs_without_payloads(
    cache, http_server, monkeypatch
):
    from unittest.mock import AsyncMock

    from fastrag.ingestion.steps.fetching import SitemapCrawler
    from fastrag.ingestion.steps.parsing import DefaultParser

    crawler = SitemapCrawler(url=f"{http_server}/sitemap.xml")

    async def run(**options):
        pipeline = Pipeline(DefaultParser()(crawler), cache=cache, **options)
        return [event async for event in pipeline]

    try:
        cold = await run()
        warm = await run()
        sources = AsyncMock(wraps=cache.get_source_entries_many)
        documents = AsyncMock(wraps=cache.get_document_entries_many)
        monkeypatch.setattr(cache, "get_source_entries_many", sources)
        monkeypatch.setattr(cache, "get_document_entries_many", documents)
        headers = await run(headers_only=True)

        assert [type(event) for event in headers] == [type(event) for event in warm]
        assert len(headers) == len(cold) > 0
        sources.assert_not_awaited()
        documents.assert_not_awaited()
    finally:
        await crawler.aclose()
