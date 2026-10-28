"""``SQLiteCache`` honours the ``CacheStore`` contract with no server at all."""

import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from fastrag.exceptions import CacheError
from fastrag.ingestion._hashing import content_hash
from fastrag.ingestion.cache import SQLiteCache
from fastrag.ingestion.models import Source
from fastrag.ingestion.steps.chunking import CharacterTextSplitter
from fastrag.ingestion.steps.fetching import LocalFileFetcher
from fastrag.ingestion.steps.parsing import (
    DefaultParser,
    DocusaurusParser,
    HtmlParser,
    ParserRouter,
    docusaurus_filter,
    html_filter,
)
from fastrag.models import Document


def source(key, content=b"page", **changes):
    return Source(
        key,
        f"https://example.test/{key}",
        content,
        "text/html",
        content_hash(content),
        **changes,
    )


def document(key, source_id="s", embedding=None):
    return Document(
        id=key,
        source_id=source_id,
        content=f"text of {key}",
        step_config_hash="cfg",
        embedding=embedding,
        metadata={"page": 1, "title": "ünïcode"},
    )


@pytest_asyncio.fixture
async def cache(tmp_path):
    async with SQLiteCache(path=tmp_path / "cache" / "fastrag.sqlite") as store:
        yield store


async def test_sources_roundtrip_misses_and_known_empty_buckets(cache, tmp_path):
    first = source(
        "a",
        etag="etag",
        last_modified=datetime(2026, 1, 2, tzinfo=UTC),
        metadata={"renderer": "http"},
        last_validated=datetime.now(UTC) - timedelta(days=1),
    )
    rendered = replace(first, content=b"rendered", metadata={"renderer": "dynamic"})
    await cache.put_source_entries("http", [first])
    await cache.put_source_entries("rendered", [rendered])
    await cache.put_source_entries("empty", [])

    results = await cache.get_source_entries_many(
        ["http", "rendered", "empty", "missing", "http"]
    )
    assert results.keys() == {"http", "rendered", "empty", "missing"}
    assert [replace(hit, cached=False) for hit in results["http"]] == [first]
    assert results["rendered"][0].content == b"rendered"
    assert results["empty"] == [] and results["missing"] is None
    assert all(hit.cached for hit in results["http"])
    assert await cache.get_source_entries_many([]) == {}

    # An unstamped source is stamped on write.
    await cache.put_source_entries("fresh", [source("b")])
    assert (await cache.get_source_entries("fresh"))[0].last_validated is not None

    # Rewriting a bucket replaces it without touching another bucket's snapshot.
    await cache.put_source_entries("http", [replace(first, content=b"updated")])
    async with SQLiteCache(path=tmp_path / "cache" / "fastrag.sqlite") as reopened:
        assert (await reopened.get_source_entries("http"))[0].content == b"updated"
        assert (await reopened.get_source_entries("rendered"))[0].content == b"rendered"


async def test_headers_skip_content_and_materialize(cache):
    await cache.put_source_entries("bucket", [source("a"), source("b", b"other")])
    headers = await cache.get_source_headers_many(["bucket", "missing"])
    assert headers["missing"] is None
    assert [header.content for header in headers["bucket"]] == [b"", b""]
    assert (await headers["bucket"][1].materialize()).content == b"other"
    assert [s.id for s in await cache.get_bucket_sources("bucket", ["b"])] == ["b"]

    docs = [document("d1", embedding=[0.1, 0.2]), document("d2")]
    await cache.put_document_entries("stage", docs)
    [h1, h2] = (await cache.get_document_headers_many(["stage"]))["stage"]
    assert h1.content == "" and h1.metadata == docs[0].metadata
    assert await h1.materialize() == replace(docs[0], cached=True)
    assert (await h2.materialize()).embedding is None


async def test_documents_roundtrip_with_exact_vectors(cache):
    docs = [document("d1", embedding=[0.1, 1 / 3, -2.5e-300]), document("d2")]
    await cache.put_document_entries("stage", [*docs, docs[0]])
    await cache.put_document_entries("empty", [])
    results = await cache.get_document_entries_many(["stage", "empty", "missing"])
    assert results == {
        "stage": [replace(doc, cached=True) for doc in docs],
        "empty": [],
        "missing": None,
    }


async def test_deletion_prunes_only_buckets_it_emptied(cache):
    await cache.put_source_entries("one", [source("a")])
    await cache.put_source_entries("two", [source("a"), source("b")])
    await cache.put_source_entries("empty", [])
    assert await cache.delete_sources(["a", "unknown"]) == 1
    assert await cache.delete_sources([]) == 0
    results = await cache.get_source_entries_many(["one", "two", "empty"])
    assert results["one"] is None and results["empty"] == []
    assert [s.id for s in results["two"]] == ["b"]

    await cache.put_document_entries(
        "chunks", [document("c1", "x"), document("c2", "y")]
    )
    await cache.put_document_entries("vectors", [document("c1", "x")])
    await cache.put_document_entries("nothing", [])
    assert await cache.delete_documents(["x"]) == 1
    assert await cache.get_document_entries_many(["chunks", "vectors", "nothing"]) == {
        "chunks": [replace(document("c2", "y"), cached=True)],
        "vectors": None,
        "nothing": [],
    }


async def test_clear_and_collect_garbage(cache):
    await cache.put_source_entries("s", [source("a"), source("b")])
    await cache.put_document_entries("d", [document("d1")])
    with pytest.raises(ValueError):
        await cache.collect_garbage(timedelta(0))
    assert await cache.collect_garbage(timedelta(days=1)) == 0

    # Age everything, then read one bucket: reads refresh the clock.
    await cache.put_document_entries("old", [document("d2")])
    await cache._run(
        lambda connection: [
            connection.execute(f"UPDATE {table} SET last_accessed = 0")
            for table in ("sources", "documents", "document_cache_entries")
        ]
    )
    await cache.get_document_entries("d")
    assert await cache.collect_garbage(timedelta(days=1)) == 3
    assert await cache.get_source_entries("s") is None
    assert await cache.get_document_entries("old") is None
    assert len(await cache.get_document_entries("d")) == 1

    await cache.put_source_entries("s", [source("a")])
    assert await cache.clear_sources() == 1
    assert await cache.clear() == 1
    assert await cache.get_document_entries("d") is None


async def test_lifecycle_and_errors(tmp_path):
    with pytest.raises(CacheError, match="async with"):
        await SQLiteCache(path=tmp_path / "unused.sqlite").get_source_entries("x")
    async with SQLiteCache(path=":memory:") as memory:
        await memory.put_source_entries("x", [source("a")])
        assert len(await memory.get_source_entries("x")) == 1

    path = tmp_path / "not-a-database.sqlite"
    path.write_bytes(b"not sqlite" * 100)
    with pytest.raises(CacheError):
        async with SQLiteCache(path=path):
            pass
    with pytest.raises(ValueError):
        SQLiteCache(path=path, timeout=-1)

    # A second process holding the write lock surfaces as a transient error.
    locked = tmp_path / "locked.sqlite"
    async with SQLiteCache(path=locked, timeout=0) as store:
        holder = sqlite3.connect(locked, isolation_level=None)
        holder.execute("BEGIN IMMEDIATE")
        try:
            from fastrag.exceptions import TransientCacheError

            with pytest.raises(TransientCacheError):
                await store.put_source_entries("x", [source("a")])
        finally:
            holder.execute("ROLLBACK")
            holder.close()
        await store.put_source_entries("x", [source("a")])


async def test_pipeline_warm_run_is_served_from_the_file(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("# Alpha\n\nAlpha grows. Apples grow.\n")
    (corpus / "b.md").write_text("# Beta\n\nBeta shines. Stars shine.\n")

    async def run():
        async with SQLiteCache(path=tmp_path / "cache.sqlite") as cache:
            fetched = LocalFileFetcher(patterns=[str(corpus)], cache=cache)
            parsed = (
                ParserRouter(cache=cache)
                .route(html_filter & docusaurus_filter, DocusaurusParser())
                .route(html_filter, HtmlParser())
                .default(DefaultParser())(fetched)
            )
            chunks = CharacterTextSplitter(chunk_size=20, chunk_overlap=0, cache=cache)
            return [chunk async for chunk in chunks(parsed)]

    cold = await run()
    warm = await run()
    assert cold and not any(chunk.cached for chunk in cold)
    assert all(chunk.cached for chunk in warm)
    assert sorted(c.id for c in warm) == sorted(c.id for c in cold)
