"""Cache performance regressions against real PostgreSQL and async scheduling."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import event, inspect, select, update

from fastrag.exceptions import CacheError, TransientCacheError
from fastrag.ingestion.cache import PostgresCache
from fastrag.ingestion.cache._batching import ReadBatcher
from fastrag.ingestion.cache.schema import (
    CacheBase,
    DocumentRecord,
    DocumentSnapshot,
    SourceRecord,
    SourceSnapshot,
)
from fastrag.ingestion.models import Source
from fastrag.models import Document


def document(key):
    return Document(
        id=key,
        source_id="https://example.test/page",
        content=key,
        embedding=[0.1, 0.2],
        metadata={"page": 1},
    )


async def test_source_buckets_own_bytes_metadata_and_validators(cache):
    from dataclasses import replace

    from fastrag.ingestion._hashing import content_hash

    first = Source(
        "shared",
        "https://example.test/page",
        b"http",
        "text/html",
        content_hash(b"http"),
        etag="http-etag",
        metadata={"renderer": "http"},
        last_validated=datetime.now(UTC) - timedelta(days=1),
    )
    rendered = replace(
        first,
        content=b"rendered",
        content_hash=content_hash(b"rendered"),
        etag=None,
        metadata={"renderer": "dynamic"},
        last_validated=datetime.now(UTC),
    )
    await cache.put_source_entries("http", [first])
    await cache.put_source_entries("rendered", [rendered])
    await cache.put_source_entries("empty", [])
    results = await cache.get_source_entries_many(
        ["http", "rendered", "empty", "missing"]
    )
    for key, expected in [("http", first), ("rendered", rendered)]:
        hit = results[key][0]
        assert replace(hit, cached=False) == expected
        header = (await cache.get_source_headers_many([key]))[key][0]
        assert header.etag == expected.etag
        assert header.last_validated == expected.last_validated
        assert await header.materialize() == hit
    assert results["empty"] == [] and results["missing"] is None
    await cache.put_source_entries("http", [replace(first, content=b"updated")])
    assert (await cache.get_source_entries("rendered"))[0].content == b"rendered"
    async with PostgresCache(url=cache._url, schema=cache._schema_name) as reopened:
        assert (await reopened.get_source_entries("rendered"))[0].content == b"rendered"
        assert (await reopened.get_source_entries("http"))[0].content == b"updated"
    assert await cache.delete_sources([first.id]) == 1
    assert await cache.get_source_entries_many(["http", "rendered", "empty"]) == {
        "http": None,
        "rendered": None,
        "empty": [],
    }


async def test_legacy_source_buckets_are_invalidated_on_snapshot_upgrade(cache):
    from sqlalchemy import text

    source = Source("s", "https://example.test", b"ambiguous", "text/plain", "hash")
    await cache.put_source_entries("http", [source])
    await cache.put_source_entries("dynamic", [source])
    await cache.put_document_entries("keep", [document("keep")])
    schema = cache._engine.dialect.identifier_preparer.quote_schema(cache._schema_name)
    async with cache._engine.begin() as connection:
        for column in SourceSnapshot.__table__.columns:
            if column.name not in {"source_cache_id", "source_id"}:
                await connection.execute(
                    text(
                        f"ALTER TABLE {schema}.source_cache_entries DROP COLUMN {column.name}"
                    )
                )
    await cache.__aenter__()
    assert await cache.get_source_entries_many(["http", "dynamic"]) == {
        "http": None,
        "dynamic": None,
    }
    assert (await cache.get_document_entries("keep"))[0].content == "keep"
    await cache.put_source_entries("http", [source])
    await cache.__aenter__()
    assert (await cache.get_source_entries("http"))[0].content == source.content


async def test_bulk_reads_and_coalescing_preserve_cache_contract(cache: PostgresCache):
    docs = [document(f"d:{i}") for i in range(64)]
    for i, doc in enumerate(docs):
        await cache.put_document_entries(f"bucket:{i}", [doc])
    await cache.put_document_entries("shared", [docs[0], docs[1]])
    await cache.put_document_entries("empty", [])
    source = Source(
        id="source",
        uri="https://example.test/page",
        content=b"page",
        content_type="text/plain",
        content_hash="hash",
    )
    await cache.put_source_entries("source:hit", [source])
    await cache.put_source_entries("source:empty", [])
    statements = []

    def record(*args):
        statements.append(args[2])

    event.listen(cache._engine.sync_engine, "before_cursor_execute", record)
    try:
        hits = await asyncio.gather(
            *[cache.get_document_entries(f"bucket:{i}") for i in range(64)]
        )
        assert len(statements) == 1
        assert [entries[0].id for entries in hits] == [doc.id for doc in docs]
        assert all(entries[0].cached for entries in hits)
        assert hits[0][0].embedding == docs[0].embedding
        assert hits[0][0].metadata == docs[0].metadata
        statements.clear()
        results = await cache.get_document_entries_many(
            ["bucket:0", "shared", "empty", "missing", "bucket:0"]
        )
        assert len(statements) == 1
        assert results["empty"] == [] and results["missing"] is None
        assert [doc.id for doc in results["shared"]] == ["d:0", "d:1"]
        assert results["bucket:0"][0].id == "d:0"
        statements.clear()
        assert await cache.get_document_entries_many([]) == {}
        assert await cache.get_source_entries_many([]) == {}
        assert statements == []
        sources = await asyncio.gather(
            *[
                cache.get_source_entries(key)
                for key in ["source:hit", "source:empty", "source:missing"]
            ]
        )
        assert len(statements) == 1
        assert sources[0][0].content == b"page"
        assert sources[0][0].last_validated is not None
        assert sources[1:] == [[], None]
        statements.clear()
        await cache.get_document_entries_many([f"missing:{i}" for i in range(257)])
        assert len(statements) == 3
    finally:
        event.remove(cache._engine.sync_engine, "before_cursor_execute", record)
    await cache.put_document_entries("bucket:0", [])
    assert await cache.get_document_entries("bucket:0") == []
    await cache.delete_documents([docs[0].source_id])
    results = await cache.get_document_entries_many(["bucket:1", "shared", "empty"])
    assert results == {"bucket:1": None, "shared": None, "empty": []}
    await cache.clear()
    assert await cache.get_document_entries("empty") is None


async def test_large_buckets_roundtrip_and_reads_refresh_gc_timestamp(
    cache: PostgresCache,
):
    docs = [document(f"large:{i:05}") for i in range(5000)]
    await cache.put_document_entries("large", [*docs, docs[0]])
    table = DocumentRecord
    old = datetime.now(UTC) - timedelta(days=3)
    async with cache._engine.begin() as connection:
        await connection.execute(update(table).values(last_accessed=old))
    result = await cache.get_document_entries_many(["large"])
    assert len(result["large"]) == 5000
    assert [doc.id for doc in result["large"]] == [doc.id for doc in docs]
    assert await cache.collect_garbage(timedelta(days=1)) == 0
    await cache.put_document_entries("large", docs[:1])
    assert len(await cache.get_document_entries("large")) == 1

    sources = [
        Source(
            id=f"source:{i}",
            uri=f"https://example.test/{i}",
            content=b"page",
            content_type="text/plain",
            content_hash=str(i),
        )
        for i in range(5000)
    ]
    await cache.put_source_entries("large:sources", sources)
    source_results = await cache.get_source_entries_many(["large:sources", "missing"])
    assert len(source_results["large:sources"]) == 5000
    assert source_results["missing"] is None


async def test_index_upgrade_on_existing_schema(cache: PostgresCache):
    indexes = [
        index
        for table in CacheBase.metadata.tables.values()
        for index in table.indexes
        if next(iter(index.columns)).name in {"source_id", "document_id"}
    ]
    assert len(indexes) == 4
    async with cache._engine.begin() as connection:
        for index in indexes:
            await connection.run_sync(index.drop)
    await cache.__aenter__()
    await cache.__aenter__()
    async with cache._engine.begin() as connection:
        for table in {index.table for index in indexes}:
            actual = await connection.run_sync(
                lambda sync, table=table: inspect(sync).get_indexes(
                    table.name, schema=cache._schema_name
                )
            )
            assert {
                tuple(column.name for column in index.columns)
                for index in indexes
                if index.table is table
            } <= {tuple(index["column_names"]) for index in actual}


async def test_batch_cancellation_failure_and_partial_flush():
    entered = asyncio.Event()
    release = asyncio.Event()
    cancelled = asyncio.Event()
    calls = []

    async def read(keys):
        calls.append(keys)
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return dict.fromkeys(keys, [])

    batcher = ReadBatcher(read, 2, 0.001)
    first = asyncio.create_task(batcher.get("same"))
    second = asyncio.create_task(batcher.get("same"))
    await asyncio.wait_for(entered.wait(), 1)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release.set()
    assert await asyncio.wait_for(second, 1) == []
    assert calls == [["same"]]
    assert not cancelled.is_set()
    assert await asyncio.wait_for(batcher.get("same"), 1) == []
    assert len(calls) == 2

    async def failing(keys):
        raise TransientCacheError("database unavailable")

    failing_batcher = ReadBatcher(failing, 8, 0.001)
    errors = await asyncio.gather(
        *[failing_batcher.get(str(i)) for i in range(3)], return_exceptions=True
    )
    assert all(isinstance(error, TransientCacheError) for error in errors)
    await failing_batcher.close()
    entered.clear()
    release.clear()
    pending = asyncio.create_task(batcher.get("pending"))
    await asyncio.wait_for(entered.wait(), 1)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await asyncio.wait_for(cancelled.wait(), 1)
    release.set()
    assert await asyncio.wait_for(batcher.get("after-timeout"), 1) == []
    entered.clear()
    release.clear()
    pending = asyncio.create_task(batcher.get("closing"))
    await asyncio.wait_for(entered.wait(), 1)
    await batcher.close()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert batcher._task is None and not batcher._pending


async def test_single_read_failure_and_reopen(cache: PostgresCache):
    await cache.__aexit__(None, None, None)
    with pytest.raises(CacheError):
        await asyncio.wait_for(cache.get_document_entries("missing"), 1)
    await cache.__aenter__()
    assert await asyncio.wait_for(cache.get_document_entries("missing"), 1) is None


async def test_snapshots_isolate_raw_chunks_and_embedding_configurations(cache):
    from dataclasses import replace

    raw = replace(document("shared-id"), embedding=None, metadata={"stage": "chunk"})
    first = replace(raw, embedding=[1.0, 2.0], metadata={"stage": "embedding-a"})
    second = replace(raw, embedding=[9.0, 8.0], metadata={"stage": "embedding-b"})
    await cache.put_document_entries("chunker:raw", [raw])
    await cache.put_document_entries("embedder:a", [first])
    await cache.put_document_entries("embedder:b", [second])
    results = await cache.get_document_entries_many(
        ["chunker:raw", "embedder:a", "embedder:b"]
    )
    assert results["chunker:raw"][0].embedding is None
    assert results["chunker:raw"][0].metadata == raw.metadata
    assert results["embedder:a"][0].embedding == first.embedding
    assert results["embedder:b"][0].embedding == second.embedding


async def test_legacy_upgrade_preserves_only_unambiguous_vectors(cache):
    from sqlalchemy import text

    await cache.put_document_entries("chunker:old", [document("legacy")])
    await cache.put_document_entries("testembedder:a", [document("legacy")])
    await cache.put_document_entries("testembedder:b", [document("ambiguous")])
    await cache.put_document_entries("testembedder:c", [document("ambiguous")])
    schema = cache._schema_name
    async with cache._engine.begin() as connection:
        await connection.execute(update(DocumentRecord).values(embedding=[0.1, 0.2]))
        await connection.execute(
            text(
                f'ALTER TABLE "{schema}".document_cache_entries DROP COLUMN payload, DROP COLUMN embedding'
            )
        )
    await cache.__aenter__()
    results = await cache.get_document_entries_many(
        ["chunker:old", "testembedder:a", "testembedder:b", "testembedder:c"]
    )
    assert results["chunker:old"] is None
    assert results["testembedder:a"][0].embedding == [0.1, 0.2]
    assert results["testembedder:b"] is None and results["testembedder:c"] is None
    await cache.__aenter__()
    assert (await cache.get_document_entries("testembedder:a"))[0].embedding == [
        0.1,
        0.2,
    ]


async def test_stage_cache_windows_are_independent_of_worker_concurrency(cache):
    from fastrag.ingestion.policy import Policy
    from fastrag.ingestion.steps.base import DocumentStep, PipelineStep

    class Inputs(PipelineStep):
        def config(self):
            return {}

        async def evaluate(self, context):
            for i in range(129):
                yield document(f"window:{i}")

    class Identity(DocumentStep):
        calls = 0

        def config(self):
            return {}

        async def transform(self, item, context):
            self.calls += 1
            yield item

    step = Identity(cache=cache, policy=Policy(concurrency=1, cache_batch_size=64))(
        Inputs()
    )
    assert len(await step.run()) == 129
    assert step.calls == 129
    statements = []

    def record(*args):
        statements.append(args[2])

    event.listen(cache._engine.sync_engine, "before_cursor_execute", record)
    try:
        results = await step.run()
    finally:
        event.remove(cache._engine.sync_engine, "before_cursor_execute", record)
    assert len(results) == 129 and all(item.cached for item in results)
    assert step.calls == 129
    assert len(statements) == 3


async def test_index_fast_path_skips_vectors_and_repairs_missing_destination(cache):
    from fastrag.ingestion.policy import Policy
    from fastrag.ingestion.steps.base import DocumentStep, PipelineStep
    from fastrag.ingestion.steps.embedding.base import Embedder
    from fastrag.vector_store import VectorStore

    class Inputs(PipelineStep):
        def config(self):
            return {}

        async def evaluate(self, context):
            for i in range(129):
                item = document(f"index:{i}")
                from dataclasses import replace

                yield replace(item, embedding=None)

    class RawCache(DocumentStep):
        def config(self):
            return {}

        async def transform(self, item, context):
            yield item

    class FakeEmbedder(Embedder):
        calls = 0
        revision = 1

        def config(self):
            return {"model": "test", "revision": self.revision}

        @property
        def dimension(self):
            return 2

        @property
        def model_id(self):
            return "test"

        async def embed(self, texts):
            assert all(text.startswith("index:") for text in texts)
            self.calls += len(texts)
            return [[1.0, 0.5] for _ in texts]

    class Store(VectorStore):
        present = set()
        checks = []

        async def ensure_ready(self, dimension):
            assert dimension == 2

        async def existing_ids(self, ids):
            self.checks.append(len(ids))
            return self.present.intersection(ids)

        async def upsert(self, documents):
            assert all(
                item.embedding == [1.0, 0.5] and item.content.startswith("index:")
                for item in documents
            )
            self.present.update(item.id for item in documents)

        async def search(self, embedding, *, k, filter=None):
            return []

    policy = Policy(concurrency=4, batch_size=4, cache_batch_size=64, on_error="raise")
    embedder = FakeEmbedder(cache=cache, policy=policy)
    store = Store(policy=policy)
    raw_step = RawCache(cache=cache, policy=policy)(Inputs())
    step = store.save(embedder(raw_step), skip_existing=True)
    from dataclasses import asdict

    from fastrag.ingestion.events import DocumentEmbedded, StepCompleted, StepStarted
    from fastrag.ingestion.steps.base import event_sink_scope

    observed = []

    async def observe(event):
        observed.append(event)
        # Public records must be ordinary DTOs, never live cache/engine handles.
        serialized = asdict(event)
        assert "_store" not in repr(serialized) and "_bucket" not in repr(serialized)

    with event_sink_scope(observe):
        assert len(await step.run()) == 129
    assert any(
        isinstance(item, StepStarted) and item.step == embedder.name
        for item in observed
    )
    assert any(
        isinstance(item, StepCompleted) and item.step == embedder.name
        for item in observed
    )
    assert embedder.calls == 129
    store.checks.clear()
    # A fully indexed rerun never even checks the embedding cache.
    statements = []

    def record(*args):
        statements.append(args[2])

    event.listen(cache._engine.sync_engine, "before_cursor_execute", record)
    observed.clear()
    try:
        with event_sink_scope(observe):
            results = await step.run()
    finally:
        event.remove(cache._engine.sync_engine, "before_cursor_execute", record)
    assert all(type(item) is Document for item in results)
    assert all(item.cached and item.embedding is None for item in results)
    embedded_events = [e for e in observed if isinstance(e, DocumentEmbedded)]
    assert len(embedded_events) == 129
    assert all(
        e.document.cached and e.document.embedding is None for e in embedded_events
    )
    assert {e.document.id for e in embedded_events} == {item.id for item in results}
    assert store.checks == [64, 64, 1]
    assert len(statements) == 3
    assert all(
        "touched.payload" not in statement and "touched.embedding" not in statement
        for statement in statements
    )
    assert embedder.calls == 129
    # A partially indexed window logs both bypassed and restored embeddings.
    missing_id = store.present.pop()
    observed.clear()
    with event_sink_scope(observe):
        assert len(await step.run()) == 129
    embedded_events = [e for e in observed if isinstance(e, DocumentEmbedded)]
    assert len(embedded_events) == 129
    assert sum(e.document.embedding is None for e in embedded_events) == 128
    assert next(
        e.document for e in embedded_events if e.document.id == missing_id
    ).embedding == [1.0, 0.5]
    assert embedder.calls == 129
    # Recreating a destination repairs it from cached vectors, not the provider.
    store.present.clear()
    assert len(await step.run()) == 129
    assert len(store.present) == 129 and embedder.calls == 129
    # A new embedding configuration/destination must hydrate cached raw text
    # before a provider call, even though the upstream was read as headers.
    store.present.clear()
    embedder.revision = 2
    assert len(await step.run()) == 129
    assert embedder.calls == 258
    assert all(item.content.startswith("index:") for item in await raw_step.run())

    from dataclasses import replace

    from fastrag.exceptions import VectorStoreError
    from fastrag.ingestion.events import ItemFailed

    async def fail(ids):
        raise VectorStoreError("destination unavailable")

    store.existing_ids = fail
    store.policy = replace(policy, on_error="skip")
    observed.clear()
    with event_sink_scope(observe):
        assert await store.save(embedder, skip_existing=True).run() == []
    assert len([item for item in observed if isinstance(item, ItemFailed)]) == 129
    assert embedder.calls == 258

    async def absent(ids):
        return set()

    async def fail_upsert(documents):
        raise VectorStoreError("write unavailable")

    store.existing_ids = absent
    store.upsert = fail_upsert
    observed.clear()
    with event_sink_scope(observe):
        assert await store.save(embedder, skip_existing=True).run() == []
    assert len([item for item in observed if isinstance(item, ItemFailed)]) == 129
    assert embedder.calls == 258


async def test_stage_owned_cleanup_preserves_other_sources_and_models(cache):
    from dataclasses import replace

    raw = document("same-content")
    await cache.put_document_entries("source:a", [replace(raw, source_id="a")])
    await cache.put_document_entries("source:b", [replace(raw, source_id="b")])
    assert await cache.delete_documents(["a"]) == 0  # Logical id still referenced by b.
    assert await cache.get_document_entries("source:a") is None
    assert (await cache.get_document_entries("source:b"))[0].source_id == "b"
    await cache.put_document_entries("model:old", [replace(raw, source_id="b")])
    async with cache._engine.begin() as connection:
        await connection.execute(
            update(DocumentSnapshot)
            .where(DocumentSnapshot.bucket_id == "model:old")
            .values(last_accessed=datetime.now(UTC) - timedelta(days=3))
        )
    assert await cache.collect_garbage(timedelta(days=1)) == 0
    assert await cache.get_document_entries("model:old") is None
    assert await cache.get_document_entries("source:b") is not None
    assert await cache.delete_documents(["b"]) == 1


async def test_compression_upgrade_and_fast_schema_check(cache):
    import json

    from sqlalchemy import text

    from fastrag.ingestion.cache._payloads import unpack_payload

    source = Source(
        id="encoding",
        uri="file:///encoding",
        content=b"repeated text " * 10000,
        content_type="text/plain",
        content_hash="unchanged",
    )
    await cache.put_source_entries("encoding:source", [source])
    doc = document("encoding:document")
    await cache.put_document_entries("encoding:document", [doc])
    schema = cache._schema_name
    async with cache._engine.begin() as connection:
        stored = (
            await connection.execute(select(DocumentSnapshot.payload))
        ).scalar_one()
        assert len(stored) < len(json.dumps(unpack_payload(stored)))
        assert (
            await connection.execute(select(SourceRecord.content))
        ).scalar_one() != source.content
        # Simulate the intermediate JSONB schema and uncompressed source bytes.
        await connection.execute(update(SourceRecord).values(content=source.content))
        await connection.execute(
            text(f'ALTER TABLE "{schema}".sources DROP COLUMN content_encoding')
        )
        await connection.execute(
            text(
                f'ALTER TABLE "{schema}".document_cache_entries '
                "ALTER COLUMN payload TYPE JSONB USING '{}'::jsonb"
            )
        )
        await connection.execute(
            text(
                f'UPDATE "{schema}".document_cache_entries '
                "SET payload = CAST(:payload AS JSONB)"
            ),
            {"payload": json.dumps(unpack_payload(stored))},
        )
    await cache.__aenter__()
    assert (await cache.get_source_entries("encoding:source"))[
        0
    ].content == source.content
    assert (await cache.get_document_entries("encoding:document"))[
        0
    ].embedding == doc.embedding
    statements = []

    def record(*args):
        statements.append(args[2])

    event.listen(cache._engine.sync_engine, "before_cursor_execute", record)
    try:
        await cache.__aenter__()
    finally:
        event.remove(cache._engine.sync_engine, "before_cursor_execute", record)
    assert len(statements) == 1


async def test_source_headers_hydrate_on_parser_miss_and_revalidation(cache):
    from fastrag.ingestion import Pipeline
    from fastrag.ingestion.cache import source_cache_id
    from fastrag.ingestion.policy import Policy
    from fastrag.ingestion.steps.base import SourceStep
    from fastrag.ingestion.steps.embedding.base import Embedder
    from fastrag.ingestion.steps.parsing import DefaultParser
    from fastrag.vector_store import VectorStore

    body = b"The original source body must survive lazy cache reads."

    class Input(SourceStep):
        validations = 0

        def config(self):
            return {}

        async def targets(self, context):
            return ["file:///header.txt"]

        async def evaluate_target(self, target, context):
            yield Source(
                id=target,
                uri=target,
                content=body,
                content_type="text/plain",
                content_hash="body",
            )

        async def evaluate(self, context):
            async for source in self.evaluate_target("file:///header.txt", context):
                yield source

        async def is_source_modified(self, *args):
            self.validations += 1
            return False

    class FakeEmbedder(Embedder):
        calls = 0

        def config(self):
            return {}

        @property
        def dimension(self):
            return 1

        @property
        def model_id(self):
            return "headers"

        async def embed(self, texts):
            assert texts == [body.decode()]
            self.calls += 1
            return [[1.0]]

    class Store(VectorStore):
        present = set()

        async def ensure_ready(self, dimension):
            pass

        async def existing_ids(self, ids):
            return set(ids).intersection(self.present)

        async def upsert(self, documents):
            self.present.update(item.id for item in documents)

        async def search(self, embedding, *, k, filter=None):
            return []

    policy = Policy(batch_size=1, concurrency=1, on_error="raise")
    source = Input(cache_validation_interval=0)
    embedder = FakeEmbedder(policy=policy)
    root = Store(policy=policy).save(
        embedder(DefaultParser()(source)), skip_existing=True
    )

    async def run():
        async with Pipeline(root, cache=cache, policy=policy) as pipeline:
            return [event async for event in pipeline]

    await run()
    await run()  # Expired source header is hydrated before validation rewrites.
    assert source.validations == 1 and embedder.calls == 1
    key = source_cache_id(source.name, "file:///header.txt", source.config_hash)
    headers = await cache.get_source_headers_many([key, "missing"])
    assert headers[key][0].content == b"" and headers["missing"] is None
    assert (await headers[key][0].materialize()).content == body
    await cache.clear_documents()
    source.cache_validation_interval = None
    await run()  # Parser miss must hydrate a fresh source header too.
    assert embedder.calls == 1
    assert (await cache.get_source_entries(key))[0].content == body

    from fastrag.ingestion.filters import Filter, MediaTypeFilter
    from fastrag.ingestion.steps.parsing import ParserRouter

    class ContentFilter(Filter):
        def matches(self, item):
            assert item.content == body
            return True

    predicate = ContentFilter() & MediaTypeFilter("text/plain")
    parser = ParserRouter().route(predicate, DefaultParser())
    routed = Store(policy=policy).save(embedder(parser(source)), skip_existing=True)
    async with Pipeline(routed, cache=cache, policy=policy) as pipeline:
        assert [item async for item in pipeline]
    assert not predicate.header_safe
    assert (~predicate).header_safe is False
    assert (predicate | MediaTypeFilter("text/plain")).header_safe is False


async def test_source_deletion_prunes_replaced_bucket_orphans(cache):
    await cache.put_document_entries("replaced", [document("orphan")])
    await cache.put_document_entries("replaced", [document("current")])
    assert await cache.delete_documents([document("current").source_id]) == 2
    assert await cache.get_document_entries("replaced") is None
    async with cache._engine.begin() as connection:
        assert (await connection.execute(select(DocumentRecord))).all() == []
