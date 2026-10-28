"""Isolated PostgreSQL structural stage snapshots and embedding cache replay."""

from dataclasses import replace

from helpers import RecordingEmbedder

from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext
from fastrag.ingestion.steps.chunking import (
    StructureAwareChunker,
    Utf8ByteTokenizer,
    build_embedding_text,
)
from fastrag.models import Document


async def stream(items):
    for item in items:
        yield item


def test_structure_chunker_inherits_pipeline_cache_by_default():
    from fastrag.ingestion.steps.base import _resolve_cache

    pipeline_cache = object()
    context = ExecutionContext("t", Policy(), cache=pipeline_cache)
    assert _resolve_cache(StructureAwareChunker(), context) is pipeline_cache
    assert _resolve_cache(StructureAwareChunker(cache=False), context) is None


async def test_structure_postgres_roundtrip_headers_and_embedding_replay(cache):
    doc = Document(
        "parent",
        "source",
        "# Root\n## A\nBody\n",
        metadata={"content_format": "markdown", "source_uri": "local:test"},
    )
    # Mimic parser snapshot projection on a fully warm pipeline.
    await cache.put_document_entries("parent-snapshot", [doc])
    headers = await cache.get_document_headers_many(["parent-snapshot"])
    reports = []
    chunker = StructureAwareChunker(
        tokenizer=Utf8ByteTokenizer(), cache=cache, diagnostics=reports.append
    )
    context = ExecutionContext("t", Policy(on_error="raise"), cache=cache)
    first = [c async for c in chunker.evaluate_items(stream([doc]), context)]
    warm_context = replace(context, cache_headers_only=True)
    second = [
        c
        async for c in chunker.evaluate_items(
            stream(headers["parent-snapshot"]), warm_context
        )
    ]
    assert second == [replace(c, cached=True) for c in first]
    assert reports[-1].cached_chunks == len(second)
    embedder = RecordingEmbedder(cache=cache)
    embedded = [c async for c in embedder.evaluate_items(stream(first), context)]
    calls = len(embedder.texts)
    replay = [c async for c in embedder.evaluate_items(stream(second), context)]
    assert replay == [replace(c, cached=True) for c in embedded]
    assert len(embedder.texts) == calls
    assert embedder.texts == [
        build_embedding_text(c.content, c.metadata) for c in first
    ]
    changed = StructureAwareChunker(
        tokenizer=Utf8ByteTokenizer(),
        include_heading_path_in_embedding=False,
        cache=cache,
    )
    changed_chunks = [c async for c in changed.evaluate_items(stream([doc]), context)]
    assert first[0].id != changed_chunks[0].id
    assert all(not c.cached for c in changed_chunks)
    await_output = [
        c async for c in embedder.evaluate_items(stream(changed_chunks), context)
    ]
    assert all(not c.cached for c in await_output)
    assert len(embedder.texts) > calls


async def test_chunker_router_validates_corrupt_structural_cache_replay(cache):
    import pytest

    from fastrag.exceptions import ChunkingError
    from fastrag.ingestion.cache import document_cache_id
    from fastrag.ingestion.steps.chunking import ChunkerRouter

    doc = Document(
        "routed-parent", "s", "# H\nbody", metadata={"content_format": "markdown"}
    )
    reports = []
    child = StructureAwareChunker(
        tokenizer=Utf8ByteTokenizer(), cache=cache, diagnostics=reports.append
    )
    router = ChunkerRouter(cache=cache).default(child)
    context = ExecutionContext("t", Policy(on_error="raise"), cache=cache)
    first = [c async for c in router.evaluate_items(stream([doc]), context)]
    second = [c async for c in router.evaluate_items(stream([doc]), context)]
    assert second == [replace(c, cached=True) for c in first]
    assert reports[-1].cached_chunks == len(second)
    bad = replace(first[0], metadata={**first[0].metadata, "embedding_token_count": 0})
    await cache.put_document_entries(
        document_cache_id(child.name, doc.id, child.config_hash), [bad]
    )
    with pytest.raises(ChunkingError):
        _ = [c async for c in router.evaluate_items(stream([doc]), context)]


async def test_structural_metadata_milvus_retrieval_and_parent_resolution(vector_store):
    from fastrag.ingestion.steps.chunking import validate_structure_chunks
    from fastrag.retrieval import SectionIndex

    doc = Document(
        "milvus-parent",
        "source",
        "# Root\n## Part\nExact body\n",
        metadata={"content_format": "markdown", "source_uri": "local:parent"},
    )
    chunker = StructureAwareChunker(tokenizer=Utf8ByteTokenizer(), cache=False)
    context = ExecutionContext("t", Policy(on_error="raise"))
    chunks = [c async for c in chunker.transform(doc, context)]
    embedded = [replace(c, embedding=[1.0, 0.0, 0.0, 0.0]) for c in chunks]
    await vector_store.ensure_ready(4)
    await vector_store.upsert(embedded)
    hits = await vector_store.search([1.0, 0.0, 0.0, 0.0], k=len(chunks))
    assert {c.id for c in hits} == {c.id for c in chunks}
    for hit in hits:
        original = next(c for c in chunks if c.id == hit.id)
        assert hit.content == original.content
        assert all(
            hit.metadata[key] == value for key, value in original.metadata.items()
        )
    normalized = chunker.normalizers.normalize(doc)
    validate_structure_chunks(
        normalized, [(c.content, c.metadata) for c in hits], chunker.tokenizer, 600
    )
    assert SectionIndex([normalized]).resolve(hits[0])[0].heading_path == (
        "Root",
        "Part",
    )


async def test_chat_refinement_postgres_cache_identity_and_replay(cache):
    from helpers import FakeChat

    from fastrag.ingestion.steps.chunking import ChatBoundaryRefiner

    document = Document(
        "refined-parent",
        "source",
        "First sentence. Second sentence.",
        metadata={"content_format": "markdown"},
    )
    model = FakeChat(['{"boundaries": [16]}', '{"boundaries": []}'])
    reports = []
    strategy = ChatBoundaryRefiner(
        llm=model, allow_external_transmission=True, identity="local:fake:v1"
    )
    chunker = StructureAwareChunker(
        tokenizer=Utf8ByteTokenizer(),
        cache=cache,
        semantic_mode="llm",
        semantic_refiner=strategy,
        diagnostics=reports.append,
    )
    context = ExecutionContext("t", Policy(on_error="raise"), cache=cache)
    first = [c async for c in chunker.evaluate_items(stream([document]), context)]
    assert len(first) == 2 and model.calls == 1
    assert any(report.model_calls == 1 for report in reports)
    second = [c async for c in chunker.evaluate_items(stream([document]), context)]
    assert second == [replace(c, cached=True) for c in first]
    assert model.calls == 1
    assert reports[-1].cached_chunks == 2
    changed = StructureAwareChunker(
        tokenizer=Utf8ByteTokenizer(),
        cache=cache,
        semantic_mode="llm",
        semantic_refiner=ChatBoundaryRefiner(
            llm=model, allow_external_transmission=True, identity="local:fake:v2"
        ),
    )
    third = [c async for c in changed.evaluate_items(stream([document]), context)]
    assert len(third) == 1 and model.calls == 2
    assert all(not c.cached for c in third)
    assert third[0].id != first[0].id


async def test_large_projected_snapshot_hydrates_each_bucket_once(cache, monkeypatch):
    """Projected siblings share one snapshot; still validate every persisted map."""
    from helpers import Characters

    doc = Document(
        "large-parent",
        "s",
        "# Guide\n\n" + "Useful documentation and examples.\n\n" * 120,
        metadata={"content_format": "markdown"},
    )

    chunker = StructureAwareChunker(tokenizer=Characters(), cache=cache)
    context = ExecutionContext("perf", Policy(on_error="raise"), cache=cache)
    chunks = [c async for c in chunker.transform(doc, context)]
    await cache.put_document_entries("large-parent-snapshot", [doc])
    await cache.put_document_entries("large-chunk-snapshot", chunks)
    snapshots = await cache.get_document_headers_many(
        ["large-parent-snapshot", "large-chunk-snapshot"]
    )
    parent = snapshots["large-parent-snapshot"][0]
    projected = sorted(
        snapshots["large-chunk-snapshot"], key=lambda c: c.metadata["chunk_index"]
    )
    reads = []
    original = cache.get_document_entries_many
    original_single = cache.get_document_entries

    async def record(keys):
        reads.extend(keys)
        return await original(keys)

    async def record_single(key):
        reads.append(key)
        return await original_single(key)

    monkeypatch.setattr(cache, "get_document_entries", record_single)
    monkeypatch.setattr(cache, "get_document_entries_many", record)
    await chunker.validate_output(parent, projected, context)
    assert projected == [replace(c, cached=True) for c in chunks]
    assert reads.count("large-chunk-snapshot") == 1
    assert reads.count("large-parent-snapshot") == 1

    # A changed snapshot cannot silently resolve an old projected identifier.
    import pytest

    from fastrag.exceptions import TransientCacheError

    await cache.put_document_entries("large-chunk-snapshot", chunks[1:])
    with pytest.raises(TransientCacheError, match="cache changed"):
        await chunker.validate_output(
            parent, list(snapshots["large-chunk-snapshot"]), context
        )
