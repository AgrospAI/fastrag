"""Real isolated PostgreSQL semantic boundary caching with fake embeddings."""

from dataclasses import replace

from helpers import FixedEmbedder, doc, stream

from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext
from fastrag.ingestion.steps.chunking import ChunkerRouter, EmbeddingSemanticChunker


async def test_semantic_cache_replay_and_provider_identity(cache):
    vectors = [[1, 0], [1, 0], [0, 1], [0, 1]]
    emb = FixedEmbedder(vectors)
    chunker = EmbeddingSemanticChunker(
        embeddings=emb, cache=cache, add_start_index=True
    )
    context = ExecutionContext("semantic-cache", Policy(on_error="raise"), cache=cache)
    source = doc("  Apples grow. Pears grow.\nStars shine. Planets orbit.  ")
    first = [c async for c in chunker.evaluate_items(stream(source), context)]
    second = [c async for c in chunker.evaluate_items(stream(source), context)]
    assert first and second == [replace(c, cached=True) for c in first]
    assert len(emb.calls) == 1
    assert "".join(c.content for c in second) == source.content
    changed_emb = FixedEmbedder(vectors, identity="different settings")
    changed = EmbeddingSemanticChunker(
        embeddings=changed_emb, cache=cache, add_start_index=True
    )
    other = [c async for c in changed.evaluate_items(stream(source), context)]
    assert len(changed_emb.calls) == 1
    assert all(not c.cached for c in other)
    assert changed.config_hash != chunker.config_hash
    assert [c.content for c in other] == [c.content for c in first]


async def test_routed_semantic_cache_keeps_source_order(cache):
    emb = FixedEmbedder([[1, 0], [1, 0], [0, 1], [0, 1]])
    child = EmbeddingSemanticChunker(embeddings=emb, cache=cache)
    router = ChunkerRouter(cache=cache).default(child)
    context = ExecutionContext("semantic-router", Policy(on_error="raise"), cache=cache)
    source = doc()
    first = [c async for c in router.evaluate_items(stream(source), context)]
    second = [c async for c in router.evaluate_items(stream(source), context)]
    assert second == [replace(c, cached=True) for c in first]
    assert "".join(c.content for c in second) == source.content
    assert len(emb.calls) == 1
