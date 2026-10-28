"""LangchainSemanticChunker delegates to LangChain with an SDK Embedder."""

from helpers import FixedEmbedder, doc

from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext
from fastrag.ingestion.steps.chunking import LangchainSemanticChunker


async def test_cuts_at_topic_change_through_sdk_embedder():
    source = doc("Apples grow. Pears grow. Stars shine. Planets orbit.")
    emb = FixedEmbedder([[1, 0], [1, 0], [0, 1], [0, 1]])
    chunker = LangchainSemanticChunker(
        embeddings=emb, buffer_size=0, add_start_index=True, batch_size=3
    )
    context = ExecutionContext("test", Policy())
    chunks = [c async for c in chunker.transform(source, context)]
    assert [c.content for c in chunks] == [
        "Apples grow. Pears grow.",
        "Stars shine. Planets orbit.",
    ]
    assert [c.metadata["start_index"] for c in chunks] == [0, 25]
    assert [c.metadata["chunk_index"] for c in chunks] == [0, 1]
    assert emb.calls == [
        ["Apples grow.", "Pears grow.", "Stars shine."],
        ["Planets orbit."],
    ]


async def test_empty_document_yields_nothing():
    emb = FixedEmbedder([])
    chunker = LangchainSemanticChunker(embeddings=emb)
    assert [p async for p in chunker.asplit(doc("   "))] == []
    assert emb.calls == []
