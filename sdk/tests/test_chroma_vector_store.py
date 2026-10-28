"""ChromaVectorStore against an embedded on-disk Chroma, and the split BM25 contract."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from fastrag.exceptions import ConfigurationError, VectorStoreError
from fastrag.lexical_store import LexicalStore
from fastrag.models import Document
from fastrag.retrieval import BM25Retriever, HybridRetriever, VectorRetriever
from fastrag.vector_store import (
    ChromaVectorStore,
    MilvusVectorStore,
    OpenSearchVectorStore,
    VectorStore,
)

pytest.importorskip("chromadb")


def doc(id, embedding, **metadata):
    return Document(
        id=id,
        source_id=f"source-{id}",
        content=f"content {id}",
        embedding=embedding,
        metadata=metadata,
        step_config_hash="hash",
    )


async def test_chroma_index_search_and_filter(tmp_path):
    async with ChromaVectorStore(path=str(tmp_path)) as store:
        await store.ensure_ready(2)
        await store.ensure_ready(2)
        await store.upsert(
            [
                doc("a", [1.0, 0.0], source_uri="https://docs/a", nested={"x": [1]}),
                doc("b", [0.0, 1.0]),
            ]
        )
        assert await store.existing_ids(["a", "missing"]) == {"a"}

        hits = await store.search([1.0, 0.1], k=5)
        assert [d.id for d in hits] == ["a", "b"]
        assert [d.metadata["rank"] for d in hits] == [0, 1]
        assert hits[0].metadata["score"] < hits[1].metadata["score"]
        assert hits[0].metadata["nested"] == {"x": [1]}
        assert hits[0].metadata["source_uri"] == "https://docs/a"
        assert hits[0].source_id == "source-a"
        assert hits[0].content == "content a"
        assert hits[0].step_config_hash == "hash"
        assert hits[1].metadata["source_uri"] == "source-b"

        filtered = await store.search(
            [1.0, 0.1], k=5, filter=json.dumps({"source_id": "source-b"})
        )
        assert [d.id for d in filtered] == ["b"]

        with pytest.raises(ConfigurationError, match="dimension"):
            await store.ensure_ready(3)
        with pytest.raises(ConfigurationError):
            await store.search([1.0, 0.0], k=0)
        with pytest.raises(ConfigurationError, match="where"):
            await store.search([1.0, 0.0], k=1, filter="[]")

    # Reopening sees the persisted collection, and rejects a different metric.
    async with ChromaVectorStore(path=str(tmp_path)) as store:
        assert await store.existing_ids(["a", "b"]) == {"a", "b"}
    async with ChromaVectorStore(path=str(tmp_path), metric_type="l2") as store:
        with pytest.raises(ConfigurationError, match="metric"):
            await store.ensure_ready(2)


async def test_chroma_unindexed_collection_and_closed_store(tmp_path):
    async with ChromaVectorStore(path=str(tmp_path), collection="other") as store:
        with pytest.raises(VectorStoreError, match="get_collection"):
            await store.search([1.0, 0.0], k=1)
    with pytest.raises(VectorStoreError, match="not open"):
        await store.existing_ids(["a"])


def test_chroma_configuration_validation():
    with pytest.raises(ConfigurationError, match="exactly one"):
        ChromaVectorStore()
    with pytest.raises(ConfigurationError, match="exactly one"):
        ChromaVectorStore(url="http://localhost:8000", path="db")
    with pytest.raises(ConfigurationError, match="HTTP"):
        ChromaVectorStore(url="grpc://localhost:8000")
    with pytest.raises(ConfigurationError, match="metric_type"):
        ChromaVectorStore(path="db", metric_type="dot")


def test_lexical_contract_is_independent_of_vector_store():
    assert not issubclass(LexicalStore, VectorStore)
    assert not issubclass(ChromaVectorStore, LexicalStore)
    assert issubclass(MilvusVectorStore, LexicalStore)
    assert issubclass(OpenSearchVectorStore, LexicalStore)


async def test_hybrid_over_separate_vector_and_lexical_stores(tmp_path):
    class KeywordStore(LexicalStore):
        def __init__(self):
            self.calls = []

        async def lexical_search(self, query, *, k, filter=None):
            self.calls.append((query, k, filter))
            return [doc("b", None, score=3.0, rank=0)][:k]

    lexical = KeywordStore()
    async with ChromaVectorStore(path=str(tmp_path)) as store:
        await store.ensure_ready(2)
        await store.upsert([doc("a", [1.0, 0.0]), doc("b", [0.0, 1.0])])
        hybrid = HybridRetriever(
            retrievers=[
                BM25Retriever(store=lexical, filter="source"),
                VectorRetriever(
                    vector_store=store,
                    embedder=MagicMock(embed=AsyncMock(return_value=[[1.0, 0.0]])),
                ),
            ],
            candidate_k=3,
        )
        result = await hybrid.retrieve("question", k=2)
    assert [d.id for d in result] == ["b", "a"]
    assert lexical.calls == [("question", 3, "source")]
