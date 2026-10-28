"""Hybrid retrieval and backend HTTP/schema contract regressions."""

import json
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from fastrag.exceptions import (
    ConfigurationError,
    TransientVectorStoreError,
    VectorStoreError,
)
from fastrag.models import Document
from fastrag.retrieval import (
    BM25Retriever,
    HybridRetriever,
    ReciprocalRankFusion,
    RerankingRetriever,
    Retriever,
    VectorRetriever,
)
from fastrag.retrieval.reranking import Reranker
from fastrag.vector_store import MilvusVectorStore, OpenSearchVectorStore


def doc(id, **metadata):
    return Document(
        id=id,
        source_id="source",
        content=f"content {id}",
        embedding=[1.0, 0.5],
        metadata=metadata,
    )


def test_rrf_scores_deduplication_and_metadata():
    a, b, c = doc("a", score=1000, rank=99, source_uri="url"), doc("b"), doc("c")
    result = ReciprocalRankFusion().fuse([[a, b, b], [b, c]], k=3)
    assert [d.id for d in result] == ["b", "a", "c"]
    assert result[0].metadata["score"] == pytest.approx(1 / 62 + 1 / 61)
    assert result[1].metadata["source_uri"] == "url"
    assert [d.metadata["rank"] for d in result] == [0, 1, 2]
    assert a.metadata == {"score": 1000, "rank": 99, "source_uri": "url"}
    assert ReciprocalRankFusion().fuse([[], []], k=5) == []


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_rrf_invalid_constant(value):
    with pytest.raises(ConfigurationError):
        ReciprocalRankFusion(rank_constant=value)


class StaticRetriever(Retriever):
    def __init__(self, documents):
        self.documents = documents
        self.requested = []

    async def retrieve(self, query, *, k=5):
        self.requested.append((query, k))
        return self.documents[:k]


class ReverseReranker(Reranker):
    @property
    def model_id(self):
        return "reverse"

    async def rerank(self, query, documents, *, top_k):
        return [
            replace(d, metadata={**d.metadata, "rank": rank})
            for rank, d in enumerate(reversed(documents))
        ][:top_k]


async def test_hybrid_reranking_and_legacy_constructor():
    lexical, dense = (
        StaticRetriever([doc("a"), doc("b")]),
        StaticRetriever([doc("b"), doc("c")]),
    )
    hybrid = HybridRetriever(retrievers=[lexical, dense], candidate_k=40)
    wrapped = RerankingRetriever(
        retriever=hybrid, reranker=ReverseReranker(), candidate_k=20
    )
    assert [d.id for d in await wrapped.retrieve("question", k=2)] == ["c", "a"]
    assert lexical.requested == dense.requested == [("question", 40)]
    store = MagicMock(search=AsyncMock(return_value=[doc("a"), doc("b")]))
    embedder = MagicMock(embed=AsyncMock(return_value=[[1.0, 0.5]]))
    legacy = RerankingRetriever(
        vector_store=store, embedder=embedder, reranker=ReverseReranker()
    )
    assert [d.id for d in await legacy.retrieve("q", k=20, top_k=1)] == ["b"]
    store.search.assert_awaited_once_with([1.0, 0.5], k=20)


async def test_opensearch_index_upsert_and_hybrid_queries(monkeypatch):
    real_client = httpx.AsyncClient
    state, mappings, requests = {}, {}, []

    def handle(request):
        requests.append(request)
        path = request.url.path
        if path == "/corpus/_mapping":
            return (
                httpx.Response(200, json={"corpus": {"mappings": mappings}})
                if mappings
                else httpx.Response(404)
            )
        if path == "/corpus" and request.method == "PUT":
            body = json.loads(request.content)
            assert body["settings"]["index"]["knn"]
            mappings.update(body["mappings"])
            return httpx.Response(200, json={"acknowledged": True})
        if path == "/_bulk":
            assert request.url.params["refresh"] == "wait_for"
            lines = request.content.decode().splitlines()
            for offset in range(0, len(lines), 2):
                action = json.loads(lines[offset])
                assert action["index"]["_index"] == "corpus"
                state[action["index"]["_id"]] = json.loads(lines[offset + 1])
            return httpx.Response(200, json={"errors": False})
        if path == "/corpus/_mget":
            return httpx.Response(
                200,
                json={
                    "docs": [
                        {"_id": id, "found": id in state}
                        for id in json.loads(request.content)["ids"]
                    ]
                },
            )
        if path == "/corpus/_search":
            body = json.loads(request.content)
            query = body["query"]
            if "knn" in query:
                assert query["knn"]["vector"]["vector"] == [1.0, 0.5]
                ids = ["b", "a"]
            else:
                assert query["match"]["content"] == "question"
                ids = ["a", "b"]
            return httpx.Response(
                200,
                json={
                    "hits": {
                        "hits": [
                            {"_id": id, "_score": 2 - rank, "_source": state[id]}
                            for rank, id in enumerate(ids[: body["size"]])
                        ]
                    }
                },
            )
        pytest.fail(f"unexpected request {request.method} {path}")

    monkeypatch.setattr(
        "fastrag.vector_store.opensearch.httpx.AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    async with OpenSearchVectorStore(
        url="https://search.example", index="corpus", user="user", password="secret"
    ) as store:
        await store.ensure_ready(2)
        await store.ensure_ready(2)
        assert len([r for r in requests if r.method == "PUT"]) == 1
        await store.upsert([doc("a", source_uri="https://docs/a"), doc("b")])
        assert await store.existing_ids(["a", "missing"]) == {"a"}
        hybrid = HybridRetriever(
            retrievers=[
                BM25Retriever(store=store),
                VectorRetriever(
                    vector_store=store,
                    embedder=MagicMock(embed=AsyncMock(return_value=[[1.0, 0.5]])),
                ),
            ]
        )
        result = await hybrid.retrieve("question", k=2)
        assert [d.id for d in result] == ["a", "b"]
        assert result[0].source_id == "source"
        assert result[0].metadata["source_uri"] == "https://docs/a"
        with pytest.raises(ConfigurationError):
            await store.ensure_ready(3)
    assert store._client is None
    with pytest.raises(VectorStoreError, match="not open"):
        await store.lexical_search("q", k=2)


async def use_transport(store, handler):
    await store._client.aclose()
    store._client = httpx.AsyncClient(
        base_url=store.url, transport=httpx.MockTransport(handler)
    )


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, VectorStoreError),
        (429, TransientVectorStoreError),
        (503, TransientVectorStoreError),
    ],
)
async def test_opensearch_error_translation(status, expected):
    async with OpenSearchVectorStore(url="https://search.example") as store:
        await use_transport(store, lambda request: httpx.Response(status))
        with pytest.raises(expected, match=str(status)):
            await store.lexical_search("q", k=2)


async def test_opensearch_bulk_partial_failure():
    async with OpenSearchVectorStore(url="https://search.example") as store:
        await use_transport(
            store,
            lambda request: httpx.Response(
                200,
                json={
                    "errors": True,
                    "items": [
                        {
                            "index": {
                                "status": 400,
                                "error": {"type": "mapper_parsing_exception"},
                            }
                        }
                    ],
                },
            ),
        )
        with pytest.raises(VectorStoreError, match="1 documents"):
            await store.upsert([doc("a")])


async def test_opensearch_native_filters_and_validation():
    bodies = []

    def handle(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"hits": {"hits": []}})

    async with OpenSearchVectorStore(url="https://search.example") as store:
        await use_transport(store, handle)
        filter = {"term": {"source_id": "source"}}
        await store.search([1.0, 0.5], k=2, filter=json.dumps(filter))
        await store.lexical_search("q", k=2, filter=json.dumps(filter))
        assert bodies[0]["query"]["knn"]["vector"]["filter"] == filter
        assert bodies[1]["query"]["bool"]["filter"] == [filter]
        with pytest.raises(ConfigurationError):
            await store.lexical_search("q", k=0)
        with pytest.raises(ConfigurationError):
            await store.search([1.0, 0.5], k=2, filter='source_id == "source"')


async def test_milvus_bm25_schema_and_branches():
    store = MilvusVectorStore(url="http://localhost:19530", enable_bm25=True)
    schema, indexes = MagicMock(), MagicMock()
    store._client = MagicMock(
        has_collection=AsyncMock(return_value=False),
        create_schema=MagicMock(return_value=schema),
        prepare_index_params=MagicMock(return_value=indexes),
        create_collection=AsyncMock(),
        load_collection=AsyncMock(),
        search=AsyncMock(
            return_value=[
                [
                    {
                        "id": "a",
                        "distance": 1.5,
                        "entity": {
                            "content": "match",
                            "metadata": {"title": "Title"},
                            "source_uri": "url",
                        },
                    }
                ]
            ]
        ),
    )
    await store.ensure_ready(2)
    schema.add_function.assert_called_once()
    assert any(
        c.kwargs.get("field_name") == "content" and c.kwargs.get("enable_analyzer")
        for c in schema.add_field.call_args_list
    )
    assert any(
        c.kwargs.get("metric_type") == "BM25" for c in indexes.add_index.call_args_list
    )
    hits = await store.lexical_search("match", k=5)
    assert hits[0].metadata == {
        "title": "Title",
        "source_uri": "url",
        "score": 1.5,
        "rank": 0,
    }
    assert store._client.search.call_args.kwargs["anns_field"] == "sparse"
    assert store._client.search.call_args.kwargs["data"] == ["match"]
    await store.search([1.0, 0.5], k=5)
    assert store._client.search.call_args.kwargs["anns_field"] == "vector"
    store.enable_bm25 = False
    with pytest.raises(ConfigurationError, match="enable_bm25"):
        await store.lexical_search("match", k=5)


async def test_milvus_rejects_old_dense_schema():
    store = MilvusVectorStore(url="http://localhost:19530", enable_bm25=True)
    store._client = MagicMock(
        has_collection=AsyncMock(return_value=True),
        describe_collection=AsyncMock(
            return_value={"fields": [{"name": "vector", "params": {"dim": 2}}]}
        ),
    )
    with pytest.raises(ConfigurationError, match="reindex"):
        await store.ensure_ready(2)
