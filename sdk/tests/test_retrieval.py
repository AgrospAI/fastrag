"""End-to-end coverage for ``fastrag.vector_store``/``fastrag.retrieval``:
indexing, search, ranking, and reranking.

``MilvusVectorStore.save`` is driven through a real ``Pipeline``, over every
embedded chunk the fixture site produces. Search then runs over the fixture
site's own isolated collection, indexed once per test through the same real
``Pipeline`` + ``MilvusVectorStore.save`` path. ``CohereReranker`` needs a
live network endpoint and is deliberately not driven here -- per this
suite's constraint against a network call, it is exercised only by
``examples/11_reranking.py``. ``RerankingRetriever`` is instead driven by
``_ReversingReranker`` below, a local stub proving ``top_k`` and the
reordering are actually the reranker's, not just the vector search's order
passed through unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from fastrag.ingestion import Pipeline
from fastrag.ingestion.cache import PostgresCache
from fastrag.ingestion.events import DocumentSaved
from fastrag.ingestion.steps.chunking import RecursiveCharacterTextSplitter
from fastrag.ingestion.steps.embedding import Model2VecEmbedder
from fastrag.ingestion.steps.fetching import SitemapCrawler
from fastrag.ingestion.steps.parsing import (
    DefaultParser,
    DocusaurusParser,
    HtmlParser,
    ParserRouter,
    docusaurus_filter,
    html_filter,
)
from fastrag.models import Document
from fastrag.retrieval import Reranker, RerankingRetriever, VectorRetriever
from fastrag.vector_store import MilvusVectorStore

CHUNK_SIZE = 120
CHUNK_OVERLAP = 20
QUERY = "AgrospAI"


async def _index(
    sitemap_url: str,
    cache: PostgresCache,
    vector_store: MilvusVectorStore,
    embedder: Model2VecEmbedder,
) -> list[Document]:
    crawler = SitemapCrawler(url=sitemap_url)
    parser = (
        ParserRouter()
        .route(html_filter & docusaurus_filter, DocusaurusParser())
        .route(html_filter, HtmlParser())
        .default(DefaultParser())(crawler)
    )
    chunker = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )(parser)
    embedded = embedder(chunker)
    saved_step = vector_store.save(embedded)
    documents: list[Document] = []
    async with Pipeline(saved_step, cache=cache) as pipeline:
        async for event in pipeline:
            if isinstance(event, DocumentSaved):
                documents.append(event.document)
    return documents


async def test_vector_store_save_upserts_every_document_and_caches_on_rerun(
    http_server: str,
    cache: PostgresCache,
    vector_store: MilvusVectorStore,
    embedder: Model2VecEmbedder,
) -> None:
    """Every embedded chunk lands in the isolated collection exactly once.

    A rerun against the same collection reports every document as
    ``cached`` (the ``existing_ids`` path) and leaves the collection's row
    count unchanged -- checked through
    :meth:`~fastrag.vector_store.base.VectorStore.existing_ids`, the store's
    own public account of what it holds, rather than a private Milvus
    row-count call: since this collection is isolated and nothing else ever
    writes to it, ``existing_ids`` returning exactly the expected id set
    *is* the row count.
    """
    sitemap_url = f"{http_server}/sitemap.xml"

    first = await _index(sitemap_url, cache, vector_store, embedder)
    assert first
    first_ids = {document.id for document in first}
    assert all(not document.cached for document in first)

    present = await vector_store.existing_ids(list(first_ids))
    assert present == first_ids

    second = await _index(sitemap_url, cache, vector_store, embedder)
    second_ids = {document.id for document in second}
    assert second_ids == first_ids
    assert all(document.cached for document in second)

    present_after_rerun = await vector_store.existing_ids(list(first_ids))
    assert present_after_rerun == first_ids


async def test_vector_retriever_ranks_by_descending_score_and_handles_overflow(
    http_server: str,
    cache: PostgresCache,
    vector_store: MilvusVectorStore,
    embedder: Model2VecEmbedder,
) -> None:
    """``VectorRetriever`` returns ``k`` best-first documents with consistent metadata.

    Also covers a ``k`` larger than the collection: it must return every
    document, not raise.
    """
    ids = {
        document.id
        for document in await _index(
            f"{http_server}/sitemap.xml", cache, vector_store, embedder
        )
    }
    assert len(ids) > 5, "the fixture site should chunk into more than 5 documents"

    retriever = VectorRetriever(vector_store=vector_store, embedder=embedder)

    documents = await retriever.retrieve(QUERY, k=3)
    assert len(documents) == 3
    assert all(document.id in ids for document in documents)

    scores = [document.metadata["score"] for document in documents]
    assert all(isinstance(score, float) for score in scores)
    assert scores == sorted(scores, reverse=True), (
        "documents must come back best-first: descending score"
    )
    for rank, document in enumerate(documents):
        assert document.metadata["rank"] == rank
    assert all("source_uri" in document.metadata for document in documents)

    overflowing = await retriever.retrieve(QUERY, k=len(ids) + 50)
    assert len(overflowing) == len(ids), (
        "a k larger than the collection must return everything, not error"
    )
    assert {document.id for document in overflowing} == ids


class _ReversingReranker(Reranker):
    """A stub :class:`Reranker` that reverses whatever order it is handed.

    Proves :class:`~fastrag.retrieval.reranking_retriever.RerankingRetriever`
    actually returns *this* reranker's order and ``top_k`` -- not the vector
    search's order passed straight through -- without a network call.
    """

    @property
    def model_id(self) -> str:
        return "reversing-stub"

    async def rerank(
        self, query: str, documents: Sequence[Document], *, top_k: int
    ) -> list[Document]:
        del query
        reordered = list(reversed(documents))[:top_k]
        total = len(reordered)
        return [
            replace(
                document,
                metadata={
                    **document.metadata,
                    "score": float(total - rank),
                    "rank": rank,
                },
            )
            for rank, document in enumerate(reordered)
        ]


async def test_reranking_retriever_honours_top_k_and_reranker_order(
    http_server: str,
    cache: PostgresCache,
    vector_store: MilvusVectorStore,
    embedder: Model2VecEmbedder,
) -> None:
    """``RerankingRetriever`` over-fetches ``k`` then narrows to ``top_k`` via the reranker."""
    ids = {
        document.id
        for document in await _index(
            f"{http_server}/sitemap.xml", cache, vector_store, embedder
        )
    }
    # A k comfortably larger than the collection, like the overflow case
    # above, so approximate-index recall does not make this flaky: both
    # calls below must see the exact same full, deterministically ordered
    # set to compare the reranker's order against.
    overfetch_k = len(ids) + 50

    over_fetch = await VectorRetriever(
        vector_store=vector_store, embedder=embedder
    ).retrieve(QUERY, k=overfetch_k)
    assert len(over_fetch) == len(ids)

    top_k = 3
    retriever = RerankingRetriever(
        vector_store=vector_store, embedder=embedder, reranker=_ReversingReranker()
    )
    documents = await retriever.retrieve(QUERY, k=overfetch_k, top_k=top_k)

    assert len(documents) == top_k
    expected_ids = [document.id for document in reversed(over_fetch)][:top_k]
    assert [document.id for document in documents] == expected_ids, (
        "must reflect the reranker's own reordering, not the vector search's"
    )
    for rank, document in enumerate(documents):
        assert document.metadata["rank"] == rank
    scores = [document.metadata["score"] for document in documents]
    assert scores == sorted(scores, reverse=True)

    # top_k defaults to k when omitted.
    defaulted = await retriever.retrieve(QUERY, k=5)
    assert len(defaulted) == 5
