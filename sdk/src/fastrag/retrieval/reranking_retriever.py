"""``RerankingRetriever``: narrowing any retrieval strategy with a reranker."""

from __future__ import annotations

from fastrag.exceptions import ConfigurationError
from fastrag.ingestion.steps.embedding import Embedder
from fastrag.models import Document
from fastrag.retrieval.base import Retriever
from fastrag.retrieval.reranking.base import Reranker
from fastrag.retrieval.vector import VectorRetriever
from fastrag.vector_store import VectorStore


class RerankingRetriever(Retriever):
    """Over-fetch with a retriever, then narrow on relevance.

    Composition, not configuration: a plain :class:`~fastrag.retrieval.vector.VectorRetriever`
    over-fetches ``k`` candidates on cosine similarity, and ``reranker`` then
    narrows them to ``top_k`` on relevance — a cross-encoder reading query
    and document together, which similarity search alone cannot do. The
    returned ``score``/``rank`` are the reranker's own, not the vector
    search's — see :meth:`~fastrag.retrieval.reranking.base.Reranker.rerank`.

    .. code-block:: python

        async with MilvusVectorStore(url=MILVUS_URI) as vector_store:
            retriever = RerankingRetriever(
                vector_store=vector_store,
                embedder=Model2VecEmbedder(model=EMBEDDING_MODEL),
                reranker=CohereReranker(
                    model="rerank-v3.5", base_url=RERANK_BASE_URL, api_key=api_key
                ),
            )
            documents = await retriever.retrieve("What is AgrospAI?", k=200, top_k=5)

    Args:
        vector_store: The collection to over-fetch from. Opened by the
            caller; this retriever neither opens nor closes it.
        embedder: Embeds the query text — see
            :class:`~fastrag.retrieval.vector.VectorRetriever`.
        retriever: Any SDK retriever, including a hybrid retriever. Mutually
            exclusive with the legacy vector_store/embedder pair.
        candidate_k: Optional over-fetch count used even when the caller
            requests only the final k results.
        reranker: Scores the over-fetched documents against the query.
    """

    def __init__(
        self,
        *,
        reranker: Reranker,
        retriever: Retriever | None = None,
        vector_store: VectorStore | None = None,
        embedder: Embedder | None = None,
        candidate_k: int | None = None,
    ) -> None:
        """Store the collaborators; no I/O happens until :meth:`retrieve`."""
        if retriever is not None:
            if vector_store is not None or embedder is not None:
                raise ConfigurationError(
                    "pass retriever or vector_store/embedder, not both"
                )
        else:
            if vector_store is None or embedder is None:
                raise ConfigurationError(
                    "provide retriever or both vector_store and embedder"
                )
            retriever = VectorRetriever(vector_store=vector_store, embedder=embedder)
        if candidate_k is not None and candidate_k < 1:
            raise ConfigurationError("candidate_k must be at least 1")
        self.retriever = retriever
        self.candidate_k = candidate_k
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker

    async def retrieve(
        self, query: str, *, k: int = 5, top_k: int | None = None
    ) -> list[Document]:
        """Fetch candidates through the wrapped retriever, then rerank to ``top_k``.

        Args:
            query: Natural-language query text.
            k: How many documents the vector store over-fetches.
            top_k: How many documents the reranker keeps, best first.
                Defaults to ``k`` when omitted.
        """
        if k < 1 or (top_k is not None and top_k < 1):
            raise ConfigurationError("k and top_k must be at least 1")
        hits = await self.retriever.retrieve(query, k=max(k, self.candidate_k or k))
        if not hits:
            return []
        return await self.reranker.rerank(
            query, hits, top_k=k if top_k is None else top_k
        )
