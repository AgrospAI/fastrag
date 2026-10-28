"""The ``Reranker`` contract: scoring documents against a query."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from fastrag.models import Document


class Reranker(ABC):
    """Scores documents against a query with a cross-encoder.

    A cross-encoder reads the query and a document together, rather than
    comparing precomputed embeddings, so it is far more accurate than vector
    similarity — and far more expensive, which is why it narrows a
    retriever's over-fetch instead of searching a whole collection; see
    :class:`~fastrag.retrieval.reranking_retriever.RerankingRetriever`.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Stable identifier of the model, for logging and events."""

    @abstractmethod
    async def rerank(
        self, query: str, documents: Sequence[Document], *, top_k: int
    ) -> list[Document]:
        """Score ``documents`` against ``query`` and return them best-first.

        Results are best-first, number at most ``top_k``, and each carries
        the reserved metadata keys ``score`` (the model's own relevance
        score, on the model's own scale — never comparable with a vector
        store's similarity score) and ``rank`` (0-based, contiguous) restated
        from this reranker — any ``score``/``rank`` a document already
        carried is overwritten, not merged. An empty ``documents`` returns
        ``[]`` without a network call.

        Args:
            query: The same query text the documents were retrieved for.
            documents: Candidates to score, already retrieved by some other
                means (typically a :class:`~fastrag.retrieval.base.Retriever`'s
                over-fetch).
            top_k: Maximum number of documents to return, best first.
        """
