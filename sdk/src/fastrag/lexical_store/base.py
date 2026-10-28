"""The ``LexicalStore`` contract: BM25 search over stored document text.

Deliberately independent of :class:`~fastrag.vector_store.base.VectorStore`.
Lexical and dense retrieval are separate capabilities, so neither forces the
other on a backend:

- A vector database with server-side BM25 inherits both
  (``class MilvusVectorStore(VectorStore, LexicalStore)``) and one open store
  answers both branches of a
  :class:`~fastrag.retrieval.hybrid.HybridRetriever`.
- A vector database without it implements only ``VectorStore``; BM25 then
  comes from a different backend implementing only ``LexicalStore`` — dense
  vectors in Milvus, say, and lexical ranking in PostgreSQL.

:class:`~fastrag.retrieval.bm25.BM25Retriever` depends on this contract only,
never on a backend client.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from fastrag.models import Document


class LexicalStore(ABC):
    """Storage able to rank its documents' text against a query with BM25.

    Opened and closed by the caller, like a vector store — a retriever holds
    an already-open one and never manages its lifecycle.
    """

    @abstractmethod
    async def lexical_search(
        self, query: str, *, k: int, filter: str | None = None
    ) -> list[Document]:
        """Return at most ``k`` BM25 matches for ``query``, best first.

        Stamps the same reserved metadata keys as
        :meth:`~fastrag.vector_store.base.VectorStore.search`: ``score`` (the
        backend's own BM25 score, never comparable across backends) and
        ``rank`` (0-based position in the returned list).

        Args:
            query: Natural-language query text, analyzed server-side.
            k: Maximum number of documents to return.
            filter: Backend-native filter expression, applied server-side.
                ``None`` searches unfiltered.
        """
        ...
