"""Lexical retrieval through the SDK's ``LexicalStore`` contract."""

from fastrag.exceptions import ConfigurationError
from fastrag.lexical_store import LexicalStore
from fastrag.models import Document
from fastrag.retrieval.base import Retriever


class BM25Retriever(Retriever):
    """Rank query terms with the store's BM25 implementation.

    ``store`` is any :class:`~fastrag.lexical_store.base.LexicalStore`: the
    same open vector store a sibling ``VectorRetriever`` searches, when the
    backend supports both, or a separate lexical database when it does not.
    """

    def __init__(self, *, store: LexicalStore, filter: str | None = None) -> None:
        """Keep the already-open store and optional native filter."""
        self.store = store
        self.filter = filter

    async def retrieve(self, query: str, *, k: int = 5) -> list[Document]:
        """Return at most ``k`` lexical matches."""
        if k < 1:
            raise ConfigurationError("retrieval k must be at least 1")
        return await self.store.lexical_search(query, k=k, filter=self.filter)
