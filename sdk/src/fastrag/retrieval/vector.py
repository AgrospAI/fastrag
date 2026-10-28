"""``VectorRetriever``: dense similarity search over a ``VectorStore``."""

from __future__ import annotations

from fastrag.ingestion.steps.embedding import Embedder
from fastrag.models import Document
from fastrag.retrieval.base import Retriever
from fastrag.vector_store import VectorStore


class VectorRetriever(Retriever):
    """Dense similarity search: the base retriever every other one over-fetches from.

    Holds its collaborators without owning their lifecycles: the caller opens
    ``vector_store`` (``async with MilvusVectorStore(...) as vector_store:``)
    and this retriever never opens or closes it. ``embedder`` is used only
    for its :meth:`~fastrag.ingestion.steps.embedding.base.Embedder.embed`
    method — never run as a pipeline step, never cached, never batched — so
    a single query embeds as one call.

    .. code-block:: python

        async with MilvusVectorStore(url=MILVUS_URI) as vector_store:
            retriever = VectorRetriever(
                vector_store=vector_store,
                embedder=Model2VecEmbedder(model=EMBEDDING_MODEL),
            )
            documents = await retriever.retrieve("What is AgrospAI?", k=5)

    Args:
        vector_store: The collection to search. Opened by the caller.
        embedder: Embeds the query text with the same model the collection
            was indexed with.
    """

    def __init__(self, *, vector_store: VectorStore, embedder: Embedder) -> None:
        """Store the collaborators; no I/O happens until :meth:`retrieve`."""
        self.vector_store = vector_store
        self.embedder = embedder

    async def retrieve(self, query: str, *, k: int = 5) -> list[Document]:
        """Embed ``query`` and return the ``k`` nearest documents, best first."""
        embedding = (await self.embedder.embed([query]))[0]
        return await self.vector_store.search(embedding, k=k)
