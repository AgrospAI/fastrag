"""``fastrag.lexical_store`` — the read contract for BM25 relevance search.

Re-exports :class:`~fastrag.lexical_store.base.LexicalStore`. It is a
capability, not a storage family: a vector store that can also rank text
(:class:`~fastrag.vector_store.milvus.MilvusVectorStore`,
:class:`~fastrag.vector_store.opensearch.OpenSearchVectorStore`) implements it
alongside :class:`~fastrag.vector_store.base.VectorStore`, while a vector
store that cannot (:class:`~fastrag.vector_store.chroma.ChromaVectorStore`)
simply does not, and lexical retrieval is served from a separate database
instead.
"""

from __future__ import annotations

from fastrag.lexical_store.base import LexicalStore

__all__ = ["LexicalStore"]
