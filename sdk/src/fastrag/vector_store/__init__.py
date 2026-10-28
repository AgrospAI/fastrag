"""``fastrag.vector_store`` — the pipeline's storage backend, read and written.

Re-exports :class:`~fastrag.vector_store.base.VectorStore`, the sink-family
base every concrete vector store implements, and
:class:`~fastrag.vector_store.chroma.ChromaVectorStore`,
:class:`~fastrag.vector_store.milvus.MilvusVectorStore` and
:class:`~fastrag.vector_store.opensearch.OpenSearchVectorStore`. Milvus and
OpenSearch also implement :class:`~fastrag.lexical_store.base.LexicalStore`
for BM25 search; Chroma does not. Lives at the top level rather than under
:mod:`fastrag.ingestion`: a vector store is queried at read time as well as
written at ingest time, so it is not an ingestion-only concern.
"""

from __future__ import annotations

from fastrag.vector_store.base import VectorStore
from fastrag.vector_store.chroma import ChromaVectorStore
from fastrag.vector_store.milvus import MilvusVectorStore
from fastrag.vector_store.opensearch import OpenSearchVectorStore

__all__ = [
    "ChromaVectorStore",
    "MilvusVectorStore",
    "OpenSearchVectorStore",
    "VectorStore",
]
