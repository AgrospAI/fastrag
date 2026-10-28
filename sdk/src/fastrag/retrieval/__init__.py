"""Retrieval: turning one query into the documents that answer it.

Retrieval is a first-class, composable layer — not an implementation detail
of prompting or chat. A retriever is fully specified as an object: ``k`` and
whatever else it needs are constructor configuration, so it can be passed
around, swapped, and benchmarked without its call sites knowing anything
about it. ``VectorRetriever`` is the base case; ``RerankingRetriever`` wraps
a vector search to narrow its over-fetch on relevance with a ``Reranker``
such as ``CohereReranker``. ``BM25Retriever`` and ``HybridRetriever`` add lexical retrieval and
replaceable rank fusion across any SDK retrievers. ``LanguageAwareRetriever``
keeps one translation of each page and prefers the question's language.
"""

from __future__ import annotations

from fastrag.retrieval.sections import SectionContext, SectionIndex
from fastrag.retrieval.base import Retriever
from fastrag.retrieval.bm25 import BM25Retriever
from fastrag.retrieval.fusion import RankFusion, ReciprocalRankFusion
from fastrag.retrieval.hybrid import HybridRetriever
from fastrag.retrieval.language import (
    LanguageAwareRetriever,
    LanguageDetector,
    LinguaLanguageDetector,
)
from fastrag.retrieval.reranking import CohereReranker, Reranker
from fastrag.retrieval.reranking_retriever import RerankingRetriever
from fastrag.retrieval.vector import VectorRetriever

__all__ = [
    "SectionContext",
    "SectionIndex",
    "BM25Retriever",
    "HybridRetriever",
    "LanguageAwareRetriever",
    "LanguageDetector",
    "LinguaLanguageDetector",
    "RankFusion",
    "ReciprocalRankFusion",
    "CohereReranker",
    "RerankingRetriever",
    "Reranker",
    "Retriever",
    "VectorRetriever",
]
