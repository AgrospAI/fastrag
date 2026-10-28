"""Reranking: scoring documents against a query."""

from __future__ import annotations

from fastrag.retrieval.reranking.base import Reranker
from fastrag.retrieval.reranking.cohere import CohereReranker

__all__ = ["CohereReranker", "Reranker"]
