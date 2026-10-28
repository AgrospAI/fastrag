"""Reciprocal Rank Fusion, independent of storage or embedding providers."""

from collections.abc import Sequence
from dataclasses import replace
from math import isfinite

from fastrag.exceptions import ConfigurationError
from fastrag.models import Document
from fastrag.retrieval.fusion.base import RankFusion


class ReciprocalRankFusion(RankFusion):
    """Sum ``1 / (rank_constant + one_based_rank)`` across input rankings.

    Each document contributes once per ranking. Input order breaks score ties
    deterministically. Original document metadata is copied, never mutated.
    """

    def __init__(self, *, rank_constant: float = 60.0) -> None:
        """Configure the positive rank smoothing constant."""
        if not isfinite(rank_constant) or rank_constant <= 0:
            raise ConfigurationError("RRF rank_constant must be finite and positive")
        self.rank_constant = rank_constant

    def fuse(self, rankings: Sequence[Sequence[Document]], *, k: int) -> list[Document]:
        """Deduplicate by document id and return descending fusion scores."""
        if k < 1:
            raise ConfigurationError("fusion k must be at least 1")
        documents: dict[str, Document] = {}
        scores: dict[str, float] = {}
        for ranking in rankings:
            seen: set[str] = set()
            for rank, document in enumerate(ranking, start=1):
                if document.id in seen:
                    continue
                seen.add(document.id)
                documents.setdefault(document.id, document)
                scores[document.id] = scores.get(document.id, 0.0) + 1 / (
                    self.rank_constant + rank
                )
        ids = sorted(scores, key=lambda document_id: scores[document_id], reverse=True)[
            :k
        ]
        return [
            replace(
                documents[document_id],
                metadata={
                    **documents[document_id].metadata,
                    "score": scores[document_id],
                    "rank": rank,
                },
            )
            for rank, document_id in enumerate(ids)
        ]
