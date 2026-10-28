"""Pluggable ranking fusion strategies."""

from fastrag.retrieval.fusion.base import RankFusion
from fastrag.retrieval.fusion.rrf import ReciprocalRankFusion

__all__ = ["RankFusion", "ReciprocalRankFusion"]
