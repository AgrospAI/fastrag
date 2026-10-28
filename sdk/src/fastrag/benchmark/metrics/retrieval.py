"""Retrieval metrics: deterministic, free, fast — reach for these first."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from fastrag.benchmark.datasets import EvalSample
from fastrag.benchmark.metrics.base import Metric, QueryOutcome
from fastrag.models import Document


@dataclass(frozen=True, kw_only=True)
class RetrievalMetric(Metric):
    """Shared machinery for the five ranking metrics below.

    Args:
        k: How many of the retrieved documents to consider, best first.
        key: Which field identifies a document for comparison against
            ``sample.relevant_ids``. ``"id"`` is the chunk id
            (``sha256(source_uri, chunk_index, content)``), which changes
            with every chunker configuration. ``"source_uri"`` compares at
            the page level, which survives a chunk-size sweep — pass it
            whenever the ground truth was authored per page rather than per
            chunk.
    """

    k: int = 5
    key: Literal["id", "source_uri"] = "id"

    needs_relevant_ids = True

    def _retrieved(self, outcome: QueryOutcome) -> list[str]:
        """Top-``k`` keys, best first, deduplicated, missing keys dropped.

        Truncated to ``k`` *before* mapping to keys and deduping: five chunks
        of one page under ``key="source_uri"`` are one retrieved item, and
        deduping before truncating would let a sixth page climb into a window
        it never ranked in.

        A document's key is read from its ``metadata`` — the retrieval layer
        stamps ``rank`` and ``source_uri`` there — so ordering relies on the
        list being best-first (which ``Retriever.retrieve`` guarantees) and
        position is read from ``metadata["rank"]``.
        """
        top = self._ranked(outcome.sources)[: self.k]
        keys = (
            getattr(document, self.key)
            if self.key == "id"
            else document.metadata.get(self.key)
            for document in top
        )
        seen: set[str] = set()
        retrieved: list[str] = []
        for key in keys:
            if key is None or key in seen:
                continue
            seen.add(key)
            retrieved.append(key)
        return retrieved

    @staticmethod
    def _ranked(sources: list[Document]) -> list[Document]:
        """Reorders ``sources`` by ``metadata["rank"]`` when a ``rank`` is stamped.

        ``Retriever.retrieve`` guarantees a best-first list, so the fallback
        (no ``rank`` stamp — e.g. a hand-built stub in a unit test) keeps the
        given order. When ranks are present but a list is out of order, the
        explicit ranks win.
        """
        if any("rank" in document.metadata for document in sources):
            ranked = sorted(
                enumerate(sources), key=lambda pair: pair[1].metadata.get("rank", 0)
            )
            return [document for _, document in ranked]
        return list(sources)


@dataclass(frozen=True, kw_only=True)
class Recall(RetrievalMetric):
    """``|relevant ∩ retrieved| / |relevant|``."""

    @property
    def name(self) -> str:
        """The summary key, e.g. ``"recall@5"``."""
        return f"recall@{self.k}"

    async def score(self, sample: EvalSample, outcome: QueryOutcome) -> float | None:
        """Recall at ``k``, or ``None`` when the sample has no relevant ids."""
        relevant = set(sample.relevant_ids)
        if not relevant:
            return None
        retrieved = set(self._retrieved(outcome))
        return len(relevant & retrieved) / len(relevant)


@dataclass(frozen=True, kw_only=True)
class Precision(RetrievalMetric):
    """``|relevant ∩ retrieved| / len(retrieved)``."""

    @property
    def name(self) -> str:
        """The summary key, e.g. ``"precision@5"``."""
        return f"precision@{self.k}"

    async def score(self, sample: EvalSample, outcome: QueryOutcome) -> float | None:
        """Precision at ``k``, or ``None`` when the sample has no relevant ids."""
        relevant = set(sample.relevant_ids)
        if not relevant:
            return None
        retrieved = self._retrieved(outcome)
        if not retrieved:
            return 0.0
        return len(relevant & set(retrieved)) / len(retrieved)


@dataclass(frozen=True, kw_only=True)
class MRR(RetrievalMetric):
    """``1 / (position of the first relevant)``, else ``0.0``."""

    @property
    def name(self) -> str:
        """The summary key, e.g. ``"mrr@5"``."""
        return f"mrr@{self.k}"

    async def score(self, sample: EvalSample, outcome: QueryOutcome) -> float | None:
        """Reciprocal rank of the first relevant hit, or ``None`` with no relevant ids."""
        relevant = set(sample.relevant_ids)
        if not relevant:
            return None
        for position, key in enumerate(self._retrieved(outcome), start=1):
            if key in relevant:
                return 1.0 / position
        return 0.0


@dataclass(frozen=True, kw_only=True)
class NDCG(RetrievalMetric):
    """Binary-gain NDCG, with the ideal ranking over ``min(len(relevant), k)``."""

    @property
    def name(self) -> str:
        """The summary key, e.g. ``"ndcg@5"``."""
        return f"ndcg@{self.k}"

    async def score(self, sample: EvalSample, outcome: QueryOutcome) -> float | None:
        """Binary-gain NDCG at ``k``, or ``None`` when the sample has no relevant ids."""
        relevant = set(sample.relevant_ids)
        if not relevant:
            return None
        retrieved = self._retrieved(outcome)
        dcg = sum(
            1.0 / math.log2(position + 2)
            for position, key in enumerate(retrieved)
            if key in relevant
        )
        ideal_hits = min(len(relevant), self.k)
        idcg = sum(1.0 / math.log2(position + 2) for position in range(ideal_hits))
        return dcg / idcg if idcg else 0.0


@dataclass(frozen=True, kw_only=True)
class HitRate(RetrievalMetric):
    """``1.0`` if any relevant document is in the window, else ``0.0``."""

    @property
    def name(self) -> str:
        """The summary key, e.g. ``"hit_rate@5"``."""
        return f"hit_rate@{self.k}"

    async def score(self, sample: EvalSample, outcome: QueryOutcome) -> float | None:
        """1.0 if any relevant document is in the window, else 0.0."""
        relevant = set(sample.relevant_ids)
        if not relevant:
            return None
        retrieved = set(self._retrieved(outcome))
        return 1.0 if relevant & retrieved else 0.0
