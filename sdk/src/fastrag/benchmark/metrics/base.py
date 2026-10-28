"""The :class:`Metric` contract, its input, and how scores aggregate.

Metrics consume the retrieval and query layers:
:class:`~fastrag.retrieval.base.Retriever` returns
:class:`~fastrag.models.Document` — there is no ``ScoredDocument`` any more —
so a metric reads a document's ``score``/``rank`` out of its ``metadata``
dict (stamped by ``VectorStore.search``) instead of off its ``.score``/
``.rank`` attributes.
"""

from __future__ import annotations

import math
import statistics
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

from fastrag.benchmark.datasets import EvalSample
from fastrag.models import Document

PERCENTILES = (50, 75, 90, 95, 99)


@dataclass(frozen=True)
class QueryOutcome:
    """What the pipeline produced for one sample — a metric's whole input.

    ``answer`` is ``""`` and ``input_tokens`` / ``output_tokens`` are ``None``
    when :meth:`~fastrag.benchmark.benchmark.Benchmark.run` targets a bare
    :class:`~fastrag.retrieval.base.Retriever` rather than a
    :class:`~fastrag.query.engine.QueryEngine`.
    """

    answer: str
    sources: list[Document]
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None


class Metric(ABC):
    """One measurement, computed per sample and aggregated over a run.

    A metric carries its own dependencies as class attributes, so
    :class:`~fastrag.benchmark.benchmark.Benchmark` can refuse an
    unsatisfiable request at construction rather than silently returning a
    number that means nothing.
    """

    needs_relevant_ids: bool = False
    """``True`` when :meth:`score` reads ``sample.relevant_ids``."""

    needs_answer: bool = False
    """``True`` when :meth:`score` reads ``outcome.answer`` — never satisfied
    by a bare :class:`~fastrag.retrieval.base.Retriever`."""

    higher_is_better: bool = True
    """Which direction improves. ``False`` for ``latency_ms``, ``total_tokens``,
    and ``cost_usd``."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Key this metric occupies in ``BenchmarkResult.summary``."""

    @abstractmethod
    async def score(self, sample: EvalSample, outcome: QueryOutcome) -> float | None:
        """This sample's score, or ``None`` when it is not measurable."""

    def aggregate(self, scores: Sequence[float | None]) -> DistributionSummary:
        """Aggregate this metric's per-sample scores. Override to change how."""
        return DistributionSummary.of(scores)


@dataclass(frozen=True)
class DistributionSummary:
    """A run's scores for one metric, never collapsed to a single mean.

    ``missing`` counts the ``None`` scores, dropped before ``mean``/``median``/
    ``std``/``min``/``max``/``percentiles`` are computed over what is left. An
    empty or all-``None`` input yields ``count=0`` and zeros everywhere, never
    a ``ZeroDivisionError``.
    """

    count: int
    missing: int
    mean: float
    median: float
    std: float
    min: float
    max: float
    percentiles: dict[str, float] = field(default_factory=dict)

    @classmethod
    def of(cls, scores: Sequence[float | None]) -> DistributionSummary:
        """Build a :class:`DistributionSummary` from raw per-sample scores."""
        present = sorted(score for score in scores if score is not None)
        missing = len(scores) - len(present)

        if not present:
            return cls(
                count=0,
                missing=missing,
                mean=0.0,
                median=0.0,
                std=0.0,
                min=0.0,
                max=0.0,
                percentiles={f"p{p}": 0.0 for p in PERCENTILES},
            )

        return cls(
            count=len(present),
            missing=missing,
            mean=statistics.fmean(present),
            median=statistics.median(present),
            std=statistics.pstdev(present),
            min=present[0],
            max=present[-1],
            percentiles={f"p{p}": _nearest_rank(present, p) for p in PERCENTILES},
        )


def _nearest_rank(sorted_scores: Sequence[float], percentile: int) -> float:
    """Nearest-rank percentile over an already-sorted, non-empty sequence.

    Exact, and defined for a single-element sequence — unlike
    :func:`statistics.quantiles`, which needs at least two points.
    """
    rank = math.ceil(percentile / 100 * len(sorted_scores))
    index = min(max(rank - 1, 0), len(sorted_scores) - 1)
    return sorted_scores[index]
