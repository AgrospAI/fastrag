"""Operational metrics: cost of running the pipeline, not quality of its output.

``TokenUsage`` and ``Cost`` return ``None`` for every sample against a bare
:class:`~fastrag.retrieval.base.Retriever` or a provider that reports no
usage — "not measurable for this sample"
(:meth:`~fastrag.benchmark.metrics.base.Metric.score`), not a zero, and
it is why a retrieval-only benchmark should not list them: every sample
would come back ``incomplete``.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastrag.benchmark.datasets import EvalSample
from fastrag.benchmark.metrics.base import Metric, QueryOutcome


class Latency(Metric):
    """Wall-clock time for one sample, in milliseconds.

    Never ``None``: :class:`~fastrag.benchmark.benchmark.Benchmark` times
    every sample itself, so this metric works against a bare
    :class:`~fastrag.retrieval.base.Retriever` too, with no chat model
    involved.
    """

    name = "latency_ms"
    higher_is_better = False

    async def score(self, sample: EvalSample, outcome: QueryOutcome) -> float | None:
        """Return the sample's wall-clock latency in milliseconds."""
        return outcome.latency_ms


class TokenUsage(Metric):
    """Input plus output tokens for one sample.

    ``None`` when either count is missing — a bare
    :class:`~fastrag.retrieval.base.Retriever` never generates, and
    :meth:`~fastrag.query.engine.QueryEngine.ask` does not report usage (in
    the new query layer ``ChatModel.invoke`` returns plain text, not a
    ``Completion`` carrying token counts).
    """

    name = "total_tokens"
    higher_is_better = False

    async def score(self, sample: EvalSample, outcome: QueryOutcome) -> float | None:
        """Total tokens, or ``None`` when the provider reported no usage."""
        if outcome.input_tokens is None or outcome.output_tokens is None:
            return None
        return float(outcome.input_tokens + outcome.output_tokens)


@dataclass(frozen=True, kw_only=True)
class Cost(Metric):
    """Dollar cost of one sample's generation call.

    Args:
        input_price: Price per 1,000,000 input tokens, in USD.
        output_price: Price per 1,000,000 output tokens, in USD.
    """

    input_price: float
    output_price: float

    name = "cost_usd"
    higher_is_better = False

    async def score(self, sample: EvalSample, outcome: QueryOutcome) -> float | None:
        """Dollar cost from token counts, or ``None`` when usage is unreported."""
        if outcome.input_tokens is None or outcome.output_tokens is None:
            return None
        return (
            outcome.input_tokens * self.input_price
            + outcome.output_tokens * self.output_price
        ) / 1_000_000
