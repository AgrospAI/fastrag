"""The :class:`Benchmark`: run a dataset through a pipeline, score every sample."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter

from fastrag.benchmark.datasets import EvalSample, QADataset
from fastrag.benchmark.metrics.base import Metric, QueryOutcome
from fastrag.benchmark.results import BenchmarkResult, SampleResult, StatusCounts
from fastrag.exceptions import ConfigurationError
from fastrag.query.engine import QueryEngine
from fastrag.retrieval.base import Retriever

BenchmarkTarget = QueryEngine | Retriever


@dataclass(kw_only=True)
class Benchmark:
    """Runs every sample in a dataset through a pipeline and scores it.

    .. code-block:: python

        from fastrag.benchmark import Benchmark, QADataset
        from fastrag.benchmark.metrics import Latency, NDCG, Recall

        benchmark = Benchmark(
            dataset=QADataset.load("eval.json"),
            metrics=[Recall(k=5), NDCG(k=5), Latency()],
            max_concurrent=4,
        )
        result = await benchmark.run(engine, name="recursive-500")

    Args:
        dataset: The samples to run.
        metrics: What to score each sample on. Every name must be unique — a
            duplicate would silently overwrite the first in
            ``BenchmarkResult.summary`` — and at least one metric's
            ``needs_relevant_ids`` must be satisfiable by the dataset.
        max_concurrent: How many samples run at once.
    """

    dataset: QADataset
    metrics: Sequence[Metric]
    max_concurrent: int = 4

    def __post_init__(self) -> None:
        """Validate that metrics are non-empty, unique, and satisfiable."""
        if not self.metrics:
            raise ConfigurationError("Benchmark.metrics cannot be empty")
        if self.max_concurrent < 1:
            raise ConfigurationError("Benchmark.max_concurrent must be at least 1")

        seen: set[str] = set()
        for metric in self.metrics:
            if metric.name in seen:
                raise ConfigurationError(
                    f"Benchmark.metrics has two metrics named {metric.name!r}; "
                    "the second would silently overwrite the first in "
                    "BenchmarkResult.summary"
                )
            seen.add(metric.name)

        needs_ids = [
            metric.name for metric in self.metrics if metric.needs_relevant_ids
        ]
        if needs_ids and not any(
            sample.relevant_ids for sample in self.dataset.samples
        ):
            raise ConfigurationError(
                f"Benchmark.metrics needs relevant_ids ({', '.join(needs_ids)}) "
                "but no sample in the dataset has any"
            )

    async def run(self, target: BenchmarkTarget, *, name: str) -> BenchmarkResult:
        """Run every sample against ``target``, scoring it with every metric.

        Samples run under a concurrency-limited semaphore, gathered back in
        dataset order. A sample whose pipeline call raises is recorded as
        ``"errored"`` with every score ``None`` — **no metric runs for it** —
        so a failure is never scored as a zero; see
        :class:`~fastrag.benchmark.results.StatusCounts`.

        Args:
            target: A :class:`~fastrag.query.engine.QueryEngine` for
                metrics that need a generated answer, or a bare
                :class:`~fastrag.retrieval.base.Retriever` for retrieval-only
                metrics — cheaper, and the signal to reach for first (see
                :mod:`fastrag.benchmark.metrics.retrieval`).
            name: This run's identity, carried on the returned result.

        Raises:
            ConfigurationError: ``target`` is a bare ``Retriever`` and a
                metric needs an answer. Raised before any sample runs, since
                it is knowable without running one.
        """
        if not isinstance(target, QueryEngine):
            needs_answer = [
                metric.name for metric in self.metrics if metric.needs_answer
            ]
            if needs_answer:
                raise ConfigurationError(
                    f"Benchmark.run: {', '.join(needs_answer)} need a generated "
                    "answer, but target is a bare Retriever, not a QueryEngine"
                )

        started = perf_counter()
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def run_one(sample: EvalSample) -> SampleResult:
            async with semaphore:
                return await self._run_sample(target, sample)

        samples = list(
            await asyncio.gather(*(run_one(sample) for sample in self.dataset.samples))
        )

        status = StatusCounts(
            successful=sum(1 for s in samples if s.status == "successful"),
            errored=sum(1 for s in samples if s.status == "errored"),
            incomplete=sum(1 for s in samples if s.status == "incomplete"),
        )
        summary = {
            metric.name: metric.aggregate([s.scores.get(metric.name) for s in samples])
            for metric in self.metrics
        }
        higher_is_better = {
            metric.name: metric.higher_is_better for metric in self.metrics
        }

        return BenchmarkResult(
            name=name,
            created_at=datetime.now(UTC),
            status=status,
            summary=summary,
            samples=samples,
            duration_s=perf_counter() - started,
            higher_is_better=higher_is_better,
        )

    async def _run_sample(
        self, target: BenchmarkTarget, sample: EvalSample
    ) -> SampleResult:
        started = perf_counter()
        try:
            if isinstance(target, QueryEngine):
                answer = await target.ask(sample.question)
                outcome = QueryOutcome(
                    answer=answer.text,
                    sources=answer.sources,
                    latency_ms=(perf_counter() - started) * 1000,
                )
            else:
                sources = await target.retrieve(sample.question)
                outcome = QueryOutcome(
                    answer="",
                    sources=sources,
                    latency_ms=(perf_counter() - started) * 1000,
                )
        except Exception as error:  # noqa: BLE001 - a failed sample must not abort the run
            return SampleResult(
                question=sample.question,
                status="errored",
                scores=dict.fromkeys(metric.name for metric in self.metrics),
                error=repr(error),
                latency_ms=(perf_counter() - started) * 1000,
            )

        scores = {
            metric.name: await metric.score(sample, outcome) for metric in self.metrics
        }
        status = (
            "incomplete"
            if any(score is None for score in scores.values())
            else "successful"
        )

        return SampleResult(
            question=sample.question,
            status=status,
            scores=scores,
            answer=outcome.answer or None,
            retrieved_ids=[document.id for document in outcome.sources if document.id],
            retrieved_uris=[
                document.metadata.get("source_uri", "")
                for document in outcome.sources
                if document.metadata.get("source_uri")
            ],
            latency_ms=outcome.latency_ms,
        )
