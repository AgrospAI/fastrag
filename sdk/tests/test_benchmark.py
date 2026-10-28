"""Unit coverage for ``fastrag.benchmark``: metric arithmetic, dataset loading,
comparison, and one end-to-end ``Benchmark.run`` -- no network.

The retrieval metrics are exercised as pure units over hand-built
:class:`~fastrag.models.Document` lists whose ``metadata`` carries the
``score``/``rank`` keys ``VectorStore.search`` stamps -- exactly the shape a
real retriever returns, with the numbers the assertions need, and no vector
store or network involved. The judge-backed generation metrics
(``Faithfulness``/``AnswerRelevancy``/``Correctness``) use a stub
:class:`~fastrag.query.llm.base.ChatModel`, so no live LLM endpoint is ever
called. The end-to-end ``Benchmark.run`` test uses a stub
:class:`~fastrag.retrieval.base.Retriever` and the same stub
:class:`~fastrag.query.llm.base.ChatModel`.
"""

from __future__ import annotations

import csv
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from fastrag.benchmark import Benchmark, ColumnMapping, Comparison, QADataset
from fastrag.benchmark.datasets import EvalSample
from fastrag.benchmark.metrics import (
    MRR,
    NDCG,
    Cost,
    Faithfulness,
    HitRate,
    Latency,
    Precision,
    Recall,
    TokenOverlap,
    TokenUsage,
)
from fastrag.benchmark.metrics.base import DistributionSummary, QueryOutcome
from fastrag.benchmark.metrics.generation import AnswerRelevancy, Correctness
from fastrag.benchmark.results import BenchmarkResult
from fastrag.exceptions import ConfigurationError
from fastrag.models import Document
from fastrag.query import QueryEngine
from fastrag.query.llm import ChatModel, Message
from fastrag.retrieval import Retriever

#: Page-level ground-truth URI the retrieval-metric units target below.
RELEVANT_URI = "https://e/relevant"
OTHER_URI = "https://e/other"


def _doc(doc_id: str, uri: str, *, rank: int) -> Document:
    """A document stamped the way ``VectorStore.search`` stamps retrieved ones."""
    return Document(
        id=doc_id,
        source_id="s",
        content=f"content of {doc_id}",
        metadata={"source_uri": uri, "score": float(10 - rank), "rank": rank},
    )


def _outcome(
    documents: list[Document], *, answer: str = "the answer", **extra
) -> QueryOutcome:
    """A :class:`QueryOutcome` over a best-first document list."""
    return QueryOutcome(answer=answer, sources=documents, latency_ms=0.0, **extra)


def _sample(*relevant_ids: str, expected_answers: Sequence[str] = ()) -> EvalSample:
    """A minimal :class:`EvalSample` with the given ``relevant_ids``."""
    return EvalSample(
        question="q",
        expected_answers=list(expected_answers),
        relevant_ids=list(relevant_ids),
    )


async def test_retrieval_metrics_score_by_rank_metadata() -> None:
    """``Recall``/``Precision``/``MRR``/``NDCG``/``HitRate`` all read
    ``metadata["rank"]`` over the ``source_uri`` ground-truth key, and every
    ``RetrievalMetric`` is unmeasurable (``None``, not ``0.0``) without any
    ``relevant_ids`` at all.
    """
    two_doc_set = [
        _doc("a", RELEVANT_URI, rank=0),
        _doc("b", RELEVANT_URI, rank=1),
        _doc("c", OTHER_URI, rank=2),
    ]
    assert (
        await Recall(k=5, key="source_uri").score(
            _sample(RELEVANT_URI), _outcome(two_doc_set)
        )
        == 1.0
    )

    ranked = [_doc("a", OTHER_URI, rank=0), _doc("b", RELEVANT_URI, rank=1)]
    assert (
        await Precision(k=5, key="source_uri").score(
            _sample(RELEVANT_URI), _outcome(ranked)
        )
        == 0.5
    )
    assert (
        await MRR(k=5, key="source_uri").score(_sample(RELEVANT_URI), _outcome(ranked))
        == 0.5
    )

    assert (
        await NDCG(k=1, key="source_uri").score(
            _sample(RELEVANT_URI, "https://e/also-relevant"),
            _outcome([_doc("a", RELEVANT_URI, rank=0)]),
        )
        == 1.0
    )

    hit_rate = HitRate(k=5, key="source_uri")
    assert (
        await hit_rate.score(
            _sample(RELEVANT_URI), _outcome([_doc("a", OTHER_URI, rank=0)])
        )
        == 0.0
    )
    assert (
        await hit_rate.score(
            _sample(RELEVANT_URI), _outcome([_doc("b", RELEVANT_URI, rank=0)])
        )
        == 1.0
    )

    assert await Recall(k=5, key="source_uri").score(_sample(), _outcome([])) is None


async def test_token_overlap_is_best_f1_with_the_answer() -> None:
    """Token overlap is the max F1 across expected answers, computed on words.

    A cheap smoke-test baseline, never accuracy: it rewards word overlap,
    not correctness.
    """
    score = await TokenOverlap().score(
        _sample(expected_answers=["the quick fox"]),
        _outcome([_doc("a", OTHER_URI, rank=0)], answer="the quick brown fox"),
    )

    assert score is not None
    assert 0.0 < score <= 1.0


async def test_operational_metrics_and_their_distribution_summary() -> None:
    """``Latency``/``TokenUsage``/``Cost`` -- never ``None`` for latency, but
    ``None`` whenever token counts are unreported (a bare
    :class:`~fastrag.retrieval.base.Retriever` never generates) -- and
    ``DistributionSummary.of`` aggregating a mix of real and missing scores.
    """
    assert Latency().higher_is_better is False
    assert (
        await Latency().score(
            _sample(), QueryOutcome(answer="", sources=[], latency_ms=12.5)
        )
        == 12.5
    )

    assert TokenUsage().higher_is_better is False
    assert await TokenUsage().score(_sample(), _outcome([])) is None
    assert (
        await TokenUsage().score(
            _sample(), _outcome([], input_tokens=100, output_tokens=50)
        )
        == 150.0
    )

    cost = Cost(input_price=1.0, output_price=2.0)
    assert cost.higher_is_better is False
    assert await cost.score(_sample(), _outcome([])) is None
    priced = await cost.score(
        _sample(), _outcome([], input_tokens=1_000_000, output_tokens=500_000)
    )
    assert priced == 1.0 * 1_000_000 / 1_000_000 + 2.0 * 500_000 / 1_000_000

    summary = DistributionSummary.of([1.0, 2.0, 3.0, None])
    assert summary.count == 3
    assert summary.missing == 1
    assert summary.mean == 2.0
    assert summary.min == 1.0
    assert summary.max == 3.0
    assert summary.percentiles["p50"] == 2.0

    empty_summary = DistributionSummary.of([None, None])
    assert empty_summary.count == 0
    assert empty_summary.missing == 2
    assert empty_summary.mean == 0.0


class _Judge(ChatModel):
    """Returns a fixed reply, so a ``JudgeMetric`` needs no network."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.received: list[Message] | None = None

    @property
    def model_id(self) -> str:
        return "stub-judge"

    async def invoke(self, messages: Sequence[Message]) -> str:
        self.received = list(messages)
        return self.reply

    async def stream(self, messages: Sequence[Message]) -> AsyncIterator[str]:
        del messages
        yield ""


async def test_judge_backed_generation_metrics_parse_and_refuse_bad_replies() -> None:
    """``Faithfulness``/``AnswerRelevancy``/``Correctness`` all ask the judge
    and parse its numeric reply through the same ``JudgeMetric`` machinery;
    an unparseable reply scores ``None``, never ``0.0`` -- a judge that
    answers "Sure! The score is high" has not judged the sample unfaithful,
    and recording that as zero would misrepresent a parsing failure as a
    measurement.
    """
    good_judge = _Judge("0.8")
    outcome = _outcome([_doc("a", OTHER_URI, rank=0)], answer="some answer")

    faithfulness = await Faithfulness(judge=good_judge).score(_sample(), outcome)
    assert faithfulness == 0.8
    assert good_judge.received is not None
    assert any(msg.role == "system" for msg in good_judge.received)

    assert await AnswerRelevancy(judge=good_judge).score(_sample(), outcome) == 0.8
    assert (
        await Correctness(judge=good_judge).score(
            _sample(expected_answers=["some answer"]), outcome
        )
        == 0.8
    )
    # No expected answers: Correctness is unmeasurable, not a guess.
    assert await Correctness(judge=good_judge).score(_sample(), outcome) is None

    bad_judge = _Judge("Sure! That looks great to me.")
    assert await Faithfulness(judge=bad_judge).score(_sample(), outcome) is None


def test_dataset_loading_auto_detects_and_column_mapping_overrides(
    tmp_path: Path,
) -> None:
    """``QADataset.from_records``/``.load`` auto-detect aliased column names
    (``answer`` for ``expected_answers``), a :class:`ColumnMapping` overrides
    auto-detection for a name that is not one of the built-in aliases, and a
    record with no resolvable question column raises ``ConfigurationError``
    rather than silently dropping it.
    """
    auto = QADataset.from_records(
        [{"question": "q1", "answer": "a1", "relevant_ids": ["r1"]}]
    )
    assert len(auto.samples) == 1
    assert auto.samples[0].question == "q1"
    assert auto.samples[0].expected_answers == ["a1"]
    assert auto.samples[0].relevant_ids == ["r1"]

    csv_path = tmp_path / "eval.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["my_question", "my_answer"])
        writer.writeheader()
        writer.writerow({"my_question": "q2", "my_answer": "a2"})

    mapped = QADataset.load(
        csv_path, ColumnMapping(question="my_question", expected_answers="my_answer")
    )
    assert len(mapped.samples) == 1
    assert mapped.samples[0].question == "q2"
    assert mapped.samples[0].expected_answers == ["a2"]

    try:
        QADataset.from_records([{"unrecognized_column": "x"}])
    except ConfigurationError:
        pass
    else:
        raise AssertionError(
            "expected ConfigurationError for a missing question column"
        )


async def test_benchmark_run_over_a_two_sample_dataset_and_comparison_ranking() -> None:
    """``Benchmark.run`` end to end: stubs produce scores and a savable
    result, whose ``samples``/``status`` carry per-sample detail; two such
    results feed ``Comparison.of``/``.metrics``/``.to_markdown``/``.best``.
    """
    dataset = QADataset.from_records(
        [
            {"question": "q1", "relevant_ids": [RELEVANT_URI]},
            {"question": "q2", "relevant_ids": [RELEVANT_URI]},
        ]
    )
    judge = _Judge("1.0")
    retriever = _StubRetriever(
        [_doc("a", RELEVANT_URI, rank=0), _doc("b", RELEVANT_URI, rank=1)]
    )
    engine = QueryEngine(retriever=retriever, llm=judge)

    strong = Benchmark(
        dataset=dataset,
        metrics=[Recall(k=5, key="source_uri"), Faithfulness(judge=judge)],
        max_concurrent=2,
    )
    strong_result = await strong.run(engine, name="strong-run")

    assert isinstance(strong_result, BenchmarkResult)
    assert strong_result.status.successful == 2
    assert strong_result.status.errored == 0
    assert "recall@5" in strong_result.summary
    assert strong_result.summary["recall@5"].mean == 1.0
    assert len(strong_result.samples) == 2
    assert strong_result.samples[0].scores["recall@5"] == 1.0

    weak_judge = _Judge("0.0")
    weak = Benchmark(
        dataset=dataset,
        metrics=[Recall(k=5, key="source_uri"), Faithfulness(judge=weak_judge)],
        max_concurrent=2,
    )
    weak_result = await weak.run(
        QueryEngine(retriever=retriever, llm=weak_judge), name="weak-run"
    )

    comparison = Comparison.of([strong_result, weak_result])
    assert set(comparison.metrics()) == {"recall@5", "faithfulness"}
    table = comparison.to_markdown()
    assert "strong-run" in table
    assert "weak-run" in table
    assert comparison.best("faithfulness").name == "strong-run"


class _StubRetriever(Retriever):
    """Returns documents in the fixed order it was handed, ignoring ``query``."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    async def retrieve(self, query: str, *, k: int = 5) -> list[Document]:
        del query
        return self.documents[:k]
