"""Generation metrics: a free bag-of-words baseline, and three judge-backed scores."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from fastrag.benchmark.datasets import EvalSample
from fastrag.benchmark.metrics.base import Metric, QueryOutcome
from fastrag.query.llm.base import ChatModel, Message

_WORD = re.compile(r"\w+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


class TokenOverlap(Metric):
    """Bag-of-words F1 between the answer and the best-matching expected answer.

    A cheap smoke-test baseline, **never accuracy**: it rewards word overlap,
    not correctness, and a fluent answer that restates the question can score
    well while being wrong. Reach for the retrieval metrics or a judge-backed
    metric when the number needs to mean something.
    """

    name = "token_overlap"
    needs_answer = True

    async def score(self, sample: EvalSample, outcome: QueryOutcome) -> float | None:
        """Best F1 across expected answers, or ``None`` when there are none."""
        if not sample.expected_answers:
            return None

        answer_words = {word.lower() for word in _WORD.findall(outcome.answer)}
        best = 0.0
        for expected in sample.expected_answers:
            expected_words = {word.lower() for word in _WORD.findall(expected)}
            if not expected_words or not answer_words:
                continue
            overlap = len(answer_words & expected_words)
            precision = overlap / len(answer_words)
            recall = overlap / len(expected_words)
            if precision + recall:
                best = max(best, 2 * precision * recall / (precision + recall))
        return best


@dataclass(frozen=True, kw_only=True)
class JudgeMetric(Metric):
    """Shared machinery for a metric scored by asking a chat model to grade.

    Args:
        judge: The chat model asked to grade each sample.
    """

    judge: ChatModel

    needs_answer = True

    async def _judge(self, messages: Sequence[Message]) -> float | None:
        """Call the judge and parse a bare ``[0, 1]`` number out of its reply.

        An unparseable reply — or one with a number outside ``[0, 1]`` — scores
        ``None``, never ``0.0``: a judge that answers "Sure! The score is
        high" has not judged the sample unfaithful, and recording that as
        zero would misrepresent a parsing failure as a measurement.

        The generation metrics are LLM-backed — the same explicit opt-in the
        old package used (only a ``Benchmark.run`` whose ``target`` is a
        :class:`~fastrag.query.engine.QueryEngine` ever invokes a
        ``JudgeMetric``). A bare
        :class:`~fastrag.retrieval.base.Retriever` as ``target`` refuses to
        run this metric before any sample does.
        """
        content = await self.judge.invoke(messages)
        return _parse_score(content)


@dataclass(frozen=True, kw_only=True)
class Faithfulness(JudgeMetric):
    """How well the answer is supported by its retrieved sources, per the judge."""

    name = "faithfulness"

    async def score(self, sample: EvalSample, outcome: QueryOutcome) -> float | None:
        """Ask the judge to score how well the answer is grounded in its sources."""
        context = "\n\n".join(document.content for document in outcome.sources)
        messages = [
            Message(
                role="system",
                content=(
                    "You judge whether an answer is faithful to — fully "
                    "supported by — the given context. Respond with only a "
                    "number between 0 and 1, where 1 means every claim is "
                    "supported by the context and 0 means none of it is."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Context:\n{context}\n\nAnswer:\n{outcome.answer}\n\n"
                    "Faithfulness score (0-1):"
                ),
            ),
        ]
        return await self._judge(messages)


@dataclass(frozen=True, kw_only=True)
class AnswerRelevancy(JudgeMetric):
    """How relevant the answer is to the question asked, per the judge."""

    name = "answer_relevancy"

    async def score(self, sample: EvalSample, outcome: QueryOutcome) -> float | None:
        """Ask the judge to score how relevant the answer is to the question."""
        messages = [
            Message(
                role="system",
                content=(
                    "You judge how relevant an answer is to the question "
                    "asked. Respond with only a number between 0 and 1, "
                    "where 1 means fully relevant and 0 means not relevant "
                    "at all."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Question:\n{sample.question}\n\nAnswer:\n{outcome.answer}\n\n"
                    "Relevancy score (0-1):"
                ),
            ),
        ]
        return await self._judge(messages)


@dataclass(frozen=True, kw_only=True)
class Correctness(JudgeMetric):
    """How correct the answer is against the sample's expected answers, per the judge."""

    name = "correctness"

    async def score(self, sample: EvalSample, outcome: QueryOutcome) -> float | None:
        """Ask the judge to score correctness, or ``None`` with no expected answers."""
        if not sample.expected_answers:
            return None

        expected = "\n".join(sample.expected_answers)
        messages = [
            Message(
                role="system",
                content=(
                    "You judge how correct a generated answer is compared to "
                    "one or more reference answers. Respond with only a "
                    "number between 0 and 1, where 1 means fully correct and "
                    "0 means completely incorrect."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Question:\n{sample.question}\n\n"
                    f"Reference answer(s):\n{expected}\n\n"
                    f"Generated answer:\n{outcome.answer}\n\n"
                    "Correctness score (0-1):"
                ),
            ),
        ]
        return await self._judge(messages)


def _parse_score(content: str) -> float | None:
    match = _NUMBER.search(content)
    if match is None:
        return None
    value = float(match.group())
    return value if 0.0 <= value <= 1.0 else None
