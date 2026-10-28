"""Evaluation datasets: a list of question/answer/relevant-id samples."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fastrag.exceptions import ConfigurationError

#: Canonical name first, so the native format needs no column mapping at all.
_ALIASES: dict[str, tuple[str, ...]] = {
    "question": ("question", "query", "input", "prompt", "instruction"),
    "expected_answers": (
        "expected_answers",
        "answer",
        "answers",
        "ground_truth",
        "reference",
        "output",
    ),
    "relevant_ids": (
        "relevant_ids",
        "contexts",
        "context_ids",
        "relevant_docs",
        "ground_truth_ids",
    ),
}

_SUPPORTED_SUFFIXES = (".json", ".jsonl", ".csv")
_DEFERRED_SUFFIXES = (".yaml", ".yml", ".parquet")


@dataclass
class EvalSample:
    """One question, with what a metric needs to score an answer to it.

    ``relevant_ids`` is what enables the retrieval metrics
    (:mod:`fastrag.benchmark.metrics.retrieval`) — a sample with none is
    still usable for generation and operational metrics, just not those.
    """

    question: str
    expected_answers: list[str] = field(default_factory=list)
    relevant_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ColumnMapping:
    """Override auto-detection for one or more of :class:`QADataset`'s columns.

    Each field left ``None`` is auto-detected by scanning its alias list
    against the loaded records' keys; passing a value overrides only that
    field. Auto-detection raises :class:`~fastrag.exceptions.ConfigurationError`
    when two alias columns are both present, rather than silently picking one.

    ``relevant_ids``'s alias list includes ``contexts`` for compatibility with
    RAGAS-style eval sets — but a RAGAS ``contexts`` column holds retrieved
    *text*, not ids. Such a column matches nothing in Fastrag's chunk ids and
    scores every retrieval metric ``0.0``: pass
    ``ColumnMapping(relevant_ids=...)`` pointing at an actual id column, or
    leave ``relevant_ids`` empty and skip the retrieval metrics for that
    dataset.
    """

    question: str | None = None
    expected_answers: str | None = None
    relevant_ids: str | None = None


@dataclass
class QADataset:
    """A list of :class:`EvalSample`, loadable from several formats.

    .. code-block:: python

        from fastrag.benchmark import ColumnMapping, QADataset

        QADataset.load("eval.csv")  # auto-detected
        QADataset.load("eval.csv", ColumnMapping(question="user_query"))
    """

    samples: list[EvalSample]

    @classmethod
    def load(cls, path: str | Path, columns: ColumnMapping | None = None) -> QADataset:
        """Load a dataset, dispatching on ``path``'s suffix.

        ``.json`` accepts either ``{"samples": [...]}`` (the native format,
        :meth:`save`'s output) or a bare ``[...]`` of records. ``.jsonl`` reads
        one record per line. ``.csv`` reads with :class:`csv.DictReader`.

        ``.yaml``/``.parquet`` and :meth:`from_huggingface` need the
        ``datasets`` extra — not yet implemented, kept behind the same gate
        this class already applies to its other deferred formats.

        Raises:
            ConfigurationError: ``path``'s suffix is ``.yaml`` or ``.parquet``
                (needs the ``datasets`` extra, not yet implemented), or is
                anything else this method does not read.
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".json":
            records = _read_json(path)
        elif suffix == ".jsonl":
            records = _read_jsonl(path)
        elif suffix == ".csv":
            records = _read_csv(path)
        elif suffix in _DEFERRED_SUFFIXES:
            raise ConfigurationError(
                f"QADataset.load({str(path)!r}) needs the 'datasets' extra: "
                'pip install "fastrag-ai[datasets]"'
            )
        else:
            raise ConfigurationError(
                f"QADataset.load: unsupported suffix {suffix!r}; supported: "
                f"{', '.join(_SUPPORTED_SUFFIXES)}"
            )

        return cls(samples=_build_samples(records, columns))

    @classmethod
    def from_huggingface(cls, path: str) -> QADataset:
        """Load a dataset from Hugging Face Hub.

        Not yet implemented — gated by the ``datasets`` extra, the same way
        the deferred ``.yaml``/``.parquet`` suffixes already are.

        Raises:
            ConfigurationError: Always — the ``datasets`` extra is not yet
                wired up, so this raises naming the install command rather
                than pretending a loading path exists.
        """
        raise ConfigurationError(
            f"QADataset.from_huggingface({path!r}) needs the 'datasets' extra: "
            'pip install "fastrag-ai[datasets]"'
        )

    @classmethod
    def from_records(cls, records: Iterable[dict[str, Any]]) -> QADataset:
        """Build a dataset from records already in memory, auto-detecting columns."""
        return cls(samples=_build_samples(records, None))

    def save(self, path: str | Path) -> None:
        """Write this dataset in the native ``{"samples": [...]}`` format."""
        payload = {"samples": [asdict(sample) for sample in self.samples]}
        Path(path).write_text(json.dumps(payload, indent=2))


def _build_samples(
    records: Iterable[dict[str, Any]], columns: ColumnMapping | None
) -> list[EvalSample]:
    """Alias-detect columns once, then map every record through them."""
    records = list(records)
    columns = columns or ColumnMapping()
    keys = {key for record in records for key in record}

    question_col = _detect("question", keys, columns.question)
    answers_col = _detect("expected_answers", keys, columns.expected_answers)
    ids_col = _detect("relevant_ids", keys, columns.relevant_ids)

    samples = []
    for record in records:
        question = record.get(question_col) if question_col else None
        if question is None:
            raise ConfigurationError(
                "QADataset: no question column found for a record; pass "
                "ColumnMapping(question=...) to name it explicitly"
            )
        metadata = record.get("metadata")
        samples.append(
            EvalSample(
                question=str(question),
                expected_answers=_as_list(
                    record.get(answers_col) if answers_col else None
                ),
                relevant_ids=_as_list(record.get(ids_col) if ids_col else None),
                metadata=metadata if isinstance(metadata, dict) else {},
            )
        )
    return samples


def _detect(field_name: str, keys: set[str], override: str | None) -> str | None:
    if override is not None:
        return override
    candidates = [alias for alias in _ALIASES[field_name] if alias in keys]
    if len(candidates) > 1:
        raise ConfigurationError(
            f"QADataset: ambiguous column for {field_name!r} — both "
            f"{candidates[0]!r} and {candidates[1]!r} are present; pass "
            f"ColumnMapping({field_name}=...) to disambiguate"
        )
    return candidates[0] if candidates else None


def _as_list(value: Any) -> list[str]:
    """Wrap a bare string into a single-element list; pass a list through."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _read_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        if "samples" not in data:
            raise ConfigurationError(
                f"{path}: a JSON object must have a top-level 'samples' key"
            )
        return data["samples"]
    if isinstance(data, list):
        return data
    raise ConfigurationError(f"{path}: expected a JSON object or a bare array")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped:
            records.append(json.loads(stripped))
    return records


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))
