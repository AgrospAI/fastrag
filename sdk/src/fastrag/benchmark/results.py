"""What one benchmark run produced, and how it is saved and reloaded."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastrag.benchmark.metrics.base import DistributionSummary
from fastrag.exceptions import ConfigurationError

FORMAT_VERSION = 1


@dataclass(frozen=True)
class StatusCounts:
    """How a run's samples broke down — never collapsed into one pass/fail count.

    A question that errored is never scored as zero: it is tallied in
    ``errored``, not folded into ``incomplete`` or ``successful``.
    """

    successful: int
    """Ran, and every metric produced a score."""

    errored: int
    """The pipeline raised."""

    incomplete: int
    """Ran, but at least one metric returned ``None``."""


@dataclass(frozen=True)
class SampleResult:
    """One sample's outcome: its scores, and enough detail to diagnose them."""

    question: str
    status: Literal["successful", "errored", "incomplete"]
    scores: dict[str, float | None]
    answer: str | None = None
    retrieved_ids: list[str] = field(default_factory=list)
    retrieved_uris: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    """The outcome of one :meth:`~fastrag.benchmark.benchmark.Benchmark.run`.

    ``name`` is the only identity a result carries — deliberately no serialized
    pipeline configuration, since the benchmark script is already the
    readable, version-controlled record of what was run. Keep the script and
    its ``results/`` directory together, and name results after their
    configuration.
    """

    name: str
    created_at: datetime
    status: StatusCounts
    summary: dict[str, DistributionSummary]
    samples: list[SampleResult]
    duration_s: float
    higher_is_better: dict[str, bool] = field(default_factory=dict)
    """Per metric name, whether a higher score is better. Read by
    :meth:`~fastrag.benchmark.comparison.Comparison.best`, which is the
    only place a result file's *meaning* — not just its numbers — is needed
    after the :class:`~fastrag.benchmark.metrics.base.Metric` objects that
    ran are gone."""

    def save(self, path: str | Path) -> None:
        """Write this result as JSON, with ``created_at`` as ISO-8601."""
        payload = {
            "v": FORMAT_VERSION,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "status": asdict(self.status),
            "summary": {name: asdict(dist) for name, dist in self.summary.items()},
            "samples": [asdict(sample) for sample in self.samples],
            "duration_s": self.duration_s,
            "higher_is_better": self.higher_is_better,
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkResult:
        """Read a result written by :meth:`save`.

        Raises:
            ConfigurationError: ``path`` is not a result file this version of
                the package can read.
        """
        payload: dict[str, Any] = json.loads(Path(path).read_text())
        if payload.get("v") != FORMAT_VERSION:
            raise ConfigurationError(
                f"{path}: unsupported benchmark result version {payload.get('v')!r}"
            )

        return cls(
            name=payload["name"],
            created_at=datetime.fromisoformat(payload["created_at"]),
            status=StatusCounts(**payload["status"]),
            summary={
                name: DistributionSummary(**dist)
                for name, dist in payload["summary"].items()
            },
            samples=[SampleResult(**sample) for sample in payload["samples"]],
            duration_s=payload["duration_s"],
            higher_is_better=payload.get("higher_is_better") or {},
        )
