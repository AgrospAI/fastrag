"""The :class:`Comparison`: any number of runs, side by side."""

from __future__ import annotations

import glob
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from fastrag.benchmark.metrics.base import DistributionSummary
from fastrag.benchmark.results import BenchmarkResult
from fastrag.exceptions import ConfigurationError

_DIST_FIELDS = frozenset({"count", "missing", "mean", "median", "std", "min", "max"})


@dataclass(frozen=True)
class Comparison:
    """Any number of :class:`~fastrag.benchmark.results.BenchmarkResult`, compared.

    .. code-block:: python

        from fastrag.benchmark import Comparison

        comparison = Comparison.load("results/*.json")  # or Comparison.of([r1, r2, r3])
        print(comparison.to_markdown())
        print(comparison.best("ndcg@5"))
    """

    results: list[BenchmarkResult]

    @classmethod
    def of(cls, results: Iterable[BenchmarkResult]) -> Comparison:
        """Build a comparison from results already in memory."""
        return cls(results=list(results))

    @classmethod
    def load(cls, pattern: str | Path) -> Comparison:
        """Load every result file matching a glob ``pattern``, sorted by path.

        Raises:
            ConfigurationError: ``pattern`` matched no files — an empty
                comparison is the silent-wrong-answer shape, so this raises
                rather than returning one.
        """
        paths = sorted(glob.glob(str(pattern)))
        if not paths:
            raise ConfigurationError(
                f"Comparison.load({str(pattern)!r}) matched no files"
            )
        return cls(results=[BenchmarkResult.load(path) for path in paths])

    def metrics(self) -> list[str]:
        """Every metric name across every result, first-seen order."""
        seen: list[str] = []
        present: set[str] = set()
        for result in self.results:
            for name in result.summary:
                if name not in present:
                    present.add(name)
                    seen.append(name)
        return seen

    def to_markdown(self, stat: str = "mean") -> str:
        """Render one row per run, one column per metric, ``stat`` of each.

        ``stat`` is any :class:`~fastrag.benchmark.metrics.base.DistributionSummary`
        field (``"mean"``, ``"p95"``, ...); an unknown one raises
        :class:`~fastrag.exceptions.ConfigurationError` rather than printing
        zeros. A run with no such metric prints ``—``. When ``stat`` is not
        ``"mean"`` every header carries it, e.g. ``latency_ms p95``.
        """
        columns = self.metrics()
        headers = ["run"] + [
            column if stat == "mean" else f"{column} {stat}" for column in columns
        ]
        rows = [
            [result.name]
            + [_format_cell(result.summary.get(column), stat) for column in columns]
            for result in self.results
        ]

        widths = [
            max(len(header), *(len(row[i]) for row in rows)) if rows else len(header)
            for i, header in enumerate(headers)
        ]

        lines = [
            _render_row(headers, widths, is_header=True),
            "| " + " | ".join("-" * width for width in widths) + " |",
        ]
        lines.extend(_render_row(row, widths, is_header=False) for row in rows)
        return "\n".join(lines)

    def best(self, metric: str, stat: str = "mean") -> BenchmarkResult:
        """The result whose ``metric`` is best, per that metric's own direction.

        Raises:
            ConfigurationError: no result carries ``metric``.
        """
        candidates = [result for result in self.results if metric in result.summary]
        if not candidates:
            raise ConfigurationError(
                f"Comparison.best: no result carries metric {metric!r}"
            )

        higher_is_better = candidates[0].higher_is_better.get(metric, True)
        pick = max if higher_is_better else min
        return pick(
            candidates, key=lambda result: _stat_value(result.summary[metric], stat)
        )

    def save(self, path: str | Path) -> None:
        """Write :meth:`to_markdown`'s default (mean) table to ``path``."""
        Path(path).write_text(self.to_markdown())


def _render_row(cells: list[str], widths: list[int], *, is_header: bool) -> str:
    # The run name (first column) and every header are left-justified; a data
    # row's metric values are right-justified so decimals line up.
    padded = [
        cell.ljust(widths[index])
        if is_header or index == 0
        else cell.rjust(widths[index])
        for index, cell in enumerate(cells)
    ]
    return "| " + " | ".join(padded) + " |"


def _format_cell(dist: DistributionSummary | None, stat: str) -> str:
    if dist is None:
        return "—"
    return f"{_stat_value(dist, stat):.2f}"


def _stat_value(dist: DistributionSummary, stat: str) -> float:
    if stat in dist.percentiles:
        return dist.percentiles[stat]
    if stat in _DIST_FIELDS:
        return getattr(dist, stat)
    raise ConfigurationError(
        f"Comparison: unknown stat {stat!r}; expected a DistributionSummary "
        "field (mean, median, std, min, max, count, missing) or a percentile "
        "key like 'p95'"
    )
