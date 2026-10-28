"""Benchmarking: load a dataset, score a pipeline against it, compare runs."""

from fastrag.benchmark.benchmark import Benchmark, BenchmarkTarget
from fastrag.benchmark.comparison import Comparison
from fastrag.benchmark.datasets import ColumnMapping, EvalSample, QADataset
from fastrag.benchmark.metrics.base import QueryOutcome
from fastrag.benchmark.results import BenchmarkResult, SampleResult, StatusCounts

__all__ = [
    "Benchmark",
    "BenchmarkResult",
    "BenchmarkTarget",
    "ColumnMapping",
    "Comparison",
    "EvalSample",
    "QADataset",
    "QueryOutcome",
    "SampleResult",
    "StatusCounts",
]
