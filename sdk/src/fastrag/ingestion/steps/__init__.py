"""Step base classes and the policy-aware evaluator.

Re-exports :class:`~fastrag.ingestion.steps.base.PipelineStep`,
:class:`~fastrag.ingestion.steps.base.ExecutionContext`,
:class:`~fastrag.ingestion.steps.base.SourceStep`,
:class:`~fastrag.ingestion.steps.base.DocumentStep`, and
:class:`~fastrag.ingestion.steps.base.RoutedStep` from
``fastrag.ingestion.steps.base``, their canonical module. Concrete steps
(fetchers, parsers, chunkers, embedders) live in sibling packages:
``fetching``, ``parsing``, ``chunking``, ``embedding``. The pipeline's
storage backend, ``VectorStore``, lives outside this package entirely, at
:mod:`fastrag.vector_store` — it is queried at read time as well as written
at ingest time, so it is not an ingestion-only concern.
"""

from __future__ import annotations

from fastrag.ingestion.steps.base import (
    DocumentStep,
    ExecutionContext,
    PipelineStep,
    RoutedStep,
    SourceStep,
)

__all__ = [
    "DocumentStep",
    "ExecutionContext",
    "PipelineStep",
    "RoutedStep",
    "SourceStep",
]
