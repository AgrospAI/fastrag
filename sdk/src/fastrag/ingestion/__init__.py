"""fastrag.ingestion — lazy, event-emitting ingestion pipeline.

A lazy pipeline API: constructing a step never executes anything, and
evaluating one emits lifecycle and domain events instead of writing to
stdout directly. This module re-exports the inert value types owned by
ingestion (``Source``, policy, events), the step contract and its evaluator
(``PipelineStep``, ``ExecutionContext``), and ``Pipeline``, the context
manager that installs the execution context and event sink for a whole
graph. ``Document`` — shared by retrieval, querying, serving and
benchmarking, not an ingestion-only concern — lives at ``fastrag.models``
instead and is deliberately not re-exported here. The cache contract lives at
``fastrag.ingestion.cache`` and is deliberately not re-exported here (see its
own module docstring); ``Logger``/``ConsoleLogger`` live at
``fastrag.ingestion.logger``, likewise not re-exported here. This package
must never import from outside its own tree, except the foundation modules
every package sits on top of (``fastrag.models``, ``fastrag.exceptions``)
and the underscore-prefixed private modules directly under ``fastrag``
(``fastrag._sql`` today) that hold plumbing shared across packages.
"""

from __future__ import annotations

from fastrag.ingestion.pipeline import Pipeline
from fastrag.ingestion.models import Source
from fastrag.ingestion.policy import Policy
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
    "Pipeline",
    "PipelineEvent",
    "PipelineStep",
    "Policy",
    "RoutedStep",
    "Source",
    "SourceStep",
]
