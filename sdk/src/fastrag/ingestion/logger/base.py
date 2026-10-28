"""``Logger``: the base class for a pipeline event stream consumer.

``docs/specification/logging.md``: a logger is a sink for a
:class:`~fastrag.ingestion.Pipeline`'s event stream. It must never alter what
the pipeline produces or how it handles errors — it only observes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from fastrag.ingestion.pipeline import Pipeline


class Logger(ABC):
    """Base class for components that consume a pipeline's event stream.

    A ``Logger`` is associated with one :class:`~fastrag.ingestion.Pipeline`
    at construction and does nothing until :meth:`run` is awaited — matching
    every :class:`~fastrag.ingestion.steps.base.PipelineStep`'s own "nothing
    happens at construction" contract.
    """

    pipeline: Pipeline

    def __init__(self, pipeline: Pipeline) -> None:
        """Associate this logger with ``pipeline``.

        Args:
            pipeline: The pipeline whose event stream :meth:`run` will
                consume. It may be bare or already entered with
                ``async with pipeline:``; iteration manages the event sink
                automatically for a bare pipeline.
        """
        self.pipeline = pipeline

    @abstractmethod
    async def run(self) -> None:
        """Consume and process every event from :attr:`pipeline`.

        Implementations iterate ``self.pipeline`` (``async for event in
        self.pipeline:``), which both drives the pipeline's root step to
        completion and yields each event as it is emitted — this is also what
        propagates a step failure out of :meth:`run`, exactly as it does out
        of :meth:`~fastrag.ingestion.Pipeline.run`, under ``policy.on_error == "raise"``.
        """
