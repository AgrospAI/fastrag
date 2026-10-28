"""``Pipeline``: the context manager that evaluates a lazy step graph.

Constructing a :class:`Pipeline` never evaluates anything — it only records
the root step and the pipeline-level defaults (``cache``/``policy``) a step
falls back to when it declares none of its own. Entering it (``async with``)
installs an event sink (via
:func:`fastrag.ingestion.steps.base.event_sink_scope`) so that *any* step
evaluated during the block — not just the ones this ``Pipeline`` drives
itself — has its events routed here. ``__aiter__``/``run()`` drive the root
step's evaluation under one
:class:`~fastrag.ingestion.steps.base.ExecutionContext` built from this
pipeline's own ``cache``/``policy``, concurrently draining the events it
produces through a queue.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Any

from fastrag.ingestion.cache import CacheStore
from fastrag.ingestion.events import PipelineEvent
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import (
    EventSink,
    ExecutionContext,
    PipelineStep,
    SourceStep,
    event_sink_scope,
    run_step,
)


class _Done:
    """Sentinel placed on the internal queue once the root step is exhausted.

    A dedicated type (rather than a bare ``object()``) so an ``isinstance``
    check on it lets the type checker narrow ``item`` back to
    :class:`~fastrag.ingestion.events.PipelineEvent` after the check, instead
    of leaving it widened to ``object``.
    """

    __slots__ = ()


_DONE = _Done()


class Pipeline(AbstractAsyncContextManager["Pipeline"]):
    """Evaluates a lazy :class:`~fastrag.ingestion.steps.base.PipelineStep` graph.

    ``Pipeline`` owns the :class:`~fastrag.ingestion.steps.base.ExecutionContext`
    for a whole run: it is responsible for creating it (with this pipeline's
    own default ``cache``/``policy``) and for installing the event sink every
    step in the graph — however deeply nested — ultimately emits into.

    .. code-block:: python

        async with Pipeline(vector_store.save(embedded_documents)) as pipeline:
            await ConsoleLogger(pipeline).run()
    """

    def __init__(
        self,
        root: PipelineStep[Any],
        *,
        cache: CacheStore | None = None,
        policy: Policy | None = None,
        source_headers_only: bool = False,
        headers_only: bool = False,
    ) -> None:
        """Create a pipeline around ``root`` without evaluating anything.

        Args:
            root: The final step in the lazy pipeline graph. Evaluating the
                pipeline runs exactly ``root``'s own upstream chain — passing
                an intermediate step here (rather than the graph's true
                terminal step) runs only the portion of the graph upstream of
                it.
            cache: Default cache backend for any step in the graph whose own
                ``cache`` is ``None`` (the "inherit" state). A step with its
                own explicit ``cache`` (including ``cache=False``) is
                unaffected.
            policy: Default execution policy for any step whose own
                ``policy`` is ``None``. Defaults to a fresh :class:`~fastrag.ingestion.policy.Policy`
                when omitted.
            source_headers_only: Read cached source identities and metadata
                without content bytes. Only valid for source-only pipelines,
                such as URL tree summaries. Cached source events have empty
                content; misses and freshness validation still fetch full bytes.
                Leave False for content inspection or transformation.
            headers_only: Like ``source_headers_only`` but for any graph:
                cached sources *and* documents are read as identities and
                metadata without payload bytes. Intended for observation
                (e.g. :class:`~fastrag.ingestion.logger.ConsoleLogger`
                cache-hit summaries); cached documents reach the consumer
                with empty content, while misses are still computed fully.
        """
        if source_headers_only and not isinstance(root, SourceStep):
            raise ValueError("source_headers_only requires a source-only pipeline")
        self._source_headers_only = source_headers_only or headers_only
        self._root = root
        self._cache = cache
        self._policy = policy or Policy()
        self._pipeline_id = str(uuid.uuid4())
        self._queue: asyncio.Queue[PipelineEvent | _Done] | None = None
        self._scope: contextlib.AbstractContextManager[None] | None = None

    @property
    def root(self) -> PipelineStep[Any]:
        """The step this pipeline evaluates — and the top of its upstream chain."""
        return self._root

    def _build_context(self, sink: EventSink) -> ExecutionContext:
        """The one :class:`~fastrag.ingestion.steps.base.ExecutionContext` this pipeline's whole run shares."""
        return ExecutionContext(
            pipeline_id=self._pipeline_id,
            policy=self._policy,
            cache=self._cache,
            event_sink=sink,
            cache_headers_only=self._source_headers_only,
        )

    async def __aenter__(self) -> Pipeline:
        """Enter the pipeline context and register the event listener queue.

        Installs an event sink, ambient for the duration of the block via
        :func:`~fastrag.ingestion.steps.base.event_sink_scope`, so a step
        evaluated anywhere inside the block — including ones several levels
        upstream of :attr:`root`, or run independently of it — has its events
        land on this pipeline's queue without threading the sink through
        explicitly.
        """
        queue: asyncio.Queue[PipelineEvent | _Done] = asyncio.Queue()
        self._queue = queue

        async def sink(event: PipelineEvent) -> None:
            await queue.put(event)

        self._scope = event_sink_scope(sink)
        self._scope.__enter__()
        return self

    async def __aexit__(
        self, exc_type: Any, exc_value: Any, exc_traceback: Any
    ) -> None:
        """Exit the pipeline context, restoring whatever sink was previously installed.

        Runs unconditionally — including when the ``async with`` body raised
        — since :func:`~fastrag.ingestion.steps.base.event_sink_scope` itself
        restores the prior contextvar value in a ``finally``.
        """
        if self._scope is not None:
            self._scope.__exit__(exc_type, exc_value, exc_traceback)
            self._scope = None
        self._queue = None

    def __aiter__(self) -> AsyncIterator[PipelineEvent]:
        """Iterate every event emitted while evaluating :attr:`root` to exhaustion.

        Drives ``root``'s evaluation in the background, concurrently with
        yielding each event to this iterator's consumer as it arrives — so a
        slow consumer never stalls event production beyond the queue filling
        up, and a fast one sees events in the exact order the graph emitted
        them. Terminates once ``root`` is exhausted. If evaluating ``root``
        raises (e.g. a :class:`~fastrag.ingestion.events.StepFailed` under
        ``policy.on_error == "raise"``), that exception is re-raised here
        after every event already produced has been yielded.

        Usable either inside ``async with Pipeline(...) as pipeline:`` (which
        installs the queue and sink for the whole block, per :meth:`__aenter__`)
        or on a bare, never-entered ``Pipeline`` — the common case of handing
        one straight to a :class:`~fastrag.ingestion.logger.Logger`, e.g.
        ``await ConsoleLogger(Pipeline(root)).run()`` — in which case a queue
        and sink scoped to just this one iteration are created and torn down
        automatically.
        """
        return self._drive()

    async def _drive(self) -> AsyncIterator[PipelineEvent]:
        owns_scope = self._queue is None
        queue: asyncio.Queue[PipelineEvent | _Done]
        if owns_scope:
            queue = asyncio.Queue()
        else:
            assert self._queue is not None
            queue = self._queue

        async def sink(event: PipelineEvent) -> None:
            await queue.put(event)

        scope = event_sink_scope(sink) if owns_scope else None
        if scope is not None:
            scope.__enter__()

        async def produce() -> None:
            try:
                context = self._build_context(sink)
                async for _ in run_step(self._root, context):
                    pass
            finally:
                await queue.put(_DONE)

        task = asyncio.ensure_future(produce())
        try:
            try:
                while True:
                    item = await queue.get()
                    if isinstance(item, _Done):
                        break
                    yield item
            finally:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
        finally:
            if scope is not None:
                scope.__exit__(None, None, None)

        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            raise error

    async def run(self) -> None:
        """Evaluate :attr:`root` while discarding every event it produces.

        For when event consumption is handled by something else — a
        :class:`~fastrag.ingestion.logger.Logger` iterating this same
        pipeline concurrently, e.g. — rather than by the caller of ``run()``
        itself.
        """
        async for _ in self:
            pass
