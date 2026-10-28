"""``PipelineStep``, ``ExecutionContext``, and the policy-aware evaluator.

This module is the canonical home for the lazy step contract every concrete
step (fetcher, parser, chunker, embedder, vector store) builds on:

- Constructing a step, and composing one with ``step(upstream)``, does no I/O
  and emits no event — only *evaluating* a step does work.
- ``PipelineStep.evaluate`` stays abstract: a ``SourceStep`` subclass (a leaf
  producer, e.g. a fetcher) implements it directly with its own fetch logic.
  ``DocumentStep`` instead implements ``evaluate`` *concretely* here, driving
  a per-item ``transform`` hook (which subclasses implement) over whatever
  upstream is bound to it — because a document transform is always "one
  upstream item in, zero or more items out", while a source has no such
  upstream shape to hang caching off of.
- The module-level ``_run_step`` is "the evaluator": the single function both
  ``PipelineStep.__aiter__`` and every ``DocumentStep``/``RoutedStep`` use to
  pull from an upstream, so lifecycle events (``StepStarted``/``StepProgress``/
  ``StepCompleted``/``StepFailed``) and the ``SourceStep`` cache wrap apply
  uniformly at every level of a composed graph, not just at the outermost
  step a caller happens to call ``run()`` on.
- ``run_concurrently`` is the shared concurrency + retry + ``ItemFailed``
  primitive. ``DocumentStep.evaluate`` is built on it, and it is exported so a
  later ``SourceStep`` (a fetcher with several targets to reach concurrently)
  can reuse it too, instead of writing its own retry loop.

Caching itself is intentionally split by step family: by default,
``SourceStep`` caches its *entire* ``evaluate()`` stream as one bucket keyed
by ``source_cache_id(name, config_hash, config_hash)``. A step that declares
an optional ``targets`` hook (duck-typed like ``is_source_modified``) opts
into per-target caching instead — one bucket per target (e.g. one per crawled
page), via ``evaluate_target(target, context)``.
``DocumentStep`` caches per upstream item, keyed by
``document_cache_id(name, item.id, config_hash)``, because document
transforms are pure and each input independently determines its output.
``RoutedStep`` (e.g. ``ParserRouter``, ``ChunkerRouter``) is the one
exception to "a step caches by its own identity": it forces its own
``cache`` to ``False`` so ``DocumentStep``'s generic per-item wrap never
runs for it, and instead caches each item by whichever *child* step the
router dispatched it to (``document_cache_id(target.name, item.id,
target.config_hash)``, in ``_dispatch_route``) — otherwise every route would
share one cache bucket keyed by the router's own config, and switching a
route would silently keep serving the old route's cached output.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Generic, Literal, Self, TypeAlias, TypeVar, cast

from fastrag.exceptions import ConfigurationError, TransientError
from fastrag.ingestion._hashing import config_hash as _compute_config_hash
from fastrag.ingestion.cache import CacheStore, document_cache_id, source_cache_id
from fastrag.ingestion.cache.base import _DocumentHeader, _SourceHeader
from fastrag.ingestion.events import (
    DocumentFetched,
    ItemFailed,
    ItemFinished,
    ItemRetrying,
    ItemStarted,
    PipelineEvent,
    StepCompleted,
    StepFailed,
    StepProgress,
    StepStarted,
)
from fastrag.ingestion.filters import Filter
from fastrag.ingestion.models import Source
from fastrag.ingestion.policy import Policy
from fastrag.models import Document

T = TypeVar("T")
R = TypeVar("R")

StepConfigHash: TypeAlias = str
"""A stable, short hex digest identifying one step's effective configuration."""

EventSink: TypeAlias = Callable[[PipelineEvent], Awaitable[None]]
"""An async callable a :class:`~fastrag.ingestion.Pipeline` installs to receive
every event emitted while it runs."""

_event_sink_var: ContextVar[EventSink | None] = ContextVar(
    "fastrag_pipelines_event_sink", default=None
)


@contextmanager
def event_sink_scope(sink: EventSink | None) -> Iterator[None]:
    """Install ``sink`` as the ambient event sink for the duration of the block.

    A step's ``run()``/``__aiter__`` builds its own :class:`~fastrag.ingestion.steps.base.ExecutionContext`
    when none is threaded to it explicitly, and reads the sink installed here
    to populate it — this is how "a step nested three levels deep reaches the
    sink without any step passing it along" holds even when a step is
    evaluated deep inside another step's own logic. A
    :class:`~fastrag.ingestion.Pipeline` (task 009) is expected to call this
    around the whole run it drives.

    Args:
        sink: The event sink to install, or ``None`` to explicitly install "no
            sink" for the scope (distinct from simply not calling this at
            all, which leaves whatever ambient sink — if any — already set).
    """
    token = _event_sink_var.set(sink)
    try:
        yield
    finally:
        _event_sink_var.reset(token)


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Runtime context passed to step evaluators.

    Attributes:
        pipeline_id: Global identifier for the executing pipeline run.
        policy: Active execution policy — a step without its own falls back
            to this.
        cache: Active cache store backend — a step whose own ``cache`` is
            ``None`` inherits this one; ``False`` disables caching regardless.
        event_sink: Where emitted events go. ``None`` means "no sink
            installed": :meth:`emit` becomes a no-op rather than raising, so a
            step run on its own works without a
            :class:`~fastrag.ingestion.Pipeline`.
    """

    pipeline_id: str
    policy: Policy
    cache: CacheStore | None = None
    event_sink: EventSink | None = None
    cache_headers_only: bool = False
    document_cache_entries: Mapping[tuple[int, str], list[Document] | None] | None = (
        None
    )

    async def emit(self, event: PipelineEvent) -> None:
        """Dispatch ``event`` to the registered sink, or do nothing.

        Args:
            event: Event instance to propagate.
        """
        if self.event_sink is not None:
            from fastrag.ingestion.cache.base import _public_item

            changes: dict[str, Any] = {}
            for name in ("source", "document"):
                item = getattr(event, name, None)
                public = _public_item(item)
                if public is not item:
                    changes[name] = public
            chunks = getattr(event, "chunks", None)
            if chunks is not None:
                changes["chunks"] = [_public_item(item) for item in chunks]
            await self.event_sink(replace(event, **changes) if changes else event)


def _default_context(step: PipelineStep[Any]) -> ExecutionContext:
    """Build the ``ExecutionContext`` a step's own entry points fall back to.

    Reuses the step's own policy (or a fresh :class:`~fastrag.ingestion.policy.Policy`), carries no
    cache of its own (a step's own explicit ``cache`` still works fine — see
    :func:`_resolve_cache`), and reads whatever sink :func:`event_sink_scope`
    currently has installed.
    """
    return ExecutionContext(
        pipeline_id=str(uuid.uuid4()),
        policy=step.policy or Policy(),
        cache=None,
        event_sink=_event_sink_var.get(),
    )


def _resolve_cache_value(
    own: CacheStore | Literal[False] | None, outer: CacheStore | None
) -> CacheStore | None:
    """Resolve a raw three-state cache value against an outer/inherited one.

    ``own is False`` (the default) disables caching outright; ``own is None``
    inherits ``outer``; any :class:`~fastrag.ingestion.cache.CacheStore` is
    used directly. Factored out of :func:`_resolve_cache` so
    :class:`RoutedStep` can resolve its *own* configured cache (kept outside
    the normal ``step.cache`` slot — see ``RoutedStep.__init__``) the same
    way, without going through a step's ``.cache`` property.
    """
    if own is False:
        return None
    if own is None:
        return outer
    return own


def _resolve_cache(
    step: PipelineStep[Any], context: ExecutionContext
) -> CacheStore | None:
    """Resolve a step's effective cache: its own, the context's, or none.

    ``step.cache is False`` disables caching outright; ``step.cache is None``
    (the default) inherits ``context.cache``; any
    :class:`~fastrag.ingestion.cache.CacheStore` is used directly.
    """
    return _resolve_cache_value(step.cache, context.cache)


def _item_identifier(item: Any) -> str:
    """A human-readable identifier for an item, for events and cache keys.

    Prefers ``.uri`` (a :class:`~fastrag.ingestion.models.Source`), falling
    back to ``.id`` (a :class:`~fastrag.models.Document`, or a
    ``Source`` accessed generically).
    """
    uri = getattr(item, "uri", None)
    if uri is not None:
        return str(uri)
    return str(getattr(item, "id", item))


def _lineage_key(item: Any) -> str:
    """The originating :class:`~fastrag.ingestion.models.Source` id ``item`` traces back to.

    Prefers ``.source_id`` (a :class:`~fastrag.models.Document`, which
    carries its originating ``Source``'s id unchanged through every later
    stage), then ``.id`` (a bare :class:`~fastrag.ingestion.models.Source`),
    then the item itself when it is a bare ``str`` (a URL a per-target
    ``SourceStep`` was mid-fetch when it failed). Returns ``""`` only for an
    item with none of the above (e.g. :class:`Sources`'s own
    ``run_concurrently`` call, whose items are child ``SourceStep``
    instances).

    Populates :attr:`~fastrag.ingestion.events.ItemFailed.source_id`, which
    :class:`~fastrag.ingestion.logger.ConsoleLogger` uses to attribute a
    downstream failure to the source row it came from.
    """
    source_id = getattr(item, "source_id", None)
    if source_id is not None:
        return str(source_id)
    item_id = getattr(item, "id", None)
    if item_id is not None:
        return str(item_id)
    if isinstance(item, str):
        return item
    if isinstance(item, tuple) and item and isinstance(item[0], str):
        # A ``(url, ...)`` work item, e.g. ``WebCrawler``'s (url, language).
        return item[0]
    return ""


class PipelineStep(ABC, Generic[T]):
    """Base class for every executable pipeline step.

    Constructing a step, and composing it with another via ``__call__``, must
    never execute work. A step can be evaluated independently with
    :meth:`run` / ``async for``, or composed into a larger graph and evaluated
    once the graph's terminal step is run.

    Every step carries two names, and every event it emits carries both:
    :attr:`name`, identifying *this instance* (its own class name by
    default), and :attr:`event_name`, identifying *the stage* its work
    belongs to. See :attr:`event_name` for how the latter is resolved.
    """

    name: str

    event_name: str
    """The stage this step's work belongs to, e.g. ``"Chunking"``.

    Declared as a plain class attribute by a *family* base class, and
    inherited by every concrete step in that family::

        class Chunker(DocumentStep, ABC):
            event_name = "Chunking"


        class ChunkerRouter(RoutedStep[Document], Chunker):
            ...  # -> "Chunking", inherited from `Chunker`

    A concrete step overrides it to opt out of its family's label::

        class ChunkerRouter(RoutedStep[Document], Chunker):
            event_name = "ChunkerRouter"  # -> "ChunkerRouter"

    Resolution walks the MRO for the *first* class that declares it in its
    own class body (so a mixin listed first, e.g.
    :class:`~fastrag.ingestion.steps.base.RoutedStep`,
    contributes nothing unless it declares one itself), and falls back to
    the class's own name when nothing in the MRO does — a step family that
    never declares an ``event_name`` still labels each of its steps
    distinctly rather than collapsing them under one blank heading. That
    fallback is computed per instance rather than assigned onto the class:
    assigning it would make it look "declared" to every subclass, which
    would then inherit their base's name instead of getting their own.
    """

    def __init__(
        self,
        *,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
    ) -> None:
        """Configure the step without executing it.

        Args:
            cache: Cache backend for this step. ``False`` disables caching,
                ``None`` (the default) inherits the execution context's
                cache, a :class:`~fastrag.ingestion.cache.CacheStore` is used
                directly.
            policy: Execution policy overriding the pipeline's default.
        """
        self._cache = cache
        self._policy = policy
        self._upstream: PipelineStep[Any] | None = None
        self.name = type(self).__name__
        self.event_name = self._resolve_event_name()

    @classmethod
    def _resolve_event_name(cls) -> str:
        """The first ``event_name`` declared in ``cls``'s MRO, or ``cls``'s own name.

        See :attr:`event_name`. Only a class body's *own* declaration counts
        (``klass.__dict__``, not ``getattr``), so the per-instance fallback
        assigned in :meth:`__init__` never masquerades as a declaration a
        subclass should inherit.
        """
        for klass in cls.__mro__:
            declared = klass.__dict__.get("event_name")
            if isinstance(declared, str) and declared:
                return declared
        return cls.__name__

    @property
    def cache(self) -> CacheStore | Literal[False] | None:
        """Return the cache explicitly configured for this step."""
        return self._cache

    @property
    def policy(self) -> Policy | None:
        """Return the policy explicitly configured for this step."""
        return self._policy

    def effective_policy(self, context: ExecutionContext) -> Policy:
        """Resolve this step's policy over the one inherited from ``context``.

        The step's own explicitly-set fields take priority; everything else
        falls through to the pipeline (or enclosing router/aggregator) policy.
        """
        if self._policy is None:
            return context.policy
        return self._policy.merge(context.policy)

    def _policy_for_item(self, item: Any, effective: Policy) -> Policy:
        """The policy governing one upstream ``item``'s transform.

        ``effective`` is this step's own resolved policy (its
        :attr:`policy`, or the context's) — what governs every item unless a
        step has a reason to vary it per item. :class:`RoutedStep` is the one
        that does: the child an item routes to may declare a policy of its
        own, and that child, not the router, is what actually does the work.

        Args:
            item: The upstream item about to be transformed.
            effective: This step's own resolved policy.
        """
        return effective

    @property
    def upstream(self) -> PipelineStep[Any] | None:
        """Return the upstream step bound via ``__call__``, or ``None``.

        Read-only counterpart to ``__call__``'s ``self._upstream``
        assignment — lets external code (e.g.
        :class:`~fastrag.ingestion.Pipeline`, walking a graph from its root
        to classify each step by position) introspect composition without
        reaching into a private attribute.
        """
        return self._upstream

    @abstractmethod
    def config(self) -> Mapping[str, Any]:
        """Return this step's own configuration, for cache-identity hashing.

        Every concrete step declares this explicitly — steps are plain
        classes, not dataclasses, so there is no automatic field
        introspection. A step that forgets it cannot be instantiated:
        ``config`` is abstract, so construction itself fails loudly rather
        than :attr:`config_hash` silently collapsing to a constant.
        """

    @property
    def config_hash(self) -> StepConfigHash:
        """A stable hash of this step's class name and :meth:`config`.

        Two identically configured steps of the same class hash equal;
        changing any declared config value changes the hash; the hash is
        stable across processes. Used as part of every cache id this step's
        evaluation reads or writes.
        """
        return _compute_config_hash(type(self).__name__, self.config())

    @abstractmethod
    def evaluate(self, context: ExecutionContext) -> AsyncIterator[T]:
        """Lazily evaluate the step within an execution context.

        Args:
            context: Execution-scoped state and services.

        Yields:
            Values produced by the step.
        """

    def __call__(self, upstream: PipelineStep[Any]) -> PipelineStep[T]:
        """Compose this step with an upstream pipeline step.

        The returned step remains lazy and performs no work until evaluated.

        Args:
            upstream: Step whose output becomes this step's input.

        Returns:
            This step, now bound to ``upstream``.
        """
        self._upstream = upstream
        return self

    async def run(self) -> list[T]:
        """Execute this step independently and collect every item.

        Builds a default :class:`~fastrag.ingestion.steps.base.ExecutionContext` when none is installed
        (this step's own :attr:`policy`, or a fresh :class:`~fastrag.ingestion.policy.Policy`). Intended
        primarily for testing and standalone use — pipeline-wide event
        observation is :class:`~fastrag.ingestion.Pipeline`'s job.
        """
        return [item async for item in self]

    def __aiter__(self) -> AsyncIterator[T]:
        """Stream this step's output lazily, evaluating it as consumed.

        Builds the same default :class:`~fastrag.ingestion.steps.base.ExecutionContext` as :meth:`run` when
        none is installed, and runs through the same evaluator — the two
        entry points differ only in collecting versus streaming.
        """
        return _run_step(self, _default_context(self))


async def _run_step(
    step: PipelineStep[T], context: ExecutionContext
) -> AsyncIterator[T]:
    r"""The policy-aware evaluator: lifecycle events plus the source cache wrap.

    Every step, at every level of a composed graph, is driven through this
    function — not just the outermost step a caller happens to call
    ``run()``/``__aiter__`` on — so a step three levels deep still gets its
    own ``StepStarted``/``StepProgress``/``StepCompleted``/``StepFailed`` and,
    if it is a :class:`~fastrag.ingestion.steps.base.SourceStep`, its own cache wrap.

    Emits ``StepStarted`` before evaluation begins, ``StepProgress`` after
    each item, ``StepCompleted`` once the stream is exhausted, or
    ``StepFailed`` (with the exception re-raised afterward) if the step
    itself cannot continue. Per-item failures are a different concern, handled
    by :func:`run_concurrently` as ``ItemFailed`` — they never reach here as
    an exception unless ``policy.on_error == "raise"``.

    Also stamps the reserved ``metadata["source_step"]`` key (the producing
    leaf ``SourceStep``'s own :attr:`~PipelineStep.name`) onto every
    ``Source`` that does not already carry one, before emitting
    ``DocumentFetched`` for it. A ``Source`` only lacks the key the first time
    it passes through *some* ``SourceStep``'s evaluation — for a
    :class:`Sources`-composed graph, that is always the innermost child's own
    :func:`run_source_step` call, since a child's stream is fully driven
    (and thus already stamped) before ``Sources.evaluate`` ever re-yields it
    up into its own, outer ``_run_step`` wrap. That outer wrap re-emits
    ``DocumentFetched`` for the same (already-stamped) ``Source`` again —
    deliberately: every ``SourceStep``, aggregate or leaf, gets its own
    lifecycle/`DocumentFetched` treatment uniformly — which is why a consumer
    counting `Source`\\ s (e.g. :class:`~fastrag.ingestion.logger.ConsoleLogger`)
    must dedupe by ``Source.id`` rather than by event count. This is the one
    reserved metadata key this module adds; see
    ``fastrag.ingestion.logger.console`` for how it is consumed.
    """
    await context.emit(StepStarted(step=step.name, event_name=step.event_name))
    processed = 0
    source = (
        cast("AsyncIterator[T]", _cached_source_stream(step, context))
        if isinstance(step, SourceStep)
        else step.evaluate(context)
    )
    try:
        async for item in source:
            processed += 1
            if isinstance(step, SourceStep):
                # `DocumentFetched` is the one domain event generic enough to
                # belong here rather than in a concrete fetcher: its fields
                # (`url`, `cached`) map 1:1 onto any `Source`, regardless of
                # which cache path produced it (fresh, single-bucket hit, or
                # per-target hit) — so it is emitted uniformly for every
                # `SourceStep`, the same way `StepProgress` already is,
                # instead of requiring every fetcher to remember to emit it
                # itself (which cached items never even pass through).
                fetched = cast("Source", item)
                if "source_step" not in fetched.metadata:
                    fetched = replace(
                        fetched, metadata={**fetched.metadata, "source_step": step.name}
                    )
                    item = cast("T", fetched)
                await context.emit(
                    DocumentFetched(
                        source=fetched,
                        step=step.name,
                        event_name=step.event_name,
                    )
                )
            yield item
            await context.emit(
                StepProgress(
                    step=step.name,
                    processed_count=processed,
                    event_name=step.event_name,
                )
            )
    except Exception as error:
        await context.emit(
            StepFailed(step=step.name, error=error, event_name=step.event_name)
        )
        raise
    else:
        await context.emit(StepCompleted(step=step.name, event_name=step.event_name))


#: Source of ``ItemStarted.item_id``: unique per process, cheap, ordered.
_item_ids = itertools.count(1)


async def _with_retries(
    operation: Callable[[], Awaitable[R]],
    policy: Policy,
    on_retry: Callable[[int, float, Exception], Awaitable[None]] | None = None,
) -> R:
    """Retry ``operation`` on a transient failure, with exponential backoff.

    Only :class:`fastrag.exceptions.TransientError` and a bare
    :class:`TimeoutError` (raised by ``policy.timeout`` expiring) are
    retried — anything else propagates on the first attempt, since a
    non-transient failure will not succeed on a second one.

    A ``policy.timeout`` that finally expires is re-raised as a
    ``TimeoutError`` *carrying a message*: ``asyncio.timeout`` raises a bare,
    argument-less one, which a consumer formatting the failure (``f"{error}"``
    in a logger or an ``ItemFailed`` handler) renders as an empty string —
    the one failure mode the pipeline could report as literally nothing. The
    type is unchanged, so anything catching ``TimeoutError`` still does.
    """
    for attempt in range(policy.retries + 1):
        try:
            async with asyncio.timeout(policy.timeout) as limit:
                return await operation()
        except (TransientError, TimeoutError) as error:
            if attempt == policy.retries:
                if isinstance(error, TimeoutError) and limit.expired():
                    raise TimeoutError(
                        f"timed out after {policy.timeout:g}s "
                        f"(Policy.timeout) on attempt {attempt + 1} "
                        f"of {policy.retries + 1}"
                    ) from error
                raise
            delay = policy.delay_for(attempt + 1)
            if on_retry is not None:
                if isinstance(error, TimeoutError) and limit.expired():
                    error = TimeoutError(f"timed out after {policy.timeout:g}s")
                await on_retry(attempt + 1, delay, error)
            await asyncio.sleep(delay)

    raise AssertionError("unreachable: the loop above always returns or raises")


async def _bounded_map(
    items: AsyncIterator[T],
    worker: Callable[[T], Awaitable[R]],
    limit: int,
) -> AsyncIterator[tuple[T, R | None, Exception | None]]:
    """Apply ``worker`` to each item of ``items`` with at most ``limit`` in flight.

    Yields ``(item, result, error)`` as calls complete — completion order, not
    input order, per ``policy.concurrency``'s documented guarantee. ``error``
    is whatever ``worker`` raised; it is reported here, never swallowed or
    re-raised — the caller decides what to do with it. Exactly one of
    ``result``/``error`` is ``None``.
    """
    iterator = items.__aiter__()
    pending: set[asyncio.Task[tuple[T, R | None, Exception | None]]] = set()
    exhausted = False

    async def run_one(item: T) -> tuple[T, R | None, Exception | None]:
        try:
            return item, await worker(item), None
        except Exception as error:
            return item, None, error

    try:
        while True:
            while not exhausted and len(pending) < limit:
                try:
                    item = await iterator.__anext__()
                except StopAsyncIteration:
                    exhausted = True
                    break
                pending.add(asyncio.create_task(run_one(item)))

            if not pending:
                return

            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                yield task.result()
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


async def run_concurrently(
    items: AsyncIterator[T],
    worker: Callable[[T], Awaitable[R]],
    *,
    context: ExecutionContext,
    policy: Policy,
    item_id: Callable[[T], str] = _item_identifier,
    step: str = "",
    event_name: str = "",
    policy_for: Callable[[T], Policy] | None = None,
) -> AsyncIterator[R]:
    """The shared concurrency, retry, and ``ItemFailed`` primitive.

    Applies ``worker`` to each item from ``items``, bounded by
    ``policy.concurrency`` calls in flight at once, retrying a call that
    raises :class:`~fastrag.exceptions.TransientError` or ``TimeoutError``
    per ``policy.retries``/``backoff``/``max_backoff``/``timeout``. A call
    that still fails after retries emits
    ``ItemFailed(uri=item_id(item), error=..., step=step)`` through
    ``context`` and is skipped (``policy.on_error in {"skip", "retry"}``) or
    re-raised (``policy.on_error == "raise"``, propagating after the
    ``ItemFailed``). Output order follows completion order, not input order.

    This is the primitive :class:`DocumentStep`'s evaluator is built on, and
    the one a concrete step with several independent targets to reach at once
    (e.g. a fetcher crawling many URLs) should call instead of writing its own
    retry loop.

    ``worker`` must not itself return ``None`` on success: ``None`` is used
    internally to mean "this call failed" (see ``_bounded_map``).

    Args:
        items: The upstream items to process.
        worker: Called once per item; its result is yielded on success.
        context: Execution-scoped state and services, used to emit
            ``ItemFailed``.
        policy: Governs concurrency, retries, backoff, timeout, and
            ``on_error`` disposition.
        item_id: Builds the human-readable identifier recorded on
            ``ItemFailed`` for a given item. Defaults to
            ``_item_identifier``.
        step: Name of the step this call is driving, recorded on any
            ``ItemFailed`` this raises so a consumer (e.g. a
            :class:`~fastrag.ingestion.logger.ConsoleLogger`) can attribute
            the failure to the right stage. Optional and empty by default —
            callers that predate this parameter keep working, just with an
            unattributed ``ItemFailed.step == ""``.
        event_name: Stage label of that step
            (:attr:`PipelineStep.event_name`), recorded on any ``ItemFailed``
            this raises. ``step`` names the instance; this names the column a
            consumer files the failure under. Optional and empty by default,
            for the same reason ``step`` is.
        policy_for: Resolves the policy governing *one* item, for a step
            whose items are not all handled the same way — a
            :class:`RoutedStep` dispatching each item to a child that may
            declare its own ``policy``. Only the per-item knobs come from it
            (``retries``, ``backoff``, ``max_backoff``, ``timeout``,
            ``on_error``); ``concurrency`` is a property of the stream, not
            of an item, and always comes from ``policy``. ``None`` (the
            default) uses ``policy`` for every item.
    """

    def policy_of(item: T) -> Policy:
        return policy if policy_for is None else policy_for(item)

    async def tracked(item: T) -> R:
        """Run one item under its policy, bracketed by lifecycle events."""
        tracking_id = next(_item_ids)
        await context.emit(
            ItemStarted(
                item_id=tracking_id,
                uri=item_id(item),
                step=step,
                event_name=event_name,
                source_id=_lineage_key(item),
            )
        )

        async def on_retry(attempt: int, delay: float, error: Exception) -> None:
            await context.emit(
                ItemRetrying(
                    item_id=tracking_id, attempt=attempt, delay=delay, error=error
                )
            )

        started = time.monotonic()
        failure: Exception | None = None
        try:
            return await _with_retries(lambda: worker(item), policy_of(item), on_retry)
        except Exception as error:
            failure = error
            raise
        finally:
            await context.emit(
                ItemFinished(
                    item_id=tracking_id,
                    elapsed=time.monotonic() - started,
                    error=failure,
                )
            )

    async for item, result, error in _bounded_map(items, tracked, policy.concurrency):
        if error is not None:
            await context.emit(
                ItemFailed(
                    uri=item_id(item),
                    error=error,
                    step=step,
                    source_id=_lineage_key(item),
                    event_name=event_name,
                )
            )
            if policy_of(item).on_error == "raise":
                raise error
            continue
        # `_bounded_map` guarantees exactly one of result/error is set.
        assert result is not None
        yield result


class SourceStep(PipelineStep[Source], ABC):
    """Base step responsible for generating raw :class:`~fastrag.ingestion.models.Source` entities.

    Source steps (fetchers, crawlers, and aggregators like
    ``docs/specification/subclasses.md``'s ``Sources``) implement
    :meth:`~PipelineStep.evaluate` directly — sourcing has no upstream item to
    hang a per-item transform off of, unlike :class:`DocumentStep`. Caching is
    still applied uniformly: ``_run_step`` wraps every ``SourceStep``'s
    evaluation in the source cache before calling ``evaluate`` at all.

    Declares ``event_name = "Fetching"`` for the whole family: sourcing *is*
    the fetching stage, whether the step is a leaf fetcher or a
    :class:`~fastrag.ingestion.steps.fetching.sources.Sources` aggregate of
    several — the aggregate must land in the same column as the fetchers it
    wraps, not one of its own. :class:`DocumentStep` deliberately declares
    none: "a document transform" is not itself a stage, so each transform
    family (``Parser``, ``Chunker``, ``Embedder``, ...) declares its own and
    anything else falls back to its own class name.
    """

    event_name = "Fetching"

    def __init__(
        self,
        *,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
        cache_validation_interval: float | None = None,
    ) -> None:
        """Initialize a ``SourceStep``.

        Args:
            cache: Cache backend, or ``False``/``None`` — see
                :class:`PipelineStep`.
            policy: Execution policy overriding the pipeline default.
            cache_validation_interval: How long a cached entry may be served
                without revalidating it against the source, in seconds.
                ``None`` means a cached entry is always considered fresh and
                is never revalidated; ``0`` means every access revalidates.
        """
        super().__init__(cache=cache, policy=policy)
        self.cache_validation_interval = cache_validation_interval


def _is_fresh(step: SourceStep, entries: Sequence[Source]) -> bool:
    """Whether ``entries``' persisted ``last_validated`` is within ``cache_validation_interval``.

    Freshness is read from each :class:`~fastrag.ingestion.models.Source`'s
    persisted ``last_validated`` column — populated by
    :meth:`~fastrag.ingestion.cache.postgres.PostgresCache.get_source_entries`
    on every read — rather than from any per-process state, so the interval
    means the same thing across a process restart.

    An empty ``entries`` or an interval of ``None`` is always fresh. Otherwise
    the bucket's age is the oldest ``last_validated`` among ``entries`` (they
    are written together, so in practice all equal); any entry missing a
    timestamp is treated as expired.
    """
    interval = step.cache_validation_interval
    if interval is None or not entries:
        return True
    known_timestamps = [entry.last_validated for entry in entries]
    if any(timestamp is None for timestamp in known_timestamps):
        return False
    oldest = min(timestamp for timestamp in known_timestamps if timestamp is not None)
    age_seconds = (datetime.now(UTC) - oldest).total_seconds()
    return age_seconds < interval


async def _touch_validated(
    cache: CacheStore, cache_id: str, entries: Sequence[Source]
) -> list[Source]:
    """Persist that ``entries`` were just reconfirmed fresh against the origin.

    Called on the "revalidated, not modified" path: the cached bytes stay,
    but ``last_validated`` is pushed forward so the next check — in this
    process or a fresh one — does not have to ask again for another
    ``cache_validation_interval``. Returns the touched entries so a caller
    yields the bumped timestamp rather than the one just superseded.
    """
    now = datetime.now(UTC)
    touched = [replace(entry, last_validated=now) for entry in entries]
    await cache.put_source_entries(cache_id, touched)
    return touched


async def _any_modified(
    validator: Callable[[str, str | None, datetime | None, str], Awaitable[bool]],
    entries: Sequence[Source],
) -> bool:
    """True if ``validator`` reports any cached entry as modified.

    Revalidation is all-or-nothing per cache bucket: entries in one
    ``source_cache_id`` were fetched together, so if any one of them changed,
    the whole bucket is treated as stale and refetched together.
    """
    for entry in entries:
        if await validator(
            entry.id, entry.etag, entry.last_modified, entry.content_hash
        ):
            return True
    return False


async def _cached_source_stream(
    step: SourceStep, context: ExecutionContext
) -> AsyncIterator[Source]:
    """Wrap ``step.evaluate(context)`` in the source cache.

    On a hit whose age is within ``cache_validation_interval``, yields cached
    entries with ``cached=True`` and never calls ``evaluate``. On an expired
    hit, calls the step's
    :class:`~fastrag.ingestion.cache.SourceFreshnessValidator` hook
    (``is_source_modified``, duck-typed — a step that does not implement it
    is an unanswerable freshness question, treated as modified) to decide
    between serving the cached entry again or refetching. On a true miss, or a
    confirmed modification, evaluates the step and writes the whole resulting
    stream through with ``put_source_entries`` once it is exhausted.

    A step that declares an optional ``targets`` hook (duck-typed, like
    ``is_source_modified``) opts out of this whole-stream bucket and into
    :func:`_cached_source_stream_by_target` instead, one cache bucket per
    target (e.g. one per crawled page) rather than one for the entire
    ``evaluate()`` run — so "add one URL to a crawl" serves every
    previously-fetched page from the cache instead of invalidating the lot.
    A step that declares no ``targets`` hook keeps the original,
    whole-stream single-bucket behavior below, keyed by
    ``source_cache_id(step.name, step.config_hash, step.config_hash)``.
    """
    cache = _resolve_cache(step, context)
    if cache is None:
        async for item in step.evaluate(context):
            yield item
        return

    targets = await _step_targets(step, context)
    if targets is not None:
        async for item in _cached_source_stream_by_target(
            step, context, cache, targets
        ):
            yield item
        return

    cache_id = source_cache_id(step.name, step.config_hash, step.config_hash)
    if step.effective_policy(context).recreate_cache:
        cached_entries = None
    else:
        cached_entries = (
            (await cache.get_source_headers_many([cache_id]))[cache_id]
            if context.cache_headers_only
            else await cache.get_source_entries(cache_id)
        )

    if cached_entries is not None:
        if _is_fresh(step, cached_entries):
            for entry in cached_entries:
                yield replace(entry, cached=True)
            return

        cached_entries = [await _materialize_item(entry) for entry in cached_entries]
        validator = getattr(step, "is_source_modified", None)
        if validator is not None and not await _any_modified(validator, cached_entries):
            touched = await _touch_validated(cache, cache_id, cached_entries)
            for entry in touched:
                yield replace(entry, cached=True)
            return

    fresh: list[Source] = []
    async for item in step.evaluate(context):
        fresh.append(item)
        yield item
    await cache.put_source_entries(cache_id, fresh)


async def _step_targets(
    step: SourceStep, context: ExecutionContext
) -> Sequence[str] | None:
    """Ask ``step`` for its independent cache targets, if it declares any.

    Duck-typed like ``is_source_modified``: a step that defines ``async def
    targets(self, context) -> Sequence[str] | None`` opts into per-target
    caching. A step that defines no such method keeps the original
    whole-stream behavior.
    """
    hook = getattr(step, "targets", None)
    if hook is None:
        return None
    return await hook(context)


async def _cached_source_stream_by_target(
    step: SourceStep,
    context: ExecutionContext,
    cache: CacheStore,
    targets: Sequence[str],
) -> AsyncIterator[Source]:
    """Cache ``step``'s output one bucket per target, fanning out concurrently.

    Mirrors :func:`_cached_source_stream`'s single-bucket logic — freshness
    check, then ``is_source_modified`` revalidation, then refetch-and-write —
    but applies it independently per target via
    ``source_cache_id(step.name, target, step.config_hash)`` rather than once
    for the whole stream. Requires ``step.evaluate_target(target, context)``,
    the per-target counterpart to ``evaluate()``, and fans out over
    ``targets`` through :func:`run_concurrently`.

    A step declaring ``streams_targets = True`` (duck-typed, like
    ``targets``) has targets too large for one unit of work — a whole crawl,
    which already bounds, retries and times out each page itself. Its cache
    misses are then streamed source by source as the crawl discovers them
    (see :func:`_stream_targets`) instead of being collected under one
    ``Policy.timeout``: downstream stages start at once, and a crawl longer
    than the timeout is not cancelled and restarted from scratch.
    """
    evaluate_target = getattr(step, "evaluate_target", None)
    if evaluate_target is None:
        raise ConfigurationError(
            f"{type(step).__name__} declares targets() but no "
            f"evaluate_target() hook for the base class to call per target"
        )
    effective_policy = step.effective_policy(context)

    prefetched_sources: dict[str, list[Source] | None] = {}
    # A bucket may hold a whole crawl while ``keeps_source`` selects a few
    # pages: read headers only, then hydrate just the pages that are kept.
    keeps = getattr(step, "keeps_source", None)
    lazy_content = keeps is not None and not context.cache_headers_only

    streams = getattr(step, "streams_targets", False)
    misses: list[str] = []

    async def process_one(target: str) -> list[Source]:
        cached = await lookup(target)
        if cached is not None:
            return cached
        if streams:
            misses.append(target)
            return []
        fresh = [item async for item in evaluate_target(target, context)]
        await cache.put_source_entries(
            source_cache_id(step.name, target, step.config_hash), fresh
        )
        return fresh

    async def lookup(target: str) -> list[Source] | None:
        """The target's cached (or revalidated) sources, or ``None`` on a miss."""
        cache_id = source_cache_id(step.name, target, step.config_hash)
        if effective_policy.recreate_cache:
            cached_entries = None
        else:
            cached_entries = (
                prefetched_sources[cache_id]
                if cache_id in prefetched_sources
                else await cache.get_source_entries(cache_id)
            )

        if cached_entries is not None:
            if _is_fresh(step, cached_entries):
                return [replace(entry, cached=True) for entry in cached_entries]

            cached_entries = await _hydrate_bucket(cache, cache_id, cached_entries)
            validator = getattr(step, "is_source_modified", None)
            if validator is not None and not await _any_modified(
                validator, cached_entries
            ):
                touched = await _touch_validated(cache, cache_id, cached_entries)
                return [replace(entry, cached=True) for entry in touched]
        return None

    for offset in range(0, len(targets), effective_policy.cache_batch_size):
        batch = targets[offset : offset + effective_policy.cache_batch_size]
        keys = [
            source_cache_id(step.name, target, step.config_hash) for target in batch
        ]
        try:
            if effective_policy.recreate_cache:
                prefetched_sources = {}
            else:
                async with asyncio.timeout(effective_policy.timeout):
                    prefetched_sources = (
                        await cache.get_source_headers_many(keys)
                        if context.cache_headers_only or lazy_content
                        else await cache.get_source_entries_many(keys)
                    )
        except Exception:
            prefetched_sources = {}

        async def _target_stream() -> AsyncIterator[str]:
            for target in batch:
                yield target

        async for produced in run_concurrently(
            _target_stream(),
            process_one,
            context=context,
            policy=effective_policy,
            item_id=str,
            step=step.name,
            event_name=step.event_name,
        ):
            kept = [s for s in produced if keeps is None or keeps(s)]
            if lazy_content:
                kept = await _hydrate_kept(cache, kept)
            for source in kept:
                yield source

    if misses:
        async for source in _stream_targets(
            step, context, cache, misses, evaluate_target, effective_policy
        ):
            if keeps is None or keeps(source):
                yield source


async def _stream_targets(
    step: SourceStep,
    context: ExecutionContext,
    cache: CacheStore,
    targets: Sequence[str],
    evaluate_target: Callable[[str, ExecutionContext], AsyncIterator[Source]],
    policy: Policy,
) -> AsyncIterator[Source]:
    """Evaluate ``targets`` concurrently, yielding each source as it arrives.

    A target's bucket is written once its evaluation completes; a target that
    fails part-way emits ``ItemFailed`` and caches nothing, so the next run
    evaluates it again rather than trusting an incomplete bucket. The queue
    is bounded, so a slow consumer pauses the crawl instead of buffering it.
    """
    queue: asyncio.Queue[Source | BaseException | None] = asyncio.Queue(
        maxsize=policy.cache_batch_size
    )
    semaphore = asyncio.Semaphore(policy.concurrency)

    async def evaluate(target: str) -> None:
        async with semaphore:
            fresh: list[Source] = []
            try:
                async for source in evaluate_target(target, context):
                    fresh.append(source)
                    await queue.put(source)
                await cache.put_source_entries(
                    source_cache_id(step.name, target, step.config_hash), fresh
                )
            except Exception as error:
                await context.emit(
                    ItemFailed(
                        uri=target,
                        error=error,
                        step=step.name,
                        source_id=target,
                        event_name=step.event_name,
                    )
                )
                if policy.on_error == "raise":
                    await queue.put(error)

    async def evaluate_all() -> None:
        try:
            await asyncio.gather(*(evaluate(target) for target in targets))
        finally:
            await queue.put(None)

    producer = asyncio.ensure_future(evaluate_all())
    try:
        while (item := await queue.get()) is not None:
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        producer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await producer


async def _hydrate_bucket(
    cache: CacheStore, cache_id: str, entries: Sequence[Source]
) -> list[Source]:
    """Load the content of every header in ``entries`` with one query."""
    headers = [e for e in entries if isinstance(e, _SourceHeader)]
    if not headers:
        return list(entries)
    full = {
        s.id: s
        for s in await cache.get_bucket_sources(cache_id, [h.id for h in headers])
    }
    return [full[e.id] if isinstance(e, _SourceHeader) else e for e in entries]


async def _hydrate_kept(cache: CacheStore, kept: list[Source]) -> list[Source]:
    """Replace lazy headers in ``kept`` by full sources, one query per bucket."""
    by_bucket: dict[str, list[str]] = {}
    for source in kept:
        if isinstance(source, _SourceHeader):
            by_bucket.setdefault(source._bucket, []).append(source.id)
    if not by_bucket:
        return kept
    full: dict[tuple[str, str], Source] = {}
    for bucket, ids in by_bucket.items():
        for loaded in await cache.get_bucket_sources(bucket, ids):
            full[bucket, loaded.id] = replace(loaded, cached=True)
    return [full[s._bucket, s.id] if isinstance(s, _SourceHeader) else s for s in kept]


async def run_source_step(
    step: SourceStep, context: ExecutionContext
) -> AsyncIterator[Source]:
    r"""Evaluate ``step`` through the same wrap ``_run_step`` applies internally.

    ``PipelineStep.__aiter__``/``run()`` always build a *fresh* default
    context (their own policy, no cache) — fine for running a step
    standalone, but wrong for an aggregator (e.g. ``Sources``) driving several
    child ``SourceStep``\ s that should inherit *its* resolved cache and policy
    when they declare none of their own. This exposes the same evaluator with
    an explicit, already-scoped :class:`~fastrag.ingestion.steps.base.ExecutionContext`.
    """
    async for item in _run_step(step, context):
        yield item


async def run_step(
    step: PipelineStep[T], context: ExecutionContext
) -> AsyncIterator[T]:
    """Evaluate any ``step`` through the same wrap ``_run_step`` applies internally.

    The generic counterpart to :func:`~fastrag.ingestion.steps.base.run_source_step`: where
    ``PipelineStep.__aiter__``/``run()`` always build a *fresh* default
    context, this lets a caller supply its own
    :class:`~fastrag.ingestion.steps.base.ExecutionContext` for the whole call — what
    :class:`~fastrag.ingestion.Pipeline` needs to drive its root step (and,
    transitively, every step in its ``upstream`` chain) under one shared
    context carrying the pipeline's own default cache, policy, and event
    sink.
    """
    async for item in _run_step(step, context):
        yield item


class DocumentStep(PipelineStep[Document], ABC):
    """Base step transforming or generating :class:`~fastrag.models.Document` instances.

    Document transformations are pure and deterministic, so a cache hit needs
    no source-freshness revalidation — unlike :class:`~fastrag.ingestion.steps.base.SourceStep`. This class
    provides a concrete :meth:`~PipelineStep.evaluate`: it pulls from
    ``self``'s upstream (bound via ``__call__``) through the evaluator, and
    for each upstream item, either serves a cached result or calls
    :meth:`transform` — concrete subclasses implement only ``transform``, the
    per-item hook, never ``evaluate`` itself.
    """

    def __init__(
        self,
        *,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
    ) -> None:
        """Initialize a ``DocumentStep``.

        Args:
            cache: Cache backend, or ``False``/``None`` — see
                :class:`PipelineStep`.
            policy: Execution policy overriding the pipeline default.
        """
        super().__init__(cache=cache, policy=policy)

    @abstractmethod
    def transform(
        self, item: Any, context: ExecutionContext
    ) -> AsyncIterator[Document]:
        """Transform one upstream item into zero or more output documents.

        Args:
            item: One item from this step's upstream (a
                :class:`~fastrag.ingestion.models.Source` for a parser, or a
                :class:`~fastrag.models.Document` for a chunker/embedder).
            context: Execution-scoped state and services.

        Yields:
            The documents this one upstream item produces.
        """

    #: Set by a step whose cached replay reads the full documents anyway
    #: (e.g. to re-validate them): its cache reads then skip the header-only
    #: projection rather than fetching every entry twice.
    replay_needs_content: bool = False

    def validate_input(self, item: Any) -> None:
        """Validate input before work or cache replay; legacy steps accept all."""
        return None

    async def _on_produced(
        self, item: Any, produced: list[Document], context: ExecutionContext
    ) -> None:
        """Hook: called once per upstream ``item`` once its documents are resolved.

        Fires the same way for a cache hit as for a fresh :meth:`transform`
        call — exactly once per upstream item, with ``produced`` already
        carrying the right ``cached`` flag either way — so a subclass that
        wants to emit a domain event here (chunking, embedding, indexing) gets
        one that reflects reality instead of only ever firing on a miss. A
        no-op by default: :class:`~fastrag.ingestion.steps.parsing.base.Parser`
        instead overrides :meth:`~PipelineStep.evaluate` itself for the same
        uniformity, since it predates this hook.
        """
        return None

    async def evaluate(self, context: ExecutionContext) -> AsyncIterator[Document]:
        """Drive :meth:`transform` over each upstream item, cached per item.

        Raises:
            fastrag.exceptions.ConfigurationError: No upstream is bound —
                call ``step(upstream)`` before evaluating.
        """
        if self._upstream is None:
            raise ConfigurationError(
                f"{type(self).__name__} has no upstream step; "
                f"call step(upstream) before evaluating it"
            )
        async for document in self.evaluate_items(
            _run_step(self._upstream, context), context
        ):
            yield document

    def _cache_lookup(
        self, item: Any, context: ExecutionContext
    ) -> tuple[CacheStore, str] | None:
        """Resolve this item's exact cache owner, including routed children."""
        cache = _resolve_cache(self, context)
        if cache is None:
            return None
        return cache, document_cache_id(self.name, item.id, self.config_hash)

    def _replay_needs_content(self) -> bool:
        """Whether this step, or any step it routes to, reads replayed content."""
        if getattr(self, "replay_needs_content", False):
            return True
        if isinstance(self, RoutedStep):
            children = [step for _, step in self.routes]
            if self._default_step is not None:
                children.append(self._default_step)
            return any(getattr(c, "replay_needs_content", False) for c in children)
        return False

    async def evaluate_items(
        self, items: AsyncIterator[Any], context: ExecutionContext
    ) -> AsyncIterator[Document]:
        """Transform a bounded cache window independently of worker concurrency."""
        needs_content = self._replay_needs_content()
        if context.cache_headers_only and needs_content:
            # Headers would be fetched now and the full entries again right
            # after; read them once (compressed payloads are smaller anyway).
            context = replace(context, cache_headers_only=False)
        cache = _resolve_cache(self, context)
        policy = self.effective_policy(context)
        batch_context = context

        async def process(item: Any) -> list[Document]:
            self.validate_input(item)
            if cache is not None:
                key = document_cache_id(self.name, item.id, self.config_hash)
                cached = (
                    None
                    if self._policy_for_item(item, policy).recreate_cache
                    else await _cached_documents(cache, key, batch_context)
                )
                if cached is not None:
                    produced = [replace(doc, cached=True) for doc in cached]
                else:
                    item = await _materialize_item(item)
                    produced = [
                        doc async for doc in self.transform(item, batch_context)
                    ]
                    await cache.put_document_entries(key, produced)
            else:
                # Routers decide whether their child needs to hydrate the input.
                if not isinstance(self, RoutedStep):
                    item = await _materialize_item(item)
                produced = [doc async for doc in self.transform(item, batch_context)]
            await self._on_produced(item, produced, batch_context)
            return produced

        async def run(stream: AsyncIterator[Any]) -> AsyncIterator[Document]:
            async for produced in run_concurrently(
                stream,
                process,
                context=batch_context,
                policy=policy,
                step=self.name,
                event_name=self.event_name,
                policy_for=lambda item: self._policy_for_item(item, policy),
            ):
                for doc in produced:
                    yield doc

        if cache is None and not isinstance(self, RoutedStep):
            async for doc in run(items):
                yield doc
            return

        async def prepare(
            batch: list[Any],
        ) -> tuple[list[Any], dict[tuple[int, str], list[Document] | None]]:
            """Read one window's cache entries (and inputs, if needed) in bulk."""
            groups: dict[int, tuple[CacheStore, list[str]]] = {}
            for item in batch:
                try:
                    lookup = self._cache_lookup(item, context)
                except Exception:
                    # Selection/dependency failures belong to the item worker/Policy.
                    continue
                if lookup is not None:
                    backend, key = lookup
                    groups.setdefault(id(backend), (backend, []))[1].append(key)
            prefetched: dict[tuple[int, str], list[Document] | None] = {}

            async def prefetch(backend: CacheStore, keys: list[str]) -> None:
                if policy.recreate_cache:
                    return
                try:
                    async with asyncio.timeout(policy.timeout):
                        results = (
                            await backend.get_document_headers_many(keys)
                            if context.cache_headers_only
                            else await backend.get_document_entries_many(keys)
                        )
                    prefetched.update(
                        {(id(backend), key): value for key, value in results.items()}
                    )
                except Exception:
                    # Retry and report cache failures under each item's own policy.
                    pass

            async def materialize() -> None:
                # Inputs replayed as header projections would otherwise each
                # be hydrated by its own round trip during validation.
                nonlocal batch
                try:
                    batch = await _materialize_window(batch)
                except Exception:
                    pass  # Each item hydrates itself, under its own policy.

            await asyncio.gather(
                *(prefetch(backend, keys) for backend, keys in groups.values()),
                *([materialize()] if needs_content else []),
            )
            return batch, prefetched

        # The next window is collected and read while this one is processed,
        # so the cache round trips of successive stages overlap.
        async for batch, prefetched in _read_ahead(
            stream_batches(items, policy.cache_batch_size), prepare
        ):
            batch_context = replace(context, document_cache_entries=prefetched)

            async def window(batch: list[Any] = batch) -> AsyncIterator[Any]:
                for item in batch:
                    yield item

            async for doc in run(window()):
                yield doc


async def _read_ahead[T, R](
    batches: AsyncIterator[T], prepare: Callable[[T], Awaitable[R]]
) -> AsyncIterator[R]:
    """Yield ``prepare(batch)`` for each batch, preparing the next one meanwhile.

    ``batches`` is only ever iterated from one background task, holding at
    most one prepared batch ahead of the consumer. An upstream failure is
    re-raised to the consumer in order, after every batch before it.
    """
    queue: asyncio.Queue[tuple[bool, Any]] = asyncio.Queue(maxsize=1)

    async def produce() -> None:
        try:
            async for batch in batches:
                await queue.put((True, await prepare(batch)))
        except Exception as error:
            await queue.put((False, error))
        else:
            await queue.put((False, None))

    producer = asyncio.ensure_future(produce())
    try:
        while True:
            ok, value = await queue.get()
            if ok:
                yield value
            elif value is None:
                return
            else:
                raise value
    finally:
        producer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await producer


async def _materialize_window(items: list[Any]) -> list[Any]:
    """Replace the document header projections in ``items`` with one read per store."""
    stores: dict[int, tuple[CacheStore, set[str]]] = {}
    for item in items:
        if isinstance(item, _DocumentHeader):
            stores.setdefault(id(item._store), (item._store, set()))[1].add(
                item._bucket
            )
    if not stores:
        return items
    full: dict[tuple[int, str, str], Document] = {}
    for store_id, (store, buckets) in stores.items():
        loaded = await store.get_document_entries_many(sorted(buckets))
        for bucket, entries in loaded.items():
            for entry in entries or []:
                full[store_id, bucket, entry.id] = entry
    return [
        full.get((id(item._store), item._bucket, item.id), item)
        if isinstance(item, _DocumentHeader)
        else item
        for item in items
    ]


async def stream_batches[T](
    items: AsyncIterator[T], size: int
) -> AsyncIterator[list[T]]:
    """Collect a bounded stream window, flushing the final partial batch."""
    batch: list[T] = []
    async for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


async def _materialize_item(item: Any) -> Any:
    """Load projected content only when an uncached transformation needs it."""
    if isinstance(item, (_DocumentHeader, _SourceHeader)):
        return await item.materialize()
    return item


async def _cached_documents(
    cache: CacheStore, key: str, context: ExecutionContext
) -> list[Document] | None:
    """Read a window's stage snapshot, falling back to a normal cache lookup."""
    entries = context.document_cache_entries
    lookup = (id(cache), key)
    if entries is not None and lookup in entries:
        return entries[lookup]
    if context.cache_headers_only:
        return (await cache.get_document_headers_many([key]))[key]
    return await cache.get_document_entries(key)


async def _dispatch_route(
    target: PipelineStep[T], item: T, context: ExecutionContext
) -> AsyncIterator[T]:
    """Feed one item to a routed child step's per-item hook, cached per child.

    Duck-typed on ``transform`` (the hook :class:`DocumentStep` defines):
    every concrete routing target in this codebase is a document-family step.

    Caches by the *matched child's* identity —
    ``document_cache_id(target.name, item.id, target.config_hash)`` — not the
    router's, so routing the same item to a differently configured child
    changes its cache identity rather than serving what another route
    produced. This is the only place a routed item's cache is consulted:
    :class:`RoutedStep` forces its own ``cache`` to ``False`` precisely so
    :class:`DocumentStep`'s generic per-item wrap (which would cache by the
    *router's* identity) never runs for it.
    """
    transform = getattr(target, "transform", None)
    if transform is None:
        raise ConfigurationError(
            f"{type(target).__name__} cannot be used as a RoutedStep target: "
            f"it defines no per-item transform() hook"
        )

    validate = getattr(target, "validate_input", None)
    if validate is not None:
        validate(item)
    cache = _resolve_cache(target, context)
    if cache is None:
        item = await _materialize_item(item)
        async for result in transform(item, context):
            yield result
        return

    cache_id = document_cache_id(target.name, cast(Any, item).id, target.config_hash)
    recreate = target.effective_policy(context).recreate_cache
    cached_entries = (
        None if recreate else await _cached_documents(cache, cache_id, context)
    )
    if cached_entries is not None:
        for entry in cached_entries:
            yield cast(T, replace(entry, cached=True))
        return

    item = await _materialize_item(item)
    produced = [result async for result in transform(item, context)]
    await cache.put_document_entries(cache_id, cast("list[Document]", produced))
    for result in produced:
        yield result


class RoutedStep(PipelineStep[T], ABC):
    """Step that conditionally delegates to child steps based on filter rules.

    A mixin, not independently instantiable: it defines :meth:`route`,
    :meth:`default`, and the per-item dispatch (:meth:`transform`, satisfying
    the hook :class:`DocumentStep` expects), but not
    :meth:`~PipelineStep.evaluate` — a concrete routed step combines this
    with :class:`DocumentStep` (routing first in the MRO, e.g. ``class
    ChunkerRouter(RoutedStep[Document], Chunker)``) so ``evaluate`` resolves to ``DocumentStep``'s concrete
    per-item driver, which calls back into this class's ``transform`` to pick
    a route.

    Deliberately declares no :attr:`~PipelineStep.event_name`, even though it
    sits *first* in a routed step's MRO: routing is not a stage, it is a way
    of choosing one. Leaving it undeclared is what lets ``class
    ChunkerRouter(RoutedStep[Document], Chunker)`` resolve to ``Chunker``'s
    ``"Chunking"`` instead of shadowing it.
    """

    #: Unknown routing overrides conservatively require complete inputs.
    #: Opt out only when built-in selection uses identity/metadata alone.
    routing_requires_content: bool = True

    def __init__(
        self,
        *,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
    ) -> None:
        """Initialize a ``RoutedStep`` with no routes registered yet.

        Args:
            cache: Cache backend, or ``False``/``None`` — see
                :class:`PipelineStep`. Kept separately, as
                ``self._router_cache``, rather than passed through to
                ``super().__init__`` (which always sees ``cache=False``
                instead): a router's own identity must never key a routed
                item's cache entry, only the *matched child's* identity may,
                so :class:`DocumentStep`'s generic per-item cache wrap (which
                would key by ``self``) must never run for a ``RoutedStep`` —
                :meth:`transform`/``_dispatch_route`` do the real, correctly
                keyed caching themselves.
            policy: Execution policy overriding the pipeline default.
        """
        super().__init__(cache=False, policy=policy)
        self._router_cache = cache
        self.routes: list[tuple[Filter[T], PipelineStep[T]]] = []
        self._default_step: PipelineStep[T] | None = None

    def route(self, filter_rule: Filter[T], step: PipelineStep[T]) -> Self:
        """Register a conditional branch step matching a filter predicate.

        Args:
            filter_rule: Filter predicate to match against incoming items.
            step: Child step to run on match.

        Returns:
            Self, for fluent chaining.
        """
        self.routes.append((filter_rule, step))
        return self

    def default(self, step: PipelineStep[T]) -> Self:
        """Register a fallback step for items that match no route filters.

        Args:
            step: Default child step.

        Returns:
            Self, for fluent chaining.
        """
        self._default_step = step
        return self

    def config(self) -> Mapping[str, Any]:
        """No identity-affecting configuration of its own.

        A routed item's cache identity is keyed by whichever child matched
        it (see ``_dispatch_route``), never by the router itself.
        """
        return {}

    def _select(self, item: T) -> PipelineStep[T] | None:
        """The first route whose filter matches ``item``, or the default."""
        for filter_rule, step in self.routes:
            if filter_rule.matches(item):
                return step
        return self._default_step

    def _policy_for_item(self, item: Any, effective: Policy) -> Policy:
        """The matched child's own policy, or this router's.

        The retry/timeout wrap lives in the evaluator driving *this* step, one
        level above the child :meth:`transform` actually dispatches to — so
        without this hook a child's own ``policy`` would be silently ignored
        for exactly the work it was set on. ``DoclingParser(policy=Policy(
        timeout=600))`` routed under a ``ParserRouter`` is the case that
        matters: a conversion that legitimately takes minutes would be
        cancelled by the router's 30-second default and skipped, and the only
        thing distinguishing that from a real parse failure is a
        ``TimeoutError``.

        Only the per-item knobs are taken from the child (see
        :func:`run_concurrently`'s ``policy_for``); ``concurrency`` stays the
        router's, since one stream has one concurrency limit however many
        children it fans out to.
        """
        try:
            target = self._select(item)
        except Exception:
            # transform reports lazy routing errors under the effective policy.
            return effective
        if target is None or target.policy is None:
            return effective
        return target.policy.merge(effective)

    def _cache_lookup(
        self, item: Any, context: ExecutionContext
    ) -> tuple[CacheStore, str] | None:
        """Prefetch by the selected child's identity and effective backend."""
        target = self._select(item)
        if target is None:
            return None
        scoped = replace(
            context,
            cache=_resolve_cache_value(self._router_cache, context.cache),
            policy=self.effective_policy(context),
        )
        lookup = getattr(target, "_cache_lookup", None)
        return None if lookup is None else lookup(item, scoped)

    async def transform(self, item: T, context: ExecutionContext) -> AsyncIterator[T]:
        """Dispatch ``item`` to the first matching route, or the default.

        An item matching no route with no default registered is skipped with
        an ``ItemFailed`` naming it — never silently dropped. A matched
        child that declares no explicit cache/policy of its own (``cache is
        None``) inherits this routed step's effective cache and policy rather
        than the outer pipeline's, since a routed child is conceptually
        scoped to its router. A child that *does* declare a policy has it
        applied to its own items, retries and timeout included — see
        ``_policy_for_item``, which is where that half happens, since the
        retry wrap sits above this method rather than inside it.
        """
        target = self._select(item)
        if target is None:
            await context.emit(
                ItemFailed(
                    uri=_item_identifier(item),
                    error=ConfigurationError(
                        f"no route matched {_item_identifier(item)!r} "
                        f"and no default step is registered"
                    ),
                    step=self.name,
                    source_id=_lineage_key(item),
                    event_name=self.event_name,
                )
            )
            return

        scoped = replace(
            context,
            cache=_resolve_cache_value(self._router_cache, context.cache),
            policy=self.effective_policy(context),
        )
        async for result in _dispatch_route(target, item, scoped):
            yield result
