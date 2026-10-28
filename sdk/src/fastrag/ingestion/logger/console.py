"""``ConsoleLogger``: renders a pipeline's event stream to the console.

``docs/specification/logging.md`` / ``docs/specification/API.md``'s "Example
output": **one row per leaf fetcher** (``LocalFileFetcher``, ``WebCrawler``,
``SitemapCrawler``, ``S3Fetcher``, ...), one column per pipeline stage
(Fetching, Parsing, Chunking, Embedding, Indexing, ...), each cell
``new/cached/errors``.

Two separate questions have to be answered to fill in a cell: *which column*
(which stage produced this event) and *which row* (which fetcher did the item
ultimately come from). They are solved differently, because the information
each needs is only available in a different place:

- **Column** is whatever the emitting step's
  :attr:`~fastrag.ingestion.steps.base.PipelineStep.event_name` resolves to,
  read straight off the event (every event carries it). A step family declares
  its own label
  (``Chunker.event_name = "Chunking"``), a step that overrides it is columned
  under its override, and a step whose family declares none falls back to its
  own class name — so a graph with a stage this SDK has never heard of grows
  a column for it rather than dropping its events. Columns are *seeded* in
  pipeline order by walking :attr:`~fastrag.ingestion.Pipeline.root`'s
  ``upstream`` chain from the source back down to the root. Columns are then
  ordered by ``column_order``, defaulting to Fetching, Parsing, Chunking,
  Embedding, Indexing. Unlisted labels follow in discovery order, including
  labels first seen on events from routed children. A step that
  emits no events at all still gets its (empty) column from that seeding.
- **Row** is the name of the leaf :class:`~fastrag.ingestion.steps.base.SourceStep`
  (fetcher) an item was originally produced by — knowable structurally only
  at the moment of fetching. ``fastrag.ingestion.steps.base._run_step`` stamps
  it there, into the reserved ``Source.metadata["source_step"]`` key, onto
  every ``Source`` that doesn't already carry one (so nested composition — a
  :class:`~fastrag.ingestion.steps.fetching.sources.Sources` aggregate
  wrapping several fetchers — never overwrites the innermost, correct stamp
  with the aggregate's own name). This module enumerates the known row names
  once at construction, by walking ``Sources.sources`` recursively
  (``_leaf_fetcher_names``) if the graph's source-most step is a ``Sources``
  aggregate, or taking that step's own name otherwise.

  A ``Document`` five stages downstream never re-reads that metadata key
  directly (parsing/chunking/embedding never touch it, so a *stale* parse or
  chunk cache entry from before this reserved key existed would carry none of
  it). Instead, this module tracks a live ``Source.id -> row name`` map,
  populated as each ``DocumentFetched`` arrives, and looks a ``Document`` up
  by its own ``source_id`` field — which every downstream step already
  carries forward unchanged — so attribution survives regardless of which
  stage's cache happens to be warm.

Rows are named after a step *instance* (``ItemFailed.step``,
``metadata["source_step"]``) and columns after a step *stage*
(``event_name``) — the two never collide, because one fetcher's row and the
``Fetching`` column answer different questions about the same event.

``new``/``cached`` are read directly off the item every domain event carries
(``DocumentFetched.source``, ``DocumentParsed``/``DocumentChunked``/
``DocumentEmbedded``/``DocumentSaved``'s ``.document``) — every one of these
events now fires the same way for a cache hit as for fresh work (see
``fastrag.ingestion.steps.base.DocumentStep._on_produced``), so
``item.cached`` is trustworthy on every event this module sees. A
``DocumentFetched`` is counted at most once per ``Source.id``: nested
``SourceStep`` composition emits it once per level (the leaf fetcher's own,
and again when the enclosing ``Sources`` aggregate is itself driven) — by
design, the same way lifecycle events double up — so this module dedupes
rather than double-counting the same fetch.

``errors`` comes from ``ItemFailed``, columned by its ``event_name`` like any
other event. Its row is resolved from its ``step`` when that directly names a
known leaf fetcher (a fetch-stage failure — a fetcher's own
``run_concurrently`` call always names itself), and otherwise from its
``source_id`` through the same live map ``DocumentFetched`` populates (a
parser/chunker/embedder/sink step is shared by every row, so only the item it
was processing says which one). A failure this module cannot attribute to a
row either way (an empty ``step`` and an unknown ``source_id``) is not
silently dropped: it is counted separately and surfaced as a trailing line
once the run ends, so an attribution gap is visible rather than invisible.

A graph whose source-most step is not a :class:`~fastrag.ingestion.steps.base.SourceStep`
at all (i.e. cannot be introspected — some hand-built test double) falls back
to the single-row behavior this module always had: every event lands in one
row, keyed ``_UNATTRIBUTED_ROW``, rather than crashing.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import sys
import threading
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from fastrag.exceptions import describe_error
from fastrag.ingestion.events import (
    DocumentChunked,
    DocumentEmbedded,
    DocumentFetched,
    DocumentParsed,
    DocumentSaved,
    DocumentsPruned,
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
from fastrag.ingestion.logger.base import Logger
from fastrag.ingestion.pipeline import Pipeline
from fastrag.ingestion.steps.base import PipelineStep, SourceStep

#: Row key used for an event this module cannot attribute to a known fetcher
#: row — either because the pipeline's root has no ``SourceStep`` anywhere in
#: its ``upstream`` chain at all (e.g. a graph built entirely of in-process
#: test doubles), or, on a graph that does, because a specific item's lineage
#: could not be resolved. Keeps the table renderable rather than crashing or
#: dropping the count.
_UNATTRIBUTED_ROW = "(unattributed)"

#: Column label for an event carrying neither an ``event_name`` nor a
#: ``step`` — only reachable for an event constructed by hand, since every
#: step stamps both (see ``fastrag.ingestion.events``).
_UNATTRIBUTED_COLUMN = "(unattributed)"

#: What an empty cell renders as: a row that never saw a given stage still
#: has to print something under that column.
_EMPTY_CELL_TEXT = "0/0/0"

#: Identical failures (same step, same error) printed live before the rest
#: are only counted.
_REPEATED_ERRORS_SHOWN = 3

_DEFAULT_COLUMN_ORDER = ("Fetching", "Parsing", "Chunking", "Embedding", "Indexing")


@dataclass
class _Cell:
    """One table cell's running counts."""

    new: int = 0
    cached: int = 0
    errors: int = 0

    def as_text(self) -> str:
        return f"{self.new}/{self.cached}/{self.errors}"


@dataclass
class _InFlight:
    """One unit of work between its ``ItemStarted`` and ``ItemFinished``."""

    event: ItemStarted
    started: float
    attempt: int = 1
    reported_slow: bool = False


@dataclass
class _StageTiming:
    """Per-stage wall-clock statistics over finished items."""

    count: int = 0
    total: float = 0.0
    slowest: list[tuple[float, str]] = field(default_factory=list)

    def add(self, elapsed: float, label: str) -> None:
        self.count += 1
        self.total += elapsed
        self.slowest.append((elapsed, label))
        self.slowest.sort(reverse=True)
        del self.slowest[3:]


def format_duration(seconds: float) -> str:
    """``8.2s``, ``2m05s``, ``1h02m``: short, fixed-ish width, human readable."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _brief_error(error: BaseException) -> str:
    """``describe_error``'s first line, bounded.

    A driver error can embed a whole SQL statement and its parameters,
    burying the progress output.
    """
    lines = describe_error(error).splitlines() or [""]
    text = lines[0] + (" […]" if len(lines) > 1 else "")
    return _shorten(text, 300)


def _shorten(text: str, limit: int = 110) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _walk_chain(root: PipelineStep[Any]) -> list[PipelineStep[Any]]:
    """Walk ``root``'s ``upstream`` chain, source-most step first.

    Follows :attr:`~fastrag.ingestion.steps.base.PipelineStep.upstream` from
    ``root`` back to whichever step has none, then reverses the result so
    index 0 is the source-most step and the last entry is ``root`` itself.
    """
    chain: list[PipelineStep[Any]] = []
    step: PipelineStep[Any] | None = root
    seen: set[int] = set()
    while step is not None and id(step) not in seen:
        chain.append(step)
        seen.add(id(step))
        step = step.upstream
    chain.reverse()
    return chain


def _leaf_fetcher_names(step: PipelineStep[Any]) -> list[str]:
    r"""Enumerate the leaf fetcher names ``step`` ultimately fetches through.

    Duck-typed on ``.sources`` — only
    :class:`~fastrag.ingestion.steps.fetching.sources.Sources` defines it,
    the same way ``targets``/``is_source_modified`` are duck-typed elsewhere
    in this package. A step without it (a plain leaf
    :class:`~fastrag.ingestion.steps.base.SourceStep`, e.g. a fetcher) is
    itself the only leaf. Recurses so a ``Sources`` aggregate nested inside
    another ``Sources`` still expands to its own leaves, depth-first, in
    ``sources``'s declared order — this is what gives ``08_sources.py``'s
    ``Sources(sources=[SitemapCrawler(...), WebCrawler(...)])`` exactly the
    two rows ``["SitemapCrawler", "WebCrawler"]``, pre-registered before a
    single event has arrived.
    """
    children = getattr(step, "sources", None)
    if children is None:
        return [step.name]
    names: list[str] = []
    for child in children:
        names.extend(_leaf_fetcher_names(child))
    return names


def _classify(root: PipelineStep[Any]) -> tuple[bool, list[str], list[str]]:
    """Return whether a source step was found, the known row names, and the seed columns.

    Columns are every distinct ``event_name`` along the walked chain, in
    pipeline order — the stage labels the steps in this graph declare for
    themselves. Deduped, because two steps of
    the same family (two chunkers in a row) share one column.

    Rows need more than the chain: when its source-most step is not actually
    a :class:`~fastrag.ingestion.steps.base.SourceStep` (or the chain is
    empty — some hand-built test double), there is no supported row mapping
    at all, and every event collapses onto the single fallback row, matching
    this module's original single-row behavior exactly. The columns are still
    seeded either way.
    """
    chain = _walk_chain(root)
    columns: list[str] = []
    for step in chain:
        label = getattr(step, "event_name", "") or step.name
        if label not in columns:
            columns.append(label)
    if not chain or not isinstance(chain[0], SourceStep):
        return False, [], columns
    return True, _leaf_fetcher_names(chain[0]), columns


class ConsoleLogger(Logger):
    """Renders a pipeline's lifecycle and domain events to the console.

    Aggregates item-level events into the per-source, per-stage progress
    table ``docs/specification/API.md``'s "Example output" shows — one row per
    leaf fetcher, one column per stage, each cell ``new/cached/errors`` —
    refreshed in place on a TTY, or printed once at the end otherwise (piping
    to a file must never see escape sequences).
    """

    show_progress: bool
    show_domain_events: bool
    show_errors: bool
    refresh_interval: float

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        show_progress: bool = True,
        show_domain_events: bool = False,
        show_errors: bool = False,
        refresh_interval: float = 0.5,
        column_order: Sequence[str] | None = None,
        stream: Any = None,
        heartbeat_interval: float = 15.0,
        slow_after: float = 60.0,
        stall_after: float = 30.0,
    ) -> None:
        """Build a console renderer for ``pipeline``.

        Args:
            pipeline: The pipeline to consume, already entered.
            show_progress: Whether to render the per-source/per-stage table
                at all.
            show_domain_events: Whether to also print one compact line per
                domain event (``DocumentFetched``, ``DocumentParsed``, ...,
                ``ItemFailed``) as it arrives — independent of the table.
            show_errors: Whether to list every ``ItemFailed`` after the final
                table, one line each, naming the stage, the originating source
                URL and the error — so the failing sources can be copied into
                a filter to re-run just them. Printed at the end (never
                mid-run) so the live table redraw is not corrupted.
            refresh_interval: Minimum seconds between live table redraws on a
                TTY. Ignored when output is not a TTY, where the table is
                only ever printed once, at the end.
            column_order: Preferred stage labels, in display order. Defaults
                to Fetching, Parsing, Chunking, Embedding, Indexing. Only
                discovered stages are displayed; unlisted stages follow in
                discovery order. Pass an empty sequence for discovery order.
            stream: Where to write output. Defaults to ``sys.stdout``.
            heartbeat_interval: Seconds between status lines when output is
                not a TTY (a TTY shows the same status live under the
                table). Each one gives elapsed time, per-stage counts, how
                many items are in flight and the oldest of them, so a piped
                run is never silent for longer than this. ``0`` disables it.
            slow_after: An item in flight for longer than this many seconds
                is listed by name (with its stage, attempt and age) until it
                finishes — the way to spot a stuck request long before its
                ``Policy.timeout`` fires. When it does finish, one line
                reports how long it took.
            stall_after: If the event loop itself makes no progress for this
                many seconds (synchronous work blocking it, which freezes
                every timeout and this logger's own output), a watchdog
                thread prints the loop thread's current stack, once per
                stall. ``0`` disables it.
        """
        super().__init__(pipeline)
        self.show_progress = show_progress
        self.show_domain_events = show_domain_events
        self.show_errors = show_errors
        self.refresh_interval = refresh_interval
        self._stream = stream if stream is not None else sys.stdout

        self._rows_known, row_names, self._columns = _classify(pipeline.root)
        order = _DEFAULT_COLUMN_ORDER if column_order is None else column_order
        self._column_order = {label: i for i, label in enumerate(dict.fromkeys(order))}
        self._sort_columns()
        if not self._rows_known:
            row_names = [_UNATTRIBUTED_ROW]
        self._rows: dict[str, dict[str, _Cell]] = {name: {} for name in row_names}
        #: ``Source.id -> row name``, populated as each ``DocumentFetched``
        #: arrives — lets a downstream ``Document`` (whose own metadata may
        #: predate the ``source_step`` reserved key, e.g. a stale parse
        #: cache) still be attributed by its stable ``source_id`` field.
        self._row_of_source_id: dict[str, str] = {}
        #: ``Source.id``\ s already counted in the fetching column — nested
        #: ``Sources`` composition emits ``DocumentFetched`` once per level
        #: for the same ``Source``, and only the first should be counted.
        self._counted_fetches: set[str] = set()
        self._unattributed_failures = 0
        #: ``Source.id -> Source.uri``, for naming the source URL of a failure.
        self._uri_of_source_id: dict[str, str] = {}
        self._failures: list[ItemFailed] = []
        self._pruned: list[DocumentsPruned] = []
        self._last_render_line_count = 0
        self._rendered_once = False
        self.heartbeat_interval = heartbeat_interval
        self.slow_after = slow_after
        self._in_flight: dict[int, _InFlight] = {}
        self._failure_counts: dict[tuple[str, str], int] = {}
        self._last_event = "none yet"
        self._retry_counts: dict[tuple[str, str], int] = {}
        self._timings: dict[str, _StageTiming] = {}
        self._started_at = time.monotonic()
        self._last_event_at = self._started_at
        self.stall_after = stall_after
        self._loop_alive_at = self._started_at
        self._lock = threading.Lock()

    def _row_for(self, item: Any) -> str:
        """The row ``item`` (a ``Source`` or ``Document``) belongs to.

        Tries ``item.metadata["source_step"]`` first (always present and
        correct for a ``Source``, freshly stamped every run regardless of its
        own cache status — see ``fastrag.ingestion.steps.base._run_step``).
        Falls back to the live ``source_id -> row`` map (built from
        ``DocumentFetched`` events) keyed by ``item.source_id`` (a
        ``Document``) or ``item.id`` (a ``Source`` accessed generically) —
        the path a ``Document`` whose own metadata predates this key, or
        never carried it in the first place, needs.
        """
        if not self._rows_known:
            return _UNATTRIBUTED_ROW
        metadata = getattr(item, "metadata", None)
        if isinstance(metadata, dict):
            candidate = metadata.get("source_step")
            if isinstance(candidate, str) and candidate in self._rows:
                return candidate
        source_id = getattr(item, "source_id", None) or getattr(item, "id", None)
        if source_id is not None:
            candidate = self._row_of_source_id.get(str(source_id))
            if candidate is not None:
                return candidate
        return _UNATTRIBUTED_ROW

    def _add_column(self, column: str) -> str:
        """Register ``column`` as a table column if it is not one already.

        This is the whole of "a new step's events grow a new column": a label
        the constructor's structural seeding never saw is placed according
        to the preferred order, and every existing row starts
        rendering an empty cell under it.
        """
        if column not in self._columns:
            self._columns.append(column)
            self._sort_columns()
        return column

    def _sort_columns(self) -> None:
        """Apply the preferred order, preserving discovery order for other labels."""
        self._columns.sort(
            key=lambda label: self._column_order.get(label, len(self._column_order))
        )

    def _cell(self, row: str, column: str) -> _Cell:
        """The cell at ``row``/``column``, creating both if this is their first event."""
        self._add_column(column)
        return self._rows.setdefault(row, {}).setdefault(column, _Cell())

    @staticmethod
    def _column_of(event: Any) -> str:
        """The stage label to file ``event`` under.

        ``event_name`` for anything a real step emitted. A hand-built event
        naming only the step it came from is columned under that instead of
        being dropped, and one naming neither lands in a visible catch-all
        column rather than an unnamed one.
        """
        return (
            getattr(event, "event_name", "")
            or getattr(event, "step", "")
            or _UNATTRIBUTED_COLUMN
        )

    def _handle(self, event: PipelineEvent) -> None:
        match event:
            case StepStarted() | StepCompleted():
                # A step that produced no items at all still gets its column
                # (the seeding at construction only covers the `upstream`
                # chain -- a routed child, or a step driven some other way,
                # announces itself here).
                self._add_column(self._column_of(event))
            case DocumentFetched(source=source):
                row = self._row_for(source)
                self._row_of_source_id.setdefault(source.id, row)
                self._uri_of_source_id.setdefault(source.id, source.uri)
                if source.id not in self._counted_fetches:
                    self._counted_fetches.add(source.id)
                    self._record(self._cell(row, self._column_of(event)), source.cached)
            case DocumentChunked(document=document, chunks=chunks):
                cell = self._cell(self._row_for(document), self._column_of(event))
                for chunk in chunks:
                    self._record(cell, chunk.cached)
            case (
                DocumentParsed(document=document)
                | DocumentEmbedded(document=document)
                | DocumentSaved(document=document)
            ):
                cell = self._cell(self._row_for(document), self._column_of(event))
                self._record(cell, document.cached)
            case ItemFailed():
                self._failures.append(event)
                self._handle_failure(event)
            case DocumentsPruned():
                self._pruned.append(event)
            case ItemStarted():
                self._in_flight[event.item_id] = _InFlight(event, time.monotonic())
            case ItemRetrying():
                flight = self._in_flight.get(event.item_id)
                if flight is not None:
                    flight.attempt = event.attempt + 1
                    self._notify_retry(flight.event, event)
            case ItemFinished():
                flight = self._in_flight.pop(event.item_id, None)
                if flight is not None:
                    stage = flight.event.event_name or flight.event.step or "?"
                    label = self._describe(flight.event, with_stage=False)
                    self._timings.setdefault(stage, _StageTiming()).add(
                        event.elapsed, label
                    )
                    if flight.reported_slow and event.error is None:
                        self._notify(
                            f"[slow]     {self._describe(flight.event)} finished "
                            f"after {format_duration(event.elapsed)}"
                        )
            case _:
                pass

    def _describe(self, event: ItemStarted, *, with_stage: bool = True) -> str:
        """``Parsing https://…/page (item: chunk-id)``: what an item is, for a line."""
        source_uri = self._uri_of_source_id.get(event.source_id) or event.uri
        item = f" (item: {event.uri})" if event.uri and event.uri != source_uri else ""
        text = _shorten(f"{source_uri}{item}")
        if not with_stage:
            return text
        return f"{event.event_name or event.step or '?'} {text}"

    def _notify(self, line: str) -> None:
        """Print one event line now, above the live table on a TTY."""
        if self._is_tty() and self._rendered_once:
            self._stream.write(f"\x1b[{self._last_render_line_count}A\x1b[J")
            self._rendered_once = False
            self._write(line)
            self._render()
        else:
            self._write(line)

    def _failure_key(self, event: ItemFailed) -> tuple[str, str]:
        return event.step or "?", self._generic_error(event.error, event)

    def _generic_error(self, error: Exception, event: Any) -> str:
        """``describe_error`` with the item's own URIs elided, so repeats group."""
        text = _brief_error(error)
        for uri in (self._uri_of_source_id.get(event.source_id), event.uri):
            if uri:
                text = text.replace(uri, "<item>")
        return text

    def _notify_retry(self, started: ItemStarted, event: ItemRetrying) -> None:
        """Print a retry as it happens; repeats of one error are only counted."""
        key = (started.step or "?", self._generic_error(event.error, started))
        seen = self._retry_counts[key] = self._retry_counts.get(key, 0) + 1
        if seen <= _REPEATED_ERRORS_SHOWN:
            self._notify(
                f"[retry]    {self._describe(started)}: attempt {event.attempt} "
                f"failed ({_brief_error(event.error)}); "
                f"retrying in {format_duration(event.delay)}"
            )
        elif seen == _REPEATED_ERRORS_SHOWN + 1:
            self._notify(
                f'[retry]    {key[0]}: more retries after "{key[1]}"; '
                "further ones are counted and summarized at the end"
            )

    def _notify_failure(self, event: ItemFailed) -> None:
        """Print a failure as it happens; repeats of one error are only counted.

        A misconfiguration (a bad API key, an unreachable service) fails
        every item the same way: the first few lines say everything, and
        hundreds more would bury the progress output.
        """
        key = self._failure_key(event)
        seen = self._failure_counts[key] = self._failure_counts.get(key, 0) + 1
        if seen <= _REPEATED_ERRORS_SHOWN:
            self._notify(self._failure_line(event, "[error]   "))
        elif seen == _REPEATED_ERRORS_SHOWN + 1:
            self._notify(
                f"[error]    {key[0]}: the error above keeps repeating; "
                "further occurrences are counted and listed at the end"
            )

    def _status_lines(self) -> list[str]:
        """Elapsed time, per-stage totals, in-flight count and the slow items."""
        now = time.monotonic()
        totals: dict[str, _Cell] = {}
        for cells in self._rows.values():
            for column, cell in cells.items():
                total = totals.setdefault(column, _Cell())
                total.new += cell.new
                total.cached += cell.cached
                total.errors += cell.errors
        stages = "  ".join(
            f"{column} {totals.get(column, _Cell()).as_text()}"
            for column in self._columns
        )
        flights = sorted(self._in_flight.values(), key=lambda f: f.started)
        if flights:
            oldest = flights[0]
            activity = (
                f"{len(flights)} in flight, oldest "
                f"{format_duration(now - oldest.started)} "
                f"({oldest.event.event_name or oldest.event.step})"
            )
        else:
            # Nothing is being worked on item by item: a step is between
            # items (bulk cache reads/writes, batching). The last event says
            # where the pipeline got to.
            activity = (
                f"idle {format_duration(now - self._last_event_at)}, "
                f"last: {self._last_event}"
            )
        lines = [
            f"[{format_duration(now - self._started_at):>7}] {stages} | {activity}"
        ]
        slow = [f for f in flights if now - f.started >= self.slow_after]
        for flight in slow[:3]:
            flight.reported_slow = True
            attempt = f" attempt {flight.attempt}" if flight.attempt > 1 else ""
            lines.append(
                f"          slow {format_duration(now - flight.started):>7}"
                f"{attempt}: {self._describe(flight.event)}"
            )
        if len(slow) > 3:
            lines.append(f"          … and {len(slow) - 3} more slow items")
        return lines

    def _timing_summary(self) -> list[str]:
        """Per-stage item counts, mean and slowest durations, for the end of a run."""
        lines = [
            f"Retries: {count} x {step}: {error}"
            for (step, error), count in self._retry_counts.items()
        ]
        if not self._timings:
            return lines
        lines.append(
            f"Timing (total {format_duration(time.monotonic() - self._started_at)}):"
        )
        for stage, timing in self._timings.items():
            mean = timing.total / timing.count
            lines.append(
                f"  {stage}: {timing.count} items, mean {format_duration(mean)}, "
                f"max {format_duration(timing.slowest[0][0])}"
            )
            for elapsed, label in timing.slowest:
                if elapsed >= 1.0:
                    lines.append(f"    {format_duration(elapsed):>7}  {label}")
        return lines

    def _handle_failure(self, event: ItemFailed) -> None:
        """Attribute one ``ItemFailed`` to a row/column cell, or count it unattributed.

        The column is the failing step's stage, like any other event. Only
        the row needs work: a ``step`` naming a known leaf fetcher directly
        (a fetch-stage failure — a fetcher's own concurrency always names
        itself, never the aggregate wrapping it) already *is* the row, while
        a step shared by every row (a parser/chunker/embedder/sink) needs
        ``source_id`` to say which one.
        """
        if self._rows_known and event.step in self._rows:
            row: str | None = event.step
        else:
            row = self._row_of_source_id.get(event.source_id) or None
        if row is None:
            self._unattributed_failures += 1
            return
        self._cell(row, self._column_of(event)).errors += 1

    @staticmethod
    def _record(cell: _Cell, cached: bool) -> None:
        if cached:
            cell.cached += 1
        else:
            cell.new += 1

    def _cell_text(self, row: str, column: str) -> str:
        """``row``/``column``'s counts, or an empty cell if that pairing never fired."""
        cell = self._rows[row].get(column)
        return cell.as_text() if cell is not None else _EMPTY_CELL_TEXT

    def _render_table(self) -> str:
        headers = ("Source", *self._columns)
        rows = [
            (name, *(self._cell_text(name, column) for column in self._columns))
            for name in sorted(self._rows)
        ]
        widths = [
            max(len(headers[i]), *(len(row[i]) for row in rows))
            if rows
            else len(headers[i])
            for i in range(len(headers))
        ]
        separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

        def format_row(cells: tuple[str, ...]) -> str:
            return (
                "| "
                + " | ".join(cell.ljust(width) for cell, width in zip(cells, widths))
                + " |"
            )

        lines = [separator, format_row(headers), separator]
        lines.extend(format_row(row) for row in rows)
        lines.append(separator)
        lines.append("Legend: new/cached/errors")
        return "\n".join(lines)

    def _is_tty(self) -> bool:
        isatty = getattr(self._stream, "isatty", None)
        return bool(isatty and isatty())

    def _write(self, text: str) -> None:
        with self._lock:
            self._stream.write(text)
            self._stream.write("\n")
            flush = getattr(self._stream, "flush", None)
            if flush is not None:
                flush()

    def _render(self, *, final: bool = False) -> None:
        table = self._render_table()
        if self._is_tty() and not final:
            table += "\n" + "\n".join(self._status_lines())
            # A line wider than the terminal wraps onto several rows, but the
            # cursor-up below counts lines: clip so the two always agree, or
            # each redraw leaves the previous render's top rows behind.
            width = shutil.get_terminal_size().columns
            table = "\n".join(
                line if len(line) < width else line[: width - 2] + "…"
                for line in table.split("\n")
            )
        if self._is_tty() and self._rendered_once:
            # Move the cursor back to the top of the previous render and
            # clear everything below it, then redraw — an in-place refresh
            # rather than an ever-growing scroll of tables. Only ever used
            # against a real TTY: a piped/redirected stream never sees these
            # escape sequences, so `| tee` output stays plain, readable text.
            self._stream.write(f"\x1b[{self._last_render_line_count}A\x1b[J")
        self._write(table)
        self._last_render_line_count = table.count("\n") + 1
        self._rendered_once = True

    def _failure_line(self, event: ItemFailed, tag: str) -> str:
        """One failure line naming the stage, source URL, item and error."""
        source_uri = self._uri_of_source_id.get(event.source_id) or event.uri
        item = f" (item: {event.uri})" if event.uri and event.uri != source_uri else ""
        return (
            f"{tag} {event.step or '?'}: {source_uri}{item}: "
            f"{_brief_error(event.error)}"
        )

    @staticmethod
    def _uri_of(document: Any) -> str:
        """The provenance URI carried by a ``Source``/``Document``, for a print line."""
        metadata = getattr(document, "metadata", {})
        return str(metadata.get("source_uri", getattr(document, "uri", document.id)))

    @classmethod
    def _marker(cls, item: Any) -> str:
        return "cached" if item.cached else "new"

    def _print_domain_event(self, event: PipelineEvent) -> None:
        match event:
            case DocumentFetched(source=source):
                marker = f" ({self._marker(source)})"
                self._write(f"[fetched]  {source.id}  {source.uri}{marker}")
            case DocumentParsed(document=document):
                marker = f" ({self._marker(document)})"
                self._write(
                    f"[parsed]   {document.id}  {self._uri_of(document)}{marker}"
                )
            case DocumentChunked(document=document, chunks=chunks):
                marker = " (cached)" if chunks and all(c.cached for c in chunks) else ""
                self._write(
                    f"[chunked]  {len(chunks):>6}  {self._uri_of(document)}{marker}"
                )
            case DocumentEmbedded(document=document):
                marker = f" ({self._marker(document)})"
                self._write(
                    f"[embedded] {document.id}  {self._uri_of(document)}{marker}"
                )
            case DocumentSaved(document=document):
                marker = f" ({self._marker(document)})"
                self._write(
                    f"[saved]    {document.id}  {self._uri_of(document)}{marker}"
                )
            case ItemFailed():
                self._write(self._failure_line(event, "[failed]  "))
            case StepFailed(step=step, error=error):
                self._write(f"[step failed] {step}: {_brief_error(error)}")
            case DocumentsPruned():
                self._write(f"[pruned]   {self._pruned_line(event)}")
            case _:
                pass

    @staticmethod
    def _pruned_line(event: DocumentsPruned) -> str:
        """What a sync deleted, or why it kept its stale documents."""
        if event.deleted or not event.stale:
            return f"{event.step}: deleted {event.deleted} stale documents"
        return (
            f"{event.step}: kept {event.stale} stale documents because "
            f"{event.failures} items failed; a clean run deletes them"
        )

    async def run(self) -> None:
        """Consume the pipeline's event stream and render progress as it arrives.

        Iterating ``self.pipeline`` both drives the root step to completion
        and yields each event; a step failure under ``policy.on_error ==
        "raise"`` propagates out of this call the same way it does out of
        ``Pipeline.run()``, after every event already produced has been
        handled. If any ``ItemFailed`` could not be attributed to a known
        row, a trailing summary line is written after the final render so
        that failure is never silently dropped.

        A background ticker keeps the output alive while no event arrives:
        on a TTY it redraws the table and status every ``refresh_interval``;
        otherwise it prints a status line every ``heartbeat_interval``.
        Failures are printed as they happen when ``show_errors`` is set, and
        listed again at the end.
        """
        is_tty = self._is_tty()
        self._started_at = self._last_event_at = time.monotonic()
        ticker = asyncio.ensure_future(self._tick(is_tty))
        alive = asyncio.ensure_future(self._keep_alive())
        done = threading.Event()
        if self.stall_after > 0:
            threading.Thread(
                target=self._watchdog,
                args=(threading.get_ident(), done),
                name="ConsoleLogger-watchdog",
                daemon=True,
            ).start()
        last_render = 0.0
        interrupted: BaseException | None = None
        try:
            async for event in self.pipeline:
                self._last_event_at = time.monotonic()
                step = getattr(event, "step", "")
                if step and not isinstance(event, StepProgress):
                    self._last_event = f"{type(event).__name__} ({step})"
                self._handle(event)
                if self.show_domain_events:
                    self._print_domain_event(event)
                elif self.show_errors and isinstance(event, ItemFailed):
                    self._notify_failure(event)
                if self.show_progress and is_tty:
                    now = time.monotonic()
                    if now - last_render >= self.refresh_interval:
                        self._render()
                        last_render = now
        except BaseException as error:
            # Ctrl-C, a timeout around the run, or on_error="raise": the
            # summary is most needed precisely when the run did not finish.
            interrupted = error
            raise
        finally:
            done.set()
            for task in (ticker, alive):
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            self._summarize(is_tty, interrupted)

    def _summarize(self, is_tty: bool, interrupted: BaseException | None) -> None:
        """Final table, timings, retries and grouped errors."""
        if interrupted is not None:
            reason = (
                "interrupted"
                if isinstance(interrupted, (asyncio.CancelledError, KeyboardInterrupt))
                else f"stopped by {_brief_error(interrupted)}"
            )
            elapsed = format_duration(time.monotonic() - self._started_at)
            self._write(f"Run {reason} after {elapsed}; partial results:")
            for line in self._status_lines()[1:]:
                self._write(line)
        if self.show_progress:
            if is_tty and self._rendered_once:
                self._stream.write(f"\x1b[{self._last_render_line_count}A\x1b[J")
                self._rendered_once = False
            self._render(final=True)
            for line in self._timing_summary():
                self._write(line)
        for event in self._pruned:
            self._write(f"Sync: {self._pruned_line(event)}")
        if self.show_errors and self._failures:
            self._write(f"Errors ({len(self._failures)}):")
            groups: dict[tuple[str, str], list[ItemFailed]] = {}
            for failure in self._failures:
                groups.setdefault(self._failure_key(failure), []).append(failure)
            for (step, error), failures in groups.items():
                if len(failures) == 1:
                    self._write(self._failure_line(failures[0], "[error]"))
                    continue
                self._write(f"[error] {step}: {error} ({len(failures)} items):")
                for failure in failures:
                    uri = self._uri_of_source_id.get(failure.source_id) or failure.uri
                    item = (
                        f" (item: {failure.uri})"
                        if failure.uri and failure.uri != uri
                        else ""
                    )
                    self._write(f"    {uri}{item}")
        if self._unattributed_failures:
            self._write(
                f"Unattributed failures: {self._unattributed_failures} "
                "(ItemFailed whose item could not be traced to a table row)"
            )

    async def _keep_alive(self) -> None:
        """Prove to the watchdog thread that the event loop is still turning."""
        while True:
            self._loop_alive_at = time.monotonic()
            await asyncio.sleep(1.0)

    def _watchdog(self, loop_thread: int, done: threading.Event) -> None:
        """Print the loop thread's stack whenever the event loop is blocked."""
        reported_at = 0.0
        while not done.wait(1.0):
            blocked = time.monotonic() - self._loop_alive_at
            if blocked < self.stall_after or self._loop_alive_at == reported_at:
                continue
            reported_at = self._loop_alive_at
            frame = sys._current_frames().get(loop_thread)
            stack = "".join(traceback.format_stack(frame)[-12:]) if frame else ""
            self._write(
                f"[stall]    event loop blocked for {format_duration(blocked)} "
                f"(synchronous work on the loop thread); it is at:\n{stack.rstrip()}"
            )

    async def _tick(self, is_tty: bool) -> None:
        """Keep the output alive between events (see :meth:`run`)."""
        if is_tty:
            if not self.show_progress:
                return
            while True:
                await asyncio.sleep(max(self.refresh_interval, 0.5))
                self._render()
        if self.heartbeat_interval <= 0:
            return
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            for line in self._status_lines():
                self._write(line)
