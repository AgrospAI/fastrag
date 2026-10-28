"""Events emitted during pipeline execution.

Two families live here: step lifecycle events (``Step*``), emitted by the
evaluator around every step regardless of what it does, and domain events
(``Document*``, ``ItemFailed``), emitted by a step itself through its
:class:`~fastrag.ingestion.steps.base.ExecutionContext` to describe what it did to one
item. Both are propagated to whatever sink a :class:`~fastrag.ingestion.Pipeline` context manager
installs.

Every event carries two names for the step that produced it, and they answer
different questions:

- ``step`` is the *instance* name — by default the producing class's own name
  (``ChunkerRouter``, ``SitemapCrawler``, ``MilvusVectorStore.save``). It
  identifies *which* object in the graph emitted this, and two differently
  configured instances of the same family are told apart by it.
- ``event_name`` is the *stage* name — the label
  :attr:`~fastrag.ingestion.steps.base.PipelineStep.event_name` resolves to
  for that class (``Fetching``, ``Parsing``, ``Chunking``, ``Embedding``,
  ``Indexing``, ...). It identifies *what kind of work* this was, so a
  consumer can group every chunker in a graph under one heading without
  knowing any concrete class. A step whose family declares no
  ``event_name`` falls back to its own class name, so this is never empty
  for an event a real step emitted.

Both are plain strings with an empty default, so an event constructed by hand
(a test double, a consumer replaying a stream) stays valid without naming
either.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fastrag.ingestion.models import Source
from fastrag.models import Document


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    """Base class for all events emitted during pipeline execution."""


@dataclass(frozen=True, slots=True)
class StepStarted(PipelineEvent):
    """Emitted when a pipeline step begins processing.

    Attributes:
        step: Name of the step that started.
        event_name: Stage label of the step that started — see this
            module's docstring.
    """

    step: str
    event_name: str = ""


@dataclass(frozen=True, slots=True)
class StepProgress(PipelineEvent):
    """Emitted incrementally as items process through a step.

    Attributes:
        step: Name of the step reporting progress.
        processed_count: Items successfully processed so far.
        event_name: Stage label of the reporting step — see this module's
            docstring.
    """

    step: str
    processed_count: int = 0
    event_name: str = ""


@dataclass(frozen=True, slots=True)
class StepCompleted(PipelineEvent):
    """Emitted when a step completes processing all items.

    Attributes:
        step: Name of the step that completed.
        event_name: Stage label of the step that completed — see this
            module's docstring.
    """

    step: str
    event_name: str = ""


@dataclass(frozen=True, slots=True)
class StepFailed(PipelineEvent):
    """Emitted when a step encounters an unrecoverable error.

    Attributes:
        step: Name of the step that failed.
        error: The exception instance that triggered failure.
        event_name: Stage label of the step that failed — see this module's
            docstring.
    """

    step: str
    error: Exception = field(default_factory=Exception)
    event_name: str = ""


# Domain-specific item events.
#
# Every event below carries the actual item it describes — a `Source` or
# `Document` — as its *first* field, so a consumer can positionally pattern
# match it out (`case DocumentFetched(document):`) and read whatever it needs
# straight off the item (`document.cached`, `document.id`,
# `document.metadata`) instead of the event replicating a subset of the
# item's own fields under different names.
#
# `step`/`event_name` come *after* that item, never before it, precisely so
# adding them left every existing positional match working unchanged.


@dataclass(frozen=True, slots=True)
class DocumentFetched(PipelineEvent):
    """Emitted when a source item has been fetched.

    Attributes:
        source: The fetched item. ``source.cached`` is ``True`` when it was
            served from the cache rather than freshly fetched.
        step: Name of the step that fetched it.
        event_name: Stage label of that step — see this module's docstring.
    """

    source: Source
    step: str = ""
    event_name: str = ""


@dataclass(frozen=True, slots=True)
class DocumentParsed(PipelineEvent):
    """Emitted when a document has been parsed.

    Attributes:
        document: The parsed document. ``document.cached`` is ``True`` when
            it was served from the cache rather than freshly parsed.
        step: Name of the step that parsed it.
        event_name: Stage label of that step — see this module's docstring.
    """

    document: Document
    step: str = ""
    event_name: str = ""


@dataclass(frozen=True, slots=True)
class DocumentChunked(PipelineEvent):
    """Emitted when a document has been split into chunks.

    Attributes:
        document: The parent document that was split.
        chunks: The chunks produced from ``document`` — retained (rather than
            just a count) so a consumer can inspect each chunk's own
            ``cached``/``id``/``metadata`` individually.
        step: Name of the step that split it.
        event_name: Stage label of that step — see this module's docstring.
    """

    document: Document
    chunks: list[Document] = field(default_factory=list)
    step: str = ""
    event_name: str = ""


@dataclass(frozen=True, slots=True)
class DocumentEmbedded(PipelineEvent):
    """Emitted when a document or chunk receives vector embeddings.

    Attributes:
        document: The embedded document. ``document.cached`` is ``True`` when
            its embedding was served from the cache rather than freshly
            computed, or when ``skip_existing`` bypassed embedding for an
            already indexed document. In the latter case, its embedding may
            be ``None``.
        step: Name of the step that embedded it.
        event_name: Stage label of that step — see this module's docstring.
    """

    document: Document
    step: str = ""
    event_name: str = ""


@dataclass(frozen=True, slots=True)
class DocumentSaved(PipelineEvent):
    """Emitted when a document is persisted into a vector store.

    Attributes:
        document: The saved document.
        step: Name of the step that saved it.
        event_name: Stage label of that step — see this module's docstring.
    """

    document: Document
    step: str = ""
    event_name: str = ""


@dataclass(frozen=True, slots=True)
class DocumentsPruned(PipelineEvent):
    """Emitted once by a vector store's sync step, after its upstream is exhausted.

    Attributes:
        step: Name of the sync step (``MilvusVectorStore.sync``).
        stale: How many stored documents the run did not yield.
        deleted: How many of them were deleted: all of them after a clean
            run, none after a run with ``failures``.
        failures: ``ItemFailed`` events the run emitted. Any failure keeps
            the stale documents, since a failed item's documents are missing
            from the run without being gone from the source.
        event_name: Stage label of that step — see this module's docstring.
    """

    step: str = ""
    stale: int = 0
    deleted: int = 0
    failures: int = 0
    event_name: str = ""


@dataclass(frozen=True, slots=True)
class ItemFailed(PipelineEvent):
    """Emitted when a single item fails within a step.

    Attributes:
        step: Name of the step the failing item was being processed by. The
            first field (rather than an afterthought default), so every
            caller names it — a consumer (e.g.
            :class:`~fastrag.ingestion.logger.ConsoleLogger`) can always
            attribute the failure to the stage it came from.
        uri: Identifier of the item that failed.
        error: The exception instance that triggered the failure.
        source_id: The originating :class:`~fastrag.ingestion.models.Source`
            id the failing item traces back to (its own ``id`` for a
            ``Source``, or ``.source_id`` for a ``Document``) — empty when the
            failing item carries neither (e.g. a whole child ``SourceStep``
            failing inside :class:`~fastrag.ingestion.steps.fetching.sources.Sources`).
            Lets a consumer attribute a failure to the row a multi-source
            table's ``step`` name alone cannot identify (a parser/chunker/
            embedder/sink step is shared by every source; only the item it
            was processing says which one).
        event_name: Stage label of the step the item failed in — what a
            consumer columns the failure under, since ``step`` alone names
            an instance rather than a stage. See this module's docstring.
    """

    step: str = ""
    uri: str = ""
    error: Exception = field(default_factory=Exception)
    source_id: str = ""
    event_name: str = ""


# Item lifecycle events.
#
# Emitted by `run_concurrently` around every unit of work it schedules, cache
# hits included, so a consumer can tell what is in flight right now and for
# how long — the only way to spot a stuck item before its timeout fires.
# `item_id` pairs an `ItemStarted` with its `ItemFinished`; `uri` alone is not
# unique (two stages work on the same document).


@dataclass(frozen=True, slots=True)
class ItemStarted(PipelineEvent):
    """Emitted when a step starts working on one item.

    Attributes:
        item_id: Unique for this unit of work; repeated on its ``ItemFinished``.
        uri: Identifier of the item.
        step: Name of the step working on it.
        event_name: Stage label of that step.
        source_id: The originating ``Source`` id, as on ``ItemFailed``.
    """

    item_id: int
    uri: str = ""
    step: str = ""
    event_name: str = ""
    source_id: str = ""


@dataclass(frozen=True, slots=True)
class ItemRetrying(PipelineEvent):
    """Emitted when a transient failure or timeout is about to be retried.

    Attributes:
        item_id: The ``ItemStarted.item_id`` of the unit of work.
        attempt: The attempt that just failed, 1-based.
        delay: Seconds before the next attempt.
        error: What the failed attempt raised.
    """

    item_id: int
    attempt: int = 1
    delay: float = 0.0
    error: Exception = field(default_factory=Exception)


@dataclass(frozen=True, slots=True)
class ItemFinished(PipelineEvent):
    """Emitted when a step is done with one item, successfully or not.

    Attributes:
        item_id: The ``ItemStarted.item_id`` of the unit of work.
        elapsed: Wall-clock seconds since its ``ItemStarted``, retries included.
        error: What it finally raised, or ``None`` on success.
    """

    item_id: int
    elapsed: float = 0.0
    error: Exception | None = None
