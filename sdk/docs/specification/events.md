> **Historical design record.** Written before implementation, as the
> pre-build design notes for the pipeline event hierarchy. It lives outside `docs/source/`, so
> it is never built into the published site and is not maintained in
> lockstep with the code. Most of what it describes shipped close to as
> written; where it and the code disagree, the code and `docs/source/` win —
> see `.agent/PENDING.md` section D for the pieces it names that were never
> built.

```python
@dataclass(frozen=True, slots=True)
class PipelineEvent:
    """Base class for all events emitted during pipeline execution."""


@dataclass(frozen=True, slots=True)
class StepStarted(PipelineEvent):
    """Emitted when a pipeline step begins processing."""

    step: str
    event_name: str = ""


@dataclass(frozen=True, slots=True)
class StepProgress(PipelineEvent):
    """Emitted incrementally as items process through a step.

    Attributes:
        processed_count: Items successfully processed so far.
    """

    step: str
    processed_count: int = 0
    event_name: str = ""


@dataclass(frozen=True, slots=True)
class StepCompleted(PipelineEvent):
    """Emitted when a step completes processing all items."""

    step: str
    event_name: str = ""


@dataclass(frozen=True, slots=True)
class StepFailed(PipelineEvent):
    """Emitted when a step encounters an unrecoverable error.

    Attributes:
        error: The exception instance that triggered failure.
    """

    step: str
    error: Exception = field(default_factory=Exception)
    event_name: str = ""


# Domain-specific Item Events
#
# Every event below carries the actual item it describes — a `Source` or
# `Document` — as its first field, so a consumer can positionally pattern
# match it out (`case DocumentFetched(document):`) and read whatever it
# needs straight off the item (`document.cached`, `document.id`,
# `document.metadata`) instead of the event replicating a subset of the
# item's own fields under different names.
#
# `step`/`event_name` come after that item, never before it, so every
# positional match keeps working.


@dataclass(frozen=True, slots=True)
class DocumentFetched(PipelineEvent):
    """Emitted when a source item has been fetched.

    Attributes:
        source: The fetched item. `source.cached` is `True` when it was
            served from the cache rather than freshly fetched.
    """

    source: Source
    step: str = ""
    event_name: str = ""


@dataclass(frozen=True, slots=True)
class DocumentParsed(PipelineEvent):
    """Emitted when a document has been parsed.

    Attributes:
        document: The parsed document. `document.cached` is `True` when it
            was served from the cache rather than freshly parsed.
    """

    document: Document
    step: str = ""
    event_name: str = ""


@dataclass(frozen=True, slots=True)
class DocumentChunked(PipelineEvent):
    """Emitted when a document has been split into chunks.

    Attributes:
        document: The parent document that was split.
        chunks: The chunks produced from `document` — retained (rather than
            just a count) so a consumer can inspect each chunk's own
            `cached`/`id`/`metadata` individually.
    """

    document: Document
    chunks: list[Document] = field(default_factory=list)
    step: str = ""
    event_name: str = ""


@dataclass(frozen=True, slots=True)
class DocumentEmbedded(PipelineEvent):
    """Emitted when a document or chunk receives vector embeddings.

    Attributes:
        document: The embedded document. `document.cached` is `True` when
            its embedding was served from the cache rather than freshly
            computed.
    """

    document: Document
    step: str = ""
    event_name: str = ""


@dataclass(frozen=True, slots=True)
class DocumentSaved(PipelineEvent):
    """Emitted when a document is persisted into a vector store.

    Attributes:
        document: The saved document.
    """

    document: Document
    step: str = ""
    event_name: str = ""


@dataclass(frozen=True, slots=True)
class ItemFailed(PipelineEvent):
    """Emitted when a single item fails within a step.

    Attributes:
        step: Name of the step the failing item was being processed by. The
            first field, so every caller now names it explicitly — a
            consumer (e.g. `ConsoleLogger`) can always attribute the
            failure to the stage it came from.
        uri: Identifier of the item that failed.
        error: The exception instance that triggered the failure.
        source_id: The originating `Source` id the failing item traces back
            to (its own `id` for a `Source`, or `.source_id` for a
            `Document`) — empty when the failing item carries neither (e.g.
            a whole child `SourceStep` failing inside `Sources`). Lets a
            consumer attribute a failure to the row a multi-source table's
            `step` name alone cannot identify.
        event_name: Stage label of the step the item failed in — what a
            consumer columns the failure under, since `step` alone names an
            instance rather than a stage.
    """

    step: str = ""
    uri: str = ""
    error: Exception = field(default_factory=Exception)
    source_id: str = ""
    event_name: str = ""
```

Every domain event fires the same way for a cache hit as for freshly
produced work: `item.cached` on the carried `Source`/`Document` is always
trustworthy, so a consumer never needs to derive "was this cached?" from
anything but the event's own item. `DocumentFetched`/`DocumentParsed` have
always worked this way; `DocumentChunked`/`DocumentEmbedded`/`DocumentSaved`
are emitted from a per-item hook (`DocumentStep._on_produced`) that fires
once an upstream item's output is resolved, whether that resolution came
from the cache or from a fresh `transform()` call — not from inside
`transform()` itself, which only ever runs on a cache miss.

Every event carries two names for the step that produced it, and they answer
different questions. `step` is the *instance* name — by default the
producing class's own name (`ChunkerRouter`, `SitemapCrawler`,
`MilvusVectorStore.save`) — and identifies *which* object in the graph
emitted this. `event_name` is the *stage* name — whatever
`PipelineStep.event_name` resolves to for that class (`Fetching`, `Parsing`,
`Chunking`, `Embedding`, `Indexing`, ...) — and identifies *what kind of
work* it was, so a consumer can group every chunker in a graph under one
heading without knowing any concrete class. A step whose family declares no
`event_name` falls back to its own class name, so neither is ever empty for
an event a real step emitted; both default to `""` only so an event built by
hand stays valid.

`ItemFailed.step` is always populated by the emitting call site; there is no
call site left that constructs one without naming the step responsible.
`ItemFailed.source_id` is likewise populated by the emitting call site (via
`run_concurrently`'s `_lineage_key`), so a consumer can attribute a
downstream failure to the source row it came from.
