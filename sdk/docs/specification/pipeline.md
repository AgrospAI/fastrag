> **Historical design record.** Written before implementation, as the
> pre-build design notes for the pipeline execution context. It lives outside `docs/source/`, so
> it is never built into the published site and is not maintained in
> lockstep with the code. Most of what it describes shipped close to as
> written; where it and the code disagree, the code and `docs/source/` win —
> see `.agent/PENDING.md` section D for the pieces it names that were never
> built.

```python
@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Runtime context passed to step evaluators to capture execution state and events.

    Attributes:
        pipeline_id: Global identifier for the executing pipeline run.
        policy: Active execution policy.
        cache: Active cache store backend.
    """

    pipeline_id: str
    policy: Policy
    cache: CacheStore | None = None
    event_sink: EventSink | None

    def emit(self, event: PipelineEvent) -> Awaitable[None]:
        """Dispatches an event to the currently registered pipeline event sink.

        Args:
            event: Event instance to propagate.
        """


class Pipeline(AbstractAsyncContextManager):
    """Evaluates a lazy PipelineStep graph.

    Pipeline is responsible for creating and installing the
    ExecutionContext for the duration of evaluation.
    """

    def __init__(
        self,
        root: PipelineStep[Any],
        *,
        cache: CacheStore | None = None,
        policy: Policy | None = None,
    ) -> None:
        """
        Create a pipeline around a root step without evaluating it.
        Args:
            root: Final step in the lazy pipeline graph.
            cache: Optional default cache.
            policy: Optional default execution policy.
        """

    async def __aenter__(self) -> Pipeline:
        """Enters pipeline context and registers the event listener queue."""
        ...

    async def __aexit__(
        self, exc_type: Any, exc_value: Any, exc_traceback: Any
    ) -> None:
        """Exits pipeline context and cleans up event sinks."""
        ...

    def __aiter__(self) -> AsyncIterator[PipelineEvent]:
        """Provides an async iterator over events emitted across all steps in the pipeline."""
        ...

    async def run(self) -> None:
        """
        Evaluate the root step while consuming its events.
        This is useful when event consumption is handled by another
        component such as a Logger.
        """


class PipelineLogger(ABC):
    """Base class for real-time visualizers or aggregators consuming pipeline event streams.

    .. note::
        Not yet built. The current codebase's event-stream consumer base is
        :class:`Logger` (see ``docs/specification/logging.md``), of which
        ``ConsoleLogger`` is the concrete implementation. ``PipelineLogger``
        is a proposed rename/generalisation that does not exist yet.
    """

    def __init__(self, pipeline: Pipeline) -> None:
        """Initializes PipelineLogger.

        Args:
            pipeline: Pipeline instance to listen to.
        """
        self.pipeline = pipeline

    @abstractmethod
    async def run(self) -> None:
        """Consumes the pipeline event iterator and renders progress/status."""
        ...


class PipelineStep(ABC, Generic[T]):
    """
    Base class for every executable pipeline step.
    Constructing a PipelineStep must never execute work. A step can either be
    evaluated independently with ``run()`` or composed with another step using
    ``__call__``.
    Evaluation receives an ExecutionContext and produces an asynchronous
    stream of output values. Lifecycle events are emitted by the evaluator,
    while domain-specific events may be emitted by individual steps.
    Every event a step produces carries both of its names: ``name``
    identifies the instance, ``event_name`` identifies the stage.
    """

    name: str

    event_name: str
    """The stage this step's work belongs to, e.g. "Chunking".

    Declared as a plain class attribute by a family base class and inherited
    by every step in that family; a concrete step may override it. Resolution
    walks the MRO for the first class declaring it in its own body, and falls
    back to the step's own class name when nothing does -- so a step family
    that declares no event_name still labels each of its steps distinctly.
    A mixin that declares none (e.g. RoutedStep) contributes nothing, which
    is what lets `class ChunkerRouter(RoutedStep[Document], Chunker)` resolve
    to Chunker's "Chunking".
    """

    def __init__(
        self,
        *,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
    ) -> None:
        """
        Configure the step without executing it.
        Args:
            cache: Optional cache backend used by this step. ``False``
                disables caching, ``None`` (the default) inherits the
                execution context's cache, a ``CacheStore`` is used directly.
            policy: Optional execution policy overriding the pipeline policy.
        """

    @property
    @abstractmethod
    def config_hash(self) -> StepConfigHash:
        """
        Return a stable hash representing the effective configuration of the
        step.
        The hash is used as part of cache identity.
        """

    @abstractmethod
    async def evaluate(
        self,
        context: ExecutionContext,
    ) -> AsyncIterator[T]:
        """
        Lazily evaluate the step within an execution context.
        Args:
            context: Execution-scoped state and services.
        Yields:
            Values produced by the step.
        """

    def __call__(
        self,
        upstream: PipelineStep[Any],
    ) -> PipelineStep[T]:
        """
        Compose this step with an upstream pipeline step.
        The returned step remains lazy and performs no work until evaluated.
        Args:
            upstream: Step whose output becomes this step's input.
        """

    async def run(self) -> list[T]:
        """
        Execute this step independently.
        This method is intended primarily for testing and standalone use.
        Pipeline-wide event observation is provided by ``Pipeline``.
        """

    @property
    def cache(self) -> CacheStore | Literal[False] | None:
        """Return the cache explicitly configured for this step."""

    @property
    def policy(self) -> Policy | None:
        """Return the policy explicitly configured for this step."""


class SourceStep(PipelineStep[Source], ABC):
    """Base pipeline step responsible for generating raw Source entities (e.g. Fetchers, Crawlers).

    Source steps validate cache freshness against upstream providers (ETag, Last-Modified, Hash).

    Declares the stage for the whole family: a Sources aggregate belongs in
    the same column as the fetchers it wraps, not one of its own.
    """

    event_name = "Fetching"

    def __init__(
        self,
        *,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
        cache_validation_interval: float | None = None,
    ) -> None:
        """Initializes SourceStep.

        Args:
            cache: Cache backend or False to disable.
            policy: Execution policy.
            cache_validation_interval: Invalidation interval for freshness checks.
        """


class DocumentStep(PipelineStep[Document], ABC):
    """Base pipeline step transforming or generating Documents (e.g. Parsers, Chunkers, Embedders).

    Document transformations are pure and deterministic: cached items do not require
    source freshness re-validation.

    Declares no event_name: "a document transform" is not itself a stage, so
    each transform family (Parser, Chunker, Embedder, ...) declares its own
    and anything else falls back to its own class name.
    """

    def __init__(
        self,
        *,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
    ) -> None:
        super().__init__(cache=cache, policy=policy)


class RoutedStep(PipelineStep[T], ABC):
    """Pipeline step that conditionally delegates to child steps based on filter rules.

    Declares no event_name, even though it sits first in a routed step's MRO:
    routing is not a stage, it is a way of choosing one.
    """

    def __init__(
        self,
        *,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
    ) -> None:
        super().__init__(cache=cache, policy=policy)
        self.routes: list[tuple[Filter[T], PipelineStep[T]]] = []
        self._default_step: PipelineStep[T] | None = None

    def route(self, filter_rule: Filter[T], step: PipelineStep[T]) -> RoutedStep[T]:
        """Registers a conditional branch step matching a specific filter predicate.

        Args:
            filter_rule: Filter predicate to match against incoming items.
            step: Child step to execute on match.

        Returns:
            Self for fluent method chaining.
        """
        ...

    def default(self, step: PipelineStep[T]) -> RoutedStep[T]:
        """Registers a fallback step for items that match no route filters.

        Args:
            step: Default step to execute.

        Returns:
            Self for fluent method chaining.
        """
        ...
```
