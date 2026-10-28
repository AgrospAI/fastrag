> **Historical design record.** Written before implementation, as the
> pre-build design notes for the step base classes. It lives outside `docs/source/`, so
> it is never built into the published site and is not maintained in
> lockstep with the code. Most of what it describes shipped close to as
> written; where it and the code disagree, the code and `docs/source/` win —
> see `.agent/PENDING.md` section D for the pieces it names that were never
> built.

```python
class Fetcher(SourceStep, ABC):
    """Base class for leaf-level data fetchers that extract raw content from a single protocol/medium."""

    event_name = "Fetching"


class Sources(SourceStep):
    """Aggregator step combining multiple Fetchers or SourceSteps into a single stream."""

    def __init__(
        self,
        sources: Sequence[SourceStep],
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
    ) -> None:
        """Initializes Sources aggregator.

        Args:
            sources: List of child SourceSteps to evaluate concurrently/sequentially.
            cache: Default cache configuration for child sources.
            policy: Default execution policy for child sources.
        """
        super().__init__(cache=cache, policy=policy)
        self.sources = sources

    def evaluate(self, ctx: ExecutionContext) -> AsyncIterator[Source]: ...

    def config(self) -> dict[str, Any]: ...


class Parser(DocumentStep, ABC):
    """Transforms raw Source streams into initial Document representations."""

    event_name = "Parsing"

    def __call__(self, upstream: SourceStep) -> DocumentStep:
        """Binds this parser to an upstream source step to construct an execution DAG.

        Args:
            upstream: Upstream SourceStep provider.
        """
        ...


class Chunker(DocumentStep, ABC):
    """Splits incoming Document representations into granular chunk Documents."""

    event_name = "Chunking"

    def __call__(self, upstream: DocumentStep) -> DocumentStep:
        """Binds this chunker to an upstream Document step.

        Args:
            upstream: Upstream DocumentStep provider.
        """
        ...


class Embedder(DocumentStep, ABC):
    """Enriches incoming Documents with dense vector embeddings."""

    event_name = "Embedding"

    def __call__(self, upstream: DocumentStep) -> DocumentStep:
        """Binds this embedder to an upstream Document step.

        Args:
            upstream: Upstream DocumentStep provider.
        """
        ...


class VectorStore(ABC):
    """Sink backend for persisting processed and embedded documents."""

    def __init__(
        self,
        policy: Policy | None = None,
    ) -> None:
        self.policy = policy or Policy()

    @abstractmethod
    def save(self, upstream: DocumentStep) -> PipelineStep[Document]:
        """Returns a terminal pipeline step that indexes incoming documents into the store.

        Args:
            upstream: Final upstream step yielding embedded Documents.
        """
        ...
```
