> **Historical design record.** Written before implementation, as the
> pre-build design notes for the logger contract. It lives outside `docs/source/`, so
> it is never built into the published site and is not maintained in
> lockstep with the code. Most of what it describes shipped close to as
> written; where it and the code disagree, the code and `docs/source/` win —
> see `.agent/PENDING.md` section D for the pieces it names that were never
> built.

```python
class Logger(ABC):
    """
    Base class for components that consume pipeline events.
    Loggers are sinks for event streams and do not alter pipeline execution.
    """

    pipeline: Pipeline

    def __init__(self, pipeline: Pipeline) -> None:
        """Associate the logger with a pipeline."""

    @abstractmethod
    async def run(self) -> None:
        """
        Consume and process all events from the associated pipeline.
        """


class ConsoleLogger(Logger):
    """
    Renders pipeline lifecycle and domain events to the console.
    The logger may aggregate item-level events into a tabular progress display
    showing new, cached, and failed items for each source and pipeline stage.

    Rows are the graph's leaf fetchers. Columns are stages, and the logger
    holds no list of them: each column is an emitting step's own
    ``event_name`` (see ``pipeline.md``), read off the event. Columns are
    seeded in pipeline order by walking the graph so the table reads
    left-to-right in the order work happens, and a stage label first seen on
    an event is appended as a new column rather than dropped -- a pipeline
    with a step family this SDK has never heard of renders it like any other.
    """

    show_progress: bool
    show_domain_events: bool
    refresh_interval: float
```
