> **Historical design record.** Written before implementation, as the
> pre-build design notes for the execution policy. It lives outside `docs/source/`, so
> it is never built into the published site and is not maintained in
> lockstep with the code. Most of what it describes shipped close to as
> written; where it and the code disagree, the code and `docs/source/` win —
> see `.agent/PENDING.md` section D for the pieces it names that were never
> built.

```python
@dataclass(frozen=True, slots=True)
class Policy:
    """Execution policy controlling operational constraints for pipeline steps.

    Attributes:
        retries: Maximum number of execution retries on failure.
        backoff: Initial exponential backoff delay in seconds.
        max_backoff: Ceiling delay for exponential backoff in seconds.
        timeout: Maximum execution duration per item or batch in seconds.
        on_error: Failure handling strategy ('skip', 'raise', 'retry').
        concurrency: Max concurrent worker tasks.
        batch_size: Processing batch size for batched transforms.
    """

    retries: int = 3
    backoff: float = 1.0
    max_backoff: float = 10.0
    timeout: float = 30.0
    on_error: Literal["skip", "raise", "retry"] = "skip"
    concurrency: int = 8
    batch_size: int = 16
```