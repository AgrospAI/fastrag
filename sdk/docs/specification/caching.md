> **Historical design record.** Written before implementation, as the
> pre-build design notes for the cache contract. It lives outside `docs/source/`, so
> it is never built into the published site and is not maintained in
> lockstep with the code. Most of what it describes shipped close to as
> written; where it and the code disagree, the code and `docs/source/` win —
> see `.agent/PENDING.md` section D for the pieces it names that were never
> built.

```python
class SourceFreshnessValidator(Protocol):
    """Protocol for checking if an upstream resource has changed."""

    async def is_source_modified(
        self,
        source_id: str,
        cached_etag: str | None,
        cached_last_modified: datetime | None,
        cached_content_hash: str,
    ) -> bool:
        """Validates whether a source entry is stale against the upstream server.

        Returns:
            True if modified, False if cached version remains fresh.
        """
        ...


class CacheStore(AbstractAsyncContextManager, ABC):
    """Pluggable abstract cache backend with distinct lifecycle support."""

    async def __aenter__(self) -> Self:
        """Open cache resources."""

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        """Release cache resources."""

    @abstractmethod
    async def get_source_entries(self, source_cache_id: str) -> list[Source] | None:
        """Fetches cached sources associated with a step run.

        Args:
            source_cache_id: Unique hash of the source step configuration and target.
        """
        ...

    @abstractmethod
    async def put_source_entries(
        self, source_cache_id: str, sources: Sequence[Source]
    ) -> None:
        """Persists fetched sources and updates source reference tables.

        Args:
            source_cache_id: Unique hash of the source step configuration.
            sources: Sequence of Source objects to persist.
        """
        ...

    @abstractmethod
    async def get_document_entries(
        self, document_cache_id: str
    ) -> list[Document] | None:
        """Fetches cached documents associated with a transformation step.

        Args:
            document_cache_id: Hash composed of document ID and step configuration.
        """
        ...

    @abstractmethod
    async def put_document_entries(
        self, document_cache_id: str, documents: Sequence[Document]
    ) -> None:
        """Persists transformed documents and updates document cache references.

        Args:
            document_cache_id: Hash composed of document ID and step configuration.
            documents: Transformed Document objects to persist.
        """
        ...
```

## `last_validated` and `cache_validation_interval` across process restarts

`Source.last_validated` is part of the persisted shape of a cached source,
not runtime-only like `Source.cached`: `put_source_entries` writes it (the
value on the `Source` if it carries one, otherwise the current time — a
source being written at all means it was just validated), and
`get_source_entries` reads it back onto every `Source` it returns.

This is what lets a `SourceStep`'s `cache_validation_interval` mean the same
thing across a process restart. The interval check reads `last_validated`
straight off the entries `get_source_entries` returned — never from
per-process state — so a process that has never seen a given cache id before
still serves it without revalidating when the persisted timestamp is within
the interval. Revalidating (calling `is_source_modified`) and finding the
source unchanged still writes `last_validated` forward to "now": the check
was answered, so the clock resets, exactly as if the source had been
refetched.

## Cache cleanup

Nothing is ever removed from a `CacheStore` automatically. `CacheStore`
declares six administration methods, each returning the number of cached
items removed, that a caller invokes explicitly on the store it already
holds:

```python
class CacheStore(AbstractAsyncContextManager, ABC):
    ...

    @abstractmethod
    async def clear_sources(self) -> int:
        """Removes every cached source and every source cache id."""
        ...

    @abstractmethod
    async def clear_documents(self) -> int:
        """Removes every cached document and every document cache id."""
        ...

    @abstractmethod
    async def clear(self) -> int:
        """Removes everything this store holds."""
        ...

    @abstractmethod
    async def delete_sources(self, source_ids: Sequence[str]) -> int:
        """Removes the named cached sources."""
        ...

    @abstractmethod
    async def delete_documents(self, source_ids: Sequence[str]) -> int:
        """Removes every cached document whose source_id is one of source_ids."""
        ...

    @abstractmethod
    async def collect_garbage(self, max_age: timedelta) -> int:
        """Removes every source and document whose last_accessed predates now - max_age."""
        ...
```

Usage stays a plain method call on the object the caller already has open:

```python
async with PostgresCache(url=DATABASE_URL) as cache:
    await cache.delete_sources(["https://old-site.example/page"])
    await cache.collect_garbage(max_age=timedelta(days=30))
```

`collect_garbage` keys purely on `last_accessed` — the same column every
`get_source_entries`/`get_document_entries` call already bumps on a hit — so
an entry read recently survives a sweep and one that has not been touched
inside `max_age` does not. `max_age` must be positive; zero or negative
raises `ValueError`.

**The empty-cache-id rule.** `get_source_entries`/`get_document_entries`
distinguish a known-but-empty result (`[]`) from a genuine miss (`None`) —
that distinction is exactly what lets a pipeline trust a cached "nothing
here" instead of refetching. Deleting a source or document removes its rows
via `ON DELETE CASCADE` on the `*_cache_entries` tables, but cascade cannot
know that a `source_cache`/`document_cache` bucket has thereby lost its last
entry: left alone, that bucket would keep answering `[]` forever. Every
deletion method — `delete_sources`, `delete_documents`, and both halves of
`collect_garbage` — therefore also removes any cache id left referencing
zero entries as a direct result of that call, so the next lookup on it
returns `None` and the pipeline refetches rather than treating a hollowed-out
bucket as a real empty result. A cache id that was already legitimately empty
before the call (a source step that genuinely produced nothing) is left
untouched, since it was never affected by the deletion.

Deletion is idempotent and tolerant: an id that is not cached is silently
skipped rather than raising, and an empty `source_ids` sequence is a no-op —
no statement is sent. Every method runs a bounded number of statements
regardless of how many rows or ids are involved, never one delete per id.

Clearing the cache never touches the vector store — `VectorStore.delete` and
any vector-store reconciliation were considered and deliberately left out of
the SDK surface; see `docs/specification/API.md`.

