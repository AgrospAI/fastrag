# Caching

Make pipeline reruns cheap by remembering what a step already produced.
Caching is explicit and per-step: each step takes a `cache=` argument —
a `CacheStore`, `False` to opt out of caching, or `None` (the default) to
inherit whatever cache the pipeline or an enclosing aggregate is running
under.

```{literalinclude} /../../examples/01_fetching.py
:language: python
:start-after: "docs:caching-open-start"
:end-before: "docs:caching-open-end"
:dedent:
```

(more steps compose onto `sources` with `cache=cache` the same way —
{doc}`ingestion` walks the whole chain.)

## Choosing a backend

Two `CacheStore` backends ship:

- `PostgresCache` self-manages its tables under a target Postgres schema,
  created on `__aenter__`. Use it when several machines share one cache, or
  the cache grows large.
- `SQLiteCache` keeps everything in one local file and needs no server and no
  extra dependency (it uses the standard-library `sqlite3`). The file and its
  parent directories are created on `__aenter__`.

```python
from fastrag.ingestion.cache import SQLiteCache

async with SQLiteCache(path=".fastrag/cache.sqlite") as cache:
    sources = SitemapCrawler(url=SITEMAP_URL, cache=cache)
```

Both honour the same contract described below, including deletion and garbage
collection. One `SQLiteCache` runs its operations one at a time on a worker
thread. Separate processes may share the file: the database runs in WAL mode,
and a writer that cannot get the lock within `timeout` seconds (default 30)
fails with `TransientCacheError`, which step retry policies treat as
retryable. `path=":memory:"` keeps a throwaway cache for the duration of the
`async with` block.

## What a key is

Cache ids are built by the identity helpers:

- `source_cache_id(step_name, source_id, config_hash)` — a source step's
  cache bucket.
- `document_cache_id(step_name, document_id, config_hash)` — a document
  step's bucket.

`config_hash` is the stable fingerprint of the step _configuration_, not its
inputs — change `chunk_size` and a chunker's cache is invalidated; change
what the fetcher URL is and only that source's bucket is. A fetched
`Source`'s freshness is revalidated against upstream (ETag → Last-Modified →
content hash) when its cached entry is older than
`cache_validation_interval`; `Source.last_validated` is persisted so the
interval means the same thing across a process restart.

## Determinism is the contract

A `Document` id derives from content plus configuration (see {doc}`ingestion`),
so a step cache keyed on id serves the identical chunk without recomputing.
That is why channel-level caching works at all — the same bytes parsed once
produce the same ids on every machine.

## Deletion and GC

Nothing is evicted automatically. `CacheStore` declares six explicit
administration methods, each returning how many items were removed:
`clear_sources()`, `clear_documents()`, `clear()`, `delete_sources(ids)`,
`delete_documents(source_ids)`, `collect_garbage(max_age)`. Deletion is
idempotent and never touches the vector store — clear the cache and the
collection still holds its rows.

## Bulk reads and automatic batching

`PostgresCache` groups concurrent single-bucket reads automatically, including
reads from routed children and source targets. Each group uses one SQL statement
to retrieve cached outputs and refresh their access timestamps. Existing
pipeline code needs no changes.

The defaults collect requests for 1 ms and read at most 128 buckets per batch:

```python
cache = PostgresCache(
    url=DATABASE_URL,
    read_batch_size=128,
    read_batch_delay=0.001,
)
```

`read_batch_delay=0` collects requests across one event-loop yield.
`read_batch_size=1` disables grouping. Batch sizes are limited to 10,000.
Direct calls group according to their arrival rate. Pipeline evaluators also
prefetch bounded windows using `Policy.cache_batch_size` (default 128),
independently of processing concurrency. Sequential direct reads still use
one statement each.

For callers that already have identifiers, `get_document_entries_many(ids)`
and `get_source_entries_many(ids)` return dictionaries keyed by every requested
cache id. Values are cached outputs, `[]` for a known-empty bucket, or `None`
for a miss. Duplicate identifiers are read once; empty requests perform no I/O.
Custom `CacheStore` backends inherit compatible single-read implementations
and can override these methods to support bulk queries.

Results are retained only while resolving a batch. Later reads query PostgreSQL
again, so writes, deletion and garbage collection require no local cache
invalidation. The bound counts buckets; a single bucket can contain many
outputs. Large writes are split into bounded SQL statements while remaining
atomic within the bucket's transaction.

## Stage snapshots and upgrades

Each source bucket owns its rendered bytes, metadata and freshness validators.
Changing a crawler's renderer, model or rendering settings selects a separate
bucket; returning to an earlier configuration reuses its own content. API keys
are excluded from renderer fingerprints.

The source snapshot upgrade invalidates legacy source buckets once: their
shared URL rows cannot identify which renderer produced the stored content.
Sources are fetched again on the next run. Document buckets are preserved.

Each document bucket owns its output snapshot, including metadata and vectors.
Chunker outputs stay vector-free and embedding configurations retain separate
vectors, even when their logical document IDs match. Text and source bodies
are compressed; vectors are stored separately from text. Reads refresh the
owning snapshot's access timestamp, and source deletion and garbage collection
use indexed snapshot lineage.

Startup checks an existing schema in one statement. Missing columns and
indexes are upgraded under a schema advisory lock. The legacy shared-document
schema cannot recover overwritten stage outputs: migration retains source
bytes and unambiguous embedding snapshots, and invalidates raw transformation
and ambiguous embedding buckets so the next run rebuilds them. This upgrade
may block writes and takes longer than a normal warm startup.

The indexing-only path described in {doc}`indexing` projects cached identifiers
and routing metadata without retrieving text or vectors. Payloads are loaded
only when downstream work needs them. Ordinary cache getters return complete
records. See {doc}`/contributing/cache-retrieval` for the investigation,
implementation plan and measured workload.

## See also

{doc}`ingestion` for the steps that take `cache=`; {doc}`indexing` for the
sink on the far side.

## Source tree summaries

When a source-only pipeline is used to summarize URLs rather than inspect
content, enable `source_headers_only` to avoid reading cached page bodies:

```python
pipeline = Pipeline(crawler, cache=cache, source_headers_only=True)
await SourceTreeLogger(pipeline, max_depth=3).run()
```

PostgreSQL serves cached identities, validators and metadata with empty content
bytes in this mode. Cache misses still fetch and cache full sources, and
expired entries still load their bodies before freshness validation. Backends
without header projections fall back to ordinary reads. Use the default
`False` for full-content loggers; transformation graphs reject this option.
Any filtering inside the source graph must work from identities or metadata.
