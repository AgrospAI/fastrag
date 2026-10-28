# Cache performance

The ingestion cache's main scaling cost was database round trips for each
upstream item, rather than finding a cache key in PostgreSQL. This investigation
records the architecture, the selected optimization, and the checks needed to
preserve cache correctness.

## Architecture and bottlenecks

The SDK evaluates a lazy graph of source, parser, chunker, embedder, and sink
steps. Source fetching is cached per target when a fetcher supplies `targets`,
or as one stream otherwise. Document transforms are cached per upstream item;
routers use the matched child's identity. The evaluator applies concurrency,
retry, timeout, event, and freshness policies around that work. These are
separate responsibilities from database storage.
[Source: step evaluator](../../../src/fastrag/ingestion/steps/base.py).

Cache identities contain a step name and a digest of its configuration and
target or upstream item identifier. They do not expose a URL that can be used
to list all buckets belonging to a crawl. Entities retain source lineage, but
the bucket-to-input relationship is encoded by that digest.
[Source: identity helpers](../../../src/fastrag/ingestion/cache/ids.py),
[source and document models](../../../src/fastrag/ingestion/models.py),
[document model](../../../src/fastrag/models.py).

Before optimization, both PostgreSQL getters used these SQL statement counts,
excluding transaction control:

| Result | Statements | Work |
| --- | ---: | --- |
| Miss | 1 | Check the bucket exists |
| Known empty | 2 | Check the bucket, then list entry identifiers |
| Populated hit | 3 | Check the bucket, list identifiers, update access time and return entities |

Thus 10,000 populated document lookups across a warm stage required 30,000 SQL
executions and 10,000 transactions. Each parser output or chunk admitted to the
next stage incurred another lookup. Concurrent evaluation overlaps requests,
but does not remove those executions; the engine's pool is shared by them.
These are counts derived from the original control flow, not a timing
benchmark. [Source: PostgreSQL cache](../../../src/fastrag/ingestion/cache/postgres.py),
[transaction and engine helpers](../../../src/fastrag/_sql.py).

Three secondary issues appeared in the same storage path:

- Large output buckets were inserted as one multirow statement. Both entity
  upserts and association inserts could exceed the driver's argument budget.
  asyncpg rejects more than 32,767 arguments for a query.
  [Source: original write helpers](../../../src/fastrag/ingestion/cache/postgres.py),
  [asyncpg bind encoding](https://github.com/MagicStack/asyncpg/blob/master/asyncpg/protocol/prepared_stmt.pyx).
- Deleting documents by `source_id` lacked an index on that predicate. Entry
  tables had composite primary keys starting with the bucket identifier, but
  lacked dedicated indexes for reverse entity lookups during deletion and
  foreign-key cascades. PostgreSQL does not automatically index referencing
  foreign-key columns; a B-tree composite index is most effective when its
  leading columns are constrained.
  [Source: cache schema](../../../src/fastrag/ingestion/cache/schema.py),
  [PostgreSQL foreign keys](https://www.postgresql.org/docs/18/ddl-constraints.html#DDL-CONSTRAINTS-FK),
  [multicolumn indexes](https://www.postgresql.org/docs/18/indexes-multicolumn.html).
- Hits update `last_accessed`, including its index. Batching reduces statement
  and transaction overhead, but these remain writes and can contend when
  concurrent pipelines share entities.
  [Source: read and garbage-collection helpers](../../../src/fastrag/ingestion/cache/postgres.py),
  [cache schema](../../../src/fastrag/ingestion/cache/schema.py).

## Selected implementation plan

Keep the existing cache identities and semantics, and optimize the storage
boundary. This reaches ordinary document steps, routed children, and source
targets without changing their retry or event behavior.
[Source: all evaluator cache call sites](../../../src/fastrag/ingestion/steps/base.py).

1. Extend `CacheStore` with bulk source and document reads, returning one result
   per requested bucket. Supply a default implementation using the existing
   single getters so third-party stores keep working.
2. Implement PostgreSQL bulk reads as one statement per bounded group. An
   `UPDATE ... RETURNING` CTE touches matched entities; an outer join from
   bucket rows to associations and returned entities preserves known-empty
   buckets, while requested buckets absent from the database become misses.
3. Coalesce concurrently requested single reads using a short configurable
   collection window and a bounded batch size. Resolve waiting callers with
   the bulk results; propagate failures and handle cancellation and context
   shutdown without leaving callers waiting.
4. Split large entity and association inserts into bounded statements inside
   their existing transaction. Preserve whole-bucket replacement and
   last-occurrence deduplication.
5. Add indexes on `documents.source_id`, `source_cache_entries.source_id`, and
   `document_cache_entries.document_id`. Ensure entry of an existing cache
   installs missing indexes as well as creating new tables.

These changes belong in the cache contract, PostgreSQL backend, and its owned
schema. The self-managed cache remains independent of the application's
Alembic history. [Source: cache contract](../../../src/fastrag/ingestion/cache/base.py),
[schema ownership](../../../src/fastrag/ingestion/cache/schema.py).

PostgreSQL CTE substatements share a snapshot. The final query must consume
the update's `RETURNING` output rather than reread the base entity table to
observe the touched rows. Update each entity once even when multiple requested
buckets reference it. [Source: PostgreSQL data-modifying CTEs](https://www.postgresql.org/docs/18/queries-with.html#QUERIES-WITH-MODIFYING).

## Why not preload every URL's identifiers

A URL-wide membership set cannot replace the cached outputs needed by each
transform. It also needs a new relationship from bucket identities back to
inputs, and rules for writes, clears, source deletion, garbage collection,
configuration changes, and updates from other processes. Loading every vector
and document for a large crawl instead would trade round trips for potentially
large memory and transfer costs. The existing model does not require either
relationship or persistent process-local state.
[Source: cache identities](../../../src/fastrag/ingestion/cache/ids.py),
[cache result and administration contract](../../../src/fastrag/ingestion/cache/base.py),
[persisted entity payloads](../../../src/fastrag/ingestion/cache/schema.py).

The chosen approach retains only pending requests and the results needed to
resolve their callers. A later read queries PostgreSQL again, so no additional
long-lived memoization invalidation protocol is introduced. Batch bounds limit
the number of requested buckets, not the number or byte size of outputs in a
single bucket. Large whole-stream source buckets remain potentially expensive.

## Validation and operational limits

Validate the optimization against a real PostgreSQL instance, counting SQL
executions after schema initialization. A group of concurrent warm reads that
fits one batch should issue one statement, and an explicit bulk request should
split only at its configured bound. Cover mixed hits, misses and known-empty
results; shared entities; duplicate identifiers; updated access times; writes
and deletion followed by new reads; large inserts; concurrent requests;
cancellation, database failure, and close/reopen behavior. Existing pipeline
tests also exercise routing identity, source freshness, cache inheritance,
and cleanup behavior.
[Source: pipeline tests](../../../tests/test_pipeline.py).

Coalescing depends on how many callers arrive together. The evaluator's
concurrency bound can keep groups smaller than the database batch limit, and
strictly sequential calls do not gain cross-call batching. The collection
window adds latency to isolated calls; compare representative warm pipelines
as well as statement counts before choosing different limits. Even sequential
populated reads benefit from collapsing three statements into one.
[Source: bounded evaluator](../../../src/fastrag/ingestion/steps/base.py),
[execution policy](../../../src/fastrag/ingestion/policy.py).

Creating missing indexes during startup is an additive schema upgrade, but
ordinary index creation blocks writes until the build finishes. Large live
caches can precreate these indexes with `CREATE INDEX CONCURRENTLY` outside a
transaction before starting the upgraded SDK; automatic initialization remains
transactional. [Source: PostgreSQL index creation](https://www.postgresql.org/docs/18/sql-createindex.html#SQL-CREATEINDEX-CONCURRENTLY).

Statement reductions are deterministic regression targets. Wall-clock speed
depends on round-trip latency, output size, concurrency, and write contention;
the architectural counts above do not establish an elapsed-time speedup.

## Subsequent retrieval optimization

The coalescer below was the first iteration. The actual cached fetching
workload required independent lookup windows, stage-owned snapshots and an
index-only header projection as well. See {doc}`cache-retrieval` for that
investigation, implementation and final example measurements.

## Measured result

A local PostgreSQL 18.4 check compared the previous backend from the Git index
with the optimized backend against the same warmed, isolated schema. It read
1,024 document buckets, each containing one text chunk and a 32-value vector,
in groups of eight concurrent requests, matching `Policy.concurrency`'s
default. Both implementations refreshed `last_accessed` and returned the
cached payload. Connection initialization and fixture writes were excluded.

| Backend | SQL executions | Elapsed time |
| --- | ---: | ---: |
| Previous per-item reads | 3,072 | 4.956 s |
| Coalesced reads | 128 | 0.536 s |

This run reduced SQL executions by 24× and elapsed time by approximately 9.2×.
Timing is specific to this machine and database; it is not a production
throughput guarantee. The regression suite directly asserts the SQL reduction
against PostgreSQL, including bounded explicit bulk reads and source reads.
[Source: cache regression tests](../../../tests/test_cache.py).
