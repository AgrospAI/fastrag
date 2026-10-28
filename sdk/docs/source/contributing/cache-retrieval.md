# Stage cache retrieval

Warm ingestion needs bounded stage reads independent of transform concurrency,
and cache outputs must belong to a stage bucket to keep embeddings correct.
An indexing-only fast path can avoid loading embeddings for chunks already in
the destination. This note records the initial investigation and the implemented
iteration below. The shared-row findings describe the original storage model.

## Why warm retrieval still takes minutes

The AgrospAI example composes source fetching, routed parsing, routed chunking,
embedding, and terminal saving. The cache's short-window coalescer only sees
the item workers admitted by the evaluator. Raising its maximum batch size
cannot admit additional requests while these workers await their current
reads. [Example](../../../agrospai-docs-chatbot/fetching.py),
[evaluator](../../../src/fastrag/ingestion/steps/base.py),
[cache backend](../../../src/fastrag/ingestion/cache/postgres.py).

For the reported workload of 3,458 chunks, four workers, and 80 ms round-trip
latency, one warm embedding lookup per four chunks implies roughly
`ceil(3458 / 4) * 0.08 = 69.2` seconds of request latency alone in that stage.
Groups of 128 lower this to approximately 28 request rounds, or 2.24 seconds
of latency, before database, decoding, payload transfer, and other stages.
These are latency estimates from the reported workload, not measurements.

The full-vector payload also matters: 3,458 vectors of 4,096 float64 elements
contain about 113 MB (108 MiB) of numeric data alone. Metadata, text, protocol,
and Python object overhead increase that cost. Full cache getters currently
return every entity column, including vectors.
[Schema](../../../src/fastrag/ingestion/cache/schema.py),
[bulk getter](../../../src/fastrag/ingestion/cache/postgres.py).

## Stage identity is also a correctness problem

The chunker creates a new identifier and stamps its configuration hash. The
embedder replaces the chunk's vector while retaining the same identifier and
configuration field. Cache buckets identify the embedder configuration, but
all buckets reference the single `documents` row for that identifier; upserts
replace its payload.
[Chunk identity](../../../src/fastrag/ingestion/steps/chunking/base.py),
[embedding transform](../../../src/fastrag/ingestion/steps/embedding/base.py),
[bucket identity](../../../src/fastrag/ingestion/cache/ids.py),
[entity writes](../../../src/fastrag/ingestion/cache/postgres.py).

Consequences derived from those operations:

- A chunk cache read after embedding can return vectors it never produced.
  The same vector can then be transferred again by the embedding cache read.
- Embedder A and B can create distinct buckets pointing to one row. Writing B
  replaces A's vector; reading A's bucket can subsequently serve B's output.
- Omitting vectors from all document reads would break valid embedder hits.
  Projection needs to be explicit and stage-aware, or the payload must be
  isolated by stage.

## Recommended architecture

Separate streaming admission from expensive item execution. Accumulate a
bounded window of upstream items, build their actual child-step cache IDs,
and call the bulk getter once for that window. Then pass the resolved hit or
miss for each item into the existing concurrency, retry, and event machinery.
Do this in both ordinary document steps and routed dispatch; a window can
contain IDs for multiple matched child configurations. Source targets already
exist as a sequence and can be fetched in bounded groups similarly.
[Bulk API](../../../src/fastrag/ingestion/cache/base.py),
[document and routed execution](../../../src/fastrag/ingestion/steps/base.py).

Bound the window by identifiers and, where possible, payload bytes. Permit
early time-based flushing for a slow upstream so the pipeline stays streaming;
avoid retaining the whole crawl. An initial window of 128 is a reasonable
experiment for this workload, not a universal optimum. Preserve retries for
bulk database failures, cancellation, per-item transform failure, source
freshness checks, known-empty results, and event attribution. Explicitly
benchmark memory and time to first output alongside total elapsed time.

Store output snapshots with the identity `(document_cache_id, document_id)`,
either in an output table or as payload columns on cache associations. Preserve
the logical document ID for vector-store deduplication, but make content,
configuration, metadata, and embedding properties of that cached output.
This allows a chunk bucket to retain its original vector-free output while
each embedder configuration retains its own vector. Source-oriented cleanup
needs a source-lineage index on these snapshots, and garbage collection must
touch the owning snapshot rather than an unrelated stage's logical entity.
[Current ownership and cleanup](../../../src/fastrag/ingestion/cache/schema.py),
[administration contract](../../../src/fastrag/ingestion/cache/base.py).

Existing globally shared entity rows cannot reconstruct older stage outputs
after overwrite. A migration must invalidate affected document buckets, or
use a fresh versioned document-output cache schema and repopulate it. Copying
the same current entity payload into every old bucket would preserve the
wrong-vector problem. Source cache rows can be retained independently.

## Avoid embedding payloads for already indexed chunks

The terminal save step already batches `existing_ids` checks, but runs them
after the embedding stage and requires an embedding even for an eventual
skip. An indexing-only orchestration path could check chunk IDs first, route
existing chunks directly to indexing completion events, and embed only missing
chunks. That eliminates embedding cache reads and vector transfer on a warm
destination. The general-purpose embedder should continue producing complete
embedded documents for callers that consume those outputs.
[Store existence contract and save step](../../../src/fastrag/vector_store/base.py),
[embedding output contract](../../../src/fastrag/ingestion/steps/embedding/base.py).

This fast path needs a correctness boundary: `existing_ids` currently answers
only logical ID presence. A changed embedding model, normalization, dimension,
or configuration must select a different index namespace or compare a stored
embedding configuration fingerprint before treating a present chunk as valid.
Otherwise the shortcut, and the current save behavior, can leave old vectors
in place for unchanged text. The embedder configuration hashes already expose
model and dimension settings; local model configuration also includes
normalization.
[Existence semantics](../../../src/fastrag/vector_store/base.py),
[OpenAI configuration](../../../src/fastrag/ingestion/steps/embedding/openai.py),
[Model2Vec configuration](../../../src/fastrag/ingestion/steps/embedding/model2vec.py).

## Implementation order and evidence

First decouple bulk read admission from transform concurrency and count actual
warm SQL rounds in the example pipeline. Next isolate stage output snapshots
and test embedder A, then B, then A again, checking exact vector equality and
chunk buckets remaining vector-free. Finally introduce the explicit
indexing-only existence gate with configuration-aware destination identity.
Measure SQL round counts, bytes read from PostgreSQL, peak memory, time to first
output, total warm runtime, and behavior after changing embedding configuration.
These checks distinguish database latency, duplicate vector transfer, and
destination work instead of attributing all warm time to the cache lookup.

## Implemented iteration and review

The current implementation collects bounded source and document windows before
item execution; document outputs now live in per-bucket associations. Snapshot
payloads retain text and metadata separately from embedding arrays. Header
reads omit large payloads and retain a private reference that can materialize
the owning snapshot before an uncached transform. Source revalidation
materializes bytes before persisting validation timestamps.
[Evaluator](../../../src/fastrag/ingestion/steps/base.py),
[header types](../../../src/fastrag/ingestion/cache/base.py),
[snapshot reads and migration](../../../src/fastrag/ingestion/cache/postgres.py),
[schema](../../../src/fastrag/ingestion/cache/schema.py).

`VectorStore.save(skip_existing=True)` is an explicit indexing-only path for a
bound embedder. It checks destination IDs over chunk windows, bypasses vectors
for present documents, and evaluates embeddings only for missing documents.
Present documents still emit cached `DocumentEmbedded` events, whose documents
may have no embedding, before their `DocumentSaved` events.
Its documented configuration requirement is a separate destination per
embedding configuration. Default saving still consumes complete embedded
documents. [Store orchestration](../../../src/fastrag/vector_store/base.py).

Focused review identified and resolved these correctness boundaries:

- Private header origins stay inside the evaluator. Public source, parser,
  chunker and save events, plus sink results, contain plain model instances.
  A serialization regression checks they expose no backend pointers.
- Route filters can inspect arbitrary entity content. Header-safe metadata
  filters work without hydration. A graph containing custom content filters
  disables header projection so route selection receives complete entities.
  Filter composition preserves this safety requirement.
- Replacing a bucket can orphan logical document rows. Source-oriented deletion
  now removes matching orphan rows immediately as well as deleting snapshots;
  snapshots belonging to another source remain intact.
- Window-wide destination failures report each affected item and preserve
  skip/retry/raise behavior. Missing-item embedding windows run through the
  usual lifecycle wrapper.

[Public filters](../../../src/fastrag/ingestion/filters/base.py),
[routing and events](../../../src/fastrag/ingestion/steps/base.py),
[snapshot cleanup](../../../src/fastrag/ingestion/cache/postgres.py),
[indexing evaluator](../../../src/fastrag/vector_store/base.py).


## Measured example run

The reported command was measured against its configured PostgreSQL and
OpenSearch services, preserving the original 100-page workload. Before the
window and snapshot changes, an instrumented cached run was stopped after
359.39 seconds without completing: it had already issued 643 document-cache
bulk calls and 606 OpenSearch requests. This is an incomplete baseline,
not a completed-run timing comparison.

After upgrading and warming the transformation snapshots, the exact command
`uv run agrospai-docs-chatbot/fetching.py` completed in **5.314 seconds** wall
clock. It reported 100 cached sources, 100 cached parsed documents, 3,292
cached chunks and 3,292 cached destination documents, with no new items or
errors. Other warm runs took about 6.7 seconds; service and network latency
vary.

A separate instrumented final run counted one source-header bulk read, two
document-header bulk reads, eight OpenSearch requests (mapping validation and
seven existence windows), no full-payload cache reads and no embedding provider
calls. It completed in 4.50 seconds after SDK imports. These measurements
include fresh cache startup; schema verification uses one SQL statement.
They are workload-specific observations, not a general throughput guarantee.

The regression suite covers 5,000-output buckets, bounded reads independent
of worker concurrency, empty and missing buckets, cancellation, stage/model
isolation, legacy migrations, lazy source validation, missing-destination
repair, public serialization, content-based routing and destination failures.
[Tests](../../../tests/test_cache.py).
