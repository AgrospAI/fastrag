# Indexing

Write embedded documents into a vector collection. Indexing is the sink end
of {doc}`ingestion`: rather than a separate sync orchestration object, you
open a `VectorStore` and call `vector_store.save(pipeline_step)` — the save
is itself a `PipelineStep` you compose onto the graph and evaluate.

```{literalinclude} /../../examples/05_indexing.py
:language: python
:start-after: "docs:indexing-save-start"
:end-before: "docs:indexing-save-end"
:dedent:
```

(`embedded_documents` is whatever the ingestion graph produced — see
{doc}`ingestion` and `examples/05_indexing.py` for the steps that build it.)

## The `VectorStore` contract

`VectorStore` is the write-path interface a sink needs, and the read path
({doc}`retrieval`) searches through. A concrete store implements:

- `ensure_ready(dimension)` — make sure the collection exists for that
  embedding dimension.
- `upsert(documents)` — write documents that are not already present.
- `existing_ids(ids)` — which of the given ids the collection already holds,
  so a batch's cached documents are skipped rather than re-upserted and
  reported `cached=True`.
- `search(embedding, *, k, filter=None)` — the query path ({doc}`retrieval`).

`sync()` also needs `all_ids()` (every id the store holds) and `delete(ids)`.
They are optional: a custom store that does not implement them supports
everything except `sync()`.

`save(documents)` is concrete on the base and returns the batched terminal
`_SaveStep` that drains the graph, dedupes by id, and upserts.
`MilvusVectorStore`, `OpenSearchVectorStore` and `ChromaVectorStore`
implement the same contract.

BM25 is a separate contract, `LexicalStore` (`fastrag.lexical_store`), with one
method: `lexical_search(query, *, k, filter=None)`. A vector store that also
ranks text implements both (`MilvusVectorStore`, `OpenSearchVectorStore`); one
that does not (`ChromaVectorStore`) implements only `VectorStore`, and lexical
retrieval comes from a separate backend implementing only `LexicalStore`.

Chroma (`pip install "fastrag-ai[chroma]"`) connects to a server with
`url=` or runs embedded against a local directory with `path=`. Document
metadata is stored as one JSON string; `source_id`, `source_uri` and
`step_config_hash` are separate keys for `where` filters. `score` is
Chroma's distance, so lower is better.

OpenSearch creates a text field with BM25 similarity and a Lucene HNSW
`knn_vector` field. Configure `url`, `index`, `user` and `password`; HTTPS
certificate verification is disabled and no certificate files are required. It uses the existing
HTTP dependency, so no additional OpenSearch extra is needed.

Milvus enables server-side BM25 with `enable_bm25=True` (Milvus 2.5+). This
creates an analyzed content field, sparse vector, BM25 function and sparse
index alongside the dense vector. Existing dense-only collections must be
reindexed into a new collection. The default remains dense-only for backward
compatibility. Both backends validate existing embedding dimensions.

## Fast cached ingestion

For a bound, identity-preserving embedder, opt into checking the destination
before loading embeddings:

```python
indexed = vector_store.save(embedder(chunker(parser(fetcher))), skip_existing=True)
```

The sink checks chunk IDs in windows of `policy.cache_batch_size`, then loads
cached vectors or calls the embedding provider only for missing IDs. Upserts
use `policy.batch_size`. The graph remains reusable and ordinary `save()` and
embedder execution continue returning complete documents.

Already indexed results carry `cached=True`. In this mode, cached events and
results can contain only identity and metadata, with empty content and no
embedding. Use ordinary execution when you need the complete payloads.
Missing-document embedding windows emit embedding lifecycle events; store
failures emit `ItemFailed` for affected documents and obey the store policy.

Use a separate collection/index for each embedding configuration. ID presence
alone does not verify that existing vectors match a changed model. Built-in
routing filters support headers; custom filters receive full cached records
unless they explicitly declare `header_safe=True` for metadata-only matching.

Whether or not `skip_existing` is set, `save()` never rewrites an ID the store
already holds — unless the policy has `recreate_cache=True`, on the store
(`MilvusVectorStore(..., policy=Policy(recreate_cache=True))`) or the whole
pipeline. Every document is then upserted over the stored one, which is how
changed metadata, such as a newly extracted site name, reaches chunks that are
already indexed. Setting it on the store alone replays every other step from
the cache.

## Syncing

`save()` only adds documents. When a page is removed or changes, the chunks
it used to produce stay in the store (a changed chunk gets a new id).
`sync()` takes the same arguments as `save()`, and once the upstream is
exhausted it deletes every stored document the run did not yield. The store
then holds exactly the run's documents:

```python
indexed = vector_store.sync(embedder(chunker(parser(fetcher))), skip_existing=True)
```

It deletes only after a clean run. A run that raises or is stopped early
deletes nothing. A run with any `ItemFailed` deletes nothing either: a page
that failed to fetch or parse is missing from the run, not from the site, and
its chunks would be deleted with the genuinely stale ones. A
`DocumentsPruned` event reports how many stale documents there were and how
many were deleted, and `ConsoleLogger` prints it at the end of the run.

Because the run defines the whole store, give each synced pipeline its own
collection/index. Two pipelines syncing into one collection delete each
other's documents.

## Strong reads

Milvus searches at strong consistency; OpenSearch bulk writes use
`refresh=wait_for` and existence checks use real-time multi-get. An index-then-retrieve in the same
program sees its own writes — the read-your-writes guarantee
`existing_ids` already provides for indexing.

## See also

{doc}`ingestion` for the steps feeding `save`; {doc}`retrieval` for the
search side; `examples/05_indexing.py` for the full pipeline to sink.
