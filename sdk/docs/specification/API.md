> **Historical design record.** Written before implementation, as the
> pre-build design notes for the API surface. It lives outside `docs/source/`, so
> it is never built into the published site and is not maintained in
> lockstep with the code. Most of what it describes shipped close to as
> written; where it and the code disagree, the code and `docs/source/` win —
> see `.agent/PENDING.md` section D for the pieces it names that were never
> built.

# New API specification

- Components:
    - Fetchers: generate a Source[] from various sources (web, local files, S3, etc.)
        - LocalFileFetcher
        - WebCrawler
        - SitemapCrawler
        - FileSource
        - S3Fetcher _(not yet built)_
    - Sources: generate a Source[] from Fetchers. Useful for aggregating multiple fetchers.
    - Parsers: generate a Document[] from Source[].
        - ParserRouter
        - DocusaurusParser
        - DefaultParser (just decodes the content as text, without any parsing)
        - DoclingParser (binary formats — spreadsheets, Word documents, PDFs — via Docling; emits Markdown or a serialized DoclingDocument, optional "docling" extra)
    - Chunkers: generate a Document[] from Document[].
        - ChunkerRouter
        - CharacterTextSplitter
        - RecursiveCharacterTextSplitter
        - DoclingChunker (native chunking of a serialized DoclingDocument — headings/tables/lists on the real structure, via Docling's own HierarchicalChunker/HybridChunker/LineBasedTokenChunker, optional "docling" extra)
        - SemanticChunker (shipped chat grouping; `llm` constructor remains compatible)
    - Embedders: generate a Document[] from Document[].
        - Model2VecEmbedder
        - OpenAIEmbedder
    - VectorStores: processes Document[].
        - MilvusVectorStore
        - ChromaVectorStore _(not yet built)_
    - Pipeline: a context manager that allows listening to events from any step. Returns an async iterator of events.
    - Loggers: processes events from a pipeline.
        - ConsoleLogger
    - Cache: a pluggable cache backend.
    - Policy: a pluggable policy backend for controlling retries, backoff, timeouts, concurrency, etc.
    - Filters: pluggable filters for filtering Document[] or Source[] based on their metadata.
        - MediaTypeFilter
        - URLFilter

> **Current status correction (T09B1).** `SemanticChunker` shipped;
> `StructureAwareChunker` and `EmbeddingSemanticChunker` now provide separate
> structural and embedding-driven algorithms. See the published structure guide.
> `S3Fetcher` and `ChromaVectorStore`
> are named here but do not exist in the current code — they are known gaps
> tracked in `.agent/PENDING.md` section D.

- Fetchers, Sources, Parsers, Chunkers, Embedders and VectorStores are all PipelineSteps. They can be composed into a pipeline.
- Constructing a PipelineStep never executes anything and never emits events (iterators). Evaluating a PipelineStep receives an ExecutionContext, and the evaluator emits lifecycle events while the step may emit domain events through that context.
- Each step should be runnable, so the user can test each step independently:
    - Fetcher().run()
    - Sources().run()
    - Parser().run()
    - Chunker().run()
    - Embedder().run()
- The events are propagated through the pipeline, so the user can listen to events at any step (leveraging contextvars). The only way to listen to events is using the Pipeline class, which is a context manager that sets the event sink for all steps in the pipeline.
- Filters can be composed, for example: htmlFilter & agrospaiFilter.
- Cache is explicit in each step, so the user can choose to cache or not cache each step independently. As cache is problematic, it is better to be explicit about it. Each step's `cache=` argument defaults to `None`, which inherits whatever cache the pipeline (or an enclosing step, e.g. `Sources`/`ParserRouter`/`ChunkerRouter`) is running under; pass `cache=False` explicitly to opt a step out of caching, or a `CacheStore` to use one directly regardless of what it inherits.
- Mental model: source → transform → transform → transform → sink
- The Document class contains all metadata to keep track of the source and to allow filtering.
- Databases schemas are created automatically if they don't exist (for both cache and vector store), and migrations are applied automatically. The user can customize the target database schema.

## Code examples

```python
html_filter = MediaTypeFilter(include_types=["text/html"])

agrospai_filter = URLFilter(
    include_patterns=[
        r"https://agrospai\.udl\.cat/en/.*",
        r"https://agrospai\.udl\.cat/es/.*",
    ],
    exclude_patterns=[
        r"https://agrospai\.udl\.cat/en/sitemap\.xml",
        r"https://agrospai\.udl\.cat/es/sitemap\.xml",
    ],
)
```

```python
async def main():
    policy = Policy(
        retries=3,
        backoff=1.0,
        max_backoff=10.0,
        timeout=30.0,
        on_error="skip",
        concurrency=8,
        batch_size=16,
    )

    async with (
        PostgresCache(...) as cache,
        MilvusVectorStore(..., policy=policy) as vector_store,
    ):
        sources = Sources(
            sources=[
                LocalFileFetcher(...),
                WebCrawler(...),
                SitemapCrawler(...),
                S3Fetcher(
                    ..., cache=False, policy=policy2
                ),  # Sources can override default cache and policy
            ],
            cache=cache,  # default cache for all sources
            policy=policy,  # default policy for all sources
        )

        parser = (
            ParserRouter(cache=cache, policy=policy)
            .route(html_filter, DocusaurusParser(...))
            .default(HtmlParser(...))
        )

        chunker = (
            ChunkerRouter(..., cache=cache, policy=policy)
            .route(html_filter, CharacterTextSplitter(...))
            .route(agrospai_filter, SemanticChunker(...))
            .default(RecursiveCharacterTextSplitter(...))
        )

        embedder = Model2VecEmbedder(..., cache=cache, policy=policy)

        fetched_documents: PipelineStep[Document] = sources
        parsed_documents: PipelineStep[Document] = parser(sources)
        chunked_documents: PipelineStep[Document] = chunker(parsed_documents)
        embedded_documents: PipelineStep[Document] = embedder(chunked_documents)

        # pipeline = Pipeline(fetched_documents) # Only run the fetcher
        # pipeline = Pipeline(parsed_documents) # Only run the fetcher and parser
        # pipeline = Pipeline(chunked_documents) # Only run the fetcher, parser, and chunker
        # pipeline = Pipeline(embedded_documents) # Only run the fetcher, parser, chunker, and embedder
        pipeline = Pipeline(
            vector_store.save(embedded_documents)
        )  # Run the entire pipeline and save to vector store

        # ^^^^^ Nothing is executed until a sink evaluates the pipeline. This is a lazy evaluation model.

        await ConsoleLogger(pipeline).run()

        # Manual event consumption (optional, for debugging)
        # async for event in pipeline:
        #     match event:
        #         case StepStarted():
        #             print(f"Step {event.step} started")
        #         case StepProgress():
        #             print(f"Step {event.step} progress: {event.processed_count} items")
        #         case StepCompleted():
        #             print(f"Step {event.step} completed")
        #         case StepFailed():
        #             print(f"[{event.step}] failed: {event.error}")
        #         # --- Debugging events for each step in the pipeline
        #         case DocumentFetched():
        #             print(f"Fetched {event.source.uri} ({'cached' if event.source.cached else 'new'})")
        #         case DocumentParsed():
        #             print(f"Parsed {event.document.source_id} ({'cached' if event.document.cached else 'new'})")
        #         case DocumentChunked():
        #             print(f"Produced {len(event.chunks)} chunks for {event.document.source_id}")
        #         case DocumentEmbedded():
        #             print(f"Produced embedding for {event.document.source_id}")
        #         case DocumentSaved():
        #             print(f"Saved {event.document.source_id} to vector store")
        #         case ItemFailed():
        #             print(f"[{event.step}] {event.uri} failed: {event.error}")
```

Example output:

```text
+-------------------+----------+----------+----------+-----------+----------+
| Source            | Fetching | Parsing  | Chunking | Embedding | Indexing |
+-------------------+----------+----------+----------+-----------+----------+
| LocalFileFetcher  | 0/2/0    | 0/10/0   | 0/20/0   | 0/0/0     | 0/0/0    |
| WebCrawler        | 0/0/0    | 0/120/0  | 0/0/0    | 0/0/0     | 0/0/0    |
| SitemapCrawler    | 0/0/0    | 0/0/0    | 0/0/0    | 0/0/0     | 0/0/0    |
| S3Fetcher         | 0/0/0    | 0/0/0    | 0/0/0    | 0/0/0     | 0/0/0    |
+-------------------+----------+----------+----------+-----------+----------+
Legend: new/cached/errors
```

## Caching behaviour

Cache validation:

WebCrawler/SitemapCrawler:

1. Try ETag
2. Try Last-Modified
3. Compare content hash

LocalFileFetcher:

1. Compare the file's current size and modification time (`stat()`, no
   read) against what was cached.
2. Reread and rehash only if either differs.

S3Fetcher _(not yet built)_:

1. S3 last modified
2. Compare content hash

---

Use Postgres for caching, a manual SQL query can delete old cache entries, or a background task can be scheduled to do that.

cache_validation_interval = how long a cached entry can be used without validating it against the source. Once the cache is older than this interval, the source is checked for changes (e.g. via ETag, Last-Modified, content hash, etc.). If the source has not changed, the cached entry remains valid and its age is reset. If null, cached entries are always considered fresh and are never validated. If 0, every access validates the cache against the source.

---

```text
Example 1
--

Run:

WebCrawler(start_urls=["a.com", "https://b.com/"], max_depth=3, debug=True)

Cached results:

source_id = normalized("a.com") = "a.com"
source_id = normalized("https://b.com/") = "b.com"

config_hash = hash(webcrawler_version, crawler config (content negotiation, authentication, etc.))

# Cache each page independently to increase cache hit rate.

hash("a.com", config_hash) = "1a2b3c4"
source_cache_id = "webcrawler:1a2b3c4" -> [Source(a.com/1), Source(a.com/2), ...]

hash("b.com", config_hash) = "5d6e7f8"
source_cache_id = "webcrawler:5d6e7f8" -> [Source(b.com/1), Source(b.com/2), ...]

Example 2
--

Run:

ChunkerRouter(...)
    .route(html_filter, CharacterTextSplitter(...))
    .route(agrospai_filter, SemanticChunker(...))
    .default(RecursiveCharacterTextSplitter(...))

Cached results:

config_hash_1 = hash(CharacterTextSplitter config)
config_hash_2 = hash(SemanticChunker config)
config_hash_3 = hash(RecursiveCharacterTextSplitter config)

# Cache each document independently to increase cache hit rate.

hash(matched_document_id, matched_config_hash) = "e21qcasg"
document_cache_id = "charactertextsplitter:e21qcasg" -> [Document(...), Document(...), ...]

Database schema (cache schema)

source_cache
id: str

source_cache_entries
source_cache_id: str
source_id: str

document_cache
id: str

document_cache_entries
document_cache_id: str
document_id: str

sources
id: str (normalized URI)
uri: str
content: bytes
etag: str | None
last_modified: datetime | None
content_hash: str
content_type: str
metadata: dict[str, Any]
last_validated: datetime
last_accessed: datetime # for database cleanup

documents
id: str (content_hash)
source_id: str
step_config_hash: str
content: str
embedding: list[float] | None
metadata: dict[str, Any]
last_accessed: datetime # for database cleanup
```
