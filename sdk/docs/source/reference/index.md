# API reference

Every public symbol has exactly one canonical module. The top-level
`fastrag` package re-exports the handful used in almost every script —
{py:class}`~fastrag.models.Document`, {py:class}`~fastrag.models.ContentFormat`,
{py:class}`~fastrag.ingestion.pipeline.Pipeline`,
{py:class}`~fastrag.ingestion.policy.Policy`,
{py:class}`~fastrag.ingestion.models.Source` — and both forms are equivalent.
Names reachable only through a `_`-prefixed module are internal and may
change without notice; see {doc}`/contributing/architecture` for why.

| Symbol          | Canonical module             |
| --------------- | ---------------------------- |
| Every exception | {py:mod}`fastrag.exceptions` |

`fastrag.ingestion` is the ingestion pipeline. Two types it produces or
depends on are shared beyond ingestion, so each lives at the top level
instead of under `fastrag.ingestion`: `Document` (parsed/chunked/embedded by
ingestion, but also what retrieval, querying, serving, and benchmarking will
consume) at `fastrag.models`, and `VectorStore` (written at ingest time, read
at query time) at `fastrag.vector_store`.

| Symbol                                                                                                                                                                                                       | Canonical module                                                                                                                                          |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Source`                                                                                                                                                                                                     | {py:mod}`fastrag.ingestion` (from {py:mod}`fastrag.ingestion.models`)                                                                                     |
| `Document`, `ContentFormat`                                                                                                                                                                                 | {py:mod}`fastrag.models` — shared by retrieval, querying, serving and benchmarking, not ingestion-only, so it is not re-exported from `fastrag.ingestion` |
| `Policy`                                                                                                                                                                                                     | {py:mod}`fastrag.ingestion` (from {py:mod}`fastrag.ingestion.policy`)                                                                                     |
| `Filter`, `MediaTypeFilter`, `URLFilter`, `ContentFormatFilter`                                                                                                                                             | {py:mod}`fastrag.ingestion.filters`                                                                                                                       |
| `PipelineEvent` and its subclasses (`StepStarted`, `StepProgress`, `StepCompleted`, `StepFailed`, `ItemFailed`, `DocumentFetched`, `DocumentParsed`, `DocumentChunked`, `DocumentEmbedded`, `DocumentSaved`, `DocumentsPruned`) | {py:mod}`fastrag.ingestion` (from {py:mod}`fastrag.ingestion.events`)                                                                                     |
| `PipelineStep`, `ExecutionContext`, `SourceStep`, `DocumentStep`, `RoutedStep`                                                                                                                               | {py:mod}`fastrag.ingestion` (from {py:mod}`fastrag.ingestion.steps.base`)                                                                                 |
| `Pipeline`                                                                                                                                                                                                   | {py:mod}`fastrag.ingestion` (from {py:mod}`fastrag.ingestion.pipeline`)                                                                                   |
| `CacheStore`, `PostgresCache`, `SQLiteCache`, `SourceFreshnessValidator`                                                                                                                                     | {py:mod}`fastrag.ingestion.cache`                                                                                                                         |
| `Logger`, `ConsoleLogger`, `DocumentsLogger`, `SourceLogger`, `SourceListLogger`, `SourceTreeLogger`, `ChunkLogger`                                                                                                                                                                                    | {py:mod}`fastrag.ingestion.logger`                                                                                                                        |
| `Fetcher`, `SitemapCrawler`, `WebCrawler`, `FileSource`, `LocalFileFetcher`, `Sources`                                                                                                                        | {py:mod}`fastrag.ingestion.steps.fetching`                                                                                                                |
| `Parser`, `ParserRouter`, `HtmlParser`, `DefaultParser`, `DocusaurusParser`, `TrafilaturaParser`, `DoclingParser`                                                                                                                   | {py:mod}`fastrag.ingestion.steps.parsing`                                                                                                                 |
| `Chunker`, `ChunkerRouter`, `CharacterTextSplitter`, `RecursiveCharacterTextSplitter`, `HeadingChunker`, `DoclingChunker`, `SemanticChunker`, `EmbeddingSemanticChunker`, `LangchainSemanticChunker`, `StructureAwareChunker`, `Tokenizer`, `TiktokenTokenizer`, `Utf8ByteTokenizer`, `build_embedding_text`, `ChatBoundaryRefiner`, `RefinementResult`, `ChunkValidation`, `validate_structure_chunks`                                                                                                        | {py:mod}`fastrag.ingestion.steps.chunking`                                                                                                                |
| `Embedder`, `Model2VecEmbedder`, `OpenAIEmbedder`                                                                                                                                                            | {py:mod}`fastrag.ingestion.steps.embedding`                                                                                                               |
| `VectorStore`, `ChromaVectorStore`, `MilvusVectorStore`, `OpenSearchVectorStore`                                                                                                                                                                            | {py:mod}`fastrag.vector_store`                                                                                                                            |

`fastrag.normalization` interprets parsed content into source-preserving trees.

| Symbol | Canonical module |
| --- | --- |
| `NormalizedDocument`, `NormalizedNode`, `SourceSpan`, `CoverageUnit`, `Exclusion`, `Diagnostic`, `Normalizer`, `NormalizerRegistry`, `TextNormalizer`, `MarkdownNormalizer`, `RstNormalizer`, `HtmlNormalizer` | {py:mod}`fastrag.normalization` |

`fastrag.retrieval` turns a query into the documents that answer it, on top
of a `VectorStore` opened by the caller.

| Symbol                         | Canonical module                                                                        |
| ------------------------------ | --------------------------------------------------------------------------------------- |
| `SectionIndex`, `SectionContext`, `Retriever`, `VectorRetriever`, `BM25Retriever`, `HybridRetriever`, `RankFusion`, `ReciprocalRankFusion` | {py:mod}`fastrag.retrieval` (`VectorRetriever` from {py:mod}`fastrag.retrieval.vector`) |
| `LexicalStore`                 | {py:mod}`fastrag.lexical_store`                                                         |
| `RerankingRetriever`           | {py:mod}`fastrag.retrieval` (from {py:mod}`fastrag.retrieval.reranking_retriever`)      |
| `Reranker`, `CohereReranker`   | {py:mod}`fastrag.retrieval.reranking`                                                   |
| `LanguageAwareRetriever`, `LanguageDetector`, `LinguaLanguageDetector` | {py:mod}`fastrag.retrieval` (from {py:mod}`fastrag.retrieval.language`) |

`fastrag.query` turns a retrieved-and-rendered prompt into a generated
answer. `QueryEngine` composes a `Retriever`, a `PromptTemplate`, and a
`ChatModel` into one `ask`/`stream` surface; chat models live under their own
`fastrag.query.llm` subpackage, the same shape as embedders under
`fastrag.ingestion.steps.embedding`.

| Symbol                                                                            | Canonical module                                                         |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `QueryEngine`, `Answer`, `SourcesChunk`, `TokenChunk`, `DoneChunk`, `StreamChunk` | {py:mod}`fastrag.query` (from {py:mod}`fastrag.query.engine`)            |
| `PromptTemplate`, `DEFAULT_RAG_PROMPT`, `group_sources`                           | {py:mod}`fastrag.query` (from {py:mod}`fastrag.query.prompts.templates`) |
| `ChatModel`, `Message`, `OpenAIChatModel`                                         | {py:mod}`fastrag.query.llm`                                              |

`fastrag.query.chat` persists the conversations a serving layer builds on
top of `QueryEngine`: anonymous sessions, the chats inside them, and the
messages inside those. Storage only — no FastAPI, no HTTP types, no SSE.

| Symbol                                                                                                              | Canonical module                                                          |
| ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `ChatStore`, `Session`, `Chat`, `ChatMessage`, `Page`, `MessageRole`, `FinishReason`, `ErrorCode`, `to_llm_history` | {py:mod}`fastrag.query.chat` (from {py:mod}`fastrag.query.chat.base`)     |
| `PostgresChatStore`                                                                                                 | {py:mod}`fastrag.query.chat` (from {py:mod}`fastrag.query.chat.postgres`) |

`fastrag.benchmark` scores a retrieval or query pipeline against a QA
dataset: load a `QADataset`, run every sample through a `Benchmark` against
a `Retriever` (retrieval metrics) or a `QueryEngine` (generation and
operational metrics too), and compare runs with `Comparison`.

| Symbol                                                                          | Canonical module                                                                           |
| ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `Benchmark`, `BenchmarkTarget`                                                  | {py:mod}`fastrag.benchmark` (from {py:mod}`fastrag.benchmark.benchmark`)                   |
| `Comparison`                                                                    | {py:mod}`fastrag.benchmark` (from {py:mod}`fastrag.benchmark.comparison`)                  |
| `QADataset`, `EvalSample`, `ColumnMapping`                                      | {py:mod}`fastrag.benchmark` (from {py:mod}`fastrag.benchmark.datasets`)                    |
| `BenchmarkResult`, `SampleResult`, `StatusCounts`                               | {py:mod}`fastrag.benchmark` (from {py:mod}`fastrag.benchmark.results`)                     |
| `Metric`, `DistributionSummary`, `QueryOutcome`                                 | {py:mod}`fastrag.benchmark.metrics` (from {py:mod}`fastrag.benchmark.metrics.base`)        |
| `Recall`, `Precision`, `MRR`, `NDCG`, `HitRate`, `RetrievalMetric`              | {py:mod}`fastrag.benchmark.metrics` (from {py:mod}`fastrag.benchmark.metrics.retrieval`)   |
| `TokenOverlap`, `Faithfulness`, `AnswerRelevancy`, `Correctness`, `JudgeMetric` | {py:mod}`fastrag.benchmark.metrics` (from {py:mod}`fastrag.benchmark.metrics.generation`)  |
| `Latency`, `TokenUsage`, `Cost`                                                 | {py:mod}`fastrag.benchmark.metrics` (from {py:mod}`fastrag.benchmark.metrics.operational`) |

`fastrag.serve` is the FastAPI app that exposes a `QueryEngine` plus a
`ChatStore` as the `/v1` chats API over SSE. The app owns no database engine
and no ORM session — it is a thin adapter over the two collaborators a
caller hands to {py:func}`~fastrag.serve.app.create_app`.

| Symbol                             | Canonical module                                               |
| ---------------------------------- | -------------------------------------------------------------- |
| `create_app`                       | {py:mod}`fastrag.serve` (from {py:mod}`fastrag.serve.app`)     |
| `ServeConfig`                      | {py:mod}`fastrag.serve` (from {py:mod}`fastrag.serve.config`)  |
| `CLI`                              | {py:mod}`fastrag.serve` (from {py:mod}`fastrag.serve.cli`)     |
| `cleanup_expired`, `CleanupReport` | {py:mod}`fastrag.serve` (from {py:mod}`fastrag.serve.cleanup`) |

The full tree, generated from docstrings:

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :recursive:

   fastrag
```
