# Conventions

Rules that hold across every module, regardless of layer. When a design
choice inside one file seems to contradict something elsewhere, one of
these is usually why — and if a change would violate one, that is a
decision to raise explicitly, not to route around silently.

## Async at process boundaries, sync everywhere else

> **A method is `async` if and only if it performs I/O to another process**
> — network, database, or disk cache. CPU-bound work is a plain function.

`Embedder.embed`, `VectorStore.*`, `CacheStore.*`, `ChatModel.invoke`/
`.stream`, `Reranker.rerank`, `SourceStep.evaluate`, `DocumentStep.evaluate`,
`Retriever.retrieve`, `QueryEngine.ask` / `.stream`, `ChatStore.*`, and
`Benchmark`'s judge-backed metrics are all `async`. `Parser.transform`,
`Chunker.split`, `PromptTemplate.render`, `DistributionSummary.of`, and
`Comparison.to_markdown` are plain functions.

**One name per operation.** There is no `run`/`arun` pair and no
`embed_documents`/`aembed_documents` pair.

Two exceptions worth knowing, so a docstring pass does not "fix" them:

- **`Metric.score` is `async def` on every metric**, judge-backed or not —
  including `Recall`, which does no I/O at all. A uniform signature lets
  `Benchmark` await every metric identically instead of branching on whether
  a particular one happens to need a network call.
- **`BenchmarkResult.save` / `.load` are plain functions.** They write a
  local file with `pathlib.Path`, which the rule above does not require to be
  async — the boundary is "another process," not "any I/O."

## Lazy by default

Every ingestion stage is a generator. `SourceStep.evaluate` yields an
`AsyncIterator[Source]`; `Parser.transform` and `Chunker.split` yield
`Document`s; composing steps with `__call__` returns a lazier `PipelineStep`
that performs no I/O until it is evaluated. A source with 50,000 documents
must never be materialized as a list. The pipeline accumulates only at the
two points a provider API forces batching — embedding and upserting —
bounded by `Policy.batch_size` ({doc}`/guide/configuration`).

## No ambient configuration

Every credential, endpoint, and model identifier is an explicit constructor
argument. Nothing importable as `fastrag.*` reads an environment variable or
loads a `.env` file — see {doc}`/guide/configuration` for what that looks
like from the caller's side. There is no exception: the cache and chat-store
schemas are self-managed (created on `__aenter__`), so nothing in the package
reads the environment on a caller's behalf.

## LangChain is an implementation detail, never a type

`fastrag` defines its own `Document`, `Embedder`, `VectorStore`, `ChatModel`,
and `Reranker`. LangChain appears only inside specific built-in
implementations — `RecursiveCharacterTextSplitter` delegates to
`langchain_text_splitters`; `OpenAIEmbedder` and `OpenAIChatModel` delegate
to `langchain_openai` — and callers never see it. No public signature in
`fastrag.*` accepts or returns a LangChain type, so an upstream breaking
change in LangChain cannot break Fastrag's own API.

## Naming convention

| Kind            | Rule                                                  | Examples                                                                          |
| --------------- | ----------------------------------------------------- | --------------------------------------------------------------------------------- |
| Interface (ABC) | Bare agent noun, no `I`, `Base`, or `Abstract` prefix | `SourceStep`, `Parser`, `Chunker`, `Embedder`, `Retriever`, `ChatStore`           |
| Implementation  | `<Qualifier><Interface>`                              | `SitemapCrawler`, `HtmlParser`, `RecursiveCharacterTextSplitter`, `PostgresCache` |
| Provider-backed | Qualifier is the provider                             | `OpenAIEmbedder`, `MilvusVectorStore`, `CohereReranker`, `OpenAIChatModel`        |
| Data artifact   | Bare noun, no suffix                                  | `Document`, `Source`, `Message`, `ChatMessage`                                    |
| Pipeline object | Bare noun naming the thing, not the act               | `Pipeline`, `QueryEngine`, `Benchmark`                                            |
| Result object   | `<Noun>Result` for evaluation                         | `BenchmarkResult`, `CleanupReport`                                                |

Two consequences worth stating:

- **One concept, one word.** "Store" means exactly one thing: a persistence
  client, prefixed by its domain — `VectorStore`, `CacheStore`,
  `ChatStore`. There is no bare `Store` façade.
- **Splitters are chunkers.** The interface is `Chunker`; every
  implementation ends in `Chunker` (or is a splitter class it delegates to),
  and the word "splitter" names nothing in Fastrag's own public surface even
  though `RecursiveCharacterTextSplitter` is publicly importable.

## Public API and stability

The public API is exactly what each package's `__init__.py` re-exports plus
the canonical modules in {doc}`/reference/index`. Anything reachable only
through a `_`-prefixed module path is internal and may change without notice
({doc}`architecture`). The package ships `py.typed`.

## See also

{doc}`architecture` for where a module sits and what it may import;
{doc}`documentation` for the equivalent rules governing where a _sentence_
belongs, not a symbol.
