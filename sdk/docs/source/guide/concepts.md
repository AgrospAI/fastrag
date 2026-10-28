# Concepts

Every symbol in `fastrag` is one of four kinds, and knowing which one you are
looking at tells you when you would ever replace it.

```{mermaid}
flowchart LR
  Source -->|SourceStep| Parser -->|Document| Chunker -->|Document| Embedder
  Embedder -->|save| VectorStore
  Retriever --> VectorStore
  QueryEngine --> Retriever
  QueryEngine --> ChatModel
  Serving[serving app] --> QueryEngine
```

| Kind         | You swap it when                             | Examples                                                                                  |
| ------------ | -------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **Artifact** | Never — it is the data passed between stages | `Document`, `Source`, `Message`, `ChatMessage`                                            |
| **Service**  | You change provider                          | `Embedder`, `VectorStore`, `ChatModel`, `Reranker`, `CacheStore`, `ChatStore`             |
| **Step**     | You change how content is transformed        | `SourceStep` (`SitemapCrawler`, `WebCrawler`, `Sources`), `Parser`, `Chunker`, `Embedder` |
| **Pipeline** | You change how the parts compose             | `Pipeline`, `Retriever`, `QueryEngine`, `Benchmark`                                       |

Two cross-cutting facilities sit outside this table because they wrap all of
it rather than belonging to one stage: a `CacheStore` in front of every step
(does this input need recomputing?) and the event stream a `Pipeline`
installs. {doc}`caching` and {doc}`configuration` cover the cache.
{doc}`ingestion` covers the event stream.

## Artifacts

Inert typed data. No I/O, no configuration, and nothing about them ever
needs swapping — a `Document` is a `Document` for every source, provider, and
pipeline.

- `Source` — raw fetched bytes, before parsing.
- `Document` — a unit of text: a parsed page, or a chunk of one.
- `Message` — one chat turn; Fastrag's own type, so a `ChatModel` never leaks
  a provider's.
- `ChatMessage` — a persisted conversation turn (id, timestamps, sources, a
  generation outcome); distinct from the model-facing `Message`.

## Services

Stateful clients for something outside the process — a model API, a vector or
chat database. You reach for a different implementation when you change
provider, not when you change what you are trying to do.

- `Embedder` — text to vectors.
- `VectorStore` — where vectors and their chunks live (`MilvusVectorStore`).
- `ChatModel` — messages to a generated answer (`OpenAIChatModel`).
- `Reranker` — scores documents against a query with a cross-encoder
  (`CohereReranker`).
- `CacheStore` / `ChatStore` — persistence: the step cache, and serving's
  conversations.

## Steps

A transformation inside the ingestion pipeline. Every `DocumentStep`
declares a `config()` over its own configuration — the contract the whole
cache layer is built on ({doc}`caching`).

- `SourceStep` — a location to raw bytes (`SitemapCrawler`, `WebCrawler`,
  `Sources`).
- `Parser` — raw bytes to text (`ParserRouter`, `HtmlParser`,
  `DefaultParser`, `DocusaurusParser`, `TrafilaturaParser`).
- `Chunker` — one document to the chunks that get embedded
  (`RecursiveCharacterTextSplitter`).
- `Embedder` — text to vectors, as a step (`Model2VecEmbedder`,
  `OpenAIEmbedder`).

## Pipelines

Orchestration that composes steps and services into something you call
directly. You change _how the parts compose_, not what any one part does.

- `Pipeline` — evaluates a lazy step graph and yields its event stream.
- `Retriever` — turns a query into ranked documents (`VectorRetriever`,
  `RerankingRetriever`).
- `QueryEngine` — retrieval plus generation, one question at a time.
- `Benchmark` — runs a dataset through either of the two above and scores it.

No module paths, no private names, no rationale here — that is what
{doc}`ingestion` through {doc}`benchmarking` are for. This page exists so you
know which chapter to open next, and {doc}`customizing` pays it off: each
kind has exactly one extension recipe.

## See also

{doc}`index` for the reading order; {doc}`customizing` to add a fifth member
to any of these lists.
