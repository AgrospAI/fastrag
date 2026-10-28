<p align="center">
  <a href="https://fastrag.agrospai.udl.cat">
    <img src="https://raw.githubusercontent.com/AgrospAI/fastrag/main/resources/logo/logo.svg" alt="fastrag" width="110">
  </a>
</p>

<h1 align="center">fastrag</h1>

<p align="center">
  <strong>Build RAG chatbots over sources that keep changing.</strong><br>
  Ingest, index, retrieve, generate, serve, and benchmark, in plain typed async Python.
</p>

<p align="center">
  <a href="https://pypi.org/project/fastrag-ai/"><img src="https://img.shields.io/pypi/v/fastrag-ai" alt="PyPI version"></a>
  <a href="https://pypi.org/project/fastrag-ai/"><img src="https://img.shields.io/pypi/pyversions/fastrag-ai" alt="Python versions"></a>
  <a href="https://fastrag.agrospai.udl.cat"><img src="https://img.shields.io/badge/docs-fastrag.agrospai.udl.cat-blue" alt="Documentation"></a>
  <a href="https://github.com/AgrospAI/fastrag/blob/main/sdk/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License: Apache 2.0"></a>
</p>

<p align="center">
  <a href="https://fastrag.agrospai.udl.cat"><b>Documentation</b></a> ·
  <a href="https://fastrag.agrospai.udl.cat/guide/quickstart.html"><b>Quickstart</b></a> ·
  <a href="https://fastrag.agrospai.udl.cat/reference/index.html"><b>API reference</b></a> ·
  <a href="https://github.com/AgrospAI/fastrag"><b>Source code</b></a>
</p>

---

**fastrag** is a Python SDK for the whole life of a retrieval-augmented
generation (RAG) system. You compose typed building blocks into one lazy
pipeline:

**fetch → parse → chunk → embed → index → retrieve → generate → serve**

It is built for knowledge bases that don't sit still, such as documentation
sites, websites, and document folders. Point it at a source and run it on a
schedule. New pages are added, changed pages are re-embedded, deleted pages
are pruned, and everything unchanged comes straight from the cache.

## Why fastrag?

- **Re-index without paying twice.** Every step is cached by its input and its
  configuration. A re-run only processes what changed, and `sync()` removes
  the chunks of pages that no longer exist.
- **A pipeline you own, not a framework you configure.** Steps are plain
  Python objects that you call on each other, evaluated lazily as an async
  stream. There is no YAML, no string registry, and no hidden global state.
- **Swap any piece.** Fetchers, parsers, chunkers, embedders, vector stores,
  retrievers, rerankers, and chat models share small typed interfaces, so
  changing provider means changing one constructor.
- **Serving included.** `create_app()` turns a query engine into a FastAPI
  app with persisted chats, cited answers streamed over Server-Sent Events,
  sessions, rate limiting, and Prometheus metrics.
- **Measure instead of guessing.** Benchmark retrieval and answers against a
  labelled QA dataset and compare configurations side by side.
- **Install only what you use.** The core is small and every integration is an
  optional extra. A missing extra fails with the exact `pip install` command
  to fix it.

## Install

```bash
pip install "fastrag-ai[html,chunkers,model2vec,chroma,serving,openai]"
```

Requires Python 3.13 or newer. These extras cover both examples below, and
the full list is under Optional extras further down.

## Quick look

### 1. Ingest a website

Crawl a sitemap, convert each page to Markdown, split it into chunks, embed
them, and keep a vector collection in sync. This runs entirely on your machine:
SQLite holds the step cache, an embedded Chroma database holds the vectors, and
a small Model2Vec model computes embeddings on the CPU. You don't need a server
or an API key.

```python
import asyncio

from fastrag.ingestion import Pipeline
from fastrag.ingestion.cache import SQLiteCache
from fastrag.ingestion.logger import ConsoleLogger
from fastrag.ingestion.steps.chunking import RecursiveCharacterTextSplitter
from fastrag.ingestion.steps.embedding import Model2VecEmbedder
from fastrag.ingestion.steps.fetching import SitemapCrawler
from fastrag.ingestion.steps.parsing import HtmlParser
from fastrag.vector_store import ChromaVectorStore


async def main():
    async with (
        SQLiteCache(path=".fastrag/cache.sqlite") as cache,
        ChromaVectorStore(path=".fastrag/vectors") as vector_store,
    ):
        crawler = SitemapCrawler(url="https://docs.example.com/sitemap.xml")
        parser = HtmlParser()
        chunker = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        embedder = Model2VecEmbedder()

        # Each step wraps the one before it. Nothing runs until the pipeline does.
        indexed = vector_store.sync(embedder(chunker(parser(crawler))))
        await ConsoleLogger(Pipeline(indexed, cache=cache)).run()


asyncio.run(main())
```

Run it again and unchanged pages come from the cache instead of being parsed
and embedded again:

```text
+----------------+----------+---------+----------+-----------+----------+
| Source         | Fetching | Parsing | Chunking | Embedding | Indexing |
+----------------+----------+---------+----------+-----------+----------+
| SitemapCrawler | 0/5/0    | 0/5/0   | 0/30/0   | 0/30/0    | 0/30/0   |
+----------------+----------+---------+----------+-----------+----------+
Legend: new/cached/errors
```

Every piece is replaceable without touching the others. For example, you can
use `MilvusVectorStore` or `OpenSearchVectorStore` instead of Chroma,
`PostgresCache` instead of SQLite, or `OpenAIEmbedder` instead of Model2Vec.

### 2. Serve it over HTTP

Wrap the same collection in a `QueryEngine` and hand it to `create_app`. You
get a FastAPI app that stores conversations in PostgreSQL and streams grounded
answers over Server-Sent Events. Generation works with any OpenAI-compatible
endpoint: OpenAI, Ollama, vLLM, or llama.cpp.

```python
import asyncio

import uvicorn

from fastrag.ingestion.steps.embedding import Model2VecEmbedder
from fastrag.query import QueryEngine
from fastrag.query.chat import PostgresChatStore
from fastrag.query.llm import OpenAIChatModel
from fastrag.retrieval import VectorRetriever
from fastrag.serve import ServeConfig, create_app
from fastrag.vector_store import ChromaVectorStore


async def main():
    async with (
        ChromaVectorStore(path=".fastrag/vectors") as vector_store,
        PostgresChatStore(
            database_url="postgresql://postgres:postgres@localhost:5432/fastrag"
        ) as chat_store,
    ):
        retriever = VectorRetriever(
            vector_store=vector_store,
            embedder=Model2VecEmbedder(),  # the same model the collection was built with
        )
        engine = QueryEngine(
            retriever=retriever,
            llm=OpenAIChatModel(
                model="llama3.2:3b",
                api_key="ollama",
                base_url="http://localhost:11434/v1",  # omit to use OpenAI itself
            ),
        )
        app = create_app(
            engine=engine,
            chat_store=chat_store,
            config=ServeConfig(allowed_origins=["http://localhost:3000"]),  # your frontend
        )
        await uvicorn.Server(uvicorn.Config(app, port=8000)).serve()


asyncio.run(main())
```

Ask it a question, and the sources arrive before the first token:

```bash
curl -N http://localhost:8000/v1/chats \
  -H 'content-type: application/json' \
  -d '{"content": "How do I get started?"}'
```

```text
event: message_start
data: {"chat_id": "01M6…", "user_message_id": "01M6…", "message_id": "01M6…"}

event: sources
data: {"message_id": "01M6…", "sources": [{"url": "https://docs.example.com/getting-started", "title": "Getting started", …}]}

event: delta
data: {"message_id": "01M6…", "delta": {"type": "text", "text": "Install"}}

…

event: message_end
data: {"message_id": "01M6…", "finish_reason": "stop"}
```

The app also serves chat history, message editing and regeneration,
`/healthz`, and Prometheus `/metrics`. See the
[HTTP API](https://fastrag.agrospai.udl.cat/guide/serving/http-api.html) and
[streaming](https://fastrag.agrospai.udl.cat/guide/serving/streaming.html)
references for the full contract.

If you don't need HTTP, the same engine answers directly from Python:

```python
answer = await engine.ask("How do I get started?")
print(answer.text)     # the answer, ending with the documents it used, e.g. [1][3]
print(answer.sources)  # the retrieved chunks, with their metadata
```

## What's in the box

| Stage     | Built-in components                                                                                                                   |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Fetch     | `SitemapCrawler`, `WebCrawler`, `LocalFileFetcher`, `YouTubeFetcher`; JavaScript-rendered pages through `PlaywrightRenderer`          |
| Parse     | `HtmlParser`, `DocusaurusParser`, `TrafilaturaParser`, `DoclingParser`, PDF parsers (`MineruParser`, `DoclingServeParser`), `ParserRouter` |
| Chunk     | `RecursiveCharacterTextSplitter`, `HeadingChunker`, `StructureAwareChunker`, `SemanticChunker`, `DoclingChunker`                      |
| Embed     | `Model2VecEmbedder` (local), `OpenAIEmbedder` (any OpenAI-compatible API)                                                             |
| Store     | `MilvusVectorStore` and `OpenSearchVectorStore` (dense and BM25), `ChromaVectorStore`                                                 |
| Cache     | `SQLiteCache`, `PostgresCache`                                                                                                        |
| Retrieve  | `VectorRetriever`, `BM25Retriever`, `HybridRetriever` (rank fusion), `RerankingRetriever` with `CohereReranker`, `LanguageAwareRetriever` |
| Generate  | `QueryEngine` with `OpenAIChatModel`, using `ask()` for a whole answer or `stream()` for tokens                                       |
| Serve     | `create_app` (FastAPI, SSE, PostgreSQL-backed chats), `CLI` terminal client                                                           |
| Benchmark | `Benchmark`, `QADataset`, `Comparison`, with retrieval, generation, and operational metrics                                           |

If none of these fit, each kind of component has a small base class to
implement. See
[Customizing](https://fastrag.agrospai.udl.cat/guide/customizing.html).

## Optional extras

| Extra           | Enables                                                                      |
| --------------- | ---------------------------------------------------------------------------- |
| `html`          | `HtmlParser`, HTML-to-Markdown conversion                                    |
| `trafilatura`   | `TrafilaturaParser`, main-content extraction for arbitrary web pages         |
| `docling`       | `DoclingParser` and `DoclingChunker` for PDFs, Office documents, spreadsheets |
| `pdf`           | `pypdf`, used by the PDF parsers                                             |
| `dynamic`       | `PlaywrightRenderer`, for pages rendered by JavaScript                       |
| `video`         | `YouTubeFetcher`                                                             |
| `chunkers`      | `RecursiveCharacterTextSplitter` and the other LangChain-based chunkers      |
| `normalization` | Token-aware structural chunking and Markdown/RST normalization               |
| `model2vec`     | `Model2VecEmbedder`, local embeddings                                        |
| `openai`        | `OpenAIEmbedder`, `OpenAIChatModel`                                          |
| `milvus`        | `MilvusVectorStore`                                                          |
| `chroma`        | `ChromaVectorStore`                                                          |
| `language`      | `LinguaLanguageDetector`, for `LanguageAwareRetriever`                       |
| `serving`       | `create_app`, `CLI`                                                          |
| `geoip`         | Country resolution for serving sessions                                      |
| `datasets`      | Reserved for YAML, Parquet, and Hugging Face `QADataset` loading (planned)   |
| `logging`       | Rich console output for the ingestion loggers                                |


## Documentation

The full documentation is at
**[fastrag.agrospai.udl.cat](https://fastrag.agrospai.udl.cat)**.

- [Quickstart](https://fastrag.agrospai.udl.cat/guide/quickstart.html): index a site and ask it a question.
- [Concepts](https://fastrag.agrospai.udl.cat/guide/concepts.html): the four kinds of object everything else builds on.
- [Guide](https://fastrag.agrospai.udl.cat/guide/index.html): [ingestion](https://fastrag.agrospai.udl.cat/guide/ingestion.html), [caching](https://fastrag.agrospai.udl.cat/guide/caching.html), [indexing and sync](https://fastrag.agrospai.udl.cat/guide/indexing.html#syncing), [retrieval](https://fastrag.agrospai.udl.cat/guide/retrieval.html), [querying](https://fastrag.agrospai.udl.cat/guide/querying.html), and [benchmarking](https://fastrag.agrospai.udl.cat/guide/benchmarking.html).
- [Serving](https://fastrag.agrospai.udl.cat/guide/serving/index.html): the HTTP API, SSE stream, metrics, and rate limiting.
- [Examples](https://fastrag.agrospai.udl.cat/guide/examples.html): runnable scripts, from a single fetch to a served, benchmarked pipeline.
- [API reference](https://fastrag.agrospai.udl.cat/reference/index.html): every public class and function.

## Project status

fastrag is in alpha. The API may change between minor releases, and each
release is recorded in the
[changelog](https://github.com/AgrospAI/fastrag/blob/main/sdk/CHANGELOG.md).
Bug reports and questions are welcome on the
[issue tracker](https://github.com/AgrospAI/fastrag/issues).

## Contributing

fastrag is developed in the open at
[github.com/AgrospAI/fastrag](https://github.com/AgrospAI/fastrag). The
[contributor guide](https://fastrag.agrospai.udl.cat/contributing/index.html)
covers the development setup, tests, conventions, and architecture. Releases
are described in the
[repository README](https://github.com/AgrospAI/fastrag#releasing-the-sdk).

## License

Licensed under the
[Apache License, Version 2.0](https://github.com/AgrospAI/fastrag/blob/main/sdk/LICENSE).
Built by [AgrospAI](https://agrospai.udl.cat) at the Universitat de Lleida.
