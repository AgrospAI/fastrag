# Changelog

All notable changes to `fastrag-ai` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until 1.0.0,
a minor release may contain breaking changes.

## [Unreleased]

## [0.2.0] - 2026-10-28

First public release.

### Added

- `fastrag.ingestion`: a lazy, cache-backed `Pipeline` of fetchers (sitemap,
  web, HTTP, local files, YouTube), parsers (HTML, PDF, Docling, Trafilatura,
  vision and audio models), chunkers (recursive, character, heading,
  structure-aware, semantic, Docling) and embedders (Model2Vec, OpenAI), with
  Postgres and SQLite caches.
- `fastrag.vector_store`: Milvus, Chroma and OpenSearch vector stores kept in
  sync with the ingestion cache.
- `fastrag.retrieval`: vector, BM25, hybrid (reciprocal rank fusion),
  language-aware and reranking retrievers, plus a Cohere reranker.
- `fastrag.query`: `QueryEngine`, prompt templates, an OpenAI-compatible chat
  model, and a Postgres chat store.
- `fastrag.serve`: a FastAPI app that streams grounded answers with citations
  over SSE, with sessions, Prometheus metrics, and optional GeoIP country
  resolution.
- `fastrag.benchmark`: retrieval, generation and operational metrics over a
  question-answer dataset.

[Unreleased]: https://github.com/AgrospAI/fastrag/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/AgrospAI/fastrag/releases/tag/v0.2.0
