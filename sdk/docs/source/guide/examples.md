# Examples

`examples/` in the repository is a runnable, numbered walk through the whole
SDK — ingest → index → retrieve → rerank → query → stream → serve → ask →
benchmark — each file introducing one new piece and reusing everything the
previous ones already built. Examples `01`–`16` read execution-time environment
overrides through `examples/_runtime.py`; defaults target the Compose services
and public sitemap. Use an isolated cache schema and collection for local smoke
runs. The Docling scripts also accept execution-time XLSX and vision settings.
The AgrospAI fetching script separately exposes CLI/environment configuration;
querying loads its adjacent dotenv at execution without replacing existing
environment values and shares fetching’s runtime configuration.
See {doc}`/contributing/setup` for repository service prerequisites.

```bash
cp .env.example .env            # optional; the defaults match Compose
docker compose up -d
task index                      # or: uv run python examples/05_indexing.py
```

The indexing examples run entirely against the Compose stack — PostgreSQL
for the cache, Milvus for the vectors, and `Model2VecEmbedder` computing
embeddings in-process — no API key, no remote endpoint. The querying and
serving examples add Ollama, which Compose also starts and pre-pulls.

| File                     | Subject                                                                                   |
| ------------------------ | ----------------------------------------------------------------------------------------- |
| `01_fetching.py`         | A `SitemapCrawler` alone: sources out of a sitemap, cached.                               |
| `02_parsing.py`          | Feed fetched sources into an `ParserRouter`.                                                |
| `03_chunking.py`         | Chunk a parsed document with `RecursiveCharacterTextSplitter`.                            |
| `04_embedding.py`        | Embed the chunks with `Model2VecEmbedder`.                                                |
| `05_indexing.py`         | The whole fetch → parse → chunk → embed → `MilvusVectorStore.save` pipeline in one graph. |
| `06_logging.py`          | The same pipeline under `Pipeline` + `ConsoleLogger`'s table.                             |
| `07_advanced_logging.py` | The raw `Pipeline` event stream.                                                          |
| `08_sources.py`          | A `Sources` aggregate over a sitemap plus a web crawl.                                    |
| `09_filters.py`          | `ParserRouter`/`ChunkerRouter` routing with `MediaTypeFilter`/`URLFilter`.                    |
| `10_retrieving.py`       | `VectorRetriever`: the top `k` chunks for one question.                                   |
| `11_reranking.py`        | `RerankingRetriever` + `CohereReranker`: over-fetch, then narrow.                         |
| `12_querying.py`         | Retrieve context, render a `PromptTemplate`, invoke `OpenAIChatModel`.                           |
| `13_streaming.py`        | Render retrieved context, then stream `OpenAIChatModel` tokens.                                    |
| `14_serving.py`          | Serve the grounded-answer API on `:8100` (`task serve`).                                  |
| `15_asking.py`           | The `CLI` client for that API (`task ask`), in another terminal.                          |
| `16_benchmarking.py`     | A chunk-size sweep scored with retrieval metrics, saved and compared.                     |
| `docling/parse_xlsx.py`  | `FileSource` + `DoclingParser`: a real XLSX converted to Markdown, its embedded images described by a remote vision model. |
| `docling/chunk_xlsx.py`  | The same XLSX chunked two ways: `RecursiveCharacterTextSplitter` versus `DoclingChunker`.  |

The two `docling/` scripts stand outside the numbered sequence: each is
self-contained (`FileSource(cache=False)`), needs no `docker compose up -d`,
and demonstrates the optional `docling` extra's `DoclingParser`/
`DoclingChunker` on a real spreadsheet rather than building on the example
before it. `parse_xlsx.py` is the one exception to "no services": describing
the spreadsheet's images needs an OpenAI-compatible vision endpoint, read
from `FASTRAG_VISION_BASE_URL`, `FASTRAG_VISION_API_KEY` and `FASTRAG_VISION_MODEL` (defaults use local Ollama).

Run any pipeline example twice to see the shape caching is supposed to have:

```text
run 1:  fetched 20 / cached 0  |  parsed 20 / cached 0  |  chunked N / cached 0
run 2:  fetched 0  / cached 20 |  parsed 0  / cached 20 |  chunked 0  / cached N
```

Parse and chunk are keyed on content, not URI ({doc}`caching`), so two
sources that happen to fetch the same bytes from two different hosts parse
them exactly once between the two.

`examples` runs against whatever is on disk in `src/` (the package is
installed editable by `uv sync`), which is what makes CI catch a public-API
break before a release ships one.

## See also

{doc}`quickstart` for the same shape compressed into one file;
{doc}`serving/index` for what `14_serving.py` and `15_asking.py` build on.

## Structural alternative

The sequence retains the character-splitter baseline for comparison. For exact
embedding budgets, source maps and parent context, see {doc}`structure`; replace
the chunking stage while preserving the rest of the lazy graph. The AgrospAI
fetching script now defaults to structural chunks and indexes unless an explicit
inspection checkpoint is selected. Remote fixture sites and provider availability
are separate from SDK correctness; the final example smoke workflow must exercise
the same main paths with bounded local inputs and isolated destinations.

## Execution-time overrides

These examples retain the real SDK crawler, parser, splitter, Model2Vec, store
and retrieval paths. They fail on empty ingestion output or missing retrieved
context; logger examples require successful destination events. Provider
failures are surfaced. Importing an example does not run its main.

| Environment variable | Purpose / default |
| --- | --- |
| `FASTRAG_SITEMAP_URL` | Public AgrospAI sitemap; select a fixture sitemap for smoke runs. |
| `FASTRAG_WEB_URL` | Second crawler root for 08/09; defaults to Pontus-X docs. |
| `FASTRAG_MAX_URLS` | Positive finite crawler limit, default 200. |
| `FASTRAG_DATABASE_URL` / `FASTRAG_CACHE_SCHEMA` | PostgreSQL endpoint / cache namespace. |
| `FASTRAG_MILVUS_URI` / `FASTRAG_COLLECTION` | Milvus endpoint / explicit collection, default `fastrag_documents`. |
| `FASTRAG_LOCAL_EMBEDDING_MODEL` | Model2Vec model, default `minishlab/potion-base-8M`. |
| `FASTRAG_RENDERER` | `http` (default) or `browser` (Playwright). |
| `FASTRAG_ROUTE_PATTERN` | URL regex for the Docusaurus route in 09; defaults to any host’s `/docs/` path. |
| `FASTRAG_QUESTION` | Retrieval/chat question, default “What is AgrospAI?”. |
| `FASTRAG_CHAT_BASE_URL` / `FASTRAG_CHAT_MODEL` / `FASTRAG_CHAT_API_KEY` | OpenAI-compatible chat endpoint/model/key for 12–14 and the optional benchmark judge. |
| `FASTRAG_RERANKER_BASE_URL` / `FASTRAG_RERANKER_MODEL` / `FASTRAG_RERANKER_API_KEY` | Cohere-compatible rerank endpoint/model/key for 11. |
| `FASTRAG_TIMEOUT` | Positive operation/provider timeout in seconds, default 60; ingestion retries are zero. |

Reuse the same collection and local embedding model when running 10–15 after
05. These examples do not delete collections or schemas. The service-backed
regressions in `tests/test_examples.py` execute the indexing, logging and
retrieval mains against generated test namespaces and the fixture HTTP site,
then independently check stored vectors and source URLs. They also execute
server health/authenticated SSE, the real interactive CLI, separate benchmark
collections and both pictured XLSX mains.


| Additional environment variable | Purpose / default |
| --- | --- |
| `FASTRAG_SERVE_HOST` / `FASTRAG_SERVE_PORT` | 14 bind address, default `127.0.0.1:8100`. |
| `FASTRAG_CHAT_SCHEMA` | 14 Postgres chat namespace, default `fastrag_chat`. |
| `FASTRAG_SERVE_SECONDS` | Optional nonnegative lifetime; zero serves until interrupted. |
| `FASTRAG_BACKEND_URL` / `FASTRAG_CLI_TOKEN_FILE` | 15 backend / owned token location; default backend `http://127.0.0.1:8100`. |
| `FASTRAG_BENCHMARK_SIZES` | Distinct integers >=2, comma separated; default `500,800,1200`. |
| `FASTRAG_BENCHMARK_DATASET` / `FASTRAG_BENCHMARK_RESULTS` | 16 QA JSON path / results directory. |
| `FASTRAG_XLSX_URL` / `FASTRAG_CHUNKS_TO_SHOW` | Docling workbook URL / positive displayed chunk limit (default 3). |
| `FASTRAG_VISION_BASE_URL` / `FASTRAG_VISION_MODEL` / `FASTRAG_VISION_API_KEY` | Explicit remote vision service for pictured XLSX; default local Ollama/llava. |

16 opens `<FASTRAG_COLLECTION>_recursive_<size>` for each configuration. Its
retriever reads that same store; the save fingerprint does not select the
collection. Each run requires nonempty successful results without errored or
incomplete samples. Comparison uses only results from the current sweep. The optional `--judge` phase
uses TokenOverlap/Faithfulness/Latency; the SDK chat contract returns plaintext,
so token usage is unavailable and no token counts are fabricated.

## Owned local smoke workflow

From the SDK root, after installing all extras:

```bash
.venv/bin/python scripts/test_services.py start
.venv/bin/python scripts/test_services.py probe
.venv/bin/python scripts/example_smoke.py
```

The workflow invokes all 20 original script entry points, and runs fetching again
for warm replay and its compatible OpenAI-embedding path in a separate collection. It starts a loopback fixture site and generated workbook with an
embedded picture, plus deterministic compatible chat/SSE/vision/embedding/rerank
HTTP APIs. These APIs validate protocol plumbing, not model quality. The numbered
embedding examples and AgrospAI pipeline use actual local Model2Vec weights;
PostgreSQL, Milvus, Docling and the optional browser renderer remain real.

Unique UUID-owned schemas/collections, benchmark destinations, token file and
results are torn down after each run, including failures. Service containers stay
available for other tests. The workflow never selects default collections or
production providers. Every script subprocess has a finite timeout; 14 must pass
health and authenticated source/text/end SSE, 15 consumes a scripted question and
`/quit`, and 14 must exit cleanly when its lifetime expires.

Select any original path using `--only` (repeatable); required indexing is performed
automatically. Browser fetching can be exercised with the installed Playwright
browser via `--renderer browser`:

```bash
.venv/bin/python scripts/example_smoke.py --only examples/09_filters.py --renderer browser
.venv/bin/python scripts/example_smoke.py --only examples/docling/parse_xlsx.py
.venv/bin/python scripts/example_smoke.py --only examples/14_serving.py --only examples/15_asking.py
```

These local fixture runs complement real external-site/provider attempts; they do
not establish the availability or answer quality of external services.
