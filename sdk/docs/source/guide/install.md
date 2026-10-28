# Install

`fastrag` is a library, not a platform: `pip install` it, construct the
pieces your pipeline needs, and call them from your own script. There is no
console command and no daemon to run.

```bash
pip install "fastrag-ai[model2vec,milvus,openai]"
```

## Requirements

- **Python 3.13 or newer.**
- **A reachable PostgreSQL server**, for anything that persists state —
  `PostgresCache` (the step cache) and `PostgresChatStore` (conversations in
  the serving layer) need one. `asyncpg` and `sqlalchemy` are core
  dependencies, so nothing extra installs it — you provide the server.
- **A vector store.** `MilvusVectorStore`
  supports dense vectors and optional BM25; `OpenSearchVectorStore` supports
  dense and lexical search through its HTTP API.
- **A chat endpoint for generation** — any OpenAI-compatible one (OpenAI,
  Ollama, vLLM, llama.cpp), via `OpenAIChatModel`.

The cache and chat-store schemas are self-managed: `PostgresCache` and
`PostgresChatStore` create their own tables on `__aenter__`, so there is no
migration step for an installed package.

## Extras

The core install stays small; every provider is an extra.

| Extra       | Installs                             | Unlocks                                                                                 |
| ----------- | ------------------------------------ | --------------------------------------------------------------------------------------- |
| `normalization` | `tiktoken`, `markdown-it-py`, `docutils` | Structural BPE counting and Markdown/RST normalization; explicit byte counting and text/HTML normalization work without these dependencies |
| `html`      | `beautifulsoup4`, `html-to-markdown` | `HtmlParser`                                                                            |
| `trafilatura` | `trafilatura`                      | `TrafilaturaParser` — heuristic main-content extraction for arbitrary web pages         |
| `docling`   | `docling`, `docling-core[chunking]`  | `DoclingParser` — spreadsheets, Word documents, PDFs, via Docling                        |
| `chunkers`  | `langchain-text-splitters`           | `RecursiveCharacterTextSplitter` — see the note below                                   |
| `openai`    | `langchain-openai`                   | `OpenAIEmbedder`, `OpenAIChatModel` (any OpenAI-compatible endpoint, not only OpenAI's) |
| `milvus`    | `pymilvus`                           | `MilvusVectorStore`                                                                     |
| `chroma`    | `chromadb`                           | `ChromaVectorStore`                                                                     |
| `model2vec` | `model2vec`                          | `Model2VecEmbedder` — local, no API key                                                 |
| `serving`   | `fastapi`, `pydantic`, `uvicorn`, `typer`, `prometheus-client` | `fastrag.serve` (`create_app`, `CLI`)                                 |
| `geoip`     | `geoip2`                             | Country resolution through `ServeConfig.geoip_database_path` (see {doc}`serving/admission`) |
| `pdf`       | `pypdf`                              | `PDFParser`                                                                             |
| `dynamic`   | `playwright`                         | `PlaywrightRenderer` and other browser-rendered fetching                                |
| `video`     | `yt-dlp`, `yt-dlp-ejs`               | `YouTubeFetcher`                                                                        |
| `language`  | `lingua-language-detector`           | `LinguaLanguageDetector`, for `LanguageAwareRetriever`                                  |
| `logging`   | `rich`                               | Rich output from the ingestion loggers                                                  |
| `datasets`  | `datasets`, `pyarrow`, `pyyaml`      | Wider `QADataset` loading                                                               |

Reranking needs no extra: `CohereReranker`
posts JSON with `httpx`, already a core dependency, rather than importing a
provider SDK — point `base_url` at any endpoint that speaks the Cohere
rerank protocol.

Importing a class whose extra is not installed raises
{py:class}`~fastrag.exceptions.ConfigurationError` naming the exact
`pip install` command, never a bare `ImportError`.

```{note}
The numbered examples currently use `RecursiveCharacterTextSplitter`, which
requires `chunkers`. Structural pipelines instead select `normalization` for
the default tokenizer and rich textual adapters; see {doc}`structure`.
```

## See also

{doc}`quickstart` to index something and ask it a question;
{doc}`configuration` for how every service takes its credentials.
