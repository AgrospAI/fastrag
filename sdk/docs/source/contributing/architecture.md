# Architecture

A module and layering view: where a change belongs, and what it may import.
{doc}`/guide/concepts` covers the same package from the _user's_ side — what
a symbol is for; this page covers the _contributor's_ side — where it lives
and what it may depend on.

## Package map

```text
src/fastrag/
├─ models.py          # Document — the artifact every layer consumes
├─ exceptions.py      # the exception hierarchy, TransientError mixin
├─ ingestion/         # lazy, event-emitting fetch/parse/chunk/embed pipeline
│  ├─ pipeline.py     # Pipeline (context manager, event sink)
│  ├─ policy.py       # Policy (retries, timeouts, concurrency)
│  ├─ events.py       # Step* lifecycle + Document* domain events + ItemFailed
│  ├─ models.py       # Source
│  ├─ cache/          # CacheStore, PostgresCache, schema, ids
│  ├─ logger/         # Logger (ABC), ConsoleLogger
│  ├─ filters/        # Filter, MediaTypeFilter, URLFilter
│  └─ steps/          # base (PipelineStep, SourceStep, DocumentStep, RoutedStep)
│     ├─ fetching/    # SitemapCrawler, WebCrawler, FileSource, LocalFileFetcher, Sources
│     ├─ parsing/     # ParserRouter, HtmlParser, DocusaurusParser, TrafilaturaParser, DefaultParser, DoclingParser
│     ├─ chunking/    # ChunkerRouter, StructureAwareChunker, character/chat/embedding-semantic/Docling chunkers
│     └─ embedding/   # Model2VecEmbedder, OpenAIEmbedder
├─ normalization/     # NormalizedDocument/Node/Span, format registry and adapters
├─ vector_store/       # VectorStore (base), MilvusVectorStore
├─ retrieval/          # Retriever, dense/BM25/hybrid/reranking retrievers, SectionIndex
├─ query/
│  ├─ engine.py        # QueryEngine (ask/stream), Answer, chunks
│  ├─ prompts/         # PromptTemplate, DEFAULT_RAG_PROMPT
│  ├─ llm/             # ChatModel, Message, OpenAIChatModel
│  └─ chat/            # ChatStore, PostgresChatStore, schema
├─ serve/              # create_app, ServeConfig, CLI, cleanup_expired
├─ benchmark/          # Benchmark, QADataset, Comparison, metrics/
└─ _sql.py, _ids.py, _expiry.py   # the shared private layer — see below
```

Every directory above is a real, currently-populated package — nothing here
is aspirational. `BACKLOG.md` tracks what is planned but not yet built; this
page only describes what exists.

## Dependency direction

Verified against the actual imports, not an intended rule: `models` and
`exceptions` depend on nothing else in the package and sit at the bottom.
`ingestion` depends on `models`/`exceptions` (plus the shared private
`fastrag._sql` and its own private `_hashing`), `vector_store` depends on
`models` + `ingestion`'s step contract, and each higher package builds on
the ones beneath:

```{mermaid}
flowchart BT
  exceptions --> models
  ingestion --> models
  vector_store --> ingestion
  retrieval --> vector_store
  query --> retrieval
  query --> models
  serve --> query
  benchmark --> retrieval
```

Two things worth knowing before you add an import:

- **One `vector_store → ingestion` edge is deliberate.** `_SaveStep` (the
  terminal sink `VectorStore.save` returns) imports `DocumentStep` from
  `fastrag.ingestion.steps.base`. Nothing under `ingestion` imports
  `vector_store`.
- **Package docstrings pin the direction.** The ingestion and vector-store
  modules carry notes on why a module is private or where a symbol's
  canonical home is; read the module docstring before moving an import.

## Structural data flow

`normalization` depends on `models` and exposes immutable source-preserving trees,
coverage units, exclusions and format adapter contracts. `StructureAwareChunker`
consumes those trees; its format-independent packing does not import ingestion
parsers. The parser capability layer records emitted representations, while
chunkers declare accepted representations for direct/routed validation.

Embedding consumes the public structural input builder without changing display
content. `SectionIndex` resolves normalized parents independently of stores. Chat
boundary refinement injects the existing `ChatModel`; embedding-semantic splitting
injects the existing `Embedder`. Both are opt-in; ordinary structural packing
requires neither provider. Fresh/cache output validation and deterministic
metadata/IDs share the same source, serialization and tokenizer contracts.

## The private-module convention

A `_`-prefixed module — `ingestion/_hashing.py`, every file under `serve/`
except `app.py`, `config.py`, `cli.py`, and `cleanup.py`, `query/chat`'s
`_cursor.py`, and `fastrag._sql`/`fastrag._ids`/`fastrag._expiry` at the
top level — is internal: it may change shape without a deprecation cycle,
and {doc}`documentation`'s build check refuses to let a `guide/` page name
one. Nothing under `fastrag.*` not re-exported by an `__init__.py` is a name
to import from.

## The shared private layer

Plumbing that more than one package needs lives in exactly one place: an
underscore-prefixed module directly under `fastrag` — `fastrag._sql`
(engine construction, the `is_transient` predicate, the
`ErrorFactory`/`Transaction` shape), `fastrag._ids` (ULID minting and
`created_at` derivation), `fastrag._expiry` (expiry computation and the
write-throttle) — instead of each package forking its own copy.
`fastrag.ingestion.cache.postgres.PostgresCache` and
`fastrag.query.chat.postgres.PostgresChatStore` both import `fastrag._sql`;
`fastrag.query.chat.postgres.PostgresChatStore` and `fastrag.serve`'s
generation loop both import `fastrag._ids`. Each caller of `fastrag._sql`
still passes in its own `ErrorFactory`, so the error type each store raises
(`CacheError`/`TransientCacheError` vs. `ChatStoreError`/
`TransientChatStoreError`) is unaffected by the shared engine plumbing
underneath it. `ingestion/cache/schema.py` and `query/chat/schema.py` are
the one deliberate exception left: each still declares its own `MetaData`
and `NAMING_CONVENTION`, because a schema helper supports exactly one
self-managing store and a store must never reach into another package's
schema.

This is the one way cross-package reuse happens now: a top-level,
underscore-prefixed module is the shared private layer package-private code
may draw on; everything else — including another package's own
`_`-prefixed modules — stays private to the package that defines it. The
older rule this replaces (copy per-package, never import cross-package) was
a deliberate call at the time, but its premise didn't hold: a fix to
`is_transient` or to the expiry write-throttle needs to land in every copy
at once, and forks drift instead. Add a name here once a second package
would otherwise need to fork it — the point where a third fork's
maintenance cost is already visible.

## The extras boundary

Every provider dependency is declared as an optional extra
({doc}`/guide/install`). The pattern, repeated in every service and step
that needs one:

```python
try:
    from langchain_openai import OpenAIEmbeddings

    _OpenAIEmbeddings = OpenAIEmbeddings
except ImportError:
    _OpenAIEmbeddings = None
```

with the constructor raising {py:exc}`~fastrag.exceptions.ConfigurationError`
naming the exact `pip install` command when the sentinel is `None`. Never a
bare `ImportError` — the message a user actually needs (which extra, spelled
correctly) is known at the point the check runs and lost by the time an
`ImportError` would otherwise propagate up through several frames of
langchain or pymilvus internals.

## See also

{doc}`conventions` for the rules that hold across every module regardless of
layer; {doc}`serving-internals` for `serve/`'s own internal shape, which
this page only maps at the package level.
