# Ingestion

Turn sources into embedded documents through a lazy pipeline of typed steps.
`fastrag.ingestion` is the fetch → parse → chunk → embed walk; every step is
a `PipelineStep` you construct (nothing executes) and compose with `__call__`
(nothing executes), so a pipeline is a graph you can inspect, reuse, and
evaluate only when a sink consumes it. `examples/05_indexing.py` is the full
graph, fetch through sink, run as a plain script:

```{literalinclude} /../../examples/05_indexing.py
:language: python
```

You would normally run this through the `ConsoleLogger`:
`Pipeline(saved_documents)` installs the execution-context and event sink,
and iterating it yields every `PipelineEvent` the graph emitted —
{doc}`configuration` and `examples/06_logging.py` show that shape.

The progress table orders discovered columns as Fetching → Parsing → Chunking
→ Embedding → Indexing. Additional stages follow in discovery order. Set
`column_order` to place custom stages where you want them:

```python
await ConsoleLogger(
    pipeline,
    column_order=[
        "Fetching",
        "Parsing",
        "Enrichment",
        "Chunking",
        "Embedding",
        "Indexing",
    ],
).run()
```

Labels match each step's `event_name`. Unlisted stages still appear after the
listed ones; an empty list uses discovery order. Only stages present in the
graph or its events are shown. With `skip_existing=True`, Embedding remains
visible even when every document is already indexed. Skipped documents emit
cached embedding events and count toward the cached total, even when their
embedding is empty.

## Summarize source URLs

`SourceTreeLogger` prints a final tree grouped by domain and path:

```python
from fastrag.ingestion.logger import SourceTreeLogger

await SourceTreeLogger(Pipeline(saved_documents), max_depth=3).run()
```

```text
example.org 2 1
└── /docs 2 1
    ├── /guide 1 1
    └── /reference 1

2 new 1 cached 0 error
```

Each line shows nonzero new, cached and error counts of unique URLs at that
path and below it, separated by spaces. Only numbers are colored; the final
summary shows the total for each status, including zeros. Path
segments start with `/`, while domain roots have no added slash. Duplicate fetch
events count once. URLs with different schemes, queries, fragments or trailing
slashes count separately within the same domain/path branch. `max_depth=1`
shows domains only; `None` (the default) shows every path level. Hidden
descendants still contribute to the status counts.

New URLs are green, cached URLs cyan, and URLs with an item failure at any
pipeline stage red. A failed URL counts only as an error, even when several
of its chunks fail. Failures without a known source URL and failed steps
are reported separately. Terminal output uses colors automatically; redirected
output is plain text. Set `color=True` or `color=False` to override this,
and `stream=` to choose the destination. No optional dependencies are needed.
Each run resets `source_count`; exceptions propagate after a partial summary.

A YouTube `/watch` URL adds its video ID as one more level below `/watch`.
`SourceListLogger` prints the same URLs as a flat list grouped by parent path.
Both loggers accept `enrich_youtube=True` to show each video's title in
parentheses:

```text
www.youtube.com 1
└── /watch 1
    └── ?v=dQw4w9WgXcQ (Video title) 1
```

A source's `title` metadata is used when present. Other titles come from
YouTube's public oEmbed endpoint, requested once per video after the pipeline
completes. Interrupted runs skip these requests, and a video whose title cannot
be fetched is shown without one.

## Inspect chunks

Install `fastrag-ai[logging]` for Rich output, then run a pipeline whose root
is the pipeline's final document stage through `DocumentsLogger`:

```python
from fastrag.ingestion.logger import DocumentsLogger, SourceLogger

await SourceLogger(Pipeline(sources)).run()
await DocumentsLogger(Pipeline(parser(sources))).run()
await DocumentsLogger(Pipeline(chunker(parser(sources)))).run()
```

`DocumentsLogger` displays complete literal text, metadata, embeddings, IDs,
lineage, configuration hashes, cache status and size statistics. By default it
inspects the root step's stage, so a chunking pipeline shows chunks without
also printing every parsed document. Pass `stages=("Parsing", "Chunking")`
to inspect several stages. Chunk records include their parent ID and position;
`document_result` records describe chunking inputs, including empty results.

`SourceLogger` displays fetched sources, deduplicated by source ID. UTF-8 bytes
are shown as literal text; other bytes are shown as base64. JSON Lines includes
`content_base64` for exact recovery regardless of text encoding.

Both support `format="jsonl"` without the Rich extra, `stream=` for file output,
and `configuration=` for run settings. Set `content_lines=3` to display only
the first and last three lines of text, with an omitted-line marker between
them. Short content is shown in full. This affects Rich and JSONL text output;
size statistics, metadata and source `content_base64` remain complete.
The default `content_lines=None` displays all content. Each run resets counters and ends with a
summary, including errors, cache hits and size totals. Document totals count
output documents (chunks included); `parent_documents` counts chunking inputs.
Statistics measure displayed items, including repeated heading context or items
shown at multiple selected stages. Failures propagate after an `interrupted`
partial summary. `ChunkLogger` remains available as a compatibility import for
chunk-only inspection anywhere in the graph.

## Structural chunks

Use {doc}`structure` when heading hierarchy, protected code/table/list structure,
exact embedding budgets and source-parent retrieval matter. `StructureAwareChunker`
normalizes explicit text/Markdown/RST/HTML representations and preserves retained
source provenance; native Docling remains a separate route. Optional semantic
boundaries reuse the existing SDK chat/embedder abstractions.

## The kinds of step

Two families share the `PipelineStep` base:

- **`SourceStep`** — produces `Source` objects. `SitemapCrawler` (one site),
  `WebCrawler` (crawl from start URLs), and `Sources` (aggregate several
  fetchers). Implementing `evaluate(context)` yields `Source`s.
- **`DocumentStep`** — transforms `Document`s. `Parser` (bytes to text),
  `Chunker` (one document to chunks), `Embedder` (adds an embedding).
  Implementing the subclass's hook (`transform`, `split`, `embed`) yields
  `Document`s, and the base drives caching, concurrency, and event emission.

`ParserRouter` / `ChunkerRouter` are `RoutedStep`s with no parsing or chunking
logic of their own: `.route(filter, step)` dispatches each item to the first
child whose `Filter` (`MediaTypeFilter`, `URLFilter`, `ContentFormatFilter`,
`html_filter`, `docusaurus_filter`, ...) matches, and `.default(step)` catches
the rest. An item matching nothing is skipped with an `ItemFailed`.

## Lazy evaluation

`Sources` can filter the combined results and limit how many matching sources
reach parsing:

```python
from fastrag.ingestion.filters import URLFilter
from fastrag.ingestion.steps.fetching import SitemapCrawler, Sources

sources = Sources(
    [SitemapCrawler(url="https://example.com/sitemap.xml")],
    filter=URLFilter(include_patterns=[r"/docs/guides/algorithms/publish/?$"]),
    max_sources=1,
)
```

Filtering happens before the limit and applies to fresh and cached results.
`max_sources=None` returns all matching sources; zero skips evaluation, and
negative values are rejected. Children complete concurrently, so a limit
selects in completion order. Children still finish fetching when a positive
result limit is reached. Use the crawler's `include`/`exclude` URL globs and
`max_urls` to select targets before downloading. `await sources.aclose()`
releases child HTTP clients after evaluation.

Constructing a step never runs any I/O or emits an event. `parser(sources)`
returns a lazy `PipelineStep[Document]`; `vector_store.save(embedded)`
returns another. Only evaluating the final step — iterating it, or running it
under a `Pipeline` with `ConsoleLogger` — actually fetches, parses, chunks,
embeds, and writes to the vector store. This is what lets a program assemble
one configurable graph for many inputs and reuse it.

## Binary documents (spreadsheets, Word, PDF)

`HtmlParser` and `DefaultParser` only understand text and HTML.
`DoclingParser` (optional `docling` extra: `pip install "fastrag-ai[docling]"`)
hands a source's bytes to [Docling](https://docling-project.github.io/docling/)
for everything else — spreadsheets, Word documents, PDFs — entirely in
memory, never through a temporary file. `office_filter`, a `MediaTypeFilter`
over the office media types (`.xlsx`, `.xls`, `.docx`, `.doc`, `.pptx`,
`.ppt`) `DoclingParser` is meant for, is the counterpart of `html_filter` for
routing it.

There are two ways to chunk what `DoclingParser` produces, and which one to
reach for depends on what `output_format` you asked it for.

**Approach 1 — Markdown, chunked by whatever you already use.** The default
(`output_format="markdown"`): Docling's converted Markdown becomes ordinary
chunk text, and any existing chunker keeps working completely unmodified —
the simplest option, and the right one when the source's exact structure
(which cell was in which row, which page a paragraph came from) does not
need to survive into retrieval.

```python
from fastrag.ingestion.steps.chunking import RecursiveCharacterTextSplitter
from fastrag.ingestion.steps.parsing import (
    DefaultParser,
    DoclingParser,
    ParserRouter,
    office_filter,
)

parser = (
    ParserRouter(cache=cache)
    .route(office_filter, DoclingParser())
    .default(DefaultParser())
)
chunker = RecursiveCharacterTextSplitter(chunk_size=500, cache=cache)
```

**Approach 2 — native Docling chunking, on the real structure.**
`output_format="docling"` instead serializes the structured `DoclingDocument`
itself into `Document.content`, tagged `content_format=DOCLING`,
and `DoclingChunker` deserializes it and chunks it with one of Docling's own
chunkers — `HierarchicalChunker` (the default: no tokenizer, no model
download), or a caller-constructed `HybridChunker`/`LineBasedTokenChunker`
(both pull a tokenizer, e.g. from Hugging Face, on first use) passed in via
`DoclingChunker(chunker=...)`. Each chunk's text is Docling's own
*contextualized* serialization (heading path plus content, joined), and its
metadata carries `headings` (`list[str]`) and, when Docling reports one,
`page_no` — provenance a flattened-text chunker has already lost by the time
it sees the string. Every chunk `DoclingChunker` emits is plain text again
(`content_format=TEXT`), so it is embeddable and storable exactly like any
other chunk, with no further translation.

```python
from fastrag.ingestion.filters import ContentFormatFilter
from fastrag.ingestion.steps.chunking import (
    ChunkerRouter,
    DoclingChunker,
    RecursiveCharacterTextSplitter,
)
from fastrag.ingestion.steps.parsing import ParserRouter, DoclingParser, office_filter
from fastrag.models import ContentFormat

parser = ParserRouter(cache=cache).route(
    office_filter, DoclingParser(output_format="docling")
)
chunker = (
    ChunkerRouter(cache=cache)
    .route(
        ContentFormatFilter(include_formats=[ContentFormat.DOCLING]),
        DoclingChunker(),
    )
    .default(RecursiveCharacterTextSplitter(chunk_size=500))
)
```

A `DoclingChunker` handed anything that is not `content_format=DOCLING`
raises `ChunkingError` rather than chunking JSON as text — always route it
behind a `ContentFormatFilter` as above. The other direction is guarded too:
an `Embedder` handed a `DOCLING`-format document (one that skipped chunking
entirely) raises `EmbeddingError` naming `DoclingChunker`, instead of
silently embedding a serialized `DoclingDocument`. A document with no
`content_format` at all — everything produced before either of these
guardrails existed — is still treated as plain text, so nothing that worked
before keeps working exactly the same.

### Describing the images inside a document

Docling exports a picture as a bare `<!-- image -->` placeholder unless
something is configured to describe it, so a spreadsheet whose content lives
in a diagram parses into nothing retrievable. `format_options` hands
per-input-format Docling `FormatOption` objects straight to its
`DocumentConverter`, which is where that (and any other Docling pipeline
option this parser does not surface itself) is turned on — here against any
OpenAI-compatible vision endpoint:

```python
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    ConvertPipelineOptions,
    PictureDescriptionApiOptions,
)
from docling.document_converter import ExcelFormatOption

from fastrag.ingestion import Policy
from fastrag.ingestion.steps.parsing import DoclingParser

pipeline_options = ConvertPipelineOptions(
    do_picture_description=True,
    picture_description_options=PictureDescriptionApiOptions(
        url="https://your-endpoint/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        params={"model": vision_model},
        picture_area_threshold=0.0,
    ),
    enable_remote_services=True,  # Docling calls no remote service without it
)

parser = DoclingParser(
    format_options={
        InputFormat.XLSX: ExcelFormatOption(pipeline_options=pipeline_options)
    },
    policy=Policy(timeout=600),
    cache=cache,
)
```

The description lands in the parsed Markdown next to the placeholder, so
every chunker downstream sees it as ordinary text. Two things are easy to get
wrong: `Policy.timeout` defaults to 30 seconds and a timed-out item is
*skipped*, not raised, so a document with several pictures silently
disappears unless the timeout is raised; and `picture_area_threshold`
defaults to `0.05`, which skips any picture covering less than 5% of its
page — most spreadsheet images.

The `policy` belongs on the parser that does the converting, including when
that parser is a route rather than the whole stage: a `ParserRouter` applies
the policy of whichever child it dispatched an item to, so
`ParserRouter(cache=cache).default(DoclingParser(policy=Policy(timeout=600)))`
gives the conversion its 600 seconds while everything else the router handles
keeps the default. `format_options` is part of the parser's
cache identity, so turning description on re-parses rather than serving the
placeholder-only text cached earlier.

### Keep Markdown sections whole

For documentation with meaningful headings, `HeadingChunker` keeps each section
body in one chunk, including complete code blocks, tables, and lists:

```python
from fastrag.ingestion.steps.chunking import HeadingChunker

chunker = HeadingChunker(cache=cache)
```

It recognizes ATX (`#` through `######`) and Setext headings, ignoring heading
syntax inside code blocks. Each chunk repeats its ancestor headings in the text
for embedding and carries their titles in `metadata["headings"]`, ordered from
outermost to innermost. Text before the first heading has an empty heading path.
A heading with only subheadings travels with its subsections. There is no size
limit or overlap; choose a different chunker when sections can exceed your
embedding model's input limit.

## Identity and determinism

A `Document`'s `id` is derived from its content and the transformation
configuration, so the same content fed through the same step configuration
produces the same id on any machine. That is the property {doc}`caching` and
{doc}`indexing` build on: a step cache keyed on id serves a chunk without
recomputing it, and a vector store upserting by id never duplicates.

## See also

{doc}`caching` for what a `CacheStore` keys on; {doc}`indexing` for the
vector-store sink these steps feed; {doc}`/guide/examples` for the numbered
run-through (`examples/01_fetching.py` up through `05_indexing.py`).

### Renderers and dynamic content

HTTP fetchers (`FileSource`, `WebCrawler`, and `SitemapCrawler`) accept a
`renderer`. Rendering finishes before source hashing, caching, and link discovery.
Sitemap XML is always fetched over HTTP without rendering.

- `HttpRenderer` (default) keeps the HTTP response unchanged.
- `PlaywrightRenderer` executes JavaScript and waits for hydration.
- `ChatRenderer` and `DecisionsRenderer` use model choices to reveal same-page
  content through load-more buttons, tabs, details, and scrolling. Decisions
  supports JEV models. Models choose observed controls, never executable code.

Install `fastrag-ai[dynamic]` and run `python -m playwright install chromium`
for browser renderers. Parsers no longer accept `render_dynamic`,
`render_timeout`, or `wait_selector`; move those settings to the fetcher's renderer.

```python
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.fetching import FileSource, DecisionsRenderer
from fastrag.ingestion.steps.parsing import DecisionsParser

renderer = DecisionsRenderer(
    model="typesafe/jev-1.13",
    api_key=api_key,
    base_url="https://openrouter.ai/api/alpha/decisions",
)
sources = FileSource(url=url, renderer=renderer, policy=Policy(timeout=600))
parser = DecisionsParser(
    model="typesafe/jev-1.13",
    api_key=api_key,
    base_url="https://openrouter.ai/api/alpha/decisions",
)
async for document in parser(sources):
    print(document.content)
```

Use `PlaywrightRenderer(timeout=30, wait_selector="#ready")` for hydration
without model interaction. Browser renderers use the downloaded response for
the initial document; scripts, assets, and JavaScript data requests resolve
against its original URL. Model-guided browser guards prevent navigation, form
submission, popups, and client-side routing. Fetchers follow discovered links.
Raw HTML retains source content from replaced panels and deduplicates repeated
states. Rendering does not clean or summarize content.

`goal`, `max_steps`, `timeout`, `render_timeout`, `settle_seconds`, and
`wait_selector` configure model-guided exploration. `max_snapshot_chars` bounds
model observations; history carries hashes and decisions rather than whole HTML
snapshots. Original source content is retained independently of that model
budget. Exhausted exploration budgets and invalid choices raise `FetchError`.
Completion is a model judgment and does not guarantee every possible application
state has been visited. Temporary provider errors can use a configured
`fallback` renderer. Parser fallbacks are configured independently.

Renderer settings participate in the fetcher's cache identity, excluding model
API keys. Browser-rendered content is re-rendered for freshness checks, because
HTML ETags cannot validate external scripts or data. `cache_validation_interval`
controls how often that work occurs. Source validators are cleared after rendering.

### Parsing supplied content

`OpenAIParser` and `DecisionsParser` only read supplied HTML and select a content
region. They remove page furniture and convert original content to Markdown;
they never launch a browser, execute scripts, click controls, or fetch the page.
They require `fastrag-ai[html]`, with no browser dependency. They make only model
API requests. The `goal` controls content selection. Invalid choices raise
`ParseError`; temporary provider failures may use a configured fallback parser.

`HtmlParser` and `DocusaurusParser` also consume supplied bytes.
Hidden tab panels and SVG labels are retained by default. Route
`html_filter & docusaurus_filter` to `DocusaurusParser` to extract a Docusaurus
article. For optional image descriptions, configure
`OpenAIVisionParser` on the parser; image enrichment fetches referenced image
assets and may rasterize SVGs locally, but does not hydrate or navigate the page.

`TrafilaturaParser` (`fastrag-ai[trafilatura]`) is the selector-free alternative
for sites whose chrome is not marked up with `<nav>`/`<footer>`: it finds the
article heuristically and carries the page's title, author, date, description,
and site name in `Document.metadata`. Unlike the parsers above it does not
reveal hidden panels, label SVGs, or describe images. Route HTML to it
explicitly:

```python
from fastrag.ingestion.steps.parsing import ParserRouter, TrafilaturaParser, html_filter

parser = ParserRouter().route(html_filter, TrafilaturaParser())
```

```python
from fastrag.ingestion.steps.fetching import FileSource, PlaywrightRenderer
from fastrag.ingestion.steps.parsing import (
    DefaultParser,
    DocusaurusParser,
    HtmlParser,
    ParserRouter,
    docusaurus_filter,
    html_filter,
)

sources = FileSource(url=url, renderer=PlaywrightRenderer())
parser = (
    ParserRouter()
    .route(html_filter & docusaurus_filter, DocusaurusParser())
    .route(html_filter, HtmlParser())
    .default(DefaultParser())
)
async for document in parser(sources):
    print(document.content)
```
