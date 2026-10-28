# Structure-aware ingestion

Use structural chunks when headings, code, tables and source provenance should
survive ingestion. The engine normalizes explicit document representations,
packs retained content, and measures the complete text sent to embeddings.

Install `fastrag-ai[normalization]` for the default BPE tokenizer and
Markdown/reStructuredText adapters. HTML normalization uses the Python standard
library. An explicitly selected byte tokenizer needs no optional dependency
for plain text or HTML.

## Choose a representation

An ingestion `Parser` converts source bytes into `Document.content` and sets
`metadata["content_format"]`. A `Normalizer` reads that already parsed string
into an ordered tree and an independent source coverage partition. Normalizers
keep the original string; they do not fetch URLs or execute markup.

| Representation | Structural adapter | Important behavior |
| --- | --- | --- |
| `text` | `TextNormalizer` | Retains the whole string; no MIME sniffing. Missing legacy format metadata uses text. |
| `markdown` | `MarkdownNormalizer` | CommonMark with tables; headings, fenced/indented code, lists, links and original syntax. |
| `rst` | `RstNormalizer` | Docutils structure with original source locations; includes/raw execution disabled. |
| `html` | `HtmlNormalizer` | Readable serialization of headings, lists, tables, code and links; script/style/template/comment exclusions recorded. |
| `docling` | Native `DoclingChunker` | Serialized `DoclingDocument`, routed separately; not a structural textual adapter. |

`StructureAwareChunker` selects a registered adapter by explicit format. Unknown
formats raise an actionable error. `ChunkerRouter` validates each selected route:
use `ContentFormatFilter` to send native Docling to `DoclingChunker`, or request
`DoclingParser(output_format="markdown")` and use structural Markdown chunks.
This second path describes the converted Markdown, not original spreadsheet
cell coordinates or PDF byte offsets. See {doc}`ingestion` for binary routing.

## Measure the actual embedding input

This complete offline recipe exercises packing and independent validation.
Its tokenizer counts UTF-8 bytes, so its limits are deliberately byte units.

<!-- executable:budget -->
```python
from fastrag.ingestion.steps.chunking import (
    StructureAwareChunker,
    Utf8ByteTokenizer,
    build_embedding_text,
    validate_structure_chunks,
)
from fastrag.models import Document

parent = Document(
    "orchard-guide",
    "local-guide",
    "# Orchard\n## Water\nWater pears weekly.\n## Soil\nTest soil each spring.\n",
    metadata={"content_format": "markdown"},
)
chunker = StructureAwareChunker(
    tokenizer=Utf8ByteTokenizer(),
    target_tokens=100,
    max_tokens=160,
    min_preferred_tokens=20,
    cache=False,
)
normalized = chunker.normalizers.normalize(parent)
pieces = list(chunker.split_normalized(normalized))
report = validate_structure_chunks(normalized, pieces, chunker.tokenizer, 160)
assert report.chunk_count == len(pieces) > 0
for content, metadata in pieces:
    embedding_text = build_embedding_text(content, metadata)
    assert chunker.tokenizer.count(embedding_text) <= 160
    assert metadata["embedding_token_count"] == chunker.tokenizer.count(embedding_text)
```

In a lazy graph, replace a character splitter with
`StructureAwareChunker(cache=cache)` and retain `embedder(chunker(parser(sources)))`.
The existing `Embedder` base uses `build_embedding_text` for marked structural
chunks while preserving display content. Legacy chunks retain their previous
embedding input. Constructing the graph still performs no work.

The default `TiktokenTokenizer` counts `cl100k_base` with `encode_ordinary`,
including literal special-token strings as ordinary content. Tiktoken may need
its encoding cache on first use; prepare that cache for offline deployments.
This is a configured-tokenizer guarantee, not a promise about an arbitrary
embedding model's tokenizer. Choose the provider's matching encoding or inject
its tokenizer. Tiktoken's [encoding API](https://github.com/openai/tiktoken/blob/main/tiktoken/core.py)
describes the underlying counting and Unicode-offset behavior.

A custom `Tokenizer` supplies a stable nonsecret `identity`, a nonnegative
integer `count(text)`, and ordered Unicode character `boundaries(text)` including
zero and the final offset. Counts need not be additive or monotonic. The engine
checks complete serialized candidates, including repeated headings, fences,
table headers and overlap. Built-in tokenizer hints reduce large-block work;
arbitrary custom tokenizers retain exhaustive feasibility checks, which can be
expensive for large documents.

## Packing, atomic structures and overlap

Fitting direct sections remain intact. Small adjacent sibling sections can
merge toward the target; crossing top-level sections requires the explicit
merge option. Oversized sections pack adjacent blocks, then split oversized
prose, code, table rows or list items with context. Code prefers logical lines
and function boundaries; tables repeat schema, and nested lists repeat parent
labels when those fit. Preservation flags relax these preferences while keeping
retained source coverage. They do not make oversized objects fit automatically.

The target is a packing preference, and the maximum measures final embedding
input. The default `split_with_context` policy enforces the maximum or raises
when mandatory context cannot fit. `preserve_and_flag` deliberately permits
oversized atomic output and records overflow; consumers must handle that before
sending it to a provider. Splitting an indivisible table row or code line can
record atomic structure loss. Read the flags when deciding whether the result
is suitable for your application.

Overlap repeats only eligible trailing prose at sentence boundaries within the
configured ceiling. It does not cross sections or add overlap to code/tables.
Repeated overlap has separate provenance and does not count as newly retained
source. Heading-only documents may yield no body chunks; empty and hierarchy-only
results are explicitly classified by validation.

## Keep provenance and parents

Offsets are half-open Unicode character ranges into the parsed parent's original
string. Node spans may overlap or be noncontiguous; `CoverageUnit` ownership plus
explicit `Exclusion` ranges partitions the source once. HTML output is a readable
serialization, so display offsets and source offsets are different coordinates.

Chunk metadata includes document/section lineage, heading paths, primary source
spans, retained intervals, serialization maps, repeated context, token counts
and overflow/atomic flags. Consecutive compatible mapping runs are coalesced
losslessly. `structure_metadata` is a lightweight node-ID/attribute projection;
complete interpreted node metadata stays on the normalized parent. Do not treat
it as a full parent document. `validate_structure_chunks` independently checks
coverage, maps, lineage, counts and ordering on fresh and persisted output.

Store normalized parents separately if retrieval needs larger context. This
standalone recipe resolves a structural hit without depending on a database.

<!-- executable:parents -->
```python
from fastrag.ingestion.steps.chunking import StructureAwareChunker, Utf8ByteTokenizer
from fastrag.models import Document
from fastrag.retrieval import SectionIndex

parent = Document(
    "orchard-guide",
    "local-guide",
    "# Orchard\n## Water\nWater pears weekly.\n",
    metadata={"content_format": "markdown"},
)
chunker = StructureAwareChunker(tokenizer=Utf8ByteTokenizer(), cache=False)
normalized = chunker.normalizers.normalize(parent)
content, metadata = next(chunker.split_normalized(normalized))
hit = Document("retrieved-hit", parent.source_id, content, metadata=metadata)
index = SectionIndex([normalized])
sections = index.resolve(hit, ancestors=True)
assert sections and all(section.document_id == parent.id for section in sections)
assert any("Water pears weekly." in section.content for section in sections)
```

`SectionIndex.resolve` returns every merged source section in source order and
can deduplicate ancestors. A missing parent/section raises `KeyError`; choose
`missing="skip"` for an available subset. Parent expansion changes neither
stored vectors nor prompts automatically: select and budget that context in your
retrieval/query application. The in-memory index has no persistence or eviction
policy; provide those in the application when parents outlive the process.

## Extend normalization

Register an adapter on the registry passed to your chunker. A stable identity
must distinguish output-changing options. Duplicate registrations require
explicit `replace=True`. Validate source slices, lineage and a complete
nonoverlapping retained/excluded coverage partition before returning.

This minimal adapter uses only public immutable models. A richer adapter adds
section/heading/list/table nodes under the same contracts.

<!-- executable:adapter -->
```python
from fastrag.ingestion.steps.chunking import StructureAwareChunker, Utf8ByteTokenizer
from fastrag.models import Document
from fastrag.normalization import (
    CoverageUnit,
    NormalizedDocument,
    NormalizedNode,
    NormalizerRegistry,
    SourceSpan,
)


class RecordNormalizer:
    identity = "records:whole-source:v1"

    def normalize(self, document):
        span = SourceSpan(0, len(document.content))
        root_id = f"{document.id}:root"
        root = NormalizedNode(root_id, "document", document.content, (span,))
        units = (
            (CoverageUnit(f"{root_id}:unit", root_id, span),)
            if document.content
            else ()
        )
        result = NormalizedDocument(document, "records", (root,), units)
        result.validate()
        return result


registry = NormalizerRegistry()
registry.register("records", RecordNormalizer())
chunker = StructureAwareChunker(
    normalizers=registry, tokenizer=Utf8ByteTokenizer(), cache=False
)
parent = Document(
    "record-1", "local", "Pear: water weekly.", metadata={"content_format": "records"}
)
pieces = list(chunker.split(parent))
assert len(pieces) == 1 and pieces[0][0] == parent.content
assert "records" in chunker.accepted_content_formats
```

{doc}`customizing` covers byte-parser and pipeline-step extension hooks. A
normalizer extension adds interpretation of a representation; it does not
require replacing the format-independent packing engine.

## Optional semantic boundaries

Heuristic structural splitting is deterministic and invokes no chat or embedding
model for boundaries. `semantic_mode="disabled"` disables sentence preferences.
For explicit chat assistance, inject `ChatBoundaryRefiner(llm=chat, identity=...,
allow_external_transmission=True)` into `StructureAwareChunker(semantic_mode="llm",
semantic_refiner=...)` and evaluate asynchronously. The refiner sends eligible
plain leaf paragraphs in Markdown/RST, never code/tables/lists/HTML or rewritten
content, and accepts only allowed source offsets in strictly validated JSON.
Timeout, retry/backoff, concurrency, input/response limits and a bounded
instance-local cache govern calls. Ordinary provider/validation failures use the
same deterministic heuristic fallback; cancellation propagates. Set the chat
provider's output-token limit too. Diagnostic calls/cache/fallback observations
are runtime reports, not identity-changing chunk metadata.

`EmbeddingSemanticChunker(embeddings=embedder)` is a separate asynchronous
algorithm: buffered sentence vectors select cosine-distance breakpoints via
percentile, standard-deviation, interquartile or gradient thresholds, or a
requested chunk count. Its sentence regex, buffer, minimum character size,
start-index and batching options map to that algorithm. It preserves original
text but supplies neither structural hard-token limits nor protected code/table
boundaries. Configure the injected SDK `Embedder` for source transmission and
cost. Neither this class nor structural refinement renames the existing
`SemanticChunker(llm=...)`, whose legacy chat-grouping targets use characters.

## Migration and limitations

Existing `Chunker.split` subclasses, `ChunkerRouter` routes, explicit caches and
lazy `Pipeline` events remain usable. Structural chunks have deterministic IDs
including content, lineage, configuration and embedding context; changing engine,
normalizer, tokenizer or refiners invalidates their cache identities. Chunk
indices are contiguous in source order per parent on fresh and cached built-in
output; concurrent parents retain completion ordering. Diagnostics do not alter
IDs. Keep complete metadata on transport; the default Milvus store rejects
metadata exceeding its supported size rather than silently discarding maps.

Normalization retains original syntax/provenance, not a rendered browser DOM or
lossless reconstruction of original binary documents. Malformed markup can have
conservative locations and diagnostics. RST includes/raw execution, HTML scripts
and style content are intentionally excluded. Custom adapters and pathological
large custom-tokenizer inputs can need application-specific limits. Optional
semantic suggestions do not establish retrieval quality: compare configurations
on your own labelled corpus with {doc}`benchmarking`.

## See also

{doc}`ingestion` for source parsing and Docling routing; {doc}`retrieval` for
retriever composition; {doc}`/reference/index` for exported APIs and options.
