"""Sibling order and content-aware routing survive real cache projections."""

import asyncio
from dataclasses import replace

import pytest
from docling_core.transforms.chunker.line_chunker import LineBasedTokenChunker
from helpers import FixedEmbedder, WordTokenizer, model

from fastrag.exceptions import ChunkingError
from fastrag.ingestion.events import DocumentChunked
from fastrag.ingestion.models import Source
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext
from fastrag.ingestion.steps.chunking import (
    ChunkerRouter,
    CharacterTextSplitter,
    Chunker,
    DoclingChunker,
    EmbeddingSemanticChunker,
    HeadingChunker,
    RecursiveCharacterTextSplitter,
    SemanticChunker,
    StructureAwareChunker,
    Utf8ByteTokenizer,
)
from fastrag.ingestion.steps.parsing import (
    DefaultParser,
    DocusaurusParser,
    HtmlParser,
    ParserRouter,
    docusaurus_filter,
    html_filter,
)
from fastrag.vector_store.base import _headers_safe


class TinyTokenizer(WordTokenizer):
    def get_max_tokens(self):
        return 4


async def stream(items):
    for item in items:
        yield item


class BlankChunker(Chunker):
    def config(self):
        return {}

    def split(self, document):
        yield " "
        yield "first"
        yield "\n"
        yield "second"
        yield ""


async def test_retained_pieces_numbered_contiguously():
    from fastrag.models import Document

    doc = Document("p", "s", "unused")
    child = BlankChunker(cache=False)
    context = ExecutionContext("blank", Policy(on_error="raise"))
    chunks = [c async for c in child.transform(doc, context)]
    assert [c.content for c in chunks] == ["first", "second"]
    assert [c.metadata["chunk_index"] for c in chunks] == [0, 1]


@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize(
    "kind",
    ["character", "recursive", "heading", "chat", "embedding", "structure", "docling"],
)
async def test_builtin_order_events_and_cached_flags(cache, kind, routed):
    from fastrag.models import Document

    text = "# A\n\nAlpha grows. Apples grow.\n\n# B\n\nBeta shines. Stars shine.\n"
    options = {
        "character": lambda: CharacterTextSplitter(chunk_size=20, chunk_overlap=0),
        "recursive": lambda: RecursiveCharacterTextSplitter(
            chunk_size=20, chunk_overlap=0
        ),
        "heading": lambda: HeadingChunker(),
        "chat": lambda: SemanticChunker(llm=model(), chunk_size=1000),
        "embedding": lambda: EmbeddingSemanticChunker(
            embeddings=FixedEmbedder([[1, 0], [1, 0], [0, 1], [0, 1]]),
            number_of_chunks=2,
        ),
        "structure": lambda: StructureAwareChunker(
            tokenizer=Utf8ByteTokenizer(),
            max_tokens=40,
            target_tokens=30,
            min_preferred_tokens=1,
            cache=None,
        ),
        "docling": lambda: DoclingChunker(
            chunker=LineBasedTokenChunker(tokenizer=TinyTokenizer())
        ),
    }
    child = options[kind]()
    from docling_core.types.doc import DocItemLabel, DoclingDocument

    tree = DoclingDocument(name="ordering")
    for paragraph in [
        "First native paragraph",
        "Second native paragraph",
        "Third native paragraph",
    ]:
        tree.add_heading(text=paragraph, level=1)
        tree.add_text(label=DocItemLabel.TEXT, text=paragraph)
    doc = Document(
        "order-parent",
        "s",
        tree.model_dump_json() if kind == "docling" else text,
        metadata={"content_format": "docling" if kind == "docling" else "markdown"},
    )
    step = ChunkerRouter().default(child) if routed else child
    events = []

    async def sink(event):
        events.append(event)

    context = ExecutionContext(
        "order", Policy(on_error="raise"), cache=cache, event_sink=sink
    )
    cold = [c async for c in step.evaluate_items(stream([doc]), context)]
    assert len(cold) > 1
    # PostgreSQL replay orders by IDs. Verify an intentionally permuted real
    # snapshot is restored too, regardless of the accidental hash ordering.
    from fastrag.ingestion.cache import document_cache_id

    await cache.put_document_entries(
        document_cache_id(child.name, doc.id, child.config_hash), list(reversed(cold))
    )
    warm = [c async for c in step.evaluate_items(stream([doc]), context)]
    assert warm == [replace(c, cached=True) for c in cold]
    assert [c.metadata["chunk_index"] for c in warm] == list(range(len(warm)))
    chunk_events = [e for e in events if isinstance(e, DocumentChunked)]
    assert len(chunk_events) == 2
    assert chunk_events[0].chunks == cold and chunk_events[1].chunks == warm
    # Destination optimization requests projected records; inspect public order
    # before hydrating content for exact snapshot comparison.
    from fastrag.ingestion.steps.base import _materialize_item

    headers = [
        c
        async for c in step.evaluate_items(
            stream([doc]), replace(context, cache_headers_only=True)
        )
    ]
    assert [c.id for c in headers] == [c.id for c in cold]
    assert all(c.cached for c in headers)
    assert [await _materialize_item(c) for c in headers] == warm
    last_chunked = [e for e in events if isinstance(e, DocumentChunked)][-1]
    assert [c.id for c in last_chunked.chunks] == [c.id for c in cold]


async def test_docusaurus_route_stays_cached_for_destination_projection(cache):
    from fastrag.ingestion.steps.fetching import LocalFileFetcher

    content = b'<html><body><article class="theme-doc-markdown"><h1>Guide</h1><p>Text</p></article></body></html>'
    source = Source("doc-source", "local:docs.html", content, "text/html", "hash")
    await cache.put_source_entries("sources", [source])
    parser = (
        ParserRouter()
        .route(html_filter & docusaurus_filter, DocusaurusParser())
        .route(html_filter, HtmlParser())
        .default(DefaultParser())
    )
    # Bind a graph only to assess the destination's production projection gate.
    graph = parser(LocalFileFetcher(patterns=["unused"]))
    assert not _headers_safe(graph)
    events = []

    async def sink(event):
        events.append(event)

    context = ExecutionContext(
        "route", Policy(on_error="raise"), cache=cache, event_sink=sink
    )
    cold = [c async for c in parser.evaluate_items(stream([source]), context)]
    warm_sources = await cache.get_source_entries("sources")
    warm = [c async for c in parser.evaluate_items(stream(warm_sources), context)]
    assert warm == [replace(c, cached=True) for c in cold]
    assert cold and all(not c.cached for c in cold)
    assert parser._select(source).name == "DocusaurusParser"


@pytest.mark.parametrize("routed", [False, True])
async def test_completion_order_across_parents_preserved(routed):
    from fastrag.models import Document

    gate = asyncio.Event()

    class Delayed(BlankChunker):
        async def asplit(self, doc):
            if doc.id == "slow":
                await gate.wait()
                await asyncio.sleep(0.01)
            else:
                gate.set()
            yield doc.id + " first"
            yield doc.id + " second"

    child = Delayed(cache=False)
    step = ChunkerRouter(cache=False).default(child) if routed else child
    context = ExecutionContext("completion", Policy(concurrency=2, on_error="raise"))
    docs = [Document("slow", "s", "x"), Document("fast", "s", "x")]
    chunks = [c async for c in step.evaluate_items(stream(docs), context)]
    assert [c.metadata["parent_id"] for c in chunks] == ["fast", "fast", "slow", "slow"]


@pytest.mark.parametrize("index", [None, True, -1, "0"])
async def test_invalid_replayed_chunk_index_rejected(cache, index):
    from fastrag.ingestion.cache import document_cache_id
    from fastrag.models import Document

    doc = Document("p", "s", "x")
    child = BlankChunker()
    corrupt = Document(
        "c", "s", "first", metadata={"parent_id": "p", "chunk_index": index}
    )
    await cache.put_document_entries(
        document_cache_id(child.name, doc.id, child.config_hash), [corrupt]
    )
    with pytest.raises(ChunkingError):
        _ = [
            c
            async for c in child.evaluate_items(
                stream([doc]),
                ExecutionContext("bad", Policy(on_error="raise"), cache=cache),
            )
        ]


async def test_custom_validation_receives_ordered_legacy_gapped_replay(cache):
    from fastrag.ingestion.cache import document_cache_id
    from fastrag.models import Document

    observations = []

    class Inspecting(BlankChunker):
        async def validate_output(self, item, produced, context):
            observations.append([c.metadata["chunk_index"] for c in produced])

    doc = Document("legacy", "s", "x")
    child = Inspecting()
    legacy = [
        Document("a", "s", "second", metadata={"chunk_index": 3}),
        Document("z", "s", "first", metadata={"chunk_index": 1}),
    ]
    await cache.put_document_entries(
        document_cache_id(child.name, doc.id, child.config_hash), legacy
    )
    warm = [
        c
        async for c in child.evaluate_items(
            stream([doc]),
            ExecutionContext("legacy", Policy(on_error="raise"), cache=cache),
        )
    ]
    assert [c.content for c in warm] == ["first", "second"]
    assert observations == [[1, 3]]
    assert all(c.cached for c in warm)


async def test_duplicate_replayed_indices_rejected(cache):
    from fastrag.ingestion.cache import document_cache_id
    from fastrag.models import Document

    doc = Document("duplicates", "s", "x")
    child = BlankChunker()
    rows = [
        Document(identity, "s", identity, metadata={"chunk_index": 0})
        for identity in ["a", "b"]
    ]
    await cache.put_document_entries(
        document_cache_id(child.name, doc.id, child.config_hash), rows
    )
    with pytest.raises(ChunkingError):
        _ = [
            c
            async for c in child.evaluate_items(
                stream([doc]),
                ExecutionContext("duplicates", Policy(on_error="raise"), cache=cache),
            )
        ]


def test_unknown_router_and_custom_content_filters_disable_headers():
    from fastrag.ingestion.filters import Filter, URLFilter
    from fastrag.ingestion.steps.parsing import DefaultParser

    class ContentFilter(Filter):
        def matches(self, item):
            return "secret" in item.content

    child = BlankChunker()
    safe = ChunkerRouter().route(URLFilter(include_patterns=["docs"]), child)
    assert _headers_safe(safe)
    unsafe = ChunkerRouter().route(ContentFilter() | URLFilter(), child)
    assert not _headers_safe(unsafe)
    # Unknown custom router selection stays conservative even with no routes.
    from fastrag.ingestion.steps.base import RoutedStep

    class CustomRouter(RoutedStep, DefaultParser):
        def _select(self, item):
            return DefaultParser() if item.content else None

    assert not _headers_safe(CustomRouter())
