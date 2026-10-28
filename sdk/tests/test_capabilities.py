"""Representation contracts through direct hooks and policy-protected routing."""

import pytest
from helpers import FixedEmbedder, document, source, stream

from fastrag.exceptions import ChunkingError, ParseError
from fastrag.ingestion.events import ItemFailed
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext
from fastrag.ingestion.steps.chunking import (
    ChunkerRouter,
    CharacterTextSplitter,
    DoclingChunker,
    EmbeddingSemanticChunker,
    HeadingChunker,
    RecursiveCharacterTextSplitter,
    StructureAwareChunker,
    Utf8ByteTokenizer,
)
from fastrag.ingestion.steps.chunking.base import Chunker
from fastrag.ingestion.filters import ContentFormatFilter
from fastrag.ingestion.steps.parsing import DefaultParser, Parser, ParserRouter
from fastrag.models import ContentFormat


async def output(step, *items, policy=None, events=None):
    context = ExecutionContext(
        "capabilities",
        policy or Policy(on_error="raise"),
        event_sink=lambda event: capture(events, event),
    )
    return [d async for d in step.evaluate_items(stream(*items), context)]


async def capture(events, event):
    from fastrag.ingestion.events import ItemFinished, ItemRetrying, ItemStarted

    # Item lifecycle events bracket every unit of work; these tests are about
    # the domain/failure events.
    lifecycle = (ItemStarted, ItemRetrying, ItemFinished)
    if events is not None and not isinstance(event, lifecycle):
        events.append(event)


@pytest.mark.parametrize(
    "mime,uri,data",
    [
        ("application/pdf", "file:///body.pdf", b"%PDF-1.7\nbody"),
        ("application/zip", "file:///body.zip", b"PK\x03\x04body"),
        ("application/octet-stream", "file:///unknown", b"\x00\xffbody"),
        ("text/plain", "file:///body.txt", b"%PDF-1.7\nbody"),
        ("application/x-private", "file:///unknown", b"readable but unknown"),
    ],
)
async def test_parser_router_refuses_binary_and_unknown(mime, uri, data):
    with pytest.raises(ParseError, match="route|parser"):
        await output(
            ParserRouter(cache=False).default(DefaultParser()), source(data, mime, uri)
        )


@pytest.mark.parametrize(
    "mime,uri,fmt",
    [
        ("text/markdown; charset=UTF-8", "file:///a.md", "markdown"),
        ("application/octet-stream", "file:///a.rst", "rst"),
        ("text/plain", "file:///a.txt", "text"),
    ],
)
async def test_default_parser_truthful_text_formats(mime, uri, fmt):
    parsed = await output(
        ParserRouter(cache=False).default(DefaultParser()),
        source(b"Heading\n=======\nbody", mime, uri),
    )
    assert parsed[0].metadata["content_format"] == fmt


@pytest.mark.parametrize(
    "chunker",
    [
        CharacterTextSplitter(),
        RecursiveCharacterTextSplitter(),
        HeadingChunker(),
        EmbeddingSemanticChunker(embeddings=FixedEmbedder([[1, 0]])),
    ],
)
async def test_known_chunkers_reject_native_trees(chunker):
    with pytest.raises(ChunkingError, match="docling.*accept|accept.*docling"):
        await output(chunker, document("docling", '{"schema_name":"DoclingDocument"}'))


@pytest.mark.parametrize("chunker", [CharacterTextSplitter(), HeadingChunker()])
def test_direct_split_rejects_unknown_format(chunker):
    with pytest.raises(ChunkingError, match="custom-tree"):
        list(chunker.split(document("custom-tree")))


class ArchiveParser(Parser):
    accepted_media_types = frozenset({"application/zip"})

    def config(self):
        return {}

    async def transform(self, item, context):
        yield self.build_document(
            item, "# Topic\n\nMarkdown body", content_format=ContentFormat.MARKDOWN
        )
        yield self.build_document(
            item, "<h1>Topic</h1><p>HTML body</p>", content_format=ContentFormat.HTML
        )
        yield self.build_document(item, "tree-value", content_format="custom-tree")


class TreeChunker(Chunker):
    accepted_content_formats = frozenset({"custom-tree"})
    content_format = ContentFormat.TEXT

    def config(self):
        return {}

    def split(self, document):
        yield "converted " + document.content


class LegacyChunker(TreeChunker):
    accepted_content_formats = None


async def test_one_archive_multiple_formats_and_custom_precedence():
    from fastrag.ingestion.filters import MediaTypeFilter

    parser = (
        ParserRouter(cache=False)
        .route(MediaTypeFilter(include_types=["application/zip"]), ArchiveParser())
        .default(DefaultParser())
    )
    parsed = await output(
        parser, source(b"PK\x03\x04", "application/zip", "file:///a.zip"), source()
    )
    assert len(parsed) == 4
    structural = StructureAwareChunker(tokenizer=Utf8ByteTokenizer())
    chunker = (
        ChunkerRouter(cache=False)
        .route(ContentFormatFilter(include_formats=["custom-tree"]), TreeChunker())
        .route(ContentFormatFilter(include_formats=["markdown", "html"]), structural)
        .default(RecursiveCharacterTextSplitter())
    )
    chunks = await output(chunker, *parsed)
    assert len(chunks) == 4
    assert any(c.content == "converted tree-value" for c in chunks)
    assert any("HTML body" in c.content and "<p>" not in c.content for c in chunks)
    assert len({c.metadata["parent_id"] for c in chunks}) == 4
    assert all(
        c.source_id
        in {
            s.id
            for s in [
                source(b"PK\x03\x04", "application/zip", "file:///a.zip"),
                source(),
            ]
        }
        for c in chunks
    )


async def test_custom_default_and_legacy_stay_permissive():
    parsed = await output(
        ParserRouter(cache=False).default(ArchiveParser()),
        source(b"PK", "application/zip"),
    )
    assert len(parsed) == 3
    assert await output(LegacyChunker(), document("new-format"))
    assert (
        await output(ChunkerRouter().default(LegacyChunker()), document("docling"))
    )[0].content == "converted body"


async def test_policy_skip_reports_each_incompatible_item_and_keeps_valid():
    events = []
    result = await output(
        ParserRouter(cache=False).default(DefaultParser()),
        source(),
        source(b"PK\x03\x04", "application/zip"),
        policy=Policy(on_error="skip"),
        events=events,
    )
    assert len(result) == 1
    failures = [e for e in events if isinstance(e, ItemFailed)]
    assert len(failures) == 1 and isinstance(failures[0].error, ParseError)


async def test_routed_native_docling_and_malformed_error():
    from docling_core.types.doc import DocItemLabel
    from docling_core.types.doc.document import DoclingDocument

    tree = DoclingDocument(name="local")
    tree.add_text(label=DocItemLabel.TEXT, text="Native tree body")
    chunks = await output(
        ChunkerRouter(cache=False).route(
            ContentFormatFilter(include_formats=["docling"]), DoclingChunker()
        ),
        document("docling", tree.model_dump_json()),
    )
    assert [c.content for c in chunks] == ["Native tree body"]
    assert chunks[0].metadata["content_format"] == "text"
    with pytest.raises(ChunkingError, match="malformed|Malformed") as error:
        list(DoclingChunker().split(document("docling", "{}")))
    assert error.value.__cause__ is not None


async def test_semantic_direct_asplit_rejects_before_embedding():
    provider = FixedEmbedder([[1, 0], [0, 1]])
    chunker = EmbeddingSemanticChunker(embeddings=provider)
    with pytest.raises(ChunkingError, match="html"):
        [p async for p in chunker.asplit(document("html", "<p>First. Second.</p>"))]
    assert provider.calls == []


async def test_declared_custom_classes_gate_and_legacy_parser_accepts_binary():
    with pytest.raises(ParseError, match="application/pdf"):
        await output(ArchiveParser(), source(b"%PDF-", "application/pdf"))
    with pytest.raises(ChunkingError, match="markdown"):
        await output(TreeChunker(), document("markdown"))

    class LegacyParser(ArchiveParser):
        accepted_media_types = None

    assert (
        len(
            await output(
                ParserRouter().default(LegacyParser()),
                source(b"%PDF-", "application/pdf"),
            )
        )
        == 3
    )


async def test_structural_custom_registry_capability():
    from fastrag.normalization import NormalizerRegistry, TextNormalizer

    registry = NormalizerRegistry(builtins=False)
    registry.register("custom-tree", TextNormalizer())
    chunker = StructureAwareChunker(normalizers=registry, tokenizer=Utf8ByteTokenizer())
    assert chunker.accepted_content_formats == {"custom-tree"}
    assert list(chunker.split(document("custom-tree")))
    with pytest.raises(ChunkingError, match="markdown"):
        list(chunker.split(document("markdown")))


async def test_direct_default_binary_and_generic_printable_vs_binary():
    context = ExecutionContext("direct", Policy(on_error="raise"))
    with pytest.raises(ParseError, match="binary"):
        [d async for d in DefaultParser().transform(source(b"PK\x03\x04"), context)]
    assert await output(
        ParserRouter().default(DefaultParser()),
        source(
            "Readable Unicode π".encode(), "application/octet-stream", "file:///unknown"
        ),
    )
    with pytest.raises(ParseError, match="binary"):
        await output(
            ParserRouter().default(DefaultParser()),
            source(b"\xff\xfe\xff", "application/octet-stream", "file:///unknown"),
        )


async def test_routed_rst_structure_and_explicit_default_precedence():
    item = document("rst", "Topic\n=====\n\nRST body\n")
    structural = ContentFormatFilter(include_formats=["html", "rst"])
    chunks = await output(
        ChunkerRouter(cache=False).route(structural, StructureAwareChunker()), item
    )
    assert chunks[0].metadata["source_format"] == "rst"
    assert chunks[0].metadata["content_format"] == "markdown"
    assert "RST body" in chunks[0].content
    html = document("html", "<p>original markup</p>")
    custom = await output(ChunkerRouter(cache=False).default(LegacyChunker()), html)
    assert custom[0].content == "converted <p>original markup</p>"


async def test_child_policy_and_unknown_metadata_errors():
    events = []
    assert (
        await output(
            ChunkerRouter().default(HeadingChunker(policy=Policy(on_error="skip"))),
            document("docling"),
            events=events,
        )
        == []
    )
    assert len(events) == 1 and isinstance(events[0].error, ChunkingError)
    with pytest.raises(ChunkingError, match="content_format"):
        await output(HeadingChunker(), document({"invalid": "format"}))
