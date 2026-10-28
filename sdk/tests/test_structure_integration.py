"""Independent persisted structure, embedding and parent-context contracts."""

import copy
import json
from dataclasses import replace

import pytest
from helpers import RecordingEmbedder

from fastrag.exceptions import ChunkingError
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext
from fastrag.ingestion.steps.chunking import (
    StructureAwareChunker,
    Utf8ByteTokenizer,
    build_embedding_text,
    validate_structure_chunks,
)
from fastrag.models import Document
from fastrag.retrieval import SectionIndex


def fixture(**kw):
    doc = Document(
        "doc",
        "src",
        "# Root\n## A\naaa\n## B\nbbb\n",
        metadata={"content_format": "markdown"},
    )
    chunker = StructureAwareChunker(tokenizer=Utf8ByteTokenizer(), cache=False, **kw)
    return doc, chunker, chunker.normalizers.normalize(doc)


async def chunks(doc, chunker):
    return [
        c async for c in chunker.transform(doc, ExecutionContext("t", policy=Policy()))
    ]


async def test_actual_embedding_preserves_display_and_legacy_input():
    doc, chunker, _ = fixture()
    output = await chunks(doc, chunker)
    embedder = RecordingEmbedder(cache=False)
    context = ExecutionContext("t", policy=Policy())
    for c in output:
        embedded = [x async for x in embedder.transform(c, context)][0]
        assert embedded.content == c.content
        assert embedder.texts[-1] == build_embedding_text(c.content, c.metadata)
    legacy = Document("old", "src", "body", metadata={"heading_path": ["ignored"]})
    assert [x async for x in embedder.transform(legacy, context)][0].content == "body"
    assert embedder.texts[-1] == "body"


async def test_structural_ids_change_with_context_config_not_diagnostics():
    doc, chunker, _ = fixture()
    output = await chunks(doc, chunker)
    changed = StructureAwareChunker(
        tokenizer=Utf8ByteTokenizer(),
        include_heading_path_in_embedding=False,
        cache=False,
    )
    other = await chunks(doc, changed)
    assert output[0].id != other[0].id
    assert output == await chunks(doc, chunker)
    reports = []
    instrumented = StructureAwareChunker(
        tokenizer=Utf8ByteTokenizer(), diagnostics=reports.append, cache=False
    )
    assert await chunks(doc, instrumented) == output
    assert reports and reports[0].chunk_count == len(output)


def test_independent_json_validation_and_faults():
    doc, engine, normalized = fixture()
    pieces = json.loads(json.dumps(list(engine.split(doc))))
    report = validate_structure_chunks(normalized, pieces, engine.tokenizer, 600)
    assert report.chunk_count == len(pieces)
    for field, value in (
        ("embedding_token_count", 0),
        ("heading_ids", ["unknown"]),
        ("source_section_ids", ["unknown"]),
        ("source_spans", []),
    ):
        broken = copy.deepcopy(pieces)
        broken[0][1][field] = value
        with pytest.raises(ChunkingError):
            validate_structure_chunks(normalized, broken, engine.tokenizer, 600)
    broken = copy.deepcopy(pieces)
    broken[0][1]["retained_intervals"][0]["output_end"] = 999
    with pytest.raises(ChunkingError):
        validate_structure_chunks(normalized, broken, engine.tokenizer, 600)
    with pytest.raises(ChunkingError):
        validate_structure_chunks(normalized, [], engine.tokenizer, 600)


async def test_section_index_merged_ancestor_missing_and_document_scope():
    doc, engine, normalized = fixture()
    output = await chunks(doc, engine)
    index = SectionIndex([normalized])
    contexts = index.resolve(output[0])
    assert [c.heading_path[-1] for c in contexts] == ["A", "B"]
    assert [c.content for c in contexts] == ["## A\naaa\n", "## B\nbbb\n"]
    expanded = index.resolve(output[0], ancestors=True)
    assert len({c.section_id for c in expanded}) == len(expanded)
    assert expanded[0].heading_path == ("Root",)
    missing = replace(
        output[0], metadata={**output[0].metadata, "document_id": "missing"}
    )
    with pytest.raises(KeyError):
        index.resolve(missing)
    assert index.resolve(missing, missing="skip") == ()


def test_heading_only_classification():
    doc, engine, _ = fixture()
    heading = replace(doc, content="# Only\n")
    normalized = engine.normalizers.normalize(heading)
    assert (
        validate_structure_chunks(normalized, [], engine.tokenizer, 600).classification
        == "hierarchy_only"
    )


async def test_persisted_ids_lineage_and_known_wrong_heading_are_rejected():
    doc, engine, normalized = fixture()
    output = await chunks(doc, engine)
    for changed in (
        replace(output[0], id="wrong"),
        replace(output[0], source_id="wrong"),
        replace(output[0], metadata={**output[0].metadata, "parent_id": "wrong"}),
    ):
        with pytest.raises(ChunkingError):
            validate_structure_chunks(normalized, [changed], engine.tokenizer, 600)
    pieces = copy.deepcopy(list(engine.split(doc)))
    section = next(
        n for n in normalized.walk() if n.kind == "section" and n.title == "A"
    )
    pieces[0][1].update(
        heading_ids=[section.id],
        heading_path=[section.title],
        heading_levels=[section.level],
    )
    with pytest.raises(ChunkingError, match="ancestry"):
        validate_structure_chunks(normalized, pieces, engine.tokenizer, 600)


def test_repeated_text_and_source_maps_are_independently_validated():
    document = Document(
        "d",
        "s",
        "| A | B |\n| --- | --- |\n" + "| value | data |\n" * 20,
        metadata={"content_format": "markdown"},
    )
    engine = StructureAwareChunker(
        tokenizer=Utf8ByteTokenizer(),
        target_tokens=70,
        max_tokens=100,
        min_preferred_tokens=10,
    )
    normalized = engine.normalizers.normalize(document)
    pieces = list(engine.split(document))
    assert any(m["repeated_context"] for _, m in pieces)
    for mutation in ("text", "map"):
        broken = copy.deepcopy(pieces)
        repeat = next(
            m["repeated_context"][0] for _, m in broken if m["repeated_context"]
        )
        if mutation == "text":
            repeat["text"] = "incorrect"
        else:
            repeat["serialization_map"] = []
        with pytest.raises(ChunkingError):
            validate_structure_chunks(normalized, broken, engine.tokenizer, 100)


async def test_legacy_chunk_id_formula_is_preserved():
    from fastrag.ingestion._hashing import content_hash
    from fastrag.ingestion.steps.chunking import CharacterTextSplitter

    doc = Document("p", "s", "legacy text")
    engine = CharacterTextSplitter(chunk_size=100, chunk_overlap=0, cache=False)
    output = await chunks(doc, engine)
    assert output[0].id == content_hash("p:0:legacy text")
