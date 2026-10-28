"""Seeded cross-format properties with independent source occupancy oracles."""

import json

import pytest
from helpers import FakeChat, generated_source, refiner

from fastrag.ingestion.steps.chunking import (
    StructureAwareChunker,
    Utf8ByteTokenizer,
    build_embedding_text,
    validate_structure_chunks,
)
from fastrag.models import Document
from fastrag.retrieval import SectionIndex


def occupancy(intervals):
    return [point for start, end in intervals for point in range(start, end)]


def check_properties(doc, engine, pieces):
    normalized = engine.normalizers.normalize(doc)
    expected = occupancy(
        (u.span.start_offset, u.span.end_offset) for u in normalized.units
    )
    actual = occupancy(
        (span["start_offset"], span["end_offset"])
        for _, meta in pieces
        for span in meta["retained_intervals"]
    )
    assert actual == expected
    validate_structure_chunks(
        normalized, pieces, engine.tokenizer, engine.options["max_tokens"]
    )
    index = SectionIndex([normalized])
    for number, (text, meta) in enumerate(pieces):
        assert meta["embedding_token_count"] == engine.tokenizer.count(
            build_embedding_text(text, meta)
        )
        assert meta["embedding_token_count"] <= engine.options["max_tokens"]
        assert all(
            engine.tokenizer.count(repeat["text"]) <= engine.options["overlap_tokens"]
            for repeat in meta["repeated_context"]
            if repeat["kind"] == "overlap"
        )
        for span in meta["source_spans"]:
            assert 0 <= span["start_offset"] < span["end_offset"] <= len(doc.content)
        hit = Document(f"hit-{number}", doc.source_id, text, metadata=meta)
        parents = index.resolve(hit, ancestors=True)
        assert parents
        assert all(p.document_id == doc.id for p in parents)
        assert all(
            p.content
            == "".join(
                doc.content[s.start_offset : s.end_offset] for s in p.source_spans
            )
            for p in parents
        )
        json.dumps(meta, ensure_ascii=False)
    rendered = "\n".join(text for text, _ in pieces)
    assert rendered.count("def harvest():") == 1
    assert "nested" in rendered and "pear" in rendered
    assert "excluded poison" not in rendered


@pytest.mark.parametrize("seed", [7, 19, 41, 83, 101, 211])
@pytest.mark.parametrize("fmt", ["markdown", "rst", "html"])
def test_seeded_cross_format_source_structure_and_determinism(seed, fmt):
    doc = Document(
        f"generated-{seed}-{fmt}",
        "generated",
        generated_source(seed, fmt),
        metadata={"content_format": fmt},
    )
    engine = StructureAwareChunker(
        tokenizer=Utf8ByteTokenizer(),
        target_tokens=400,
        max_tokens=600,
        overlap_tokens=30,
        cache=False,
    )
    pieces = list(engine.split(doc))
    assert pieces == list(engine.split(doc))
    check_properties(doc, engine, pieces)
    assert all(
        not m["overlap_spans"]
        for _, m in pieces
        if m["chunk_type"] in {"table", "code_block"}
    )


@pytest.mark.parametrize("size", [0, 1, 99, 100, 399, 400, 600, 601, 1200, 1201, 2401])
def test_exact_configured_byte_boundaries(size):
    engine = StructureAwareChunker(tokenizer=Utf8ByteTokenizer(), cache=False)
    doc = Document("boundary", "generated", "x" * size)
    pieces = list(engine.split(doc))
    assert "".join(text for text, _ in pieces) == doc.content
    expected = (
        []
        if size == 0
        else [size]
        if size <= 600
        else [400] * ((size - 201) // 400) + [size - 400 * ((size - 201) // 400)]
    )
    assert [m["embedding_token_count"] for _, m in pieces] == expected


@pytest.mark.parametrize("size", [99, 100, 399, 400, 600, 601, 1201])
def test_multibyte_budget_safe_lossless_cuts(size):
    doc = Document("unicode", "generated", "界😀" * size)
    engine = StructureAwareChunker(tokenizer=Utf8ByteTokenizer(), cache=False)
    pieces = list(engine.split(doc))
    assert "".join(text for text, _ in pieces) == doc.content
    assert all(
        len(build_embedding_text(t, m).encode("utf-8")) <= 600 for t, m in pieces
    )
    assert all("�" not in t for t, _ in pieces)


@pytest.mark.parametrize("seed", [7, 41, 101])
async def test_seeded_invalid_provider_exact_fallback(seed):
    doc = Document(
        "provider",
        "generated",
        generated_source(seed, "markdown"),
        metadata={"content_format": "markdown"},
    )
    baseline = StructureAwareChunker(tokenizer=Utf8ByteTokenizer(), cache=False)
    expected = list(baseline.split(doc))
    model = FakeChat(
        ["bad json", RuntimeError("local unavailable"), '{"boundaries":[true]}'] * 8
    )
    reports = []
    engine = StructureAwareChunker(
        tokenizer=Utf8ByteTokenizer(),
        cache=False,
        semantic_mode="llm",
        semantic_refiner=refiner(model),
        diagnostics=reports.append,
    )
    actual = [pair async for pair in engine.asplit(doc)]
    assert actual == expected
    assert reports[-1].model_calls > 0 and reports[-1].model_fallbacks
