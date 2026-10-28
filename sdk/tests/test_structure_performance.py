"""Ownership and indexed mapping invariants for the performance audit."""

import random

import pytest
from helpers import Characters

from fastrag.exceptions import ChunkingError
from fastrag.ingestion.steps.chunking import (
    StructureAwareChunker,
    validate_structure_chunks,
)
from fastrag.models import Document
from fastrag.normalization.models import Exclusion, SourceSpan, _Node, finish


def test_overlap_ownership_matches_last_covering_preorder_node():
    rng = random.Random(108)
    for _ in range(30):
        blocks = []
        for _ in range(50):
            a, b = sorted(rng.sample(range(201), 2))
            blocks.append(_Node("paragraph", a, b))
        result = finish(
            Document("d", "s", "x" * 200),
            "text",
            blocks,
            [
                Exclusion(SourceSpan(70, 90), "policy"),
                Exclusion(SourceSpan(80, 100), "nested"),
            ],
        )
        nodes = list(result.walk())
        for unit in result.units:
            expected = next(
                n
                for n in reversed(nodes)
                if any(
                    s.start_offset
                    <= unit.span.start_offset
                    < unit.span.end_offset
                    <= s.end_offset
                    for s in n.spans
                )
            )
            assert unit.node_id == expected.id
        assert result.exclusions == (Exclusion(SourceSpan(70, 100), "policy"),)


@pytest.mark.parametrize(
    "fmt,source",
    [
        ("markdown", "# Guide\n\n" + "Small café paragraph.\n\n" * 120),
        ("html", "<h1>Guide</h1>" + "<p>Small &amp; café paragraph.</p>" * 120),
        ("rst", "Guide\n=====\n\n" + "Small café paragraph.\n\n" * 120),
    ],
)
def test_large_mapping_checks_exact_coordinates_and_roles(fmt, source):
    doc = Document("d", "s", source, metadata={"content_format": fmt})
    engine = StructureAwareChunker(tokenizer=Characters(), cache=False)
    normalized = engine.normalizers.normalize(doc)
    pieces = list(engine.split_normalized(normalized))
    validate_structure_chunks(normalized, pieces, engine.tokenizer, 600)
    content, metadata = pieces[0]
    for field, value in [("role", "fabricated"), ("output_start", len(content))]:
        corrupt = {
            **metadata,
            "retained_intervals": [dict(v) for v in metadata["retained_intervals"]],
        }
        corrupt["retained_intervals"][0][field] = value
        with pytest.raises(ChunkingError):
            validate_structure_chunks(
                normalized, [(content, corrupt), *pieces[1:]], engine.tokenizer, 600
            )


def test_nonmonotonic_prefix_search_counts_unique_complete_inputs():
    from fastrag.ingestion.steps.chunking.structure import _Atom, _Piece

    class Nonmonotonic(Characters):
        identity = "test:nonmonotonic"

        def __init__(self):
            self.inputs = []

        def count(self, text):
            self.inputs.append(text)
            return 3 if len(text) == 17 else len(text)

    tokenizer = Nonmonotonic()
    engine = StructureAwareChunker(
        tokenizer=tokenizer,
        target_tokens=4,
        max_tokens=6,
        min_preferred_tokens=1,
        include_heading_path_in_embedding=False,
        cache=False,
    )
    normalized = engine.normalizers.normalize(
        Document("d", "s", "abcdefghijklmnopqrst")
    )
    root = normalized.roots[0]
    template = _Piece(
        [_Atom(c, (SourceSpan(i, i + 1),)) for i, c in enumerate(normalized.source)],
        root,
        [],
    )
    pieces = engine._split_atoms(template, prose=True)
    assert pieces[0].text == "abcdefghijklmnopq"
    assert "".join(p.text for p in pieces) == normalized.source
    assert tokenizer.inputs.count(normalized.source) == 1


@pytest.mark.parametrize("kind", ["code", "prose"])
def test_builtin_large_block_uses_bounded_exact_candidate_counts(monkeypatch, kind):
    from fastrag.ingestion.steps.chunking import TiktokenTokenizer

    tokenizer = TiktokenTokenizer()
    count = tokenizer.count
    calls = []

    def recording(text):
        calls.append(len(text))
        # Fail fast on the former exhaustive algorithm rather than a minute-long test.
        assert len(calls) <= 1500, "exhaustive full-prefix recounting"
        return count(text)

    monkeypatch.setattr(tokenizer, "count", recording)
    body = "value = 'Useful code documentation'\n" * 1800
    source = "# Huge code\n\n" + (
        "```python\n" + body + "```\n" if kind == "code" else body
    )
    assert len(body.encode()) > 60000
    engine = StructureAwareChunker(tokenizer=tokenizer, cache=False)
    normalized = engine.normalizers.normalize(
        Document("large", "local", source, metadata={"content_format": "markdown"})
    )
    pieces = list(engine.split_normalized(normalized))
    validate_structure_chunks(normalized, pieces, tokenizer, 600)
    assert len(pieces) > 10
    assert all(meta["embedding_token_count"] <= 600 for _, meta in pieces)
    if kind == "code":
        fragments = [
            text for text, meta in pieces if meta["chunk_type"] == "code_block"
        ]
        assert all(
            text.startswith("```python\n") and text.endswith("```")
            for text in fragments
        )
        assert "".join(
            text.removeprefix("```python\n").removesuffix("```") for text in fragments
        ).replace("\n", "") == body.replace("\n", "")


@pytest.mark.parametrize("name", ["bpe", "bytes"])
def test_builtin_hints_preserve_unicode_context_overlap_and_exact_inputs(
    name, monkeypatch
):
    from fastrag.ingestion.steps.chunking import TiktokenTokenizer, Utf8ByteTokenizer
    from fastrag.ingestion.steps.chunking.structure import build_embedding_text

    tokenizer = TiktokenTokenizer() if name == "bpe" else Utf8ByteTokenizer()
    original = tokenizer.count
    inputs = set()

    def count(text):
        inputs.add(text)
        return original(text)

    monkeypatch.setattr(tokenizer, "count", count)
    source = "# Unicode 界\n\n" + ("Café 😀 é. Helpful Unicode sentence 界.\n" * 70)
    engine = StructureAwareChunker(
        tokenizer=tokenizer,
        cache=False,
        target_tokens=80,
        max_tokens=110,
        min_preferred_tokens=20,
        overlap_tokens=25,
        max_overlap_tokens=30,
    )
    normalized = engine.normalizers.normalize(
        Document("unicode", "local", source, metadata={"content_format": "markdown"})
    )
    pieces = list(engine.split_normalized(normalized))
    validate_structure_chunks(normalized, pieces, tokenizer, 110, 30)
    assert any(meta["overlap_spans"] for _, meta in pieces)
    assert all(build_embedding_text(text, meta) in inputs for text, meta in pieces)
    assert all(
        original(build_embedding_text(text, meta)) <= 110 for text, meta in pieces
    )


def test_builtin_subclass_keeps_arbitrary_nonmonotonic_fallback():
    from fastrag.ingestion.steps.chunking import Utf8ByteTokenizer
    from fastrag.ingestion.steps.chunking.structure import _Atom, _Piece

    class Arbitrary(Utf8ByteTokenizer):
        identity = "custom:nonmonotonic-byte-subclass"

        def count(self, text):
            return 3 if len(text) == 17 else len(text)

    engine = StructureAwareChunker(
        tokenizer=Arbitrary(),
        target_tokens=4,
        max_tokens=6,
        min_preferred_tokens=1,
        include_heading_path_in_embedding=False,
        cache=False,
    )
    normalized = engine.normalizers.normalize(
        Document("d", "s", "abcdefghijklmnopqrst")
    )
    pieces = engine._split_atoms(
        _Piece(
            [
                _Atom(c, (SourceSpan(i, i + 1),))
                for i, c in enumerate(normalized.source)
            ],
            normalized.roots[0],
            [],
        ),
        prose=True,
    )
    assert pieces[0].text == "abcdefghijklmnopq"
    assert "".join(piece.text for piece in pieces) == normalized.source


@pytest.mark.parametrize("name", ["bpe", "bytes"])
def test_builtin_impossible_heading_context_reports_clear_error(name):
    from fastrag.ingestion.steps.chunking import TiktokenTokenizer, Utf8ByteTokenizer

    tokenizer = TiktokenTokenizer() if name == "bpe" else Utf8ByteTokenizer()
    engine = StructureAwareChunker(
        tokenizer=tokenizer,
        target_tokens=8,
        max_tokens=10,
        min_preferred_tokens=1,
        cache=False,
    )
    source = "# " + "Distinct heading words " * 15 + "\n\nOrdinary body."
    with pytest.raises(ChunkingError, match="Mandatory context|cannot fit"):
        list(
            engine.split(
                Document("d", "s", source, metadata={"content_format": "markdown"})
            )
        )
