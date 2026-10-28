"""Local exact-count structural engine and source-fragment tests."""

import pytest
from helpers import Characters, assert_coverage

from fastrag.exceptions import ChunkingError
from fastrag.ingestion.steps.chunking import StructureAwareChunker
from fastrag.ingestion.steps.chunking.structure import (
    Utf8ByteTokenizer,
    build_embedding_text,
)
from fastrag.models import Document
from fastrag.normalization import NormalizerRegistry


def chunk(text, fmt="markdown", **options):
    document = Document(
        id="doc", source_id="source", content=text, metadata={"content_format": fmt}
    )
    engine = StructureAwareChunker(
        tokenizer=Characters(),
        target_tokens=80,
        max_tokens=120,
        min_preferred_tokens=20,
        cache=False,
        **options,
    )
    return document, engine, list(engine.split(document))


def test_fitting_section_above_target_and_hierarchy():
    text = "# Guide\n### Part\n" + "x" * 85
    document, engine, pieces = chunk(text)
    assert len(pieces) == 1
    assert pieces[0][1]["heading_path"] == ["Guide", "Part"]
    assert "x" * 85 in pieces[0][0]
    assert_coverage(document, pieces)
    assert list(engine.split(document)) == pieces


def test_empty_heading_only_and_duplicate_sections():
    for text in ("", " \n", "# Guide\n## Empty"):
        assert chunk(text)[2] == []
    document, _, pieces = chunk("# Same\na\n# Same\nb\n")
    assert len(pieces) == 2
    assert pieces[0][1]["source_section_ids"] != pieces[1][1]["source_section_ids"]
    assert_coverage(document, pieces)


def test_small_siblings_merge_with_common_parent():
    document, _, pieces = chunk("# Guide\n## A\na\n## B\nb\n## C\nc\n")
    assert len(pieces) == 1
    assert pieces[0][1]["heading_path"] == ["Guide"]
    assert len(pieces[0][1]["source_section_ids"]) == 3
    assert "## A" in pieces[0][0] and "## C" in pieces[0][0]
    assert_coverage(document, pieces)


@pytest.mark.parametrize(
    "fmt,text",
    [
        ("text", "界😀" * 180),
        ("markdown", "# H\n" + "Long sentence without punctuation " * 30),
        (
            "html",
            "<h1>H</h1><p>"
            + "Hello &amp; <b>world</b>. " * 30
            + "</p><script>omit</script>",
        ),
    ],
)
def test_oversized_prose_preserves_every_actual_slice(fmt, text):
    document, _, pieces = chunk(text, fmt)
    assert len(pieces) > 1
    assert_coverage(document, pieces)
    assert all(len(m["source_spans"]) for _, m in pieces)
    if fmt == "html":
        assert all("<b>" not in text and "<p>" not in text for text, _ in pieces)
        assert "&" in "".join(text for text, _ in pieces)


def test_code_fragments_have_fences_language_and_exact_coverage():
    document, _, pieces = chunk(
        "# Code\n```python\ndef first():\n"
        + "    call()\n" * 25
        + "def next():\n    finish()\n```\n"
    )
    assert len(pieces) > 2
    assert all(
        text.startswith("```python\n") and text.endswith("```") for text, _ in pieces
    )
    assert all(m["chunk_type"] == "code_block" for _, m in pieces)
    assert_coverage(document, pieces)


def test_table_rows_repeat_header_and_oversized_row_is_flagged():
    text = (
        "# Table\n| A | B |\n| --- | --- |\n"
        + "| n | value |\n" * 18
        + "| long | "
        + "z" * 220
        + " |\n"
    )
    document, _, pieces = chunk(text)
    assert len(pieces) > 2
    assert any(m["atomic_structure_lost"] for _, m in pieces)
    assert all("| A | B |" in body for body, _ in pieces)
    assert any(m["repeated_context"] for _, m in pieces)
    assert_coverage(document, pieces)


def test_nested_list_items_and_oversized_item():
    document, _, pieces = chunk(
        "# List\n"
        + "- parent\n  - nested detail\n" * 12
        + "- giant "
        + "a" * 250
        + "\n"
    )
    assert any("parent\n  - nested detail" in text for text, _ in pieces)
    assert any(m["atomic_structure_lost"] for _, m in pieces)
    assert_coverage(document, pieces)


def test_preserve_atomic_overflow_and_impossible_context():
    document, _, pieces = chunk(
        "# H\n```\n" + "a" * 400 + "\n```", oversized_block_policy="preserve_and_flag"
    )
    assert len(pieces) == 1 and pieces[0][1]["overflow"]
    assert_coverage(document, pieces)
    with pytest.raises(ChunkingError, match="context"):
        chunk("# " + "H" * 200 + "\nbody")


def test_sentence_overlap_is_source_backed_and_bounded():
    document, _, pieces = chunk(
        "# H\n" + "One short sentence. " * 20, overlap_tokens=25
    )
    assert any(m["overlap_spans"] for _, m in pieces[1:])
    assert all(
        len(
            "".join(
                document.content[s["start_offset"] : s["end_offset"]]
                for s in m["overlap_spans"]
            )
        )
        <= 25
        for _, m in pieces
    )
    assert_coverage(document, pieces)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_tokens": 0},
        {"target_tokens": True},
        {"min_preferred_tokens": 500},
        {"overlap_tokens": 51},
        {"semantic_mode": "llm"},
        {"oversized_block_policy": "discard"},
        {"preserve_code_blocks": 1},
    ],
)
def test_validated_options(kwargs):
    with pytest.raises(ValueError):
        StructureAwareChunker(tokenizer=Characters(), cache=False, **kwargs)


def test_tokenizer_identity_and_byte_fallback():
    assert Utf8ByteTokenizer().count("界😀") == 7
    first = StructureAwareChunker(tokenizer=Characters(), cache=False)

    class Other(Characters):
        identity = "other"

    assert (
        first.config_hash
        != StructureAwareChunker(tokenizer=Other(), cache=False).config_hash
    )


def test_nonadditive_exact_budget():
    class Nonadditive(Characters):
        identity = "nonadditive"

        def count(self, text):
            return len(text) + (40 if "\n\n" in text else 0)

    doc = Document(
        id="d",
        source_id="s",
        content="# H\n" + "a sentence. " * 40,
        metadata={"content_format": "markdown"},
    )
    engine = StructureAwareChunker(
        tokenizer=Nonadditive(),
        target_tokens=70,
        max_tokens=90,
        min_preferred_tokens=10,
        cache=False,
    )
    pieces = list(engine.split(doc))
    assert all(Nonadditive().count(build_embedding_text(t, m)) <= 90 for t, m in pieces)


def test_unused_overlap_ceiling_allows_small_budget():
    engine = StructureAwareChunker(
        tokenizer=Characters(),
        target_tokens=10,
        max_tokens=10,
        min_preferred_tokens=1,
        cache=False,
    )
    doc = Document(id="d", source_id="s", content="abcdefghijk")
    assert [t for t, _ in engine.split(doc)] == ["abcdefghij", "k"]


def test_reference_definition_is_readable_and_in_source_order():
    doc, _, pieces = chunk("# H\nbody [ref]\n\n[ref]: https://example.com\n")
    assert len(pieces) == 1
    assert "[ref]: https://example.com" in pieces[0][0]
    assert_coverage(doc, pieces)


def test_html_fitting_code_language_and_indentation():
    doc, _, pieces = chunk(
        '<h1>H</h1><pre><code class="language-python">def f():\n    pass\n</code></pre>',
        "html",
    )
    assert len(pieces) == 1
    assert "```python\ndef f():\n    pass\n```" in pieces[0][0]
    assert_coverage(doc, pieces)


def test_html_nested_ordered_list_fitting_and_split():
    body = '<ol start="3"><li>parent<ul><li>child</li></ul></li><li>next</li></ol>'
    doc, _, pieces = chunk(body, "html")
    assert "3. parent\n  - child" in pieces[0][0]
    assert "4. next" in pieces[0][0]
    assert_coverage(doc, pieces)
    doc, _, pieces = chunk(body * 10, "html")
    assert any("3. parent\n  - child" in t for t, _ in pieces)
    assert_coverage(doc, pieces)


def test_table_repeat_includes_valid_delimiter_and_repeat_mapping():
    doc, _, pieces = chunk("| A | B |\n| :--- | ---: |\n" + "| n | value |\n" * 20)
    for text, meta in pieces:
        assert "| A | B |\n| :--- | ---: |" in text
        for repeat in meta["repeated_context"]:
            assert text[repeat["output_start"] : repeat["output_end"]] == repeat["text"]
            assert repeat["serialization_map"]
    assert_coverage(doc, pieces)


def test_html_table_schema_caption_and_cell_spans():
    doc, _, pieces = chunk(
        "<table><caption>Results</caption><tr><th>A</th><th>B</th></tr>"
        '<tr><td colspan="2">value</td></tr></table>',
        "html",
    )
    text = pieces[0][0]
    assert "Results" in text and "| A | B |\n| --- | --- |" in text
    assert "colspan=2" in text and "value" in text
    assert_coverage(doc, pieces)


def test_preserve_flags_change_oversized_serialization():
    text = "```python\n" + "    call()\n" * 30 + "```"
    _, _, preserved = chunk(text)
    _, _, relaxed = chunk(text, preserve_code_blocks=False)
    assert preserved != relaxed
    assert all(t.startswith("```python") for t, _ in preserved)
    assert any(not t.startswith("```python") for t, _ in relaxed)
    assert any(m["atomic_structure_lost"] for _, m in relaxed)
    text = "| A | B |\n| --- | --- |\n" + "| n | value |\n" * 30
    doc, _, relaxed = chunk(text, preserve_table_rows=False)
    assert any("| A | B |" not in t for t, _ in relaxed)
    assert all(m["atomic_structure_lost"] for _, m in relaxed)
    assert_coverage(doc, relaxed)


@pytest.mark.parametrize("size", [0, 1, 99, 100, 399, 400, 600, 601, 1200])
def test_exact_boundaries(size):
    engine = StructureAwareChunker(tokenizer=Characters(), cache=False)
    doc = Document(id="d", source_id="s", content="x" * size)
    pieces = list(engine.split(doc))
    assert "".join(t for t, _ in pieces) == doc.content
    assert len(pieces) == (
        0 if size == 0 else 1 if size <= 600 else 2 if size <= 800 else 3
    )
    assert all(m["embedding_token_count"] <= 600 for _, m in pieces)


def test_rst_code_and_conservative_table_are_displayed_once():
    text = (
        "Guide\n=====\n\n.. code-block:: python\n\n    def f():\n        pass\n\n"
        + "+-----+-----+\n| A   | B   |\n+=====+=====+\n"
        + ("| aaa | bbb |\n+-----+-----+\n" * 15)
    )
    doc, _, pieces = chunk(text, "rst")
    rendered = "\n".join(t for t, _ in pieces)
    assert "def f():" in rendered and rendered.count("pass") == 1
    assert "aaa" in rendered and "bbb" in rendered
    assert_coverage(doc, pieces)


def test_html_oversized_code_ordered_list_and_table_context():
    samples = [
        (
            '<pre><code class="language-python">'
            + "    call()\n" * 30
            + "</code></pre>",
            "```python",
            "code_block",
        ),
        (
            '<ol start="7">'
            + "<li>parent<ul><li>nested detail</li></ul></li>" * 20
            + "</ol>",
            "7. parent",
            "list",
        ),
        (
            "<table><caption>Caption</caption><tr><th>A</th><th>B</th></tr>"
            + "<tr><td>hello</td><td>world</td></tr>" * 20
            + "</table>",
            "| A | B |\n| --- | --- |",
            "table",
        ),
    ]
    for text, expected, kind in samples:
        doc, _, pieces = chunk(text, "html")
        assert len(pieces) > 1
        assert any(expected in t for t, _ in pieces)
        assert all(m["chunk_type"] == kind for _, m in pieces)
        assert_coverage(doc, pieces)
        if kind == "table":
            assert all("Caption" in t and expected in t for t, _ in pieces)
            for t, m in pieces:
                for repeat in m["repeated_context"]:
                    assert (
                        t[repeat["output_start"] : repeat["output_end"]]
                        == repeat["text"]
                    )


def test_html_giant_label_repeat_is_mapped_to_actual_tag():
    source = '<ol start="5"><li>' + "label " * 40 + "<ul><li>tail</li></ul></li></ol>"
    doc, _, pieces = chunk(source, "html")
    repeats = [r for _, m in pieces for r in m["repeated_context"]]
    assert repeats and all(r["text"] == "5." for r in repeats)
    assert all(
        doc.content[s["start_offset"] : s["end_offset"]] == "<li>"
        for r in repeats
        for s in r["source_spans"]
    )
    assert_coverage(doc, pieces)
    _, _, relaxed = chunk(source, "html", preserve_list_hierarchy=False)
    assert not any(m["repeated_context"] for _, m in relaxed)


def test_links_images_caption_malformed_html_and_references_remain_readable():
    source = '<h1>H</h1><p><a href="https://example.com">site</a><img src="image.png" alt="diagram"><p>after'
    doc, _, pieces = chunk(source, "html")
    rendered = "\n".join(t for t, _ in pieces)
    assert "[site](https://example.com)" in rendered
    assert "![diagram](image.png)" in rendered and "after" in rendered
    assert_coverage(doc, pieces)


def test_rst_oversized_code_and_list_retains_source_semantics():
    source = (
        "Guide\n=====\n\n.. code-block:: python\n\n"
        + "    call()\n" * 30
        + "\n"
        + "* parent\n\n  * nested\n\n" * 20
    )
    doc, _, pieces = chunk(source, "rst")
    rendered = "\n".join(t for t, _ in pieces)
    assert rendered.count("call()") == 30
    assert rendered.count("nested") == 20
    assert_coverage(doc, pieces)


def test_heuristic_sentence_boundary_differs_from_disabled():
    source = "Short sentence. " + "x" * 190
    _, _, heuristic = chunk(source)
    _, _, disabled = chunk(source, semantic_mode="disabled")
    assert heuristic[0][0] == "Short sentence."
    assert len(disabled[0][0]) == 80


def test_default_tokenizer_exact_actual_budget_and_determinism():
    doc = Document(id="d", source_id="s", content="word " * 1200)
    engine = StructureAwareChunker(cache=False)
    first = list(engine.split(doc))
    assert first == list(engine.split(doc))
    assert len(first) > 1
    assert all(
        engine.tokenizer.count(build_embedding_text(t, m)) <= 600 for t, m in first
    )


async def test_registered_adapter_and_base_document_api():
    from fastrag.normalization import TextNormalizer

    registry = NormalizerRegistry(builtins=False)
    registry.register("custom", TextNormalizer())
    doc = Document(
        id="custom",
        source_id="s",
        content="body",
        metadata={"content_format": "custom"},
    )
    engine = StructureAwareChunker(
        tokenizer=Characters(), normalizers=registry, cache=False
    )
    assert list(engine.split(doc))[0][0] == "body"
    from fastrag.ingestion.policy import Policy
    from fastrag.ingestion.steps.base import ExecutionContext

    context = ExecutionContext(pipeline_id="test", policy=Policy())
    output = [item async for item in engine.transform(doc, context)]
    again = [item async for item in engine.transform(doc, context)]
    assert output == again
    assert output[0].metadata["parent_id"] == doc.id
    assert output[0].metadata["chunk_index"] == 0
    assert len(output) == 1
    assert output[0].content == "body"
    assert output[0].metadata["position"] == 0


@pytest.mark.parametrize("defect", ["duplicate", "missing", "invisible", "all_missing"])
def test_operation_rejects_broken_primary_serialization(monkeypatch, defect):
    from fastrag.ingestion.steps.chunking.structure import _Atom

    engine = StructureAwareChunker(tokenizer=Characters(), cache=False)
    original = engine._raw_render

    def broken(normalized, node):
        atoms = original(normalized, node)
        if node.kind == "paragraph" and atoms:
            if defect == "duplicate":
                return [*atoms, atoms[0]]
            if defect == "missing":
                return atoms[:-1]
            if defect == "all_missing":
                return []
            return [*atoms[:-1], _Atom("", atoms[-1].spans, "body")]
        return atoms

    monkeypatch.setattr(engine, "_raw_render", broken)
    doc = Document(
        id="d", source_id="s", content="body", metadata={"content_format": "text"}
    )
    with pytest.raises(ChunkingError, match="Duplicate|not serialized|readable"):
        list(engine.split(doc))


def test_custom_tree_source_order_around_subsections():
    from fastrag.normalization import (
        CoverageUnit,
        NormalizedDocument,
        NormalizedNode,
        SourceSpan,
    )

    source = "# P\nbefore\n## S\ninside\nafter\n"

    def leaf(ident, kind, start, end, parent):
        return NormalizedNode(
            ident, kind, source[start:end], (SourceSpan(start, end),), parent_id=parent
        )

    hp = leaf("hp", "heading", 0, 4, "p")
    before = leaf("before", "paragraph", 4, 11, "p")
    hs = leaf("hs", "heading", 11, 16, "s")
    inside = leaf("inside", "paragraph", 16, 23, "s")
    after = leaf("after", "paragraph", 23, len(source), "p")
    sub = NormalizedNode(
        "s",
        "section",
        source[11:23],
        (SourceSpan(11, 23),),
        children=(hs, inside),
        parent_id="p",
        title="S",
        level=2,
    )
    parent = NormalizedNode(
        "p",
        "section",
        source,
        (SourceSpan(0, len(source)),),
        children=(hp, before, sub, after),
        parent_id="root",
        title="P",
        level=1,
    )
    root = NormalizedNode(
        "root", "document", source, (SourceSpan(0, len(source)),), children=(parent,)
    )
    doc = Document(id="d", source_id="s", content=source)
    units = tuple(
        CoverageUnit(n.id, n.id, n.spans[0]) for n in (hp, before, hs, inside, after)
    )
    normalized = NormalizedDocument(doc, "text", (root,), units)
    pieces = list(
        StructureAwareChunker(tokenizer=Characters(), cache=False).split_normalized(
            normalized
        )
    )
    assert ["before" in t for t, _ in pieces] == [True, False, False]
    assert ["inside" in t for t, _ in pieces] == [False, True, False]
    assert ["after" in t for t, _ in pieces] == [False, False, True]


def test_headerless_html_table_has_explicit_generated_schema():
    doc, _, pieces = chunk(
        "<table>" + "<tr><td>a</td><td>b</td></tr>" * 20 + "</table>", "html"
    )
    assert all("| Column 1 | Column 2 |\n| --- | --- |" in t for t, _ in pieces)
    assert_coverage(doc, pieces)
    doc, _, pieces = chunk("<table><tr><td>a</td><td>b</td></tr></table>", "html")
    assert "| Column 1 | Column 2 |\n| --- | --- |" in pieces[0][0]
    assert_coverage(doc, pieces)
