from dataclasses import replace

import pytest

from fastrag.models import Document
from fastrag.normalization import NormalizerRegistry, SourceSpan


def doc(text, fmt):
    return Document("d", "s", text, metadata={"content_format": fmt})


def check(result):
    result.validate()
    spans = sorted(
        [u.span for u in result.units] + [e.span for e in result.exclusions],
        key=lambda s: s.start_offset,
    )
    assert (
        "".join(result.source[s.start_offset : s.end_offset] for s in spans)
        == result.source
    )
    assert len({n.id for n in result.walk()}) == len(list(result.walk()))
    assert result == NormalizerRegistry().normalize(result.document)


def test_markdown_structure_crlf_unicode():
    result = NormalizerRegistry().normalize(
        doc(
            "# Café\r\n\r\n### Jump\r\n\r\n- parent\r\n  - nested [link](https://example.org)\r\n\r\n```py\r\n  # code\r\n```\r\n\r\n| A | B |\r\n|---|---|\r\n| α | β |\r\n\r\n<div>keep</div>\r\n",
            "markdown",
        )
    )
    check(result)
    kinds = {n.kind for n in result.walk()}
    assert {
        "section",
        "heading",
        "list",
        "list_item",
        "code_block",
        "table",
        "table_row",
        "raw_block",
    } <= kinds
    sections = [n for n in result.walk() if n.kind == "section"]
    assert sections[1].parent_id == sections[0].id
    assert sections[0].title == "Café"
    assert any(n.metadata.get("language") == "py" for n in result.walk())
    assert any("https://example.org" in str(n.metadata) for n in result.walk())


def test_rst_real_parser_and_safe_directives(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("DO NOT LOAD")
    text = f"Title\n=====\n\n.. include:: {secret}\n\n.. imaginary:: keep me\n\n.. note:: hello\n\nTerm\n  Definition\n\n:field: value\n\n.. code:: python\n\n   x = 1\n\n+---+---+\n| A | B |\n+===+===+\n| α | β |\n+---+---+\n"
    result = NormalizerRegistry().normalize(doc(text, "rst"))
    check(result)
    assert "DO NOT LOAD" not in str(result)
    assert result.diagnostics
    assert {"admonition", "code_block", "table", "table_row", "list", "raw_block"} <= {
        n.kind for n in result.walk()
    }
    assert "imaginary" in "".join(n.content for n in result.walk())


def test_html_exclusions_recovery_and_tab_preservation():
    text = '<article><h1>Caf&eacute;</h1><div hidden><p>Tab &amp; text</p></div><ul><li>one<li>two</ul><table><caption>C</caption><tr><th colspan="2">H</th></tr><tr><td>A<td>B</table><pre>  x\n y</pre><figure><img src="x" alt="alt"><figcaption>cap</figcaption></figure><script>bad()</script><style>x{}</style><a href="/link">link</a></article>'
    result = NormalizerRegistry().normalize(doc(text, "html"))
    check(result)
    assert len(result.exclusions) == 2
    assert any(n.title == "Café" for n in result.walk())
    assert any("Tab &amp; text" in n.content for n in result.walk())
    assert {"list_item", "table_row", "caption", "code_block", "figure", "image"} <= {
        n.kind for n in result.walk()
    }
    assert any("colspan" in str(n.metadata) for n in result.walk())
    assert any("/link" in str(n.metadata) for n in result.walk())


def test_registry_extensions_and_format_routing():
    registry = NormalizerRegistry()
    assert (
        registry.normalize(
            Document(
                "d", "s", "<h1>literal</h1>", metadata={"content_type": "text/html"}
            )
        ).source_format
        == "text"
    )
    with pytest.raises(ValueError, match="registered"):
        registry.register("text", registry.adapter("text"))
    before = registry.identity
    registry.register("custom", registry.adapter("text"))
    assert registry.identity != before
    assert registry.normalize(doc("hello", "custom")).source == "hello"
    with pytest.raises(ValueError, match="docling"):
        registry.normalize(doc("{}", "docling"))
    with pytest.raises(ValueError):
        SourceSpan(3, 2)
    result = registry.normalize(doc("abc", "text"))
    with pytest.raises(ValueError):
        replace(result, units=()).validate()


def test_exclusion_splits_container_spans_without_fabricating_source_interval():
    from fastrag.normalization import HtmlNormalizer

    text = '<p>left<script>excluded</script>right<img src="x" alt="y"></p>'
    result = HtmlNormalizer().normalize(doc(text, "html"))
    result.validate()
    paragraph = next(n for n in result.walk() if n.kind == "paragraph")
    assert len(paragraph.spans) == 2
    assert "excluded" not in paragraph.content
    assert paragraph.metadata["images"] == [{"src": "x", "alt": "y"}]
    assert "excluded" not in paragraph.metadata["text"]


def test_rst_headings_and_table_source_locations():
    text = "Title\n=====\n\nIntro.\n\nSub\n---\n\nBody.\n\n+---+---+\n| A | B |\n+===+===+\n| x | y |\n+---+---+\n"
    result = NormalizerRegistry().normalize(doc(text, "rst"))
    headings = [n for n in result.walk() if n.kind == "heading"]
    assert headings[0].content.startswith("Title\n=====")
    assert headings[1].content.startswith("Sub\n---")
    table = next(n for n in result.walk() if n.kind == "table")
    assert table.content.startswith("+---+---+")
    assert "Body." not in table.content
    rows = [n for n in table.walk() if n.kind == "table_row"]
    assert rows[0].spans[0].start_offset < rows[1].spans[0].start_offset


@pytest.mark.parametrize(
    "fmt,text",
    [
        ("markdown", "Title\n===\n\n    code\n\n> quote\n\n---\n\n![alt](x)\n"),
        (
            "rst",
            "Title\n=====\n\n- parent\n\n  - child\n\nLiteral::\n\n  code\n\n===== =====\nA     B\n===== =====\nx     y\n===== =====\n",
        ),
        (
            "html",
            "<section><h1>Title</h1><article><h3>Sub</h3><p>body</article></section>",
        ),
    ],
)
def test_additional_parser_features(fmt, text):
    result = NormalizerRegistry().normalize(doc(text, fmt))
    check(result)
    assert any(n.kind == "section" for n in result.walk())


def test_html_configured_boilerplate_and_hidden_exclusions():
    from fastrag.normalization import HtmlNormalizer

    result = HtmlNormalizer(
        exclude_tags=("script", "style", "nav"), exclude_hidden=True
    ).normalize(doc("<nav>nav</nav><div hidden>hidden</div><p>visible</p>", "html"))
    result.validate()
    assert len(result.exclusions) == 2
    assert result.roots[0].content == "<p>visible</p>"


def test_registry_identity_does_not_construct_optional_adapters(monkeypatch):
    registry = NormalizerRegistry()

    def forbidden():
        raise AssertionError("unselected adapter constructed")

    monkeypatch.setitem(registry._factories, "markdown", forbidden)
    monkeypatch.setitem(registry._factories, "rst", forbidden)
    before = registry.identity
    assert registry.normalize(doc("plain", "text")).source == "plain"
    assert registry.identity == before


def test_normalized_metadata_is_json_compatible():
    import json
    from dataclasses import asdict

    for fmt, text in [
        ("markdown", "[link](https://example.org)\n"),
        ("rst", "`link <https://example.org>`_\n"),
        ("html", '<p><a href="/x">link</a></p>'),
    ]:
        result = NormalizerRegistry().normalize(doc(text, fmt))
        assert json.loads(json.dumps(asdict(result)))["document"]["content"] == text


def test_rst_never_executes_registered_custom_directive(monkeypatch):
    from docutils.parsers.rst import Directive, directives

    class Dangerous(Directive):
        def run(self):
            raise AssertionError("custom directive executed")

    monkeypatch.setitem(directives._directives, "dangerous", Dangerous)
    result = NormalizerRegistry().normalize(doc(".. dangerous::\n\n   keep\n", "rst"))
    assert result.diagnostics
    assert "dangerous" in result.source
    result.validate()


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize(
    "directive", ["code-block:: python", "code:: python", "parsed-literal::"]
)
def test_rst_heading_directive_and_body_have_exact_authoritative_slices(
    newline, directive
):
    heading = "Guide\n=====\n"
    block = f".. {directive}\n\n    def f():\n        pass\n"
    body = "Links `site <https://example.com>`_."
    text = (heading + "\n" + block + "\n" + body + "\n").replace("\n", newline)
    result = NormalizerRegistry().normalize(doc(text, "rst"))
    check(result)
    nodes = [
        n for n in result.walk() if n.kind in {"heading", "code_block", "paragraph"}
    ]
    assert [(n.kind, n.content) for n in nodes] == [
        ("heading", heading.replace("\n", newline)),
        ("code_block", block.replace("\n", newline)),
        ("paragraph", body),
    ]
    assert nodes[0].spans[-1].end_offset <= nodes[1].spans[0].start_offset
    assert nodes[1].spans[-1].end_offset <= nodes[2].spans[0].start_offset


@pytest.mark.parametrize(
    "directive", ["imaginary::", "raw:: html", "include:: missing", "code:: python"]
)
def test_rst_failed_directive_retained_once_without_swallowing_neighbors(directive):
    block = f".. {directive}\n"
    text = "Title\n=====\n\n" + block + "\nAfter.\n"
    result = NormalizerRegistry().normalize(doc(text, "rst"))
    check(result)
    raw = [n for n in result.walk() if n.kind == "raw_block"]
    assert len(raw) == 1
    assert raw[0].content == block
    assert (
        next(n for n in result.walk() if n.kind == "heading").content
        == "Title\n=====\n"
    )
    assert next(n for n in result.walk() if n.kind == "paragraph").content == "After."
    assert result.diagnostics


def test_rst_literal_repeated_code_nested_directive_and_empty_section_boundaries():
    text = "=====\nTitle\n=====\n\nLiteral::\n\n   same\n\n.. code::\n\n   same\n\n.. note::\n\n   .. code:: python\n\n      nested()\n\nAfter.\n\nEmpty\n-----\n\nLast\n----\n"
    result = NormalizerRegistry().normalize(doc(text, "rst"))
    check(result)
    headings = [n.content for n in result.walk() if n.kind == "heading"]
    assert headings == ["=====\nTitle\n=====\n", "Empty\n-----\n", "Last\n----\n"]
    codes = [n for n in result.walk() if n.kind == "code_block"]
    assert [n.content for n in codes] == [
        "   same\n",
        ".. code::\n\n   same\n",
        "   .. code:: python\n\n      nested()\n",
    ]
    assert (
        next(n for n in result.walk() if n.kind == "admonition").content
        == ".. note::\n\n   .. code:: python\n\n      nested()\n"
    )
    assert codes[0].spans[-1].end_offset <= codes[1].spans[0].start_offset


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_rst_unknown_body_adjacent_directives_and_quoted_literal(newline):
    unknown = ".. imaginary:: hi\n   :option: yes\n\n   body\n"
    image1 = ".. image:: first.png\n"
    image2 = ".. image:: second.png\n"
    quoted = "> first\n> second\n"
    text = (
        "Title\n=====\n\n"
        + unknown
        + "\n"
        + image1
        + image2
        + "\nQuoted::\n\n"
        + quoted
        + "\nAfter.\n"
    ).replace("\n", newline)
    result = NormalizerRegistry().normalize(doc(text, "rst"))
    check(result)
    blocks = [n for n in result.walk() if n.kind not in {"document", "section"}]
    assert [n.content for n in blocks] == [
        part.replace("\n", newline)
        for part in [
            "Title\n=====\n",
            unknown,
            image1,
            image2,
            "Quoted::",
            quoted,
            "After.",
        ]
    ]
    assert all(
        left.spans[-1].end_offset <= right.spans[0].start_offset
        for left, right in zip(blocks, blocks[1:])
    )


def test_rst_duplicate_markup_titles_diagnostic_does_not_steal_adornment():
    text = "*Title*\n=======\n\n.. code::\n\n   x\n\n*Title*\n=======\n\nAfter.\n"
    result = NormalizerRegistry().normalize(doc(text, "rst"))
    check(result)
    assert [n.content for n in result.walk() if n.kind == "heading"] == [
        "*Title*\n=======\n"
    ] * 2
    assert not any(n.kind == "raw_block" for n in result.walk())
    assert result.diagnostics


@pytest.mark.parametrize(
    "wrapper", ["Title\n=====\n\n{}", ".. note::\n\n{}", "- item\n\n{}"]
)
def test_rst_custom_directive_never_executes_in_nested_parsing(monkeypatch, wrapper):
    from docutils.parsers.rst import Directive, directives

    class Dangerous(Directive):
        def run(self):
            raise AssertionError("nested custom directive executed")

    monkeypatch.setitem(directives._directives, "dangerous", Dangerous)
    nested = ".. dangerous::\n\n   keep\n"
    if wrapper.startswith(("..", "-")):
        nested = "\n".join("   " + line if line else "" for line in nested.split("\n"))
    result = NormalizerRegistry().normalize(doc(wrapper.format(nested), "rst"))
    check(result)
    raw = [n for n in result.walk() if n.kind == "raw_block"]
    assert len(raw) == 1
    assert ".. dangerous::" in raw[0].content and "keep" in raw[0].content


def test_rst_identity_stays_stable_after_new_source_location_policy_resolution():
    registry = NormalizerRegistry()
    identity = registry.identity
    registry.normalize(doc("Title\n=====\n", "rst"))
    assert registry.identity == identity


@pytest.mark.parametrize("adornment", ["---", "==="])
def test_rst_adjacent_heading_adornments_do_not_become_next_overline(adornment):
    text = f"One\n===\nTwo\n{adornment}\n\nBody.\n"
    result = NormalizerRegistry().normalize(doc(text, "rst"))
    check(result)
    assert [n.content for n in result.walk() if n.kind == "heading"] == [
        "One\n===\n",
        f"Two\n{adornment}\n",
    ]
