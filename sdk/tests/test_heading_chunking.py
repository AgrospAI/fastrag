"""Whole-section splitting and heading ancestry without external services."""

import pytest

from fastrag.ingestion.steps.chunking import HeadingChunker
from fastrag.models import Document


def split(text):
    return list(
        HeadingChunker().split(Document(id="doc", source_id="source", content=text))
    )


def test_sections_keep_code_tables_and_ancestry():
    code = "```python\n# a comment\n" + "call()\n" * 200 + "```"
    table = "| A | B |\n| --- | --- |\n| 1 | 2 |"
    pieces = split(
        f"# Guide\n## Build\n{code}\n{table}\n### Deploy\n- Upload\n- Publish\n## Verify\nCheck status."
    )
    assert len(pieces) == 3
    assert pieces[0] == (
        f"# Guide\n\n## Build\n\n{code}\n{table}",
        {"headings": ["Guide", "Build"]},
    )
    assert pieces[1] == (
        "# Guide\n\n## Build\n\n### Deploy\n\n- Upload\n- Publish",
        {"headings": ["Guide", "Build", "Deploy"]},
    )
    assert pieces[2][1] == {"headings": ["Guide", "Verify"]}


def test_setext_skipped_levels_and_duplicate_headings():
    pieces = split(
        "Preamble.\n\nGuide\n=====\n\nSection\n-------\nBody.\n##### Detail ###\nDeep.\n## Section\nOther body."
    )
    assert [metadata["headings"] for _, metadata in pieces] == [
        [],
        ["Guide", "Section"],
        ["Guide", "Section", "Detail"],
        ["Guide", "Section"],
    ]
    assert pieces[0][0] == "Preamble."
    assert "Guide\n=====" in pieces[1][0]
    assert "Other body." in pieces[-1][0]


def test_code_fences_and_indentation_do_not_create_sections():
    text = "# Guide\n~~~~bash\n## comment\n~~~\n# still in code\n~~~~\n    # indented comment\n    Text\n    ----\n\n```python\n# unclosed fence\n## still code"
    pieces = split(text)
    assert len(pieces) == 1
    assert pieces[0][1] == {"headings": ["Guide"]}
    assert "# still in code" in pieces[0][0]
    assert "# unclosed fence" in pieces[0][0]


@pytest.mark.parametrize("text", ["", " \n\t"])
def test_empty_document(text):
    assert split(text) == []


def test_heading_only_sections_are_preserved():
    pieces = split("# Guide\n## Empty\n## Other\n### Last")
    assert [metadata["headings"] for _, metadata in pieces] == [
        ["Guide", "Empty"],
        ["Guide", "Other", "Last"],
    ]
