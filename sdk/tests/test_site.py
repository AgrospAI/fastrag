"""Site identity: the ``site_name``/``favicon_url`` read from a page's ``<head>``."""

from __future__ import annotations

from fastrag.ingestion._site import (
    MAX_FAVICON_URL_CHARS,
    inherit_site_metadata,
    site_metadata,
)
from fastrag.ingestion.models import Source

PAGE = "https://example.test/en/docs/intro/"


def _head(*tags: str) -> bytes:
    return f"<html><head>{''.join(tags)}</head><body></body></html>".encode()


def test_reads_site_name_and_resolves_icon_against_the_page() -> None:
    content = _head(
        '<meta property="og:site_name" content="  Example \n Docs ">',
        '<link rel="icon" href="/en/img/favicon.ico">',
    )
    assert site_metadata(content, "text/html; charset=utf-8", PAGE) == {
        "site_name": "Example Docs",
        "favicon_url": "https://example.test/en/img/favicon.ico",
    }


def test_falls_back_to_application_name_and_touch_icon() -> None:
    content = _head(
        '<link rel="apple-touch-icon" href="touch.png">',
        '<meta name="application-name" content="Example">',
    )
    assert site_metadata(content, "text/html", PAGE) == {
        "site_name": "Example",
        "favicon_url": "https://example.test/en/docs/intro/touch.png",
    }


def test_prefers_icon_over_touch_icon_and_honours_base_href() -> None:
    content = _head(
        '<base href="https://cdn.example.test/assets/">',
        '<link rel="apple-touch-icon" href="touch.png">',
        '<link rel="shortcut icon" href="favicon.ico">',
    )
    assert site_metadata(content, "text/html", PAGE) == {
        "favicon_url": "https://cdn.example.test/assets/favicon.ico",
    }


def test_drops_inline_and_oversized_icons() -> None:
    inline = _head('<link rel="icon" href="data:image/svg+xml,<svg></svg>">')
    assert site_metadata(inline, "text/html", PAGE) == {}

    oversized = _head(f'<link rel="icon" href="/{"a" * MAX_FAVICON_URL_CHARS}.ico">')
    assert site_metadata(oversized, "text/html", PAGE) == {}


def test_ignores_body_tags_and_non_html() -> None:
    body_only = (
        b"<html><head></head><body>"
        b'<meta property="og:site_name" content="Late">'
        b'<link rel="icon" href="/late.ico"></body></html>'
    )
    assert site_metadata(body_only, "text/html", PAGE) == {}

    pdf = _head('<meta property="og:site_name" content="Example">')
    assert site_metadata(pdf, "application/pdf", PAGE) == {}


def test_inherit_keeps_a_source_own_values() -> None:
    site = {"site_name": "Example", "favicon_url": "https://example.test/f.ico"}
    assert inherit_site_metadata({"kind": "pdf", "site_name": "Own"}, site) == {
        "kind": "pdf",
        "site_name": "Own",
        "favicon_url": "https://example.test/f.ico",
    }


def test_parsed_documents_carry_the_source_site_identity() -> None:
    from fastrag.ingestion.steps.parsing import DefaultParser

    source = Source(
        id="page",
        uri=PAGE,
        content=b"Text",
        content_type="text/plain",
        content_hash="hash",
        metadata={
            "status_code": 200,
            "site_name": "Example",
            "favicon_url": "https://example.test/f.ico",
        },
    )
    parser = DefaultParser()
    document = parser.build_document(source, "Text")
    assert document.metadata["site_name"] == "Example"
    assert document.metadata["favicon_url"] == "https://example.test/f.ico"
    assert "status_code" not in document.metadata

    overridden = parser.build_document(source, "Text", {"site_name": "Parsed"})
    assert overridden.metadata["site_name"] == "Parsed"
