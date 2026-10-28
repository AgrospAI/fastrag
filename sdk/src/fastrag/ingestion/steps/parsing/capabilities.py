"""MIME capability helpers; source MIME and output representation stay distinct."""

from __future__ import annotations

import mimetypes
from fnmatch import fnmatchcase
from urllib.parse import urlsplit


def source_media_type(source) -> str:
    """Use declared MIME, or extension for missing/generic declarations."""
    declared = (source.content_type or "").split(";", 1)[0].strip().lower()
    if declared and declared != "application/octet-stream":
        return declared
    path = urlsplit(source.uri).path
    # Platform mimetypes tables disagree on reStructuredText and Markdown.
    suffix = path.rsplit(".", 1)[-1].lower()
    known = {"rst": "text/x-rst", "md": "text/markdown", "markdown": "text/markdown"}
    return (
        known.get(suffix)
        or mimetypes.guess_type(path)[0]
        or declared
        or "application/octet-stream"
    )


def accepts_media_type(actual: str, accepted: frozenset[str]) -> bool:
    """Match explicit MIME values and family wildcards."""
    return any(fnmatchcase(actual, pattern.lower()) for pattern in accepted)


TEXT_MEDIA_TYPES = frozenset(
    {
        "text/*",
        "application/json",
        "application/xml",
        "application/xhtml+xml",
        "application/javascript",
        "application/x-yaml",
        "application/yaml",
        "application/octet-stream",
    }
)


def text_format(source) -> str:
    """Describe retained raw textual syntax without converting it."""
    media = source_media_type(source)
    if media in {"text/markdown", "text/x-markdown"}:
        return "markdown"
    if media in {"text/x-rst", "text/prs.fallenstein.rst", "text/restructuredtext"}:
        return "rst"
    if media in {"text/html", "application/xhtml+xml"}:
        return "html"
    return "text"
