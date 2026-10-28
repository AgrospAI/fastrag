"""Markdown sections kept whole, with their heading ancestry for retrieval."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import Any

from fastrag.models import Document
from fastrag.ingestion.steps.chunking.base import Chunker, SplitPiece

_ATX = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*)|[ \t]*)$")
_SETEXT = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


class HeadingChunker(Chunker):
    """Split Markdown at headings without splitting section bodies.

    Each chunk includes its ancestor headings in text and a ``headings``
    metadata list ordered from outermost to innermost. ATX and Setext headings
    are recognized; headings inside fenced or indented code are ignored.
    Consecutive ancestor headings travel with their first subsection instead
    of becoming isolated title chunks. Text before the first heading is kept
    with an empty heading path. Sections have no size cap or overlap, so code,
    lists and tables stay intact even when a section is long.

    Inherits ``Chunker``'s ``cache`` and ``policy`` constructor arguments.
    """

    accepted_content_formats = frozenset(["text", "markdown"])

    def config(self) -> Mapping[str, Any]:
        """Return the section splitting configuration for cache identity."""
        return {"headings": "atx_and_setext", "include_ancestors": True}

    def split(self, document: Document) -> Iterator[SplitPiece]:
        """Yield whole sections with their heading paths and original Markdown."""
        self.validate_input(document)
        # (level, title, original heading Markdown)
        headings: list[tuple[int, str, str]] = []
        body: list[str] = []
        fence: str | None = None
        lines = document.content.splitlines(keepends=True)
        index = 0

        def section() -> SplitPiece:
            prefix = "\n\n".join(heading[2] for heading in headings)
            text = "\n\n".join(part for part in (prefix, "".join(body).strip()) if part)
            return text, {"headings": [heading[1] for heading in headings]}

        while index < len(lines):
            line = lines[index].rstrip("\r\n")
            marker = _FENCE.match(line)
            if fence is not None:
                body.append(lines[index])
                if (
                    marker
                    and marker[1][0] == fence[0]
                    and len(marker[1]) >= len(fence)
                    and not marker[2].strip()
                ):
                    fence = None
                index += 1
                continue
            if marker and not (marker[1][0] == "`" and "`" in marker[2]):
                fence = marker[1]
                body.append(lines[index])
                index += 1
                continue

            heading = _ATX.match(line)
            underline = (
                _SETEXT.match(lines[index + 1].rstrip("\r\n"))
                if index + 1 < len(lines)
                and line.strip()
                and not line.startswith(("    ", "\t"))
                else None
            )
            if heading:
                level = len(heading[1])
                title = re.sub(r"[ \t]+#+[ \t]*$", "", heading[2] or "").strip()
                original = line
                consumed = 1
            elif underline:
                level = 1 if underline[1][0] == "=" else 2
                title = line.strip()
                original = line + "\n" + lines[index + 1].rstrip("\r\n")
                consumed = 2
            else:
                body.append(lines[index])
                index += 1
                continue

            if "".join(body).strip() or (headings and level <= headings[-1][0]):
                yield section()
            body = []
            headings = [entry for entry in headings if entry[0] < level]
            headings.append((level, title, original))
            index += consumed

        if headings or "".join(body).strip():
            yield section()
