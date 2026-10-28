"""Shared preservation of interactive panels, SVG labels and image descriptions."""

from __future__ import annotations

import base64
import re
from typing import Any
from urllib.parse import unquote_to_bytes, urljoin

import httpx

from fastrag.exceptions import ParseError
from fastrag.ingestion.steps.parsing.vision import OpenAIVisionParser


def reveal_content(root: Any) -> None:
    """Keep every hidden panel in Markdown, including CSS-hidden alternatives."""
    for element in root.select("[hidden], [aria-hidden], [style], details"):
        element.attrs.pop("hidden", None)
        element.attrs.pop("aria-hidden", None)
        if element.name == "details":
            element["open"] = ""
        if element.has_attr("style"):
            element["style"] = re.sub(
                r"(?:display\s*:\s*none|visibility\s*:\s*hidden)\s*(?:!important)?\s*;?",
                "",
                element["style"],
                flags=re.I,
            )


async def enrich_content(
    root: Any, soup: Any, uri: str, vision: OpenAIVisionParser | None
) -> None:
    """Replace meaningful SVGs and annotate images in their original positions."""
    for svg in list(root.select("svg")):
        if svg.find_parent(class_=re.compile(r"icon", re.I)):
            svg.decompose()
            continue
        labels = list(
            dict.fromkeys(
                label.get_text(" ", strip=True)
                for label in svg.select("title, desc, text, foreignobject")
                if label.get_text(" ", strip=True)
            )
        )
        description = ""
        if vision:
            png = svg.get("data-fastrag-png")
            if png:
                description = await vision.describe(base64.b64decode(png), "image/png")
            else:
                description = await vision.describe(str(svg).encode(), "image/svg+xml")
        if labels or description:
            replacement = soup.new_tag("p")
            replacement.string = (
                "Diagram: "
                + " | ".join(labels)
                + ("\n" + description if description else "")
            )
            svg.replace_with(replacement)
        else:
            svg.decompose()
    if not vision:
        for image in root.select('img[src^="data:"]'):
            image.decompose()
        return
    descriptions: dict[str, str] = {}
    async with httpx.AsyncClient(
        timeout=vision.timeout, follow_redirects=True
    ) as client:
        for image in list(root.select("img")):
            src = image.get("src") or image.get("data-src")
            if (
                not src
                or image.get("role") == "presentation"
                or image.get("aria-hidden") == "true"
            ):
                continue
            # Prefer the largest responsive source over the thumbnail.
            srcset = image.get("srcset")
            if srcset and not src.startswith("data:"):
                candidates = [part.strip().split() for part in srcset.split(",")]
                if all(len(part) == 2 for part in candidates):
                    src = max(candidates, key=lambda part: float(part[1][:-1]))[0]
            src = urljoin(uri, src)
            if src not in descriptions:
                try:
                    if src.startswith("data:"):
                        header, data = src.split(",", 1)
                        mime = header[5:].split(";")[0]
                        content = (
                            base64.b64decode(data)
                            if ";base64" in header
                            else unquote_to_bytes(data)
                        )
                    else:
                        response = await client.get(src)
                        response.raise_for_status()
                        content = response.content
                        mime = response.headers.get("content-type", "image/png").split(
                            ";"
                        )[0]
                    descriptions[src] = await vision.describe(content, mime)
                except Exception as error:
                    raise ParseError(
                        f"Image parsing failed for {uri}: {type(error).__name__}"
                    ) from error
            image["src"] = src
            paragraph = soup.new_tag("p")
            paragraph.string = "Image description: " + descriptions[src]
            image.insert_after(paragraph)
            # Never embed a base64 image blob in the searchable document.
            if src.startswith("data:"):
                image.decompose()
