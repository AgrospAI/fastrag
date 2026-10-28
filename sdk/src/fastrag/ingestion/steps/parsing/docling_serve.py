"""PDF parsing through a remote `docling-serve <https://github.com/docling-project/docling-serve>`_.

Not to be confused with :class:`~fastrag.ingestion.steps.parsing.docling.DoclingParser`,
which runs Docling locally and handles every office format; this one only
calls a remote server and only accepts PDFs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import httpx

from fastrag.exceptions import ParseError
from fastrag.ingestion.cache import CacheStore
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.parsing._remote_pdf import post_pdf
from fastrag.ingestion.steps.parsing.pdf import (
    PDFParseResult,
    PDFParser,
    ParsedPage,
    Pages,
    contiguous_runs,
)
from fastrag.models import ContentFormat

_PAGE_BREAK = "<<<FASTRAG_PAGE_BREAK>>>"


class DoclingServeParser(PDFParser):
    """Parse PDFs to Markdown (one document per page) with docling-serve.

    The server is asked to put a page-break marker between pages, and the
    Markdown is split on it. Non-contiguous ``pages`` cost one request per
    contiguous run.

    .. code-block:: python

        parser = DoclingServeParser(api_key=DOCLING_API_KEY, base_url=DOCLING_URL)
    """

    content_format = ContentFormat.MARKDOWN

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://docling.agrospai.udl.cat",
        timeout: float = 600,
        pages: Pages | None = None,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
    ) -> None:
        """Initialize ``DoclingServeParser`` (``api_key`` is not cached)."""
        super().__init__(pages=pages, cache=cache, policy=policy)
        self.api_key, self.base_url, self.timeout = api_key, base_url, timeout

    def config(self) -> Mapping[str, Any]:
        """Parse-affecting settings, without credentials."""
        return {**self._page_config(), "base_url": self.base_url}

    async def parse_pdf(
        self, content: bytes, filename: str, pages: list[int] | None
    ) -> PDFParseResult:
        """Convert the PDF and split the Markdown on page breaks."""
        runs = contiguous_runs(pages) if pages else [(1, None)]
        parsed: list[ParsedPage] = []
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers={"X-Api-Key": self.api_key},
            timeout=self.timeout,
        ) as client:
            for start, end in runs:
                data: dict[str, Any] = {
                    "to_formats": "md",
                    "md_page_break_placeholder": _PAGE_BREAK,
                    "image_export_mode": "placeholder",
                }
                if end is not None:
                    data["page_range"] = [str(start), str(end)]
                body = await post_pdf(
                    client,
                    "docling-serve",
                    "/v1/convert/file",
                    data=data,
                    files={"files": (filename, content, "application/pdf")},
                )
                if body.get("status") != "success":
                    raise ParseError(f"docling-serve failed on {filename}")
                markdown = (body.get("document") or {}).get("md_content") or ""
                for offset, text in enumerate(markdown.split(_PAGE_BREAK)):
                    # Image placeholders carry no information for retrieval.
                    text = text.replace("<!-- image -->", "").strip()
                    parsed.append(ParsedPage(start + offset, text))
        if pages:
            allowed = set(pages)
            parsed = [page for page in parsed if page.page_no in allowed]
        return PDFParseResult(parsed)
