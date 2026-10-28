"""PDF parsing through a remote Chandra OCR API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import httpx

from fastrag.exceptions import ConfigurationError, ParseError
from fastrag.ingestion.cache import CacheStore
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.parsing._remote_pdf import post_pdf
from fastrag.ingestion.steps.parsing.pdf import (
    PDFParseResult,
    PDFParser,
    ParsedPage,
    Pages,
)
from fastrag.models import ContentFormat

OutputFormat = Literal["markdown", "html"]
_FORMATS: dict[str, ContentFormat] = {
    "markdown": ContentFormat.MARKDOWN,
    "html": ContentFormat.HTML,
}


class ChandraParser(PDFParser):
    """OCR PDFs with Chandra; one Markdown (or HTML) document per page.

    .. code-block:: python

        parser = ChandraParser(api_key=CHANDRA_API_KEY, base_url=CHANDRA_URL)
        parser = ChandraParser(..., output_format="html")
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://chandra.agrospai.udl.cat",
        output_format: OutputFormat = "markdown",
        timeout: float = 900,
        pages: Pages | None = None,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
    ) -> None:
        """Initialize ``ChandraParser``.

        Args:
            api_key: ``X-Api-Key`` credential; excluded from cache identity.
            base_url: Chandra API root.
            output_format: ``"markdown"`` or ``"html"`` page content.
            timeout: Request timeout in seconds.
            pages: 1-based page selection (sent 0-based to Chandra).
            cache: Cache backend, or ``False``/``None``.
            policy: Execution policy overriding the pipeline default.
        """
        if output_format not in _FORMATS:
            raise ConfigurationError(
                f"ChandraParser.output_format must be 'markdown' or 'html', "
                f"got {output_format!r}"
            )
        super().__init__(pages=pages, cache=cache, policy=policy)
        self.api_key, self.base_url, self.timeout = api_key, base_url, timeout
        self.output_format = output_format
        self.content_format = _FORMATS[output_format]

    def config(self) -> Mapping[str, Any]:
        """Parse-affecting settings, without credentials."""
        return {
            **self._page_config(),
            "base_url": self.base_url,
            "output_format": self.output_format,
        }

    async def parse_pdf(
        self, content: bytes, filename: str, pages: list[int] | None
    ) -> PDFParseResult:
        """OCR the PDF and map Chandra's per-page results."""
        data = (
            {"page_range": ",".join(str(page - 1) for page in pages)} if pages else {}
        )
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers={"X-Api-Key": self.api_key},
            timeout=self.timeout,
        ) as client:
            body = await post_pdf(
                client,
                "Chandra",
                "/v1/ocr",
                data=data,
                files={"file": (filename, content, "application/pdf")},
            )
        results = body.get("pages")
        if not isinstance(results, list):
            raise ParseError(f"Chandra returned no per-page results for {filename}")
        parsed: list[ParsedPage] = []
        for index, entry in enumerate(results):
            if entry.get("error"):
                continue
            # Chandra numbers pages 1-based *within the requested selection*.
            page_no = pages[index] if pages else int(entry.get("page", index + 1))
            parsed.append(
                ParsedPage(
                    page_no,
                    entry.get(self.output_format) or "",
                    self.content_format,
                    {"token_count": entry.get("token_count")},
                )
            )
        return PDFParseResult(
            parsed,
            page_count=None if pages else body.get("num_pages"),
            metadata={"failed_pages": body.get("errors") or []}
            if body.get("errors")
            else {},
        )
