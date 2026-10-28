"""PDF parsing through a remote `MinerU <https://github.com/opendatalab/MinerU>`_ API."""

from __future__ import annotations

import asyncio
import re
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
    pdf_page_count,
)
from fastrag.models import ContentFormat

_DATA_IMAGE = re.compile(r"!\[[^\]]*\]\(data:[^)]*\)")


class MineruParser(PDFParser):
    """Parse PDFs to Markdown (one document per page) with a MinerU API.

    MinerU returns one Markdown blob per request, so the PDF's pages are
    requested individually (``concurrency`` at a time) to keep page
    provenance. Needs the ``pdf`` extra (``pypdf``) to count pages.

    .. code-block:: python

        parser = MineruParser(api_key=MINERU_API_KEY, base_url=MINERU_URL)
    """

    content_format = ContentFormat.MARKDOWN

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://mineru.agrospai.udl.cat",
        tier: str = "standard",
        keep_images: bool = False,
        concurrency: int = 4,
        timeout: float = 1000,
        pages: Pages | None = None,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
    ) -> None:
        """Initialize ``MineruParser``.

        Args:
            api_key: Bearer token; excluded from cache identity.
            base_url: MinerU API root.
            tier: MinerU quality tier (``"standard"``, ...).
            keep_images: Keep embedded base64 images in the Markdown. Off by
                default: they are huge and useless for retrieval.
            concurrency: Simultaneous page requests per PDF.
            timeout: Per-request timeout in seconds.
            pages: 1-based page selection; see :class:`PDFParser`.
            cache: Cache backend, or ``False``/``None``.
            policy: Execution policy overriding the pipeline default.
        """
        super().__init__(pages=pages, cache=cache, policy=policy)
        self.api_key, self.base_url, self.tier = api_key, base_url, tier
        self.keep_images, self.concurrency, self.timeout = (
            keep_images,
            concurrency,
            timeout,
        )

    def config(self) -> Mapping[str, Any]:
        """Parse-affecting settings, without credentials."""
        return {
            **self._page_config(),
            "base_url": self.base_url,
            "tier": self.tier,
            "keep_images": self.keep_images,
        }

    async def parse_pdf(
        self, content: bytes, filename: str, pages: list[int] | None
    ) -> PDFParseResult:
        """Request each page and return its Markdown."""
        total = pdf_page_count(content)
        selected = [page for page in (pages or range(1, total + 1)) if page <= total]
        semaphore = asyncio.Semaphore(max(1, self.concurrency))

        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        ) as client:

            async def one(page: int) -> ParsedPage:
                async with semaphore:
                    body = await post_pdf(
                        client,
                        "MinerU",
                        "/v1/file_parse",
                        params={
                            "tier": self.tier,
                            "wait_timeout": int(self.timeout) - 100,
                            "page_range": str(page),
                        },
                        files={"files": (filename, content, "application/pdf")},
                    )
                files = body.get("files") or []
                if body.get("status") != "completed" or not files:
                    raise ParseError(f"MinerU failed on {filename} page {page}")
                markdown = ((files[0] or {}).get("content") or {}).get("markdown") or ""
                if not self.keep_images:
                    markdown = re.sub(r"\n{3,}", "\n\n", _DATA_IMAGE.sub("", markdown))
                return ParsedPage(page, markdown.strip())

            parsed = await asyncio.gather(*(one(page) for page in selected))
        return PDFParseResult(
            parsed, page_count=total, metadata={"mineru_tier": self.tier}
        )
