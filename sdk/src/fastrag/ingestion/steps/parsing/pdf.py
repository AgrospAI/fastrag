"""``PDFParser``: the shared interface of every PDF-to-text parser.

Concrete PDF parsers (``MineruParser``, ``DoclingServeParser``,
``ChandraParser``) differ only in *how* they turn PDF bytes into text. They all
share this contract, so they can be swapped without touching the pipeline:

- The input is always a PDF (``application/pdf``).
- The output is **one** :class:`~fastrag.models.Document` **per PDF page**, so
  the source and the page are tracked in every document's metadata
  (``source_uri``, ``page_no``, ``page_count``, ``source_url``).
- ``metadata["content_format"]`` states what each page's ``content`` is
  (Markdown, HTML, ...) so a downstream step can pick the matching chunker.

Because pages are emitted as separate, ordered, 1-based-numbered documents
linked by ``source_id``, a later chunker can join pages back together (e.g. to
keep a paragraph or table that spans a page break in one chunk) without the
parser having to know about it.

A new parser only implements :meth:`PDFParser.parse_pdf`.
"""

from __future__ import annotations

import dataclasses
import io
from abc import abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Literal
from urllib.parse import urlparse

from fastrag.exceptions import ConfigurationError, ParseError
from fastrag.ingestion._hashing import content_hash
from fastrag.ingestion.cache import CacheStore
from fastrag.ingestion.filters import MediaTypeFilter
from fastrag.ingestion.models import Source
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext
from fastrag.ingestion.steps.parsing.base import Parser
from fastrag.models import ContentFormat, Document

#: Page selection accepted by every PDF parser: ``"1-3,5"`` or ``[1, 2, 3, 5]``.
#: Always **1-based**; each parser converts to its API's own convention.
Pages = str | Sequence[int]

#: The filter a :class:`~fastrag.ingestion.steps.parsing.router.ParserRouter`
#: route to a :class:`PDFParser` matches against.
#:
#: .. code-block:: python
#:
#:     parser = ParserRouter().route(pdf_filter, ChandraParser(...))
pdf_filter = MediaTypeFilter(include_types=["application/pdf", "application/x-pdf"])


@dataclass(frozen=True, slots=True)
class ParsedPage:
    """One page's parsed content, as returned by :meth:`PDFParser.parse_pdf`."""

    #: 1-based page number in the original PDF.
    page_no: int
    content: str
    #: What ``content`` is; becomes ``metadata["content_format"]``.
    content_format: ContentFormat = ContentFormat.MARKDOWN
    #: Extra per-page metadata (e.g. a page-level OCR confidence).
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PDFParseResult:
    """What a backend produced for a whole PDF."""

    pages: Sequence[ParsedPage]
    #: Total pages in the PDF if the backend knows it.
    page_count: int | None = None
    #: Extra metadata shared by every page (e.g. the backend's model/version).
    metadata: Mapping[str, Any] = field(default_factory=dict)


def normalize_pages(pages: Pages | None) -> list[int] | None:
    """Turn ``"1-3,5"`` / ``[1, 2, 3, 5]`` into a sorted, de-duplicated list.

    Raises:
        fastrag.exceptions.ConfigurationError: Malformed or non-positive pages.
    """
    if pages is None:
        return None
    numbers: set[int] = set()
    try:
        if isinstance(pages, str):
            for part in pages.split(","):
                part = part.strip()
                if not part:
                    continue
                start, _, end = part.partition("-")
                low, high = int(start), int(end or start)
                if high < low:
                    raise ValueError(part)
                numbers.update(range(low, high + 1))
        else:
            numbers.update(int(page) for page in pages)
    except ValueError as error:
        raise ConfigurationError(
            f"Invalid 1-based page selection {pages!r}; use e.g. '1-3,5' or [1, 2, 3, 5]"
        ) from error
    if not numbers or min(numbers) < 1:
        raise ConfigurationError(f"Pages are 1-based and non-empty, got {pages!r}")
    return sorted(numbers)


def contiguous_runs(pages: Sequence[int]) -> list[tuple[int, int]]:
    """``[1, 2, 3, 7, 8]`` -> ``[(1, 3), (7, 8)]``."""
    runs: list[tuple[int, int]] = []
    for page in pages:
        if runs and page == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], page)
        else:
            runs.append((page, page))
    return runs


def pdf_page_count(content: bytes) -> int:
    """Count a PDF's pages (needs the ``pdf`` extra: ``pypdf``)."""
    try:
        from pypdf import PdfReader
    except ImportError as error:  # pragma: no cover - installs without the extra
        raise ConfigurationError(
            'This PDF parser needs the "pdf" extra: pip install "fastrag-ai[pdf]"'
        ) from error
    import logging

    # pypdf warns once per malformed object; the count is unaffected.
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    try:
        return len(PdfReader(io.BytesIO(content)).pages)
    except Exception as error:
        raise ParseError(f"Could not read the PDF page count: {error}") from error


class PDFParser(Parser):
    """Base class of every PDF parser: one document per page.

    Subclasses implement :meth:`parse_pdf` and :meth:`config`; they inherit
    media-type validation (``application/pdf`` only), per-page document
    construction, page metadata and per-page identities.
    """

    accepted_media_types = frozenset({"application/pdf"})

    def __init__(
        self,
        *,
        pages: Pages | None = None,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
    ) -> None:
        """Initialize the shared PDF parser options.

        Args:
            pages: Restrict parsing to these **1-based** pages (``"1-3,5"``
                or ``[1, 2, 3, 5]``). ``None`` parses every page.
            cache: Cache backend, or ``False``/``None`` — see
                :class:`~fastrag.ingestion.steps.base.PipelineStep`.
            policy: Execution policy overriding the pipeline default.
        """
        super().__init__(cache=cache, policy=policy)
        self.pages = normalize_pages(pages)

    @abstractmethod
    async def parse_pdf(
        self, content: bytes, filename: str, pages: list[int] | None
    ) -> PDFParseResult:
        """Parse ``content`` (PDF bytes), returning per-page results.

        ``pages`` is the sorted 1-based selection or ``None`` for all pages.
        Pages with no text may be omitted or returned empty.
        """

    def _page_config(self) -> dict[str, Any]:
        """Shared cache-identity entries; subclasses merge them into ``config``."""
        return {"pages": self.pages, "paged_documents": 1}

    async def transform(
        self, item: Source, context: ExecutionContext
    ) -> AsyncIterator[Document]:
        """Yield one document per non-empty PDF page, in page order."""
        self.validate_input(item)
        if not item.content:
            return
        name = PurePosixPath(urlparse(item.uri).path).name or "document.pdf"
        result = await self.parse_pdf(item.content, name, self.pages)
        base_uri = item.uri.split("#", 1)[0]
        for page in sorted(result.pages, key=lambda parsed: parsed.page_no):
            text = page.content.strip()
            if not text:
                continue
            metadata: dict[str, Any] = {
                **result.metadata,
                **page.metadata,
                "parser": type(self).__name__,
                "page_no": page.page_no,
                "source_url": f"{base_uri}#page={page.page_no}",
            }
            if result.page_count is not None:
                metadata["page_count"] = result.page_count
            built = self.build_document(
                item, text, metadata, content_format=page.content_format
            )
            # Identical pages of one source must not collapse into one id.
            yield dataclasses.replace(
                built,
                id=content_hash(f"{item.id}#page={page.page_no}:{text}"),
            )
