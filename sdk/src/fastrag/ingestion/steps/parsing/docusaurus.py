"""Docusaurus documentation parsing: the article, not the site chrome.

A Docusaurus site (``docs.pontus-x.eu``, ``agrospai.udl.cat/*/docs/*``) wraps
its actual article in a container ``div`` carrying a ``markdown`` class,
surrounded by navigation, sidebars, breadcrumbs, footers, and "edit this
page" links that :class:`~fastrag.ingestion.steps.parsing.html.HtmlParser`
has no way to distinguish from content. Feeding the whole page to it buries
the article in boilerplate that then gets chunked and embedded — noise a RAG
index does not want. This parser extracts only the first such ``div`` and
converts that subtree; a page carrying no such container is parsed whole,
the way ``HtmlParser`` does, with a metadata marker recording the fallback.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
import re
from typing import Any, Literal
from urllib.parse import urljoin

from fastrag.ingestion.steps.parsing.vision import OpenAIVisionParser
from fastrag.ingestion.steps.parsing._content import reveal_content, enrich_content

from fastrag.exceptions import ConfigurationError, ParseError
from fastrag.ingestion.cache import CacheStore
from fastrag.ingestion.filters import Filter
from fastrag.ingestion.models import Source
from fastrag.models import ContentFormat, Document
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext
from fastrag.ingestion.steps.parsing.base import Parser
from fastrag.ingestion.steps.parsing.html import (
    CONVERSION,
    DEFAULT_STRIP_SELECTORS,
    MISSING_EXTRA,
    BeautifulSoup,
    convert,
)

#: Article headers contain the page heading; permalink anchors are UI chrome.
DEFAULT_DOCUSAURUS_SELECTORS = tuple(
    selector for selector in DEFAULT_STRIP_SELECTORS if selector != "header"
) + ("a.hash-link",)

#: Metadata key set on every produced document, ``True`` when the page
#: carried no ``markdown``-classed ``div`` and this parser fell back to
#: parsing the whole document instead — the marker a caller reads to tell the
#: two cases apart.
FALLBACK_METADATA_KEY = "docusaurus_fallback"


class _DocusaurusFilter(Filter[Source | Document]):
    """Matches HTML carrying Docusaurus markup, not mere mentions of it."""

    def matches(self, item: Source | Document) -> bool:
        """True when ``item`` carries a Docusaurus article or generator tag."""
        content = item.content
        if isinstance(content, str):
            content = content.encode()
        return b"theme-doc-markdown" in content or bool(
            re.search(rb"<meta\b[^>]*content=[\"']Docusaurus", content, re.I)
        )


#: The filter a :class:`~fastrag.ingestion.steps.parsing.router.ParserRouter`
#: routes Docusaurus pages with. It inspects content, so it is not
#: ``header_safe``::
#:
#:     parser = ParserRouter(cache=cache).route(
#:         html_filter & docusaurus_filter, DocusaurusParser()
#:     )
docusaurus_filter: Filter[Source | Document] = _DocusaurusFilter()


class DocusaurusParser(Parser):
    """Extract a Docusaurus page's article, discarding the surrounding chrome.

    Finds the first ``div`` whose class list includes ``markdown`` (Docusaurus
    themes typically render it as ``class="theme-doc-markdown markdown"``) and
    converts only that subtree to Markdown, using the same
    :class:`~fastrag.ingestion.steps.parsing.html.HtmlParser` machinery
    (strip selectors, link absolutization, HTML-to-Markdown conversion) —
    just scoped to the article rather than the whole document. A page with no
    such ``div`` is parsed whole, exactly as ``HtmlParser`` would, and the
    produced document's metadata records that the fallback happened.

    .. code-block:: python

        from fastrag.ingestion.steps.parsing import DocusaurusParser, ParserRouter

        parser = ParserRouter(cache=cache).route(docs_filter, DocusaurusParser())
    """

    content_format = ContentFormat.MARKDOWN

    accepted_media_types = frozenset({"text/html", "application/xhtml+xml"})

    def __init__(
        self,
        *,
        strip_selectors: Sequence[str] = DEFAULT_DOCUSAURUS_SELECTORS,
        encoding: str | None = None,
        vision: OpenAIVisionParser | None = None,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
    ) -> None:
        """Initialize ``DocusaurusParser``.

        Args:
            strip_selectors: CSS selectors removed from the extracted subtree
                (or the whole document, on fallback) before conversion.
            encoding: Force an encoding. ``None`` lets the document's own
                declaration win.
            vision: Optional image and diagram description model.
            cache: Cache backend, or ``False``/``None`` — see
                :class:`~fastrag.ingestion.steps.base.PipelineStep`.
            policy: Execution policy overriding the pipeline default.

        Raises:
            fastrag.exceptions.ConfigurationError: The optional "html" extra
                is not installed.
        """
        if BeautifulSoup is None or convert is None:
            raise ConfigurationError(MISSING_EXTRA)
        super().__init__(cache=cache, policy=policy)
        self.strip_selectors = tuple(strip_selectors)
        self.encoding = encoding
        self.vision = vision

    def config(self) -> Mapping[str, Any]:
        """This parser's cache-identity configuration.

        ``strip_selectors``/``encoding`` change what a given page's bytes
        parse into, so both are part of the fingerprint. Which ``div`` is
        matched is not configurable, so it is not part of it.
        """
        return {
            "strip_selectors": list(self.strip_selectors),
            "encoding": self.encoding,
            "include_hidden": True,
            "svg_labels": True,
            "vision": dict(self.vision.config()) if self.vision else None,
        }

    async def transform(
        self, item: Source, context: ExecutionContext
    ) -> AsyncIterator[Document]:
        """Yield one Markdown document per page, extracted or whole."""
        self.validate_input(item)
        if not item.content.strip():
            return

        soup = self._soup(item)
        markdown_div = soup.find("div", class_="markdown")
        fallback = markdown_div is None
        root = soup if fallback else markdown_div

        reveal_content(root)
        for selector in self.strip_selectors:
            for element in root.select(selector):
                element.decompose()

        await enrich_content(root, soup, item.uri, self.vision)
        self._absolutize(root, item.uri)

        content = convert(str(root), options=CONVERSION).content.strip()
        if not content:
            return

        title = soup.title.string if soup.title and soup.title.string else None
        metadata: dict[str, Any] = {FALLBACK_METADATA_KEY: fallback}
        if title:
            metadata["title"] = title.strip()
        yield self.build_document(item, content, metadata)

    def _soup(self, source: Source) -> Any:
        try:
            return BeautifulSoup(
                source.content, "html.parser", from_encoding=self.encoding
            )
        except Exception as error:  # bs4 raises whatever the underlying codec does
            raise ParseError(f"{source.uri} is not parseable HTML: {error}") from error

    @staticmethod
    def _absolutize(root: Any, base_uri: str) -> None:
        """Rewrite relative ``href`` and ``src`` values against the page's URI."""
        for attribute, selector in (("href", "a[href]"), ("src", "img[src]")):
            for element in root.select(selector):
                value = element.get(attribute)
                if isinstance(value, str):
                    element[attribute] = urljoin(base_uri, value)
