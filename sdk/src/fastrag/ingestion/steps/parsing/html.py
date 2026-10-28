"""HTML to Markdown with hidden panels, SVG labels and optional vision descriptions."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Literal
from urllib.parse import urljoin

from fastrag.ingestion.steps.parsing.vision import OpenAIVisionParser
from fastrag.ingestion.steps.parsing._content import reveal_content, enrich_content

from fastrag.exceptions import ConfigurationError, ParseError
from fastrag.ingestion.cache import CacheStore
from fastrag.ingestion.filters import MediaTypeFilter
from fastrag.ingestion.models import Source
from fastrag.models import ContentFormat, Document
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext
from fastrag.ingestion.steps.parsing.base import Parser

try:
    from bs4 import BeautifulSoup as _BeautifulSoup
    from html_to_markdown import ConversionOptions
    from html_to_markdown import convert as _convert

    _BS4_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by installs without the extra
    _BS4_AVAILABLE = False

#: The parser dependencies, or unusable placeholders when the optional "html"
#: extra is missing; :meth:`HtmlParser.__init__` refuses to run until they
#: exist.
BeautifulSoup: Any = _BeautifulSoup if _BS4_AVAILABLE else None
convert: Any = _convert if _BS4_AVAILABLE else None
#: The converter prepends a YAML block of every ``<meta>`` tag by default.
#: That is page metadata, not page text: embedding it puts the same
#: generator string and locale in front of every chunk 0 on the site, so the
#: title travels in ``Document.metadata`` instead.
CONVERSION: Any = ConversionOptions(extract_metadata=False) if _BS4_AVAILABLE else None

#: Repeated page furniture removed before content extraction. Meaningful SVGs
#: and images are handled separately by the shared content preservation code.
DEFAULT_STRIP_SELECTORS: tuple[str, ...] = (
    "script",
    "style",
    "noscript",
    "nav",
    "header",
    "footer",
    "aside",
)

MISSING_EXTRA = 'HtmlParser needs the "html" extra: pip install "fastrag-ai[html]"'

#: The media types this parser understands.
HTML_MEDIA_TYPES: frozenset[str] = frozenset({"text/html", "application/xhtml+xml"})

#: The filter a :class:`~fastrag.ingestion.steps.parsing.router.ParserRouter`
#: routes HTML sources with::
#:
#:     parser = ParserRouter(cache=cache).route(html_filter, HtmlParser())
html_filter = MediaTypeFilter(include_types=list(HTML_MEDIA_TYPES))


class HtmlParser(Parser):
    """Convert HTML to Markdown, dropping the page furniture first.

    Markdown rather than stripped text, because the heading levels, lists,
    and tables it preserves are structure worth keeping. Relative links are
    resolved against the source's URI, so a chunk quoted back to a user
    carries links that still work.

    .. code-block:: python

        from fastrag.ingestion.steps.parsing import HtmlParser

        parser = HtmlParser(strip_selectors=("script", "style", ".cookie-banner"))
    """

    content_format = ContentFormat.MARKDOWN

    accepted_media_types = frozenset(HTML_MEDIA_TYPES)

    def __init__(
        self,
        *,
        strip_selectors: Sequence[str] = DEFAULT_STRIP_SELECTORS,
        encoding: str | None = None,
        vision: OpenAIVisionParser | None = None,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
    ) -> None:
        """Initialize ``HtmlParser``.

        Args:
            strip_selectors: CSS selectors removed before conversion. Pass
                your own to keep a site's ``<aside>``, or to drop a cookie
                banner this default knows nothing about.
            encoding: Force an encoding. ``None`` lets the document's own
                declaration win, which is right far more often than a guess
                from outside it.
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
        parse into, so both are part of the fingerprint.
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
        """Yield one Markdown document per HTML page, or nothing if it is blank."""
        self.validate_input(item)
        if not item.content.strip():
            return

        soup = self._soup(item)

        reveal_content(soup)
        for selector in self.strip_selectors:
            for element in soup.select(selector):
                element.decompose()

        await enrich_content(soup, soup, item.uri, self.vision)
        self._absolutize(soup, item.uri)

        content = convert(str(soup), options=CONVERSION).content.strip()
        if not content:
            return

        title = soup.title.string if soup.title and soup.title.string else None
        yield self.build_document(
            item, content, {"title": title.strip()} if title else None
        )

    def _soup(self, source: Source) -> Any:
        try:
            return BeautifulSoup(
                source.content, "html.parser", from_encoding=self.encoding
            )
        except Exception as error:  # bs4 raises whatever the underlying codec does
            raise ParseError(f"{source.uri} is not parseable HTML: {error}") from error

    @staticmethod
    def _absolutize(soup: Any, base_uri: str) -> None:
        """Rewrite relative ``href`` and ``src`` values against the page's URI."""
        for attribute, selector in (("href", "a[href]"), ("src", "img[src]")):
            for element in soup.select(selector):
                value = element.get(attribute)
                if isinstance(value, str):
                    element[attribute] = urljoin(base_uri, value)
