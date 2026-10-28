"""Main-content extraction for arbitrary web pages, via Trafilatura.

:class:`~fastrag.ingestion.steps.parsing.html.HtmlParser` removes page
furniture by CSS selector, which works when a site marks its chrome up with
``<nav>``/``<footer>`` and fails on the many that build it out of anonymous
``div``s. :class:`~fastrag.ingestion.steps.parsing.docusaurus.DocusaurusParser`
solves that for one site generator. This parser solves it heuristically for
any page: Trafilatura scores the DOM by text density and link ratio to find
the article, so no per-site selector is needed, at the price of the usual
heuristic risk of dropping a block a selector would have kept.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Literal

from fastrag.exceptions import ConfigurationError, ParseError
from fastrag.ingestion.cache import CacheStore
from fastrag.ingestion.models import Source
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext
from fastrag.ingestion.steps.parsing.base import Parser
from fastrag.ingestion.steps.parsing.html import HTML_MEDIA_TYPES
from fastrag.models import ContentFormat, Document

try:
    from trafilatura import extract as _extract
    from trafilatura import extract_metadata as _extract_metadata
    from trafilatura import load_html as _load_html

    _TRAFILATURA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by installs without the extra
    _TRAFILATURA_AVAILABLE = False

#: The Trafilatura entry points, or unusable placeholders when the optional
#: "trafilatura" extra is missing; :meth:`TrafilaturaParser.__init__` refuses
#: to run until they exist.
extract: Any = _extract if _TRAFILATURA_AVAILABLE else None
extract_metadata: Any = _extract_metadata if _TRAFILATURA_AVAILABLE else None
load_html: Any = _load_html if _TRAFILATURA_AVAILABLE else None

MISSING_EXTRA = (
    'TrafilaturaParser needs the "trafilatura" extra: '
    'pip install "fastrag-ai[trafilatura]"'
)

#: Page metadata Trafilatura extracts that is carried into
#: ``Document.metadata`` when present, rather than into the embedded text.
METADATA_FIELDS: tuple[str, ...] = (
    "title",
    "author",
    "date",
    "description",
    "sitename",
)


class TrafilaturaParser(Parser):
    """Extract a web page's main content as Markdown, with no per-site selectors.

    Trafilatura locates the article heuristically and discards navigation,
    sidebars, footers, and related-link blocks around it. Relative links are
    resolved against the source's URI, and the page's title, author, date,
    description, and site name travel in ``Document.metadata``.

    .. code-block:: python

        from fastrag.ingestion.steps.parsing import (
            ParserRouter,
            TrafilaturaParser,
            html_filter,
        )

        parser = ParserRouter(cache=cache).route(
            html_filter, TrafilaturaParser(favor_precision=True)
        )
    """

    content_format = ContentFormat.MARKDOWN

    accepted_media_types = frozenset(HTML_MEDIA_TYPES)

    def __init__(
        self,
        *,
        favor_precision: bool = False,
        favor_recall: bool = False,
        include_tables: bool = True,
        include_links: bool = True,
        include_images: bool = False,
        include_comments: bool = False,
        prune_xpath: Sequence[str] = (),
        encoding: str | None = None,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
    ) -> None:
        """Initialize ``TrafilaturaParser``.

        Args:
            favor_precision: Prefer dropping doubtful blocks, for less noise
                at the cost of occasionally losing content.
            favor_recall: Prefer keeping doubtful blocks, for more content at
                the cost of occasionally keeping boilerplate.
            include_tables: Keep tables, rendered as Markdown tables.
            include_links: Keep links as Markdown links rather than bare
                anchor text.
            include_images: Keep images as Markdown image references.
            include_comments: Keep a page's reader-comment section, appended
                after the article.
            prune_xpath: XPath expressions for elements removed before
                extraction, for furniture the heuristics keep on a
                particular site.
            encoding: Force an encoding. ``None`` lets Trafilatura detect it
                from the document's own declaration and bytes.
            cache: Cache backend, or ``False``/``None`` — see
                :class:`~fastrag.ingestion.steps.base.PipelineStep`.
            policy: Execution policy overriding the pipeline default.

        Raises:
            fastrag.exceptions.ConfigurationError: The optional "trafilatura"
                extra is not installed, or both ``favor_precision`` and
                ``favor_recall`` are set.
        """
        if extract is None:
            raise ConfigurationError(MISSING_EXTRA)
        if favor_precision and favor_recall:
            raise ConfigurationError(
                "TrafilaturaParser: favor_precision and favor_recall are "
                "mutually exclusive"
            )
        super().__init__(cache=cache, policy=policy)
        self.favor_precision = favor_precision
        self.favor_recall = favor_recall
        self.include_tables = include_tables
        self.include_links = include_links
        self.include_images = include_images
        self.include_comments = include_comments
        self.prune_xpath = tuple(prune_xpath)
        self.encoding = encoding

    def config(self) -> Mapping[str, Any]:
        """This parser's cache-identity configuration.

        Every option changes what a given page's bytes parse into, so all of
        them are part of the fingerprint.
        """
        return {
            "favor_precision": self.favor_precision,
            "favor_recall": self.favor_recall,
            "include_tables": self.include_tables,
            "include_links": self.include_links,
            "include_images": self.include_images,
            "include_comments": self.include_comments,
            "prune_xpath": list(self.prune_xpath),
            "encoding": self.encoding,
        }

    async def transform(
        self, item: Source, context: ExecutionContext
    ) -> AsyncIterator[Document]:
        """Yield one Markdown document per page, or nothing if it has no content."""
        self.validate_input(item)
        if not item.content.strip():
            return

        # Extraction is CPU-bound lxml work; keep it off the event loop.
        extracted = await asyncio.to_thread(self._extract, item)
        if extracted is None:
            return
        content, metadata = extracted
        yield self.build_document(item, content, metadata)

    def _extract(self, source: Source) -> tuple[str, dict[str, Any]] | None:
        try:
            tree = load_html(
                source.content.decode(self.encoding)
                if self.encoding
                else source.content
            )
        except Exception as error:  # a forced codec that does not fit the bytes
            raise ParseError(f"{source.uri} is not parseable HTML: {error}") from error
        if tree is None:
            raise ParseError(f"{source.uri} is not parseable HTML")

        try:
            # Trafilatura resolves relative links against the host root, not
            # the page, so a sibling "page.html" would lose its directory.
            tree.make_links_absolute(source.uri, resolve_base_href=True)
            # Metadata first: extraction prunes the tree it is given.
            page = extract_metadata(tree, default_url=source.uri)
            content = extract(
                tree,
                url=source.uri,
                output_format="markdown",
                favor_precision=self.favor_precision,
                favor_recall=self.favor_recall,
                include_tables=self.include_tables,
                include_links=self.include_links,
                include_images=self.include_images,
                include_comments=self.include_comments,
                prune_xpath=list(self.prune_xpath) or None,
            )
        except Exception as error:
            raise ParseError(
                f"Trafilatura extraction failed for {source.uri}: {error}"
            ) from error

        content = (content or "").strip()
        if not content:
            return None

        metadata = {
            field: value
            for field in METADATA_FIELDS
            if (value := getattr(page, field, None))
        }
        return content, metadata
