"""Link-following crawling: start from one or more URLs, follow what they link to.

Unlike :class:`~fastrag.ingestion.steps.fetching.sitemap.SitemapCrawler`
(which trusts a sitemap to enumerate every page up front), ``WebCrawler``
discovers pages by fetching a start URL, extracting its links, and following
them within the configured bounds — the right tool for a site with no
sitemap, or where only a subtree reachable from one URL matters.

Caching is bucketed **per start URL**, not per page: ``targets()`` returns
the (normalized) start URLs, and ``evaluate_target()`` crawls the whole
subtree reachable from one of them, exactly
``docs/specification/API.md``'s "Example 1"
(``source_cache_id = "webcrawler:1a2b3c4" -> [Source(a.com/1), Source(a.com/2), ...]``).
This means a start URL whose linked pages changed at all is revalidated (and,
if stale, recrawled) as one unit — the granularity API.md's own example
describes, not a per-page bucket the way ``SitemapCrawler`` uses. See
``.agent/results/013-web-crawler.md`` for why this differs from
``SitemapCrawler``'s per-page bucketing despite both opting into the same
``targets``/``evaluate_target`` hooks.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from fnmatch import fnmatch
from typing import Any, Literal
from urllib.parse import urljoin, urlsplit

from fastrag.exceptions import ConfigurationError
from fastrag.ingestion._hashing import content_hash
from fastrag.ingestion._site import (
    SITE_METADATA_KEYS,
    inherit_site_metadata,
    site_metadata,
    site_of,
)
from fastrag.ingestion.cache import CacheStore
from fastrag.ingestion.models import Source
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext, run_concurrently
from fastrag.ingestion.steps.fetching.base import Fetcher
from fastrag.ingestion.steps.fetching.renderers import Renderer, HttpRenderer
from fastrag.ingestion.steps.fetching.youtube import fetch_embeds
from fastrag.ingestion.steps.fetching.http import (
    DEFAULT_USER_AGENT,
    _HttpTransport,
    _parse_http_date,
    normalize_uri,
)

try:
    from bs4 import BeautifulSoup as _BeautifulSoup

    _BS4_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by installs without the extra
    _BS4_AVAILABLE = False

#: Link extraction needs a real HTML parser; a placeholder when the optional
#: "html" extra is missing. :meth:`WebCrawler.__init__` refuses to run until
#: it exists — see the ``_X_AVAILABLE``/``Any``-placeholder pattern in
#: ``steps/parsing/html.py``.
BeautifulSoup: Any = _BeautifulSoup if _BS4_AVAILABLE else None

MISSING_EXTRA = 'WebCrawler needs the "html" extra: pip install "fastrag-ai[html]"'

#: Media types worth parsing for outgoing links. Anything else (a PDF, an
#: image) is still fetched and yielded as a ``Source``, just never scanned
#: for links.
_HTML_MEDIA_TYPES = frozenset({"text/html", "application/xhtml+xml"})


class WebCrawler(Fetcher):
    """Follow links from one or more start URLs, within configured bounds.

    Accepts either a single ``url=`` (the form
    ``examples/08_sources.py``/``09_filters.py`` use) or several
    ``start_urls=`` (``docs/specification/API.md``'s multi-start form) —
    exactly one of the two must be given. Both are stored the same way
    internally (``self.start_urls``), so the rest of this class does not
    distinguish them.

    .. code-block:: python

        from fastrag.ingestion.steps.fetching import WebCrawler

        crawler = WebCrawler(url="https://docs.pontus-x.eu/", max_urls=200)
        crawler = WebCrawler(start_urls=["a.com", "https://b.com/"], max_depth=3)
    """

    #: A start URL's crawl bounds, retries and times out each page itself, so
    #: the cache wrap streams it rather than running it as one timed item.
    streams_targets = True

    def __init__(
        self,
        *,
        url: str | None = None,
        start_urls: Sequence[str] | None = None,
        max_urls: int | None = None,
        max_depth: int | None = None,
        include: str | Sequence[str] | None = None,
        exclude: str | Sequence[str] | None = None,
        same_origin: bool = True,
        renderer: Renderer | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        follow_redirects: bool = True,
        user_agent: str = DEFAULT_USER_AGENT,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
        cache_validation_interval: float | None = None,
    ) -> None:
        """Initialize a ``WebCrawler``.

        Args:
            url: A single start URL. Mutually exclusive with ``start_urls``.
            start_urls: Several start URLs, crawled independently (and cached
                independently — see the module docstring). Mutually
                exclusive with ``url``.
            max_urls: Stop each start URL's crawl after this many pages.
                Useful while developing against a large site.
            max_depth: Maximum number of link hops to follow from a start
                URL. ``0`` fetches only the start URL itself. ``None`` follows links indefinitely.
            include: A URL or URL glob patterns a discovered link must match to
                be followed, and a page must match to be yielded (the start
                URL is crawled for links but yielded only if it matches). ``None`` keeps everything (subject to the other
                bounds). Same glob style as
                :class:`~fastrag.ingestion.steps.fetching.sitemap.SitemapCrawler`.
            exclude: URL glob patterns a discovered link must not match,
                applied after ``include``.
            same_origin: Whether a discovered link must share its start
                URL's scheme, host, and port to be followed. ``True`` by
                default — a crawl does not wander off its own site unless
                asked to.
            renderer: Raw-content renderer; defaults to HTTP passthrough.
                The ``<iframe>`` embeds of each page that have a route (see
                :meth:`~Fetcher.route`, e.g. ``youtube_filter`` to
                ``YouTubeFetcher()``) are fetched and yielded as their own
                ``Source``; unrouted embeds are ignored. Links to routed
                URLs are fetched by their route instead of crawled.
            headers: Extra request headers — part of the cache fingerprint,
                since e.g. ``Accept-Language`` can change what a URI returns.
            timeout: Per-request timeout, in seconds.
            follow_redirects: Whether to follow HTTP redirects.
            user_agent: The ``User-Agent`` header to send.
            cache: Cache backend, or ``False``/``None`` — see
                :class:`~fastrag.ingestion.steps.base.PipelineStep`.
            policy: Execution policy overriding the pipeline default —
                ``policy.concurrency`` bounds how many pages are fetched in
                parallel, per crawl level.
            cache_validation_interval: How long a cached start URL's bucket
                may be served without revalidating it against the origin —
                see :class:`~fastrag.ingestion.steps.base.SourceStep`.

        Raises:
            fastrag.exceptions.ConfigurationError: Neither or both of
                ``url``/``start_urls`` were given, or the optional "html"
                extra is not installed.
        """
        if (url is None) == (start_urls is None):
            raise ConfigurationError(
                "WebCrawler needs exactly one of url= or start_urls="
            )
        if BeautifulSoup is None:
            raise ConfigurationError(MISSING_EXTRA)
        super().__init__(
            cache=cache,
            policy=policy,
            cache_validation_interval=cache_validation_interval,
        )
        self.start_urls: list[str]
        if start_urls is not None:
            self.start_urls = list(start_urls)
        else:
            assert url is not None  # guaranteed by the exactly-one check above
            self.start_urls = [url]
        self.max_urls = max_urls
        self.max_depth = max_depth
        self.include = [include] if isinstance(include, str) else include
        self.exclude = [exclude] if isinstance(exclude, str) else exclude
        self.same_origin = same_origin
        self.renderer = renderer or HttpRenderer()
        self._transport = _HttpTransport(
            headers=headers,
            timeout=timeout,
            follow_redirects=follow_redirects,
            user_agent=user_agent,
        )

    def config(self) -> dict[str, Any]:
        """This crawler's cache fingerprint: only what changes a page's bytes.

        ``start_urls``, ``max_urls``, ``max_depth``, ``include``,
        ``exclude``, and ``same_origin`` decide *which* pages are discovered,
        not what a given page returns, so they are deliberately excluded
        here — the same principle
        :meth:`~fastrag.ingestion.steps.fetching.sitemap.SitemapCrawler.config`
        follows, for the same reason. Per-start-URL cache ids already encode
        the start URL itself as their ``target``.
        """
        return {
            **self._transport.config(),
            "renderer": dict(self.renderer.config()),
            "routes": self.routes_config(),
        }

    async def targets(self, context: ExecutionContext) -> list[str]:
        """Return the normalized start URLs — one independent cache bucket each."""
        return _unique([normalize_uri(u) for u in self.start_urls])

    async def evaluate_target(
        self, target: str, context: ExecutionContext
    ) -> AsyncIterator[Source]:
        """Crawl the subtree reachable from one start URL, within bounds.

        Fetches breadth-first, one level of link depth at a time, each
        level's pages fetched concurrently (bounded by
        ``policy.concurrency``) via :func:`~fastrag.ingestion.steps.base.run_concurrently` — so a page
        that fails to fetch after retries emits an attributed ``ItemFailed``
        and is skipped (``policy.on_error == "skip"``) rather than aborting
        the whole crawl. A ``visited`` set, seeded with ``target`` itself,
        guarantees every URL is fetched at most once per call even when
        several pages link to it, and pages are deduplicated again by the
        ``Source.id`` they actually resolve to, so two URLs that redirect to
        one page are yielded (and cached) once.
        """
        effective_policy = self.effective_policy(context)
        origin = _origin(target)
        # A page is identified by (url, request language): a server that picks
        # the locale from ``Accept-Language`` serves different pages, or
        # redirects to another one, for the same URL.
        visited: set[tuple[str, str | None]] = {(target, None)}
        emitted: set[str] = set()
        #: depth -> pending (url, request language). A language-switcher link
        #: is queued at its page's own depth: another translation of a page is
        #: not "one hop deeper" into the site.
        levels: dict[int, list[tuple[str, str | None]]] = {0: [(target, None)]}
        fetched_count = 0
        #: origin -> the site name and icon its pages declared, first seen
        #: wins; lent to that origin's sources declaring none (e.g. a PDF).
        sites: dict[str, dict[str, str]] = {}

        while levels:
            depth = min(levels)
            frontier = levels.pop(depth)
            if self.max_urls is not None:
                budget = self.max_urls - fetched_count
                if budget <= 0:
                    break
                frontier = frontier[:budget]

            extract_links = self.max_depth is None or depth < self.max_depth

            async def _frontier_stream(
                pages: list[tuple[str, str | None]] = frontier,
            ) -> AsyncIterator[tuple[str, str | None]]:
                for page in pages:
                    yield page

            discovered: list[tuple[str | None, str | None, list[_Link]]] = []
            async for sources, links, request_lang, page_lang in run_concurrently(
                _frontier_stream(),
                lambda page: self._fetch_page(
                    page[0], language=page[1], extract_links=extract_links
                ),
                context=context,
                policy=effective_policy,
                item_id=lambda page: page[0],
                step=self.name,
                event_name=self.event_name,
            ):
                fetched_count += 1
                source, *extras = sources
                # The resolved id may differ from the URL that was requested
                # (a redirect); marking it visited stops a later level from
                # queueing the redirect target under its own name.
                visited.add((source.id, request_lang))
                if page_lang is not None:
                    visited.add((source.id, page_lang))
                if source.id in emitted:
                    # Two frontier URLs resolved to one page: a redirect
                    # (``/a`` -> ``/b``), or two spellings that normalize
                    # alike. Both carry the same ``Source.id``, and a whole
                    # crawl is written to the cache as one batch, where a
                    # repeated id is a hard error ("ON CONFLICT DO UPDATE
                    # command cannot affect row a second time"). The repeat
                    # is the same page, links included, so dropping it loses
                    # nothing.
                    continue
                emitted.add(source.id)
                yield _with_site(source, sites)
                for extra in extras:
                    if extra.id not in emitted:
                        emitted.add(extra.id)
                        yield _with_site(extra, sites)
                discovered.append((request_lang, page_lang, links))

            if not extract_links:
                continue

            for request_lang, page_lang, links in discovered:
                for link, link_lang in links:
                    switches = link_lang is not None and link_lang != page_lang
                    lang = link_lang if switches else request_lang
                    if (link, lang) in visited or not self._keeps(link, origin):
                        continue
                    visited.add((link, lang))
                    levels.setdefault(depth if switches else depth + 1, []).append(
                        (link, lang)
                    )

    async def evaluate(self, context: ExecutionContext) -> AsyncIterator[Source]:
        """Crawl every start URL, uncached, bounded by ``policy.concurrency``.

        Used directly when no cache is in effect — the per-target cache wrap
        (:meth:`targets`/:meth:`evaluate_target`) is what applies when one
        is. Start URLs are themselves fetched concurrently, mirroring
        :meth:`evaluate_target`'s own per-level concurrency.
        """
        targets = await self.targets(context)
        effective_policy = self.effective_policy(context)

        async def _collect(target: str) -> list[Source]:
            return [source async for source in self.evaluate_target(target, context)]

        async def _targets() -> AsyncIterator[str]:
            for target in targets:
                yield target

        async for sources in run_concurrently(
            _targets(),
            _collect,
            context=context,
            policy=effective_policy,
            item_id=str,
            step=self.name,
            event_name=self.event_name,
        ):
            for source in sources:
                yield source

    async def is_source_modified(
        self,
        source_id: str,
        cached_etag: str | None,
        cached_last_modified: datetime | None,
        cached_content_hash: str,
    ) -> bool:
        """Conditional freshness check: ETag, then Last-Modified, then content hash.

        Identical ladder to
        :meth:`~fastrag.ingestion.steps.fetching.sitemap.SitemapCrawler.is_source_modified`,
        applied here to one page within a start URL's cached bucket — the
        base class's ``_any_modified`` calls this once per cached page, and
        treats the whole bucket as stale (recrawling the entire start URL via
        :meth:`evaluate_target`) if any one page reports modified.
        """
        routed = self.route_for(source_id)
        if routed is not None:
            # Duck-typed: a routed fetcher without a validator always refetches.
            validator = getattr(routed, "is_source_modified", None)
            if validator is None:
                return True
            return await validator(
                source_id, cached_etag, cached_last_modified, cached_content_hash
            )
        if not isinstance(self.renderer, HttpRenderer):
            fetched = await self.renderer.render(await self._transport.fetch(source_id))
            return content_hash(fetched.content) != cached_content_hash
        validators = await self._transport.validators(source_id)
        if validators is not None:
            etag = validators.get("etag")
            if etag is not None and cached_etag is not None:
                return etag != cached_etag

            last_modified = validators.get("last_modified")
            if last_modified is not None and cached_last_modified is not None:
                parsed = _parse_http_date(last_modified)
                if parsed is not None:
                    return parsed != cached_last_modified

        fetched = await self._transport.fetch(source_id)
        return content_hash(fetched.content) != cached_content_hash

    async def aclose(self) -> None:
        """Release the shared HTTP client and renderer resources. Idempotent."""
        try:
            await self._transport.aclose()
        finally:
            await self.renderer.aclose()

    async def _fetch_page(
        self, uri: str, *, language: str | None, extract_links: bool
    ) -> tuple[list[Source], list[_Link], str | None, str | None]:
        """Fetch one page; return its sources, links, request and page language."""
        routed = self.route_for(uri)
        if routed is not None:
            return await routed.fetch(uri), [], language, None
        headers = {"accept-language": language} if language else None
        raw = await self._transport.fetch(uri, headers=headers)
        fetched = await self.renderer.render(raw)
        source = Source(
            id=normalize_uri(fetched.uri),
            uri=fetched.uri,
            content=fetched.content,
            content_type=fetched.content_type,
            content_hash=content_hash(fetched.content),
            etag=fetched.etag,
            last_modified=fetched.last_modified,
            metadata={
                "status_code": fetched.status_code,
                **site_metadata(raw.content, raw.content_type, raw.uri),
            },
        )
        links: list[_Link] = []
        page_lang: str | None = None
        if fetched.content_type in _HTML_MEDIA_TYPES:
            page_lang = _page_language(raw.content)
            if extract_links:
                # The raw HTML too: a renderer drops hidden nodes, and with
                # them links in collapsed menus (e.g. a language-switcher
                # dropdown).
                links = _unique(
                    self._extract_links(fetched.content, fetched.uri)
                    + self._extract_links(raw.content, raw.uri)
                )
        sources = [source]
        # The raw HTML too: a renderer may strip the embeds.
        sources += await fetch_embeds(
            self, source.uri, (raw.content, raw.uri), (fetched.content, fetched.uri)
        )
        return sources, links, language, page_lang

    def _extract_links(self, content: bytes, base_uri: str) -> list[_Link]:
        """Return ``(url, language)`` for each link; ``language`` is its ``lang``/``hreflang``."""
        soup = BeautifulSoup(content, "html.parser")
        links: list[_Link] = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"].strip()
            if not href or href.startswith("#"):
                continue
            absolute = urljoin(base_uri, href)
            if urlsplit(absolute).scheme not in ("http", "https"):
                continue
            language = _primary_language(anchor.get("hreflang") or anchor.get("lang"))
            links.append((normalize_uri(absolute), language))
        return _unique(links)

    def keeps_source(self, source: Source) -> bool:
        """Whether ``source`` passes ``include``/``exclude`` (applied to cached output too)."""
        return self._matches_filters(source.id)

    def _matches_filters(self, url: str) -> bool:
        if self.include and not any(fnmatch(url, pattern) for pattern in self.include):
            return False
        return not (
            self.exclude and any(fnmatch(url, pattern) for pattern in self.exclude)
        )

    def _keeps(self, url: str, origin: str) -> bool:
        if self.same_origin and _origin(url) != origin:
            return False
        return self._matches_filters(url)


def _origin(url: str) -> str:
    """Return ``scheme://host:port`` for ``url``, the unit ``same_origin`` compares."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def _with_site(source: Source, sites: dict[str, dict[str, str]]) -> Source:
    """Record ``source``'s site name and icon for its origin, and fill in any it lacks."""
    site = sites.setdefault(_origin(source.uri), {})
    for key, value in site_of(source.metadata).items():
        site.setdefault(key, value)
    if not site or all(key in source.metadata for key in SITE_METADATA_KEYS):
        return source
    return replace(source, metadata=inherit_site_metadata(source.metadata, site))


def _unique[T](items: list[T]) -> list[T]:
    """Drop duplicates, keeping first-seen order."""
    return list(dict.fromkeys(items))


#: A discovered link: its normalized URL and the language it declares (if any).
_Link = tuple[str, str | None]


def _primary_language(tag: str | None) -> str | None:
    """``"ca-ES"`` -> ``"ca"``: servers negotiate on the bare language subtag."""
    if not tag or not tag.strip():
        return None
    return tag.strip().split("-")[0].lower()


def _page_language(content: bytes) -> str | None:
    """The primary language an HTML page declares on ``<html lang>``."""
    html = BeautifulSoup(content, "html.parser").find("html")
    return _primary_language(html.get("lang")) if html else None
