"""Sitemap-driven fetching: expand ``sitemap.xml``, then fetch what it lists.

Ported from ``src/fastrag/steps/fetching/sitemap.py`` onto the new
``Fetcher``/``SourceStep`` base. Caching is per page (one bucket per crawled
URL, via the ``targets``/``evaluate_target`` hooks
``fastrag.ingestion.steps.base`` fans out over) rather than per crawl, so a
sitemap that gains one URL still serves every previously-fetched page from
the cache — see ``docs/specification/API.md``'s "cache each page independently
to increase cache hit rate."
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from fnmatch import fnmatch
from typing import Any, Literal
from urllib.parse import urljoin, urlsplit
from xml.etree import ElementTree

from fastrag.exceptions import FetchError
from fastrag.ingestion._hashing import content_hash
from fastrag.ingestion._site import inherit_site_metadata, site_metadata, site_of
from fastrag.ingestion.cache import CacheStore
from fastrag.ingestion.models import Source
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext, run_concurrently
from fastrag.ingestion.steps.fetching.base import Fetcher
from fastrag.ingestion.steps.fetching.youtube import fetch_embeds
from fastrag.ingestion.steps.fetching.renderers import Renderer, HttpRenderer
from fastrag.ingestion.steps.fetching.web import _origin
from fastrag.ingestion.steps.fetching.http import (
    DEFAULT_USER_AGENT,
    _HttpTransport,
    _parse_http_date,
    normalize_uri,
)

#: How many nested ``<sitemapindex>`` levels to follow before giving up. Real
#: sites use one; a deeper chain is far more likely to be a loop.
MAX_INDEX_DEPTH = 3

#: ``src``/``data`` of ``<embed>``, ``<iframe>`` and ``<object>`` tags.
_EMBED_REF = re.compile(
    r"<(?:embed|iframe|object)\b[^>]*?\b(?:src|data)\s*=\s*[\"']([^\"']+)[\"']",
    re.I | re.S,
)


class SitemapCrawler(Fetcher):
    """Expand ``sitemap.xml``, then fetch what it lists.

    The right default for a website: the site itself publishes the canonical
    page list, so a rerun picks up new pages without editing any code. Nested
    ``<sitemapindex>`` documents are followed.

    .. code-block:: python

        from fastrag.ingestion.steps.fetching import SitemapCrawler

        crawler = SitemapCrawler(
            url="https://example.com/sitemap.xml",
            include=["https://example.com/docs/*"],
            cache=cache,
        )
    """

    def __init__(
        self,
        *,
        url: str,
        include: str | Sequence[str] | None = None,
        exclude: str | Sequence[str] | None = None,
        max_urls: int | None = None,
        renderer: Renderer | None = None,
        extract_pdfs: bool = True,
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        follow_redirects: bool = True,
        user_agent: str = DEFAULT_USER_AGENT,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
        cache_validation_interval: float | None = None,
    ) -> None:
        """Initialize a ``SitemapCrawler``.

        Args:
            url: The sitemap URL.
            include: A URL or sequence of URL glob patterns to keep.
                Exact URLs match normalized spellings (such as trailing
                slashes). ``None`` keeps everything.
            exclude: A URL or sequence of URL glob patterns to drop,
                applied after ``include`` with the same matching rules.
            max_urls: Stop after this many URLs. Useful while developing
                against a large site.
            renderer: Raw-content renderer; defaults to HTTP passthrough.
                Its model and extraction settings participate in each page's
                cache key; changing either selects a separate cache bucket.
                The ``<iframe>`` embeds of each page that have a route (see
                :meth:`~Fetcher.route`, e.g. ``youtube_filter`` to
                ``YouTubeFetcher()``) are fetched and yielded as their own
                ``Source`` alongside the page's own ``Source``; unrouted embeds
                are ignored.
            extract_pdfs: Whether to find PDFs embedded in each page via
                ``<embed>``, ``<iframe>`` or ``<object>`` and yield each as
                its own ``Source`` whose ``content`` is the PDF bytes,
                alongside the page's own ``Source``.
            headers: Extra request headers — part of the cache fingerprint,
                since e.g. ``Accept-Language`` can change what a URI returns.
            timeout: Per-request timeout, in seconds.
            follow_redirects: Whether to follow HTTP redirects.
            user_agent: The ``User-Agent`` header to send.
            cache: Cache backend, or ``False``/``None`` — see
                :class:`~fastrag.ingestion.steps.base.PipelineStep`.
            policy: Execution policy overriding the pipeline default.
            cache_validation_interval: How long a cached page may be served
                without revalidating it against the origin — see
                :class:`~fastrag.ingestion.steps.base.SourceStep`.
        """
        super().__init__(
            cache=cache,
            policy=policy,
            cache_validation_interval=cache_validation_interval,
        )
        self.url = url
        self.include = [include] if isinstance(include, str) else include
        self.exclude = [exclude] if isinstance(exclude, str) else exclude
        self.max_urls = max_urls
        self.renderer = renderer or HttpRenderer()
        self.extract_pdfs = extract_pdfs
        self._transport = _HttpTransport(
            headers=headers,
            timeout=timeout,
            follow_redirects=follow_redirects,
            user_agent=user_agent,
        )

    def config(self) -> dict[str, Any]:
        """This crawler's cache fingerprint: only what changes a page's bytes.

        ``url``, ``include``, ``exclude``, and ``max_urls`` decide *which*
        pages are discovered, not what a given page returns, so they are
        deliberately excluded here — otherwise raising ``max_urls`` would
        invalidate every already-cached page's bucket for no reason. Per-page
        cache ids already encode the page's own URL as their ``target``.
        """
        return {
            **self._transport.config(),
            "renderer": dict(self.renderer.config()),
            "extract_pdfs": self.extract_pdfs,
            "routes": self.routes_config(),
        }

    async def targets(self, context: ExecutionContext) -> list[str]:
        """Return the filtered page URLs the sitemap lists, normalized and deduplicated.

        Costs one request per sitemap document, and no page downloads. Called
        by the base class's per-target cache wrap before deciding which pages
        need :meth:`evaluate_target`.

        Sitemaps routinely list one page under several spellings (``/audit``
        and ``/audit/``, a stray ``#fragment``, an explicit ``:443``). Those
        all collapse to one
        :func:`~fastrag.ingestion.steps.fetching.http.normalize_uri`, and so
        to one ``Source.id``, so they are collapsed here too — into a single
        target and a single cache bucket, rather than fetched repeatedly and
        written over each other. ``include``/``exclude`` still see the
        sitemap's own spelling for glob patterns; exact URLs match normalized
        spellings too.
        """
        urls = await self._expand(self.url, depth=0, seen=set())
        kept = _unique([normalize_uri(url) for url in urls if self._keeps(url)])
        return kept if self.max_urls is None else kept[: self.max_urls]

    async def evaluate_target(
        self, target: str, context: ExecutionContext
    ) -> AsyncIterator[Source]:
        """Fetch one page URL, yielding its ``Source`` then one per routed embed."""
        for source in await self._fetch_sources(target):
            yield source

    async def evaluate(self, context: ExecutionContext) -> AsyncIterator[Source]:
        """Fetch every listed page, uncached, bounded by ``policy.concurrency``.

        Used directly when no cache is in effect — the per-target cache wrap
        (:meth:`targets`/:meth:`evaluate_target`) is what applies when one is.
        """
        targets = await self.targets(context)
        effective_policy = self.effective_policy(context)

        async def _targets() -> AsyncIterator[str]:
            for target in targets:
                yield target

        async for sources in run_concurrently(
            _targets(),
            self._fetch_sources,
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

        ``source_id`` is the cached entry's normalized id, which — unlike
        ``API.md``'s illustrative example — still carries a scheme, so it is
        directly usable as a fetchable URI here. A server that answers none
        of the three (refuses ``HEAD`` and gives no reusable validator) reads
        as modified only implicitly: the refetch-and-compare fallback below
        always produces a definite answer.
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
        if not isinstance(self.renderer, HttpRenderer) and not _is_pdf_uri(source_id):
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

    async def _fetch_sources(self, uri: str) -> list[Source]:
        """Fetch a page, then the routed embeds it contains, as separate sources."""
        routed = self.route_for(uri)
        if routed is not None:
            return await routed.fetch(uri)
        raw = await self._transport.fetch(uri)
        fetched = await self.renderer.render(raw)
        page = self._page_source(
            fetched, site_metadata(raw.content, raw.content_type, raw.uri)
        )
        sources = [page]

        # Look in the raw HTML too: a renderer may return markdown or
        # extracted text with the embeds already stripped.
        sources += await fetch_embeds(
            self,
            page.uri,
            (raw.content, raw.uri),
            (fetched.content, fetched.uri),
        )
        if self.extract_pdfs:
            pdf_urls = _unique(
                _pdf_urls(raw.content, raw.uri)
                + _pdf_urls(fetched.content, fetched.uri)
            )
            for pdf_url in pdf_urls:
                if normalize_uri(pdf_url) != page.id:
                    pdf = await self._fetch_pdf(pdf_url, page.uri)
                    if _origin(pdf.uri) == _origin(page.uri):
                        # Hosted by the page's own site: cited under its name and icon.
                        pdf = replace(
                            pdf,
                            metadata=inherit_site_metadata(
                                pdf.metadata, site_of(page.metadata)
                            ),
                        )
                    sources.append(pdf)
        return sources

    async def _fetch_pdf(self, url: str, page_uri: str) -> Source:
        fetched = await self._transport.fetch(url)
        content_type = (fetched.content_type or "").split(";")[0].strip().lower()
        if content_type in ("", "application/octet-stream"):
            content_type = "application/pdf"
        return Source(
            id=normalize_uri(fetched.uri),
            uri=fetched.uri,
            content=fetched.content,
            content_type=content_type,
            content_hash=content_hash(fetched.content),
            etag=fetched.etag,
            last_modified=fetched.last_modified,
            metadata={
                "kind": "pdf",
                "page_uri": page_uri,
                "status_code": fetched.status_code,
            },
        )

    def _page_source(self, fetched: Any, site: Mapping[str, str]) -> Source:
        return Source(
            id=normalize_uri(fetched.uri),
            uri=fetched.uri,
            content=fetched.content,
            content_type=fetched.content_type,
            content_hash=content_hash(fetched.content),
            etag=fetched.etag,
            last_modified=fetched.last_modified,
            metadata={"status_code": fetched.status_code, **site},
        )

    async def _expand(self, url: str, *, depth: int, seen: set[str]) -> list[str]:
        if depth >= MAX_INDEX_DEPTH or url in seen:
            return []
        seen.add(url)

        response = await self._transport.request("GET", url)
        root = _parse(response.content, url)
        locations = [
            element.text.strip()
            for element in root.iterfind(".//{*}loc")
            if element.text and element.text.strip()
        ]

        if not _is_index(root):
            return _unique(locations)

        nested: list[str] = []
        for location in locations:
            nested.extend(await self._expand(location, depth=depth + 1, seen=seen))
        return _unique(nested)

    def _keeps(self, url: str) -> bool:
        if self.include and not any(_matches(url, pattern) for pattern in self.include):
            return False
        return not (
            self.exclude and any(_matches(url, pattern) for pattern in self.exclude)
        )


def _matches(url: str, pattern: str) -> bool:
    """Match globs as written and exact URLs by their normalized identity."""
    if any(character in pattern for character in "*?["):
        return fnmatch(url, pattern)
    return normalize_uri(url) == normalize_uri(pattern)


def _parse(content: bytes, url: str) -> ElementTree.Element:
    try:
        return ElementTree.fromstring(content)
    except ElementTree.ParseError as error:
        raise FetchError(f"{url} is not a parseable sitemap: {error}") from error


def _is_index(root: ElementTree.Element) -> bool:
    """True for ``<sitemapindex>``, which lists sitemaps rather than pages."""
    return root.tag.rsplit("}", 1)[-1] == "sitemapindex"


def _is_pdf_uri(uri: str) -> bool:
    return urlsplit(uri).path.lower().endswith(".pdf")


def _pdf_urls(content: bytes, base_uri: str) -> list[str]:
    """Absolute URLs of every PDF referenced by an embed/iframe/object tag."""
    html = content.decode("utf-8", errors="replace")
    urls: list[str] = []
    for ref in _EMBED_REF.findall(html):
        url = urljoin(base_uri, ref.strip())
        if url.startswith(("http://", "https://")) and _is_pdf_uri(url):
            urls.append(url)
    return urls


def _unique(urls: list[str]) -> list[str]:
    """Drop duplicates, keeping the sitemap's order."""
    return list(dict.fromkeys(urls))
