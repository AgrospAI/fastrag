r"""Single-URL fetching: exactly one file at exactly one URL.

Where :class:`~fastrag.ingestion.steps.fetching.sitemap.SitemapCrawler`
expands a sitemap and :class:`~fastrag.ingestion.steps.fetching.web.WebCrawler`
follows links — both discovering *many* pages — ``FileSource`` expresses the
simplest possible intent: "ingest this one file, at this one URL." The right
tool for a single published PDF, spreadsheet, or report, and for anything
that would otherwise need a one-entry sitemap. Aggregating several
``FileSource``\\ s (or mixing one into a crawl) is
:class:`~fastrag.ingestion.steps.fetching.sources.Sources`'s job, not this
class's — it deliberately takes one ``url``, not a list.

Caching is one bucket for the URL, via the same ``targets``/``evaluate_target``
hooks :class:`~fastrag.ingestion.steps.fetching.sitemap.SitemapCrawler` uses
for its per-page buckets — here there is only ever one target, but the shape
(and so ``cache_validation_interval``'s meaning) is identical.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from datetime import datetime
from typing import Any, Literal

from fastrag.ingestion._hashing import content_hash
from fastrag.ingestion._site import site_metadata
from fastrag.ingestion.cache import CacheStore
from fastrag.ingestion.models import Source
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext
from fastrag.ingestion.steps.fetching.base import Fetcher
from fastrag.ingestion.steps.fetching.renderers import Renderer, HttpRenderer
from fastrag.ingestion.steps.fetching.http import (
    DEFAULT_USER_AGENT,
    _HttpTransport,
    _parse_http_date,
    normalize_uri,
)


class FileSource(Fetcher):
    """Fetch exactly one URL, yielding exactly one ``Source``.

    .. code-block:: python

        from fastrag.ingestion.steps.fetching import FileSource

        source = FileSource(
            url="https://example.com/report.xlsx",
            cache=cache,
        )
    """

    def __init__(
        self,
        *,
        url: str,
        renderer: Renderer | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        follow_redirects: bool = True,
        user_agent: str = DEFAULT_USER_AGENT,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
        cache_validation_interval: float | None = None,
    ) -> None:
        """Initialize a ``FileSource``.

        Args:
            url: The URL to fetch. Exactly one — aggregate several with
                :class:`~fastrag.ingestion.steps.fetching.sources.Sources`.
            renderer: Raw-content renderer; defaults to HTTP passthrough.
            headers: Extra request headers — part of the cache fingerprint,
                since e.g. ``Accept-Language`` can change what a URI returns.
            timeout: Per-request timeout, in seconds.
            follow_redirects: Whether to follow HTTP redirects.
            user_agent: The ``User-Agent`` header to send.
            cache: Cache backend, or ``False``/``None`` — see
                :class:`~fastrag.ingestion.steps.base.PipelineStep`.
            policy: Execution policy overriding the pipeline default.
            cache_validation_interval: How long the cached file may be served
                without revalidating it against the origin — see
                :class:`~fastrag.ingestion.steps.base.SourceStep`.
        """
        super().__init__(
            cache=cache,
            policy=policy,
            cache_validation_interval=cache_validation_interval,
        )
        self.url = url
        self.renderer = renderer or HttpRenderer()
        self._transport = _HttpTransport(
            headers=headers,
            timeout=timeout,
            follow_redirects=follow_redirects,
            user_agent=user_agent,
        )

    def config(self) -> dict[str, Any]:
        """This fetcher's cache fingerprint: only what changes the fetched bytes.

        ``url`` is deliberately excluded, exactly as
        :meth:`~fastrag.ingestion.steps.fetching.sitemap.SitemapCrawler.config`
        excludes the pages it discovers — the URL is already encoded in the
        per-target cache id (see :meth:`targets`), so including it here too
        would only make the fingerprint redundant, not more correct.
        """
        return {**self._transport.config(), "renderer": dict(self.renderer.config())}

    async def targets(self, context: ExecutionContext) -> list[str]:
        """Return the one normalized URL this fetcher targets.

        Called by the base class's per-target cache wrap before deciding
        whether :meth:`evaluate_target` needs to run.
        """
        return [normalize_uri(self.url)]

    async def evaluate_target(
        self, target: str, context: ExecutionContext
    ) -> AsyncIterator[Source]:
        """Fetch the URL, yielding the single ``Source`` it produces."""
        yield await self._fetch_source(target)

    async def evaluate(self, context: ExecutionContext) -> AsyncIterator[Source]:
        """Fetch the URL, uncached.

        Used directly when no cache is in effect — the per-target cache wrap
        (:meth:`targets`/:meth:`evaluate_target`) is what applies when one is.
        """
        yield await self._fetch_source(normalize_uri(self.url))

    async def is_source_modified(
        self,
        source_id: str,
        cached_etag: str | None,
        cached_last_modified: datetime | None,
        cached_content_hash: str,
    ) -> bool:
        """Conditional freshness check: ETag, then Last-Modified, then content hash.

        Identical ladder to
        :meth:`~fastrag.ingestion.steps.fetching.sitemap.SitemapCrawler.is_source_modified`:
        ``source_id`` is the cached entry's normalized id, still carrying a
        scheme, so it is directly usable as a fetchable URI here. A server
        that answers none of the three (refuses ``HEAD`` and gives no
        reusable validator) reads as modified only implicitly: the
        refetch-and-compare fallback below always produces a definite
        answer.
        """
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
        """Release the shared HTTP client. Idempotent."""
        await self._transport.aclose()

    async def _fetch_source(self, uri: str) -> Source:
        raw = await self._transport.fetch(uri)
        fetched = await self.renderer.render(raw)
        return Source(
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
