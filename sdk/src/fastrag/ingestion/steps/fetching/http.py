"""Shared HTTP transport for the fetching steps that speak HTTP.

Non-public: ``_HttpTransport`` and its helpers are plumbing shared by
concrete fetchers (:class:`~fastrag.ingestion.steps.fetching.sitemap.SitemapCrawler`
and any future HTTP-speaking ``Fetcher``), not part of this package's public
surface. Ported from ``src/fastrag/steps/fetching/http.py`` — connection
reuse, user agent, timeouts, redirect following, and transient-status
classification are unchanged; only the shape around it differs, since the
new ``Fetcher`` is a plain ``SourceStep`` rather than a fingerprinted
dataclass, so this transport is held as an attribute instead of mixed in via
inheritance.
"""

from __future__ import annotations

import mimetypes
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit

import httpx

from fastrag.exceptions import FetchError, TransientFetchError

DEFAULT_USER_AGENT = "fastrag/0.1 (+https://github.com/AgrospAI/fastrag)"

#: Status codes worth retrying: rate limiting, and anything the server calls
#: its own fault.
_TRANSIENT_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

#: Ports implied by a scheme, so ``http://a.com:80/`` and ``http://a.com/``
#: normalize to the same id.
_DEFAULT_PORTS = {"http": 80, "https": 443}


@dataclass(frozen=True, slots=True)
class FetchedPage:
    """The raw HTTP response for one fetched URI, before it becomes a ``Source``."""

    uri: str
    """The URI actually fetched, after redirects."""

    content: bytes
    content_type: str
    etag: str | None
    last_modified: datetime | None
    status_code: int


class _HttpTransport:
    """Lazily created, reused ``httpx.AsyncClient`` plus HTTP-specific logic.

    Holds one client so that fetching hundreds of pages reuses connections
    instead of completing a TLS handshake per page. Configuration here
    (headers, timeout, redirects, user agent) is exactly what can change what
    bytes a given URI returns — the part of a fetcher's configuration that
    belongs in its cache fingerprint (see :meth:`config`); *which* URIs get
    fetched (a sitemap URL, an include/exclude filter, a URL cap) does not,
    and stays out of this class.
    """

    def __init__(
        self,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        follow_redirects: bool = True,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self.headers = headers
        self.timeout = timeout
        self.follow_redirects = follow_redirects
        self.user_agent = user_agent
        self._client: httpx.AsyncClient | None = None

    def config(self) -> dict[str, Any]:
        """The subset of this transport's configuration that changes fetched bytes."""
        return {
            "headers": dict(self.headers) if self.headers else None,
            "timeout": self.timeout,
            "follow_redirects": self.follow_redirects,
            "user_agent": self.user_agent,
        }

    async def fetch(
        self, uri: str, *, headers: Mapping[str, str] | None = None
    ) -> FetchedPage:
        """``GET`` one URI and return its content plus validators.

        ``headers`` are sent for this request only, on top of the client's own.
        """
        response = await self.request("GET", uri, headers=headers)
        return FetchedPage(
            uri=str(response.url),
            content=response.content,
            content_type=_content_type(response, uri),
            etag=response.headers.get("etag"),
            last_modified=_parse_http_date(response.headers.get("last-modified")),
            status_code=response.status_code,
        )

    async def validators(self, uri: str) -> dict[str, str] | None:
        """``HEAD`` one URI and return whatever validators it advertises.

        Returns ``None`` when the server advertises none, or refuses ``HEAD``
        entirely: an unanswerable freshness question must read as "refetch",
        never as "unchanged".
        """
        try:
            response = await self.request("HEAD", uri)
        except FetchError:
            return None

        found = {
            "etag": response.headers.get("etag"),
            "last_modified": response.headers.get("last-modified"),
        }
        validators = {name: value for name, value in found.items() if value}
        return validators or None

    async def aclose(self) -> None:
        """Close the shared HTTP client. Idempotent."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def request(
        self, method: str, uri: str, *, headers: Mapping[str, str] | None = None
    ) -> httpx.Response:
        client = self._http_client()
        try:
            response = await client.request(method, uri, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            message = f"{method} {uri} failed with HTTP {status}"
            if status in _TRANSIENT_STATUS:
                raise TransientFetchError(message) from error
            raise FetchError(message) from error
        except httpx.HTTPError as error:
            # Connection resets, DNS hiccups, and timeouts are all worth a retry.
            raise TransientFetchError(f"{method} {uri} failed: {error}") from error

        return response

    def _http_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {"user-agent": self.user_agent, **(self.headers or {})}
            self._client = httpx.AsyncClient(
                headers=headers,
                timeout=self.timeout,
                follow_redirects=self.follow_redirects,
            )
        return self._client


def normalize_uri(uri: str) -> str:
    """Return a normalized identifier for ``uri``, used as ``Source.id``.

    Lowercases the scheme and host, adds ``https://`` when the caller typed a
    bare host (no ``://``), drops a default port for the scheme, drops the
    fragment, and drops a trailing slash from the path. The scheme is kept
    (unlike ``API.md``'s illustrative ``normalized("https://b.com/") ==
    "b.com"``): dropping it would collide ``http://a.com`` and
    ``https://a.com``, which are not the same resource.
    """
    candidate = uri if "://" in uri else f"https://{uri}"
    parts = urlsplit(candidate)
    scheme = parts.scheme.lower()
    netloc = _normalize_netloc(parts, scheme)
    path = _normalize_path(parts.path)
    return urlunsplit((scheme, netloc, path, parts.query, ""))


def _normalize_netloc(parts: SplitResult, scheme: str) -> str:
    hostname = (parts.hostname or "").lower()
    port = parts.port
    if port is not None and port == _DEFAULT_PORTS.get(scheme):
        port = None

    netloc = f"{hostname}:{port}" if port is not None else hostname
    if parts.username:
        userinfo = parts.username
        if parts.password:
            userinfo = f"{userinfo}:{parts.password}"
        netloc = f"{userinfo}@{netloc}"
    return netloc


def _normalize_path(path: str) -> str:
    if path in ("", "/"):
        return ""
    return path.rstrip("/") if path.endswith("/") else path


def _parse_http_date(value: str | None) -> datetime | None:
    """Parse an RFC 2822 HTTP-date header into an aware ``datetime``, or ``None``."""
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _content_type(response: httpx.Response, uri: str) -> str:
    """Return the media type, ignoring parameters like ``charset``.

    Falls back to the URI extension, because a server sending
    ``application/octet-stream`` for a PDF is common enough to plan for.
    """
    header = response.headers.get("content-type", "")
    media_type = header.split(";", 1)[0].strip().lower()
    if media_type and media_type != "application/octet-stream":
        return media_type

    guessed, _ = mimetypes.guess_type(uri)
    return guessed or media_type or "application/octet-stream"
