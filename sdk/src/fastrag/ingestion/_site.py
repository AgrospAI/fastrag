"""Site identity — a page's site name and favicon — read from its HTML ``<head>``.

The HTTP fetchers store it on a page's ``Source.metadata`` under
:data:`SITE_METADATA_KEYS`, and :meth:`~fastrag.ingestion.steps.parsing.base.Parser.build_document`
carries it into every document (and so every chunk) parsed from that page.
A client citing a retrieved chunk can then show which site it came from
without fetching the page itself. Read from the raw HTTP response, never a
renderer's output: renderers return the page body, so the ``<head>`` is gone
by then.
"""

from __future__ import annotations

from collections.abc import Mapping
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit

#: The ``Source``/``Document`` metadata keys this module produces.
SITE_METADATA_KEYS = ("site_name", "favicon_url")

#: Both values end up in every chunk's metadata and every persisted chat
#: message's sources, so anything longer is dropped rather than stored.
MAX_SITE_NAME_CHARS = 100
MAX_FAVICON_URL_CHARS = 512

#: A ``<head>`` sits at the start of a page; scanning further only costs time.
_MAX_SCANNED_BYTES = 256 * 1024

_HTML_MEDIA_TYPES = frozenset({"text/html", "application/xhtml+xml"})

#: ``<meta>`` names for the site's own name, most specific first.
_SITE_NAME_KEYS = ("og:site_name", "application-name")


def site_metadata(content: bytes, content_type: str, base_uri: str) -> dict[str, str]:
    """Return the ``site_name``/``favicon_url`` an HTML page declares.

    ``site_name`` comes from ``<meta property="og:site_name">``, falling back
    to ``<meta name="application-name">``. ``favicon_url`` is the first
    ``<link rel="icon">`` (``shortcut icon`` included), falling back to
    ``apple-touch-icon``, resolved against ``base_uri`` (or the page's
    ``<base href>``); only ``http(s)`` URLs are kept, so an inline ``data:``
    icon is dropped. A key is omitted when the page declares nothing usable,
    and a non-HTML ``content_type`` returns ``{}``.
    """
    if content_type.split(";")[0].strip().lower() not in _HTML_MEDIA_TYPES:
        return {}
    parser = _HeadParser()
    try:
        parser.feed(content[:_MAX_SCANNED_BYTES].decode("utf-8", errors="replace"))
        parser.close()
    except Exception:  # noqa: BLE001 - malformed markup just yields no metadata
        pass

    metadata: dict[str, str] = {}
    site_name = next(
        (parser.meta[key] for key in _SITE_NAME_KEYS if key in parser.meta), None
    )
    if site_name is not None:
        site_name = " ".join(site_name.split())
        if 0 < len(site_name) <= MAX_SITE_NAME_CHARS:
            metadata["site_name"] = site_name

    icons = parser.icons + parser.touch_icons
    if icons:
        favicon_url = urljoin(urljoin(base_uri, parser.base_href or ""), icons[0])
        if (
            urlsplit(favicon_url).scheme in ("http", "https")
            and len(favicon_url) <= MAX_FAVICON_URL_CHARS
        ):
            metadata["favicon_url"] = favicon_url
    return metadata


def inherit_site_metadata(
    metadata: Mapping[str, Any], site: Mapping[str, str]
) -> dict[str, Any]:
    """Fill in whichever :data:`SITE_METADATA_KEYS` ``metadata`` lacks from ``site``.

    For a source fetched alongside an HTML page of the same site — a PDF it
    links to, or a page that declares no name or icon of its own — so it is
    cited under that site's name and icon too.
    """
    return {
        **{key: site[key] for key in SITE_METADATA_KEYS if key in site},
        **metadata,
    }


def site_of(metadata: Mapping[str, Any]) -> dict[str, str]:
    """The :data:`SITE_METADATA_KEYS` subset of ``metadata``."""
    return {key: metadata[key] for key in SITE_METADATA_KEYS if key in metadata}


class _HeadParser(HTMLParser):
    """Collects ``<meta>``, ``<link rel=icon>`` and ``<base>`` up to ``</head>``."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.icons: list[str] = []
        self.touch_icons: list[str] = []
        self.base_href: str | None = None
        self._done = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._done:
            return
        if tag == "body":
            self._done = True
            return
        values = {name.lower(): (value or "").strip() for name, value in attrs}
        if tag == "meta":
            key = (values.get("property") or values.get("name") or "").lower()
            if key in _SITE_NAME_KEYS and values.get("content"):
                self.meta.setdefault(key, values["content"])
        elif tag == "link" and values.get("href"):
            rels = values.get("rel", "").lower().split()
            if "icon" in rels:
                self.icons.append(values["href"])
            elif "apple-touch-icon" in rels:
                self.touch_icons.append(values["href"])
        elif tag == "base" and values.get("href") and self.base_href is None:
            self.base_href = values["href"]

    def handle_endtag(self, tag: str) -> None:
        if tag == "head":
            self._done = True
