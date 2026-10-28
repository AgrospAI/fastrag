r"""Fetching: locations in, ``Source`` objects out.

``Fetcher`` is the abstract leaf-level ``SourceStep`` concrete fetchers
extend; ``Sources`` aggregates several of them (or other ``SourceStep``\ s)
into one stream; ``SitemapCrawler``, ``WebCrawler``, ``FileSource``, and
``LocalFileFetcher`` are the concrete fetchers this package ships —
``FileSource`` for exactly one URL, ``LocalFileFetcher`` for local files
matched by glob, the other two for discovering many pages over HTTP.
``S3Fetcher`` is named in ``docs/specification/API.md`` but has no old
counterpart to port and is a later plan's territory.
"""

from __future__ import annotations

from fastrag.ingestion.steps.fetching.base import Fetcher
from fastrag.ingestion.steps.fetching.file import FileSource
from fastrag.ingestion.steps.fetching.local import LocalFileFetcher
from fastrag.ingestion.steps.fetching.renderers import (
    ChatRenderer,
    DecisionsRenderer,
    HttpRenderer,
    PlaywrightRenderer,
    Renderer,
)
from fastrag.ingestion.steps.fetching.sitemap import SitemapCrawler
from fastrag.ingestion.steps.fetching.sources import Sources
from fastrag.ingestion.steps.fetching.web import WebCrawler
from fastrag.ingestion.steps.fetching.youtube import YouTubeFetcher, youtube_filter

__all__ = [
    "Renderer",
    "HttpRenderer",
    "PlaywrightRenderer",
    "ChatRenderer",
    "DecisionsRenderer",
    "Fetcher",
    "FileSource",
    "LocalFileFetcher",
    "SitemapCrawler",
    "Sources",
    "WebCrawler",
    "YouTubeFetcher",
    "youtube_filter",
]
