"""Shared URL accounting and YouTube titles for final source summaries."""

from __future__ import annotations

import asyncio
import sys
from abc import abstractmethod
from collections.abc import Iterable, Mapping
from typing import Literal, TextIO

import httpx

from fastrag.ingestion.events import DocumentFetched, ItemFailed, StepFailed
from fastrag.ingestion.logger.base import Logger
from fastrag.ingestion.pipeline import Pipeline
from fastrag.ingestion.steps.fetching.http import DEFAULT_USER_AGENT
from fastrag.ingestion.steps.fetching.youtube import youtube_video_id

_Status = Literal["new", "cached", "error"]
_COLORS: dict[_Status, str] = {"new": "32", "cached": "36", "error": "31"}

_OEMBED_URL = "https://www.youtube.com/oembed"
_TITLE_CONCURRENCY = 8
_TITLE_TIMEOUT = 10.0


def _literal(label: str) -> str:
    # Untrusted URLs must not inject terminal escapes or additional tree lines.
    return "".join(
        f"\\x{ord(char):02x}" if ord(char) < 32 or ord(char) == 127 else char
        for char in label
    )


def _title(title: str | None) -> str:
    return f" ({_literal(title)})" if title else ""


def _video_id(uri: str) -> str | None:
    try:
        return youtube_video_id(uri)
    except ValueError:
        return None


async def _youtube_titles(uris: Iterable[str]) -> dict[str, str]:
    """Titles of the YouTube videos in ``uris``, from YouTube's public oEmbed.

    Each video is requested once, however many URLs spell it. Private, removed
    or unreachable videos are left out rather than failing the summary.
    """
    videos = {uri: video_id for uri in uris if (video_id := _video_id(uri))}
    if not videos:
        return {}
    semaphore = asyncio.Semaphore(_TITLE_CONCURRENCY)

    async def lookup(client: httpx.AsyncClient, video_id: str) -> str | None:
        async with semaphore:
            try:
                response = await client.get(
                    _OEMBED_URL,
                    params={
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                        "format": "json",
                    },
                )
                response.raise_for_status()
                data = response.json()
            except (httpx.HTTPError, ValueError):
                return None
        title = data.get("title") if isinstance(data, dict) else None
        return title if isinstance(title, str) and title else None

    ids = sorted(set(videos.values()))
    async with httpx.AsyncClient(
        timeout=_TITLE_TIMEOUT,
        headers={"user-agent": DEFAULT_USER_AGENT},
        follow_redirects=True,
    ) as client:
        found = await asyncio.gather(*(lookup(client, video_id) for video_id in ids))
    titles = dict(zip(ids, found, strict=True))
    return {
        uri: title for uri, video_id in videos.items() if (title := titles[video_id])
    }


class _SourceSummaryLogger(Logger):
    """Collect unique source URLs and their status, then render them once.

    Fetch cache status determines new/cached. An item failure at any stage
    marks its originating URL as error, counted once even if several chunks
    fail. Errors take precedence over new, which takes precedence over cached.
    Unattributed item failures and failed steps are reported separately.
    Runs reset ``source_count`` and print partial summaries before propagating
    exceptions.

    With ``enrich_youtube=True``, YouTube video URLs carry their title. A
    source's ``title`` metadata is used when present; other videos are looked
    up on YouTube's oEmbed endpoint once the pipeline completes, so an
    interrupted run never waits on the network.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        stream: TextIO | None = None,
        color: bool | None = None,
        enrich_youtube: bool = False,
    ) -> None:
        """Configure the destination, terminal colors and video titles.

        Args:
            pipeline: Pipeline whose complete event stream will be consumed.
            stream: Output destination; defaults to sys.stdout.
            color: None enables ANSI colors only on a TTY. True forces colors
                and False disables them. New is green, cached cyan, error red.
            enrich_youtube: Show each YouTube video's title in parentheses
                after its URL, looking up titles that sources do not include.
        """
        super().__init__(pipeline)
        self.stream = stream if stream is not None else sys.stdout
        self.color = color
        self.enrich_youtube = enrich_youtube
        self.source_count = 0

    def _use_color(self) -> bool:
        if self.color is None:
            return bool(getattr(self.stream, "isatty", lambda: False)())
        return self.color

    def _number(self, count: int, status: _Status, color: bool) -> str:
        text = str(count)
        return f"\x1b[{_COLORS[status]}m{text}\x1b[0m" if color else text

    def _summary(self, counts: Mapping[_Status, int], color: bool) -> str:
        return " ".join(
            f"{self._number(counts[status], status, color)} {status}"
            for status in _COLORS
        )

    @staticmethod
    def _failure_lines(unattributed: int, failed_steps: int) -> list[str]:
        lines = []
        if unattributed:
            lines.append(f"Unattributed item failures: {unattributed}")
        if failed_steps:
            lines.append(f"Failed steps: {failed_steps}")
        return lines

    @abstractmethod
    def _render(
        self,
        urls: dict[str, _Status],
        titles: dict[str, str],
        unattributed: int,
        failed_steps: int,
    ) -> None:
        """Print the final summary of ``urls``, titled by ``titles``."""

    async def run(self) -> None:
        """Consume events, then print the complete or interrupted summary."""
        self.source_count = 0
        urls: dict[str, _Status] = {}
        titles: dict[str, str] = {}
        source_uris: dict[str, str] = {}
        failures: list[ItemFailed] = []
        failed_steps = 0
        completed = False
        try:
            async for event in self.pipeline:
                if isinstance(event, DocumentFetched):
                    source = event.source
                    source_uris[source.id] = source.uri
                    if source.uri:
                        if not source.cached or source.uri not in urls:
                            urls[source.uri] = "cached" if source.cached else "new"
                        title = source.metadata.get("title")
                        if (
                            self.enrich_youtube
                            and isinstance(title, str)
                            and title
                            and _video_id(source.uri)
                        ):
                            titles[source.uri] = title
                elif isinstance(event, ItemFailed):
                    failures.append(event)
                elif isinstance(event, StepFailed):
                    failed_steps += 1
            completed = True
        finally:
            unattributed = 0
            for failure in failures:
                uri = source_uris.get(failure.source_id)
                if uri is None and failure.uri:
                    # Unknown document IDs are not new source URLs.
                    if failure.uri in urls or "://" in failure.uri:
                        uri = failure.uri
                if uri:
                    urls[uri] = "error"
                else:
                    unattributed += 1
            self.source_count = len(urls)
            try:
                if completed and self.enrich_youtube:
                    titles.update(
                        await _youtube_titles(uri for uri in urls if uri not in titles)
                    )
            finally:
                self._render(urls, titles, unattributed, failed_steps)
