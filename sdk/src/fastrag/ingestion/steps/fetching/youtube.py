"""YouTube video fetching, and the iframe-embed discovery crawlers route through it.

``YouTubeFetcher`` downloads one video (or just its audio track) with yt-dlp.
It is a normal ``Fetcher`` usable on its own, and the target of
:meth:`~fastrag.ingestion.steps.fetching.base.Fetcher.route` on crawlers:

.. code-block:: python

    crawler = WebCrawler(url="https://example.com").route(
        youtube_filter, YouTubeFetcher(audio_only=True)
    )
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import re
import shutil
import tempfile
from collections.abc import AsyncIterator, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, urljoin, urlsplit

from fastrag.exceptions import ConfigurationError, FetchError
from fastrag.ingestion._hashing import content_hash
from fastrag.ingestion.cache import CacheStore
from fastrag.ingestion.filters import URLFilter
from fastrag.ingestion.models import Source
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext
from fastrag.ingestion.steps.fetching.base import Fetcher

MISSING_VIDEO_EXTRA = (
    'Video extraction needs the "video" extra: pip install "fastrag-ai[video]"'
)

_LOGGER = logging.getLogger(__name__)

_YOUTUBE_ID = r"[A-Za-z0-9_-]{11}"
_YOUTUBE_HOST = r"(?:www\.|m\.)?youtube(?:-nocookie)?\.com"
_WATCH_PREFIX = "https://www.youtube.com/watch?v="

#: ``youtube.com/embed/ID``, ``youtube.com/shorts/ID``, ``youtu.be/ID``.
_PATH_FORMS = re.compile(
    rf"^(?:https?:)?//(?:{_YOUTUBE_HOST}/(?:embed|shorts|live)/|youtu\.be/)({_YOUTUBE_ID})(?:[/?#]|$)"
)

#: Matches any URL :class:`YouTubeFetcher` can download.
youtube_filter = URLFilter(
    include_patterns=[
        rf"^(?:https?:)?//{_YOUTUBE_HOST}/(?:watch\?|embed/|shorts/|live/)",
        r"^(?:https?:)?//youtu\.be/",
    ]
)

#: ``<iframe src="...">`` (either quote style) in raw or rendered HTML.
_IFRAME_SRC = re.compile(r"<iframe\b[^>]*?\bsrc\s*=\s*[\"']([^\"']+)[\"']", re.I | re.S)

#: yt-dlp ``info`` fields kept as video source metadata (``None`` dropped).
_VIDEO_METADATA_FIELDS = (
    "title",
    "description",
    "uploader",
    "channel",
    "channel_url",
    "upload_date",
    "duration",
    "view_count",
    "like_count",
    "tags",
    "categories",
    "language",
    "webpage_url",
)


def youtube_video_id(uri: str) -> str | None:
    """The video id of a YouTube watch/embed/short/``youtu.be`` URL, else ``None``."""
    match = _PATH_FORMS.match(uri)
    if match:
        return match.group(1)
    parts = urlsplit(uri)
    if re.fullmatch(_YOUTUBE_HOST, parts.hostname or "") and parts.path == "/watch":
        video_id = parse_qs(parts.query).get("v", [""])[0]
        if re.fullmatch(_YOUTUBE_ID, video_id):
            return video_id
    return None


class YouTubeFetcher(Fetcher):
    """Download one YouTube video, yielding its media bytes as a ``Source``.

    Needs the "video" extra (``yt-dlp``).
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        audio_only: bool = False,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
        cache_validation_interval: float | None = None,
    ) -> None:
        """Initialize a ``YouTubeFetcher``.

        Args:
            url: The video to fetch when used standalone. Not needed when
                used as a :meth:`~Fetcher.route` target.
            audio_only: Download only the audio track rather than the best
                combined video+audio stream.
            cache: Cache backend, or ``False``/``None`` — see
                :class:`~fastrag.ingestion.steps.base.PipelineStep`.
            policy: Execution policy overriding the pipeline default.
            cache_validation_interval: See
                :class:`~fastrag.ingestion.steps.base.SourceStep`.
        """
        super().__init__(
            cache=cache,
            policy=policy,
            cache_validation_interval=cache_validation_interval,
        )
        self.url = url
        self.audio_only = audio_only

    def config(self) -> dict[str, Any]:
        """Cache fingerprint: whether the audio track alone is downloaded."""
        return {"audio_only": self.audio_only}

    def canonical_uri(self, uri: str) -> str:
        """The ``watch?v=`` URL for any spelling of a video URL."""
        video_id = youtube_video_id(uri)
        return uri if video_id is None else f"{_WATCH_PREFIX}{video_id}"

    async def targets(self, context: ExecutionContext) -> list[str]:
        """The one video this fetcher targets."""
        return [self._own_url()]

    async def evaluate_target(
        self, target: str, context: ExecutionContext
    ) -> AsyncIterator[Source]:
        """Download the video, yielding its ``Source``."""
        for source in await self.fetch(target):
            yield source

    async def evaluate(self, context: ExecutionContext) -> AsyncIterator[Source]:
        """Download the video, uncached."""
        for source in await self.fetch(self._own_url()):
            yield source

    async def is_source_modified(
        self,
        source_id: str,
        cached_etag: str | None,
        cached_last_modified: datetime | None,
        cached_content_hash: str,
    ) -> bool:
        """Never: a video's bytes are keyed by its id and do not change."""
        return False

    async def fetch(self, uri: str) -> list[Source]:
        """Download the video at ``uri`` (any YouTube URL form)."""
        url = self.canonical_uri(uri)
        if youtube_video_id(url) is None:
            raise FetchError(f"not a YouTube video URL: {uri}")
        content, extension, info = await asyncio.to_thread(
            _download, url, audio_only=self.audio_only
        )
        content_type = mimetypes.guess_type(f"x.{extension}")[0] or (
            "audio/mp4" if self.audio_only else "video/mp4"
        )
        if self.audio_only and not content_type.startswith("audio/"):
            # An audio-only stream (e.g. ``.webm``/``.mp4``) is reported with a
            # ``video/*`` type by extension; it must route as audio.
            content_type = "audio/" + content_type.split("/", 1)[1]
        return [
            Source(
                id=url,
                uri=url,
                content=content,
                content_type=content_type,
                content_hash=content_hash(content),
                metadata={"kind": "video", "audio_only": self.audio_only, **info},
            )
        ]

    def _own_url(self) -> str:
        if self.url is None:
            raise ConfigurationError(
                "YouTubeFetcher needs url= unless used as a route target"
            )
        return self.canonical_uri(self.url)


def embed_urls(content: bytes, base_uri: str) -> list[str]:
    """Absolute ``src`` URLs of every ``<iframe>`` in ``content``."""
    html = content.decode("utf-8", errors="replace")
    return [urljoin(base_uri, src.strip()) for src in _IFRAME_SRC.findall(html)]


async def fetch_embeds(
    owner: Fetcher, page_uri: str, *contents: tuple[bytes, str]
) -> list[Source]:
    """Fetch the page's ``<iframe>`` embeds that ``owner`` has a route for.

    ``contents`` are ``(bytes, base_uri)`` pairs — typically the raw and the
    rendered page, since a renderer may strip embeds. Equivalent spellings of
    one embed are fetched once. A dead or private embed is skipped with a
    warning rather than failing the page.
    """
    sources: list[Source] = []
    seen: set[str] = set()
    for content, base_uri in contents:
        for url in embed_urls(content, base_uri):
            fetcher = owner.route_for(url)
            if fetcher is None:
                if url not in seen and youtube_filter.matches(
                    Source(
                        id=url, uri=url, content=b"", content_type="", content_hash=""
                    )
                ):
                    seen.add(url)
                    _LOGGER.warning(
                        "YouTube embed %s (found on page %s) is NOT being fetched: "
                        "no route handles YouTube URLs. Add "
                        ".route(youtube_filter, YouTubeFetcher()) to fetch it; "
                        "otherwise it is skipped.",
                        url,
                        page_uri,
                    )
                continue
            canonical = fetcher.canonical_uri(url)
            if canonical in seen:
                continue
            seen.add(canonical)
            try:
                fetched = await fetcher.fetch(url)
            except FetchError as error:
                _LOGGER.warning(
                    "skipping unavailable embed (found on page %s): %s", page_uri, error
                )
                continue
            for source in fetched:
                source.metadata.setdefault("page_uri", page_uri)
                sources.append(source)
    return sources


def _yt_dlp() -> Any:
    try:
        import yt_dlp
    except ImportError:
        return None
    return yt_dlp


def _js_runtimes() -> dict[str, dict[str, Any]]:
    """Installed JS runtimes yt-dlp can use to solve YouTube's signature/``n`` challenges.

    yt-dlp enables only ``deno`` by default; without any runtime most videos
    report "This video is not available".
    """
    return {name: {} for name in ("deno", "node", "bun") if shutil.which(name)}


def _video_metadata(info: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: info[field]
        for field in _VIDEO_METADATA_FIELDS
        if info.get(field) not in (None, "", [])
    }


def _download(url: str, *, audio_only: bool) -> tuple[bytes, str, dict[str, Any]]:
    """Download ``url`` with yt-dlp, returning ``(bytes, extension, metadata)``.

    Blocking; call through ``asyncio.to_thread``. Audio-only takes the best
    audio stream as-is, so ffmpeg is not required for it.
    """
    yt_dlp = _yt_dlp()
    if yt_dlp is None:
        raise ConfigurationError(MISSING_VIDEO_EXTRA)
    with tempfile.TemporaryDirectory() as directory:
        options = {
            "format": "bestaudio/best" if audio_only else "bv*+ba/b",
            "outtmpl": str(Path(directory) / "%(id)s.%(ext)s"),
            "quiet": True,
            # ``quiet`` alone still draws a progress bar on stderr.
            "noprogress": True,
            "no_warnings": True,
            "noplaylist": True,
            "js_runtimes": _js_runtimes(),
        }
        try:
            with yt_dlp.YoutubeDL(options) as downloader:
                info = downloader.extract_info(url, download=True)
        except Exception as error:
            raise FetchError(f"could not download video {url}: {error}") from error
        files = [path for path in Path(directory).iterdir() if path.is_file()]
        if not files:
            raise FetchError(f"video {url} produced no file")
        path = max(files, key=lambda candidate: candidate.stat().st_size)
        return path.read_bytes(), path.suffix.lstrip("."), _video_metadata(info)
