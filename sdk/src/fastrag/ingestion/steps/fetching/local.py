r"""Local-disk fetching: files matched by glob, one ``Source`` per file.

Where :class:`~fastrag.ingestion.steps.fetching.file.FileSource` fetches one
URL and the crawlers discover pages over HTTP, ``LocalFileFetcher`` reads
whatever a corpus that already lives on disk needs — "every spreadsheet
under ``data/``" — without publishing it over HTTP first. A pattern is a
plain file path, a directory path (expanded to the files under it,
recursively), or a glob (including ``**``); several patterns may be given at
once and their results are deduplicated.

Caching is one bucket per file, via the same ``targets``/``evaluate_target``
hooks :class:`~fastrag.ingestion.steps.fetching.sitemap.SitemapCrawler` uses
for its per-page buckets — one new file appearing under a watched directory
never invalidates the files already cached. Freshness is checked cheaply,
without rereading a file's bytes: a file's size doubles as its "ETag" (an
opaque validator, same slot HTTP's ETag occupies) and its mtime as its
"Last-Modified" — both available from a single ``stat()`` call. Content is
only reread when either changes.
"""

from __future__ import annotations

import asyncio
import glob
import mimetypes
import os
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

from fastrag.exceptions import FetchError
from fastrag.ingestion._hashing import content_hash
from fastrag.ingestion.cache import CacheStore
from fastrag.ingestion.models import Source
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext, run_concurrently
from fastrag.ingestion.steps.fetching.base import Fetcher

#: Extensions worth pinning explicitly rather than trusting ``mimetypes``,
#: whose registered types vary across Python versions and host OSes — a
#: consumer relying on ``.xlsx``/``.docx``/``.pptx``/``.md`` always getting
#: the same media type should not depend on that.
_MEDIA_TYPES: dict[str, str] = {
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".md": "text/markdown",
}


class LocalFileFetcher(Fetcher):
    """Fetch local files matched by glob, yielding one ``Source`` per file.

    .. code-block:: python

        from fastrag.ingestion.steps.fetching import LocalFileFetcher

        fetcher = LocalFileFetcher(
            patterns=["data/**/*.xlsx"],
            exclude=["data/**/~$*"],
            cache=cache,
        )

    Each element of ``patterns`` may be a plain file path, a directory path
    (every file under it, recursively), or a glob pattern (``*``, ``?``,
    ``[...]``, and ``**`` when used with a directory-spanning segment — see
    :func:`glob.glob`). A pattern that matches no files is an error, not a
    silent no-op: a typo'd path should fail loudly rather than quietly
    ingest nothing.
    """

    def __init__(
        self,
        *,
        patterns: Sequence[str],
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        max_files: int | None = None,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
        cache_validation_interval: float | None = None,
    ) -> None:
        """Initialize a ``LocalFileFetcher``.

        Args:
            patterns: File paths, directory paths, and/or glob patterns to
                expand. At least one is required.
            include: Path glob patterns to keep, matched against each
                discovered file's resolved absolute path. ``None`` keeps
                everything.
            exclude: Path glob patterns to drop, applied after ``include``.
            max_files: Stop after this many files. Useful while developing
                against a large tree.
            cache: Cache backend, or ``False``/``None`` — see
                :class:`~fastrag.ingestion.steps.base.PipelineStep`.
            policy: Execution policy overriding the pipeline default.
            cache_validation_interval: How long a cached file may be served
                without revalidating it against disk — see
                :class:`~fastrag.ingestion.steps.base.SourceStep`.
        """
        super().__init__(
            cache=cache,
            policy=policy,
            cache_validation_interval=cache_validation_interval,
        )
        self.patterns = list(patterns)
        self.include = include
        self.exclude = exclude
        self.max_files = max_files

    def config(self) -> dict[str, Any]:
        """This fetcher's cache fingerprint.

        Empty: unlike the HTTP-speaking fetchers, nothing here changes what
        bytes a given file returns — that is entirely up to the filesystem.
        ``patterns``/``include``/``exclude``/``max_files`` decide *which*
        files are discovered, not what a given file's bytes are, the same
        reasoning
        :meth:`~fastrag.ingestion.steps.fetching.sitemap.SitemapCrawler.config`
        documents for its own URL selection — so they are deliberately
        excluded here too. Per-file cache ids already encode the file's own
        URI as their ``target``.
        """
        return {}

    async def targets(self, context: ExecutionContext) -> list[str]:
        """Return the filtered, deduplicated ``file://`` URIs this fetcher targets.

        Called by the base class's per-target cache wrap before deciding
        which files need :meth:`evaluate_target`. Expansion (globbing,
        directory walks, and the ``stat`` calls needed to tell a file from a
        directory) runs in a worker thread so it never blocks the event
        loop.

        Raises:
            fastrag.exceptions.FetchError: A pattern in ``patterns`` matched
                no files.
        """
        matches = await asyncio.to_thread(self._expand)
        kept = [path for path in matches if self._keeps(str(path))]
        unique = list(dict.fromkeys(kept))
        uris = [path.as_uri() for path in unique]
        return uris if self.max_files is None else uris[: self.max_files]

    async def evaluate_target(
        self, target: str, context: ExecutionContext
    ) -> AsyncIterator[Source]:
        """Read one file, yielding the single ``Source`` it produces."""
        yield await self._fetch_source(target)

    async def evaluate(self, context: ExecutionContext) -> AsyncIterator[Source]:
        """Read every matched file, uncached, bounded by ``policy.concurrency``.

        Used directly when no cache is in effect — the per-target cache wrap
        (:meth:`targets`/:meth:`evaluate_target`) is what applies when one is.
        """
        targets = await self.targets(context)
        effective_policy = self.effective_policy(context)

        async def _targets() -> AsyncIterator[str]:
            for target in targets:
                yield target

        async for source in run_concurrently(
            _targets(),
            self._fetch_source,
            context=context,
            policy=effective_policy,
            step=self.name,
            event_name=self.event_name,
        ):
            yield source

    async def is_source_modified(
        self,
        source_id: str,
        cached_etag: str | None,
        cached_last_modified: datetime | None,
        cached_content_hash: str,
    ) -> bool:
        """Conditional freshness check: file size, then mtime — never rereading bytes.

        ``cached_etag`` holds the file's size at the time it was cached
        (this fetcher's stand-in for an HTTP ``ETag``, an opaque origin
        validator, populated the same way by ``_fetch_source``) and
        ``cached_last_modified`` its mtime. A file that no longer exists, or
        can no longer be ``stat``-ed, reads as modified — the same
        "unanswerable freshness question" logic
        :meth:`~fastrag.ingestion.steps.fetching.sitemap.SitemapCrawler.is_source_modified`
        applies to a server that refuses ``HEAD``: an attempted refetch
        always produces a definite answer (here, a ``FetchError``), rather
        than silently keeping stale bytes.
        """
        path = _path_from_uri(source_id)
        try:
            stat = await asyncio.to_thread(path.stat)
        except OSError:
            return True

        if cached_last_modified is not None and _mtime(stat) != cached_last_modified:
            return True
        return cached_etag is not None and cached_etag != str(stat.st_size)

    def _expand(self) -> list[Path]:
        matches: list[Path] = []
        for pattern in self.patterns:
            matches.extend(_expand_pattern(pattern))
        return matches

    def _keeps(self, value: str) -> bool:
        if self.include and not any(
            fnmatch(value, pattern) for pattern in self.include
        ):
            return False
        return not (
            self.exclude and any(fnmatch(value, pattern) for pattern in self.exclude)
        )

    async def _fetch_source(self, uri: str) -> Source:
        path = _path_from_uri(uri)
        try:
            content, stat = await asyncio.to_thread(_read, path)
        except OSError as error:
            raise FetchError(f"cannot read {path}: {error}") from error

        return Source(
            id=uri,
            uri=uri,
            content=content,
            content_type=_content_type(path),
            content_hash=content_hash(content),
            etag=str(stat.st_size),
            last_modified=_mtime(stat),
            metadata={},
        )


def _expand_pattern(pattern: str) -> list[Path]:
    """Expand one pattern into the (unresolved-order, resolved-path) files it names.

    A directory expands to every file under it, recursively; anything else
    goes through :func:`glob.glob` (which already handles a plain, literal
    file path — it exists, so it is returned unchanged), filtered down to
    files (never directories, even when a glob matches one).
    """
    path = Path(pattern)
    if path.is_dir():
        found = sorted(
            candidate for candidate in path.rglob("*") if candidate.is_file()
        )
    else:
        found = sorted(
            candidate
            for raw in glob.glob(pattern, recursive=True)
            if (candidate := Path(raw)).is_file()
        )
    if not found:
        raise FetchError(f"pattern matched no files: {pattern!r}")
    return [candidate.resolve() for candidate in found]


def _content_type(path: Path) -> str:
    """Return the media type for ``path``, by extension.

    Checks :data:`_MEDIA_TYPES` first, then falls back to :mod:`mimetypes`,
    then ``application/octet-stream`` for anything neither recognizes.
    """
    override = _MEDIA_TYPES.get(path.suffix.lower())
    if override is not None:
        return override
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def _mtime(stat: os.stat_result) -> datetime:
    return datetime.fromtimestamp(stat.st_mtime, tz=UTC)


def _read(path: Path) -> tuple[bytes, os.stat_result]:
    """Read ``path``'s bytes and ``stat`` it, in one worker-thread hop."""
    return path.read_bytes(), path.stat()


def _path_from_uri(uri: str) -> Path:
    """Invert :meth:`pathlib.Path.as_uri`: a ``file://`` URI back to a ``Path``."""
    return Path(unquote(urlsplit(uri).path))
