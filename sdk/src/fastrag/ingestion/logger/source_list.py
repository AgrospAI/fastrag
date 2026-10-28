"""Final flat URL list grouped by prefix."""

from __future__ import annotations

from urllib.parse import urlsplit

from fastrag.ingestion.logger._source_summary import (
    _COLORS,
    _SourceSummaryLogger,
    _Status,
    _literal,
    _title,
)


def _prefix(uri: str) -> str:
    try:
        parsed = urlsplit(uri)
    except ValueError:
        return "(other)"
    if parsed.netloc:
        path = parsed.path.rstrip("/").rsplit("/", 1)[0]
        return f"{parsed.scheme}://{parsed.netloc}{path}"
    return uri.rstrip("/").rsplit("/", 1)[0] if "/" in uri.rstrip("/") else ""


class SourceListLogger(_SourceSummaryLogger):
    """Print a final list of unique source URLs, one per line, grouped by prefix.

    The prefix of a URL is everything before its last path segment (scheme,
    domain and parent path). Groups are sorted by prefix and separated by one
    empty line; URLs inside a group are sorted. Each URL is colored by status:
    new is green, cached cyan and error red. With ``enrich_youtube=True``, a
    YouTube video's title follows its URL in parentheses. The last line, after
    an empty line, is the same summary printed by ``SourceTreeLogger``. Event
    consumption and status rules are shared with ``SourceTreeLogger``.
    """

    def _render(
        self,
        urls: dict[str, _Status],
        titles: dict[str, str],
        unattributed: int,
        failed_steps: int,
    ) -> None:
        color = self._use_color()
        groups: dict[str, list[str]] = {}
        counts: dict[_Status, int] = {"new": 0, "cached": 0, "error": 0}
        for uri, status in sorted(urls.items()):
            counts[status] += 1
            text = _literal(uri)
            if color:
                text = f"\x1b[{_COLORS[status]}m{text}\x1b[0m"
            groups.setdefault(_prefix(uri), []).append(text + _title(titles.get(uri)))

        lines: list[str] = []
        for prefix in sorted(groups):
            if lines:
                lines.append("")
            lines.extend(groups[prefix])
        extra = self._failure_lines(unattributed, failed_steps)
        if extra:
            lines.append("")
            lines.extend(extra)
        if lines:
            lines.append("")
        lines.append(self._summary(counts, color))
        print("\n".join(lines), file=self.stream, flush=True)
