"""Final URL tree summaries grouped by domain and path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TextIO
from urllib.parse import urlsplit

from fastrag.ingestion.logger._source_summary import (
    _SourceSummaryLogger,
    _Status,
    _literal,
    _title,
    _video_id,
)
from fastrag.ingestion.pipeline import Pipeline


@dataclass
class _Node:
    children: dict[str, _Node] = field(default_factory=dict)
    counts: dict[_Status, int] = field(
        default_factory=lambda: {"new": 0, "cached": 0, "error": 0}
    )
    title: str | None = None


def _parts(uri: str) -> list[str]:
    try:
        parsed = urlsplit(uri)
    except ValueError:
        return ["(other)", f"/{uri}"]
    domain = parsed.netloc.lower() or (
        f"{parsed.scheme}://" if parsed.scheme else "(local)"
    )
    parts = [domain, *(f"/{part}" for part in parsed.path.split("/") if part)]
    video_id = _video_id(uri)
    if video_id and parsed.path == "/watch":
        # A watch URL names its video only in the query, which paths drop.
        parts.append(f"?v={video_id}")
    return parts


class SourceTreeLogger(_SourceSummaryLogger):
    """Print a final tree of unique source URLs and their subtree totals.

    Domain nodes have depth 1; each path segment adds one level. Schemes,
    queries, fragments and trailing slashes distinguish URLs for counting,
    but share the same domain/path branch. A YouTube ``/watch`` URL adds its
    ``?v=`` video ID as one more level. Every node includes URLs at that path
    and below it, even when descendants are hidden by ``max_depth``. Path
    labels below domain roots start with a slash. Each line shows nonzero
    colored status numbers; a final summary names and totals them.

    Status rules, ``enrich_youtube`` titles (shown in parentheses after the
    video's label) and failure handling are those of the shared source summary
    base. No optional logging dependency is required.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        max_depth: int | None = None,
        stream: TextIO | None = None,
        color: bool | None = None,
        enrich_youtube: bool = False,
    ) -> None:
        """Configure tree depth, destination, terminal colors and video titles.

        Args:
            pipeline: Pipeline whose complete event stream will be consumed.
            max_depth: Positive depth limit, or None to show all path levels.
                A value of 1 shows only domains.
            stream: Output destination; defaults to sys.stdout.
            color: None enables ANSI colors only on a TTY. True forces colors
                and False disables them. New is green, cached cyan, error red.
            enrich_youtube: Show each YouTube video's title in parentheses
                after its ``?v=`` label, looking up titles that sources do
                not include.
        """
        super().__init__(
            pipeline, stream=stream, color=color, enrich_youtube=enrich_youtube
        )
        if max_depth is not None and (
            isinstance(max_depth, bool)
            or not isinstance(max_depth, int)
            or max_depth < 1
        ):
            raise ValueError("max_depth must be a positive integer or None")
        self.max_depth = max_depth

    def _label(self, label: str, node: _Node, color: bool) -> str:
        counts = []
        for status, count in node.counts.items():
            if count:
                counts.append(self._number(count, status, color))
        return " ".join([_literal(label) + _title(node.title), *counts])

    def _render(
        self,
        urls: dict[str, _Status],
        titles: dict[str, str],
        unattributed: int,
        failed_steps: int,
    ) -> None:
        root = _Node()
        for uri, outcome in urls.items():
            node = root
            node.counts[outcome] += 1
            for part in _parts(uri):
                node = node.children.setdefault(part, _Node())
                node.counts[outcome] += 1
            node.title = titles.get(uri, node.title)
        color = self._use_color()
        lines: list[str] = []

        # Iterative traversal also handles URLs with very deep paths.
        def branches(
            parent: _Node, prefix: str, depth: int
        ) -> list[tuple[str, _Node, str, int, bool]]:
            children = sorted(parent.children.items())
            return [
                (label, child, prefix, depth, index == len(children) - 1)
                for index, (label, child) in reversed(list(enumerate(children)))
            ]

        pending = branches(root, "", 1)
        while pending:
            label, node, prefix, depth, last = pending.pop()
            lines.append(
                prefix
                + (("└── " if last else "├── ") if depth > 1 else "")
                + self._label(label, node, color)
            )
            if self.max_depth is not None and depth >= self.max_depth:
                continue
            pending.extend(
                branches(
                    node,
                    prefix + (("    " if last else "│   ") if depth > 1 else ""),
                    depth + 1,
                )
            )
        lines.extend(self._failure_lines(unattributed, failed_steps))
        if lines:
            lines.append("")
        lines.append(self._summary(root.counts, color))
        print("\n".join(lines), file=self.stream, flush=True)
