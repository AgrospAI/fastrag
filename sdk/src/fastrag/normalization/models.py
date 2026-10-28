"""Source-preserving structural models independent of ingestion and packing."""

from __future__ import annotations

from bisect import bisect_right
from heapq import heappop, heappush
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Iterator, Protocol

from fastrag.models import Document


@dataclass(frozen=True, slots=True, order=True)
class SourceSpan:
    """Half-open Unicode offsets into the original Document content."""

    start_offset: int
    end_offset: int

    def __post_init__(self) -> None:
        """Reject inverted and negative offsets."""
        if type(self.start_offset) is not int or type(self.end_offset) is not int:
            raise ValueError("Source offsets must be integers")
        if self.start_offset < 0 or self.end_offset < self.start_offset:
            raise ValueError("Invalid source span")


@dataclass(frozen=True, slots=True)
class NormalizedNode:
    """Ordered structure; content is original syntax, not rewritten prose."""

    id: str
    kind: str
    content: str
    spans: tuple[SourceSpan, ...]
    children: tuple[NormalizedNode, ...] = ()
    parent_id: str | None = None
    level: int | None = None
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def walk(self) -> Iterator[NormalizedNode]:
        """Yield this node and descendants in source order."""
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass(frozen=True, slots=True)
class CoverageUnit:
    """A nonoverlapping retained source interval owned by a structural node."""

    id: str
    node_id: str
    span: SourceSpan


@dataclass(frozen=True, slots=True)
class Exclusion:
    """Explicitly omitted source with the policy reason."""

    span: SourceSpan
    reason: str


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """Stable normalization diagnostic; no timing or secret values."""

    code: str
    message: str
    span: SourceSpan | None = None


@dataclass(frozen=True, slots=True)
class NormalizedDocument:
    """Tree and independent exact source coverage partition."""

    document: Document
    source_format: str
    roots: tuple[NormalizedNode, ...]
    units: tuple[CoverageUnit, ...]
    exclusions: tuple[Exclusion, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def source(self) -> str:
        """Return the unchanged original string."""
        return self.document.content

    def walk(self) -> Iterator[NormalizedNode]:
        """Yield all nodes in preorder."""
        for root in self.roots:
            yield from root.walk()

    def validate(self) -> None:
        """Check lineage, source slices and complete nonduplicated coverage."""
        nodes = {n.id: n for n in self.walk()}
        if len(nodes) != len(list(self.walk())):
            raise ValueError("Duplicate node IDs")
        for node in nodes.values():
            if node.parent_id is not None and node.parent_id not in nodes:
                raise ValueError("Unknown parent")
            if any(s.end_offset > len(self.source) for s in node.spans):
                raise ValueError("Node outside source")
            if node.content != "".join(
                self.source[s.start_offset : s.end_offset] for s in node.spans
            ):
                raise ValueError("Node content differs from source")
            if any(c.parent_id != node.id for c in node.children):
                raise ValueError("Broken child lineage")
        intervals = sorted(
            [u.span for u in self.units] + [e.span for e in self.exclusions]
        )
        cursor = 0
        for span in intervals:
            if span.start_offset != cursor or span.end_offset > len(self.source):
                raise ValueError("Coverage gap, overlap or outside source")
            cursor = span.end_offset
        if cursor != len(self.source):
            raise ValueError("Incomplete source coverage")
        if any(u.node_id not in nodes for u in self.units):
            raise ValueError("Unknown coverage node")
        for unit in self.units:
            if not any(
                span.start_offset <= unit.span.start_offset
                and unit.span.end_offset <= span.end_offset
                for span in nodes[unit.node_id].spans
            ):
                raise ValueError("Coverage unit outside owning node")
        if tuple(sorted(self.units, key=lambda u: u.span)) != self.units:
            raise ValueError("Coverage units out of source order")


class Normalizer(Protocol):
    """Injectable format adapter with stable nonsecret cache identity."""

    @property
    def identity(self) -> str:
        """Describe every output-changing adapter option."""
        ...

    def normalize(self, document: Document) -> NormalizedDocument:
        """Normalize a document without executing its markup."""
        ...


@dataclass
class _Node:
    kind: str
    start: int
    end: int
    children: list[_Node] = field(default_factory=list)
    level: int | None = None
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def line_offsets(source: str) -> list[int]:
    """Build character offsets without normalizing source line endings."""
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def finish(
    document: Document,
    fmt: str,
    blocks: list[_Node],
    exclusions: list[Exclusion] | None = None,
    diagnostics: list[Diagnostic] | None = None,
) -> NormalizedDocument:
    """Resolve heading hierarchy, freeze nodes and partition retained source."""
    source = document.content
    root = _Node("document", 0, len(source))
    stack = [root]
    for block in blocks:
        if block.kind == "heading":
            while len(stack) > 1 and (stack[-1].level or 0) >= (block.level or 0):
                stack.pop()
            section = _Node(
                "section", block.start, block.end, [block], block.level, block.title
            )
            stack[-1].children.append(section)
            stack.append(section)
        else:
            stack[-1].children.append(block)

    def extend(node: _Node) -> None:
        for child in node.children:
            extend(child)
        if node.children:
            node.end = max(node.end, *(c.end for c in node.children))

    extend(root)
    excluded = sorted(exclusions or [], key=lambda e: e.span)
    # Collapse nested policy exclusions; overlapping exclusions are one policy interval.
    clean: list[Exclusion] = []
    for item in excluded:
        if clean and item.span.start_offset < clean[-1].span.end_offset:
            prior = clean.pop()
            clean.append(
                Exclusion(
                    SourceSpan(
                        prior.span.start_offset,
                        max(prior.span.end_offset, item.span.end_offset),
                    ),
                    prior.reason,
                )
            )
        else:
            clean.append(item)
    exclusion_ends = [e.span.end_offset for e in clean]
    ordinal = 0

    def freeze(node: _Node, parent: str | None) -> NormalizedNode:
        nonlocal ordinal
        nid = sha256(
            f"normalization-v1:{document.id}:{ordinal}:{node.kind}:{node.start}:{node.end}".encode()
        ).hexdigest()
        ordinal += 1
        spans: list[SourceSpan] = []
        cursor = node.start
        for exclusion_index in range(bisect_right(exclusion_ends, cursor), len(clean)):
            exclusion = clean[exclusion_index]
            a, b = exclusion.span.start_offset, exclusion.span.end_offset
            if a >= node.end:
                break
            if cursor < a:
                spans.append(SourceSpan(cursor, min(a, node.end)))
            cursor = max(cursor, b)
        if cursor < node.end:
            spans.append(SourceSpan(cursor, node.end))
        return NormalizedNode(
            nid,
            node.kind,
            "".join(source[s.start_offset : s.end_offset] for s in spans),
            tuple(spans),
            tuple(freeze(c, nid) for c in node.children),
            parent,
            node.level,
            node.title,
            node.metadata,
        )

    frozen = freeze(root, None)
    nodes = list(frozen.walk())
    bounds = sorted(
        {
            0,
            len(source),
            *(s.start_offset for n in nodes for s in n.spans),
            *(s.end_offset for n in nodes for s in n.spans),
            *(e.span.start_offset for e in clean),
            *(e.span.end_offset for e in clean),
        }
    )
    # Highest preorder ordinal wins, even for conservative crossing spans.
    starts = sorted(
        (span.start_offset, span.end_offset, ordinal)
        for ordinal, node in enumerate(nodes)
        for span in node.spans
        if span.start_offset < span.end_offset
    )
    active: list[tuple[int, int]] = []
    position = 0
    exclusion_index = 0
    units = []
    for a, b in zip(bounds, bounds[1:]):
        while position < len(starts) and starts[position][0] <= a:
            _, end, ordinal = starts[position]
            heappush(active, (-ordinal, end))
            position += 1
        while active and active[0][1] <= a:
            heappop(active)
        while (
            exclusion_index < len(clean) and clean[exclusion_index].span.end_offset <= a
        ):
            exclusion_index += 1
        if (
            exclusion_index < len(clean)
            and clean[exclusion_index].span.start_offset <= a
        ):
            continue
        owner = nodes[-active[0][0]]
        units.append(CoverageUnit(f"{owner.id}:{a}:{b}", owner.id, SourceSpan(a, b)))
    result = NormalizedDocument(
        document, fmt, (frozen,), tuple(units), tuple(clean), tuple(diagnostics or [])
    )
    result.validate()
    return result
