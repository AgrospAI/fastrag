"""Deterministic normalized-tree packing with exact serialized token budgets."""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from bisect import bisect_left, bisect_right, insort
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from time import perf_counter
from html import unescape
from html.parser import HTMLParser
from importlib.metadata import version
from typing import Any, Protocol

from fastrag.exceptions import ChunkingError
from fastrag.ingestion.steps.chunking.base import Chunker, SplitPiece
from fastrag.ingestion.steps.chunking.refinement import ChatBoundaryRefiner
from fastrag.models import ContentFormat, Document
from fastrag.normalization import (
    NormalizedDocument,
    NormalizedNode,
    NormalizerRegistry,
    SourceSpan,
)

_REFINEMENT: ContextVar[dict[str, tuple[int, ...]]] = ContextVar(
    "structural_refinement", default={}
)
_MODEL_DIAGNOSTICS: ContextVar[dict[str, Any]] = ContextVar(
    "structural_model_diagnostics", default={}
)

EMBEDDING_TEXT_VERSION = "structure-embedding:v1"

_LOGGER = logging.getLogger(__name__)


def _compact_maps(
    values: list[dict[str, Any]], source: str, text: str
) -> list[dict[str, Any]]:
    """Coalesce identity translations and constant invisible runs, losslessly.

    Other transformations keep their atom boundaries; adjacency alone is not
    evidence that their interior source/display relationship is affine.
    """
    result: list[dict[str, Any]] = []
    for original in values:
        value: dict[str, Any] = {
            **original,
            "source_spans": [dict(s) for s in original["source_spans"]],
        }
        spans = value["source_spans"]
        a, b = value["output_start"], value["output_end"]
        affine = (
            len(spans) == 1
            and b > a
            and (source[spans[0]["start_offset"] : spans[0]["end_offset"]] == text[a:b])
        )
        if affine:
            value["mapping_kind"] = "affine"
        if result and len(spans) == 1 and len(result[-1]["source_spans"]) == 1:
            prev = result[-1]
            previous_span = prev["source_spans"][0]
            constant = prev["output_start"] == prev["output_end"] == a == b
            if (
                prev["role"] == value["role"]
                and previous_span["end_offset"] == spans[0]["start_offset"]
                and prev["output_end"] == a
                and ((affine and prev.get("mapping_kind") == "affine") or constant)
            ):
                previous_span["end_offset"] = spans[0]["end_offset"]
                prev["output_end"] = b
                continue
        result.append(value)
    return result


def _structure_attributes(node: NormalizedNode) -> dict[str, Any]:
    """Reference the parent interpretation, retaining lightweight structure.

    Whole-node prose/code/inline interpretations belong to the normalized
    parent. Display plus exact maps preserve chunk content independently.
    """
    omitted = {"inline", "text", "code", "links", "images", "references"}
    result = {key: value for key, value in node.metadata.items() if key not in omitted}
    if "cells" in result:
        result["cells"] = [
            {key: value for key, value in cell.items() if key not in omitted}
            for cell in result["cells"]
        ]
    return {**result, "node_id": node.id}


class Tokenizer(Protocol):
    """Exact counting and lossless Unicode boundaries, with cache identity."""

    identity: str

    def count(self, text: str) -> int:
        """Count complete serialized text; counts need not be additive."""
        ...

    def boundaries(self, text: str) -> Sequence[int]:
        """Return ordered safe offsets including zero and the final offset."""
        ...


class Utf8ByteTokenizer:
    """Explicit offline byte budget, not an embedding-model token estimate."""

    identity = "utf8-bytes:v1"

    def count(self, text: str) -> int:
        """Count bytes, conservatively including multibyte Unicode."""
        return len(text.encode("utf-8"))

    def boundaries(self, text: str) -> Sequence[int]:
        """Never split a Unicode code point."""
        return range(len(text) + 1)


class TiktokenTokenizer:
    """Exact ordinary-text BPE count for an explicitly named encoding."""

    def __init__(self, encoding: str = "cl100k_base") -> None:
        """Load the encoding; first use may populate tiktoken's resource cache."""
        try:
            import tiktoken
        except ImportError as error:
            raise ImportError(
                "Install fastrag[normalization] for TiktokenTokenizer, or select Utf8ByteTokenizer"
            ) from error
        self.encoding = tiktoken.get_encoding(encoding)
        self.identity = f"tiktoken:{version('tiktoken')}:{encoding}:ordinary:v1"

    def count(self, text: str) -> int:
        """Treat literal special-token spellings as ordinary source text."""
        return len(self.encoding.encode_ordinary(text))

    def boundaries(self, text: str) -> Sequence[int]:
        """Use original Unicode offsets rather than decoding partial BPE bytes."""
        return range(len(text) + 1)


def _builtin_token_starts(tokenizer: Tokenizer, text: str) -> list[int] | None:
    """Source-offset hints only; subclasses retain arbitrary tokenizer semantics."""
    if type(tokenizer) is TiktokenTokenizer:
        decoded, starts = tokenizer.encoding.decode_with_offsets(
            tokenizer.encoding.encode_ordinary(text)
        )
        # Tiktoken can replace lone surrogates. Never map replaced text to source.
        return [*starts, len(text)] if decoded == text else None
    if type(tokenizer) is Utf8ByteTokenizer:
        return [
            *(i for i, char in enumerate(text) for _ in char.encode("utf-8")),
            len(text),
        ]
    return None


def build_embedding_text(content: str, metadata: Mapping[str, Any]) -> str:
    """Build the exact versioned embedding serialization independently.

    The engine records which heading IDs are already displayed. Checking IDs
    rather than title strings handles duplicate titles and headings in code.
    Generic documents without this version marker retain content-only input.
    """
    if metadata.get("embedding_text_version") != EMBEDDING_TEXT_VERSION:
        return content
    if not metadata.get("include_heading_path_in_embedding", True):
        return content
    displayed = set(metadata.get("displayed_heading_ids", []))
    titles = metadata.get("heading_path", [])
    levels = metadata.get("heading_levels", [])
    ids = metadata.get("heading_ids", [])
    prefix = [
        f"{'#' * level} {title}"
        for title, level, ident in zip(titles, levels, ids)
        if ident not in displayed
    ]
    return "\n\n".join([*prefix, content]) if prefix else content


@dataclass(frozen=True)
class _Atom:
    text: str
    spans: tuple[SourceSpan, ...]
    role: str = "body"


@dataclass
class _Piece:
    atoms: list[_Atom]
    section: NormalizedNode
    path: list[NormalizedNode]
    kind: str = "mixed"
    reason: str = "section_fits"
    prefix: str = ""
    suffix: str = ""
    repeats: list[dict[str, Any]] = field(default_factory=list)
    overlap: list[SourceSpan] = field(default_factory=list)
    context_primary: list[_Atom] = field(default_factory=list)
    displayed: list[str] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)
    lost: bool = False
    overflow: bool = False
    structure: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return (self.prefix + "".join(a.text for a in self.atoms) + self.suffix).strip()


def _span(span: SourceSpan) -> dict[str, int]:
    return {"start_offset": span.start_offset, "end_offset": span.end_offset}


def _coalesce(spans: list[SourceSpan]) -> list[SourceSpan]:
    result: list[SourceSpan] = []
    for span in sorted(spans):
        if result and result[-1].end_offset == span.start_offset:
            previous = result.pop()
            result.append(SourceSpan(previous.start_offset, span.end_offset))
        elif span.end_offset > span.start_offset:
            result.append(span)
    return result


class _HtmlRender(HTMLParser):
    """Event serialization; each event keeps its precise original slice."""

    def __init__(self, source: str, offset: int) -> None:
        super().__init__(convert_charrefs=False)
        self.source, self.source_base = source, offset
        self.lines = [0]
        for line in source.splitlines(keepends=True):
            self.lines.append(self.lines[-1] + len(line))
        self.atoms: list[_Atom] = []
        self.links: list[str] = []
        self.pre = 0
        self.covered = 0
        self.lists: list[dict[str, Any]] = []
        self.row_cells = 0
        self.row_header = False
        self.table_depth = 0
        self.table_schema = False
        self.row_start = 0

    def position(self) -> int:
        line, col = self.getpos()
        return self.lines[line - 1] + col

    def add(self, text: str, start: int, end: int, role: str = "syntax") -> None:
        if start > self.covered:
            self.atoms.append(
                _Atom(
                    self.source[self.covered : start],
                    (
                        SourceSpan(
                            self.source_base + self.covered, self.source_base + start
                        ),
                    ),
                )
            )
        self.atoms.append(
            _Atom(
                text,
                (SourceSpan(self.source_base + start, self.source_base + end),),
                role,
            )
        )
        self.covered = end

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        start = self.position()
        end = start + len(self.get_starttag_text() or "")
        options = dict(attrs)
        text = ""
        if re.fullmatch(r"h[1-6]", tag):
            text = "\n" + "#" * int(tag[1]) + " "
        elif tag in {"ol", "ul"}:
            try:
                number = int(options.get("start") or "1")
            except ValueError:
                number = 1
            self.lists.append({"ordered": tag == "ol", "next": number})
        elif tag == "li":
            listing = self.lists[-1] if self.lists else {"ordered": False}
            if listing["ordered"]:
                try:
                    number = int(options.get("value") or listing["next"])
                except ValueError:
                    number = listing["next"]
                listing["next"] = number + 1
                marker = f"{number}. "
            else:
                marker = "- "
            text = "\n" + "  " * max(0, len(self.lists) - 1) + marker
        elif tag == "table":
            self.table_depth += 1
            self.table_schema = False
        elif tag == "tr":
            text = "\n|"
            self.row_start = len(self.atoms)
            self.row_cells = 0
            self.row_header = False
        elif tag in {
            "p",
            "div",
            "section",
            "article",
            "br",
            "tr",
            "blockquote",
            "figcaption",
            "caption",
        }:
            text = "\n"
        elif tag in {"td", "th"}:
            self.row_cells += 1
            self.row_header = self.row_header or tag == "th"
            text = " "
            attributes = [
                f"{key}={options[key]}"
                for key in ("rowspan", "colspan")
                if options.get(key) and options[key] != "1"
            ]
            if attributes:
                text += "[" + ", ".join(attributes) + "] "
        elif tag == "pre":
            self.pre += 1
        elif tag == "a":
            self.links.append(options.get("href") or "")
            text = "["
        elif tag == "img":
            text = f"![{options.get('alt') or ''}]({options.get('src') or ''})"
        self.add(text, start, end)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        start = self.position()
        end = self.source.find(">", start)
        end = len(self.source) if end < 0 else end + 1
        text = ""
        if tag in {"ol", "ul"}:
            if self.lists:
                self.lists.pop()
        elif tag in {"td", "th"}:
            text = " |"
        elif tag == "table":
            self.table_depth = max(0, self.table_depth - 1)
        elif tag == "tr":
            text = "\n"
            if self.row_header:
                text += "|" + " --- |" * self.row_cells + "\n"
                self.table_schema = True
            elif self.table_depth and not self.table_schema:
                labels = "|" + "".join(
                    f" Column {i + 1} |" for i in range(self.row_cells)
                )
                schema = "|" + " --- |" * self.row_cells
                self.atoms.insert(
                    self.row_start, _Atom(labels + "\n" + schema, (), "synthetic")
                )
                self.table_schema = True
        elif tag == "a":
            text = f"]({self.links.pop() if self.links else ''})"
        elif tag == "pre":
            self.pre = max(0, self.pre - 1)
        elif tag in {
            "p",
            "li",
            "tr",
            "div",
            "section",
            "article",
            "caption",
            "figcaption",
        } or re.fullmatch(r"h[1-6]", tag):
            text = "\n"
        self.add(text, start, end)

    def handle_data(self, data: str) -> None:
        start = self.position()
        for index, character in enumerate(data):
            self.add(character, start + index, start + index + 1, "body")

    def handle_entityref(self, name: str) -> None:
        start = self.position()
        match = re.match(r"&[^\s<&;]+;?", self.source[start:])
        raw = match.group(0) if match else "&" + name
        self.add(unescape(raw), start, start + len(raw), "body")

    def handle_charref(self, name: str) -> None:
        self.handle_entityref("#" + name)

    def handle_comment(self, data: str) -> None:
        start = self.position()
        end = self.source.find("-->", start)
        self.add("", start, len(self.source) if end < 0 else end + 3)

    def handle_decl(self, decl: str) -> None:
        start = self.position()
        end = self.source.find(">", start)
        self.add("", start, len(self.source) if end < 0 else end + 1)

    def finish(self) -> list[_Atom]:
        self.feed(self.source)
        self.close()
        if self.covered < len(self.source):
            self.atoms.append(
                _Atom(
                    self.source[self.covered :],
                    (
                        SourceSpan(
                            self.source_base + self.covered,
                            self.source_base + len(self.source),
                        ),
                    ),
                )
            )
        return self.atoms


class StructureAwareChunker(Chunker):
    """Pack normalized sections and structures using a hard embedding budget.

    Defaults never invoke a model. Exact budgets guarantee only the injected
    tokenizer; select the embedding model's tokenizer explicitly when needed.
    See the living specification for serialization and overflow contracts.
    """

    content_format = ContentFormat.MARKDOWN

    def __init__(
        self,
        *,
        tokenizer: Tokenizer | None = None,
        normalizers: NormalizerRegistry | None = None,
        target_tokens: int = 400,
        max_tokens: int = 600,
        min_preferred_tokens: int = 100,
        overlap_tokens: int = 0,
        max_overlap_tokens: int = 50,
        merge_small_sections: bool = True,
        allow_cross_top_level_merge: bool = False,
        preserve_heading_hierarchy: bool = True,
        preserve_code_blocks: bool = True,
        preserve_table_rows: bool = True,
        preserve_list_hierarchy: bool = True,
        include_heading_path_in_embedding: bool = True,
        semantic_mode: str = "heuristic",
        semantic_refiner: ChatBoundaryRefiner | None = None,
        oversized_block_policy: str = "split_with_context",
        cache=None,
        policy=None,
        diagnostics=None,
    ) -> None:
        """Validate options before creating optional tokenizer resources."""
        numbers = {
            "target_tokens": target_tokens,
            "max_tokens": max_tokens,
            "min_preferred_tokens": min_preferred_tokens,
            "overlap_tokens": overlap_tokens,
            "max_overlap_tokens": max_overlap_tokens,
        }
        if any(type(value) is not int for value in numbers.values()):
            raise ValueError("Token limits must be integers, not booleans")
        if not 0 < min_preferred_tokens <= target_tokens <= max_tokens:
            raise ValueError(
                "Require 0 < min_preferred_tokens <= target_tokens <= max_tokens"
            )
        if not 0 <= overlap_tokens <= max_overlap_tokens:
            raise ValueError("Require 0 <= overlap_tokens <= max_overlap_tokens")
        flags = {
            "merge_small_sections": merge_small_sections,
            "allow_cross_top_level_merge": allow_cross_top_level_merge,
            "preserve_heading_hierarchy": preserve_heading_hierarchy,
            "preserve_code_blocks": preserve_code_blocks,
            "preserve_table_rows": preserve_table_rows,
            "preserve_list_hierarchy": preserve_list_hierarchy,
            "include_heading_path_in_embedding": include_heading_path_in_embedding,
        }
        if any(type(value) is not bool for value in flags.values()):
            raise ValueError("Structural options must be booleans")
        if semantic_mode not in {"disabled", "heuristic", "llm"}:
            raise ValueError("semantic_mode must be disabled, heuristic or llm")
        if semantic_mode == "llm" and not isinstance(
            semantic_refiner, ChatBoundaryRefiner
        ):
            raise ValueError("llm mode requires an opt-in ChatBoundaryRefiner")
        self.semantic_refiner = semantic_refiner
        if oversized_block_policy not in {"split_with_context", "preserve_and_flag"}:
            raise ValueError("Unknown oversized_block_policy")
        super().__init__(cache=cache, policy=policy)
        self.diagnostics = diagnostics
        self.tokenizer = tokenizer if tokenizer is not None else TiktokenTokenizer()
        if not isinstance(self.tokenizer.identity, str) or not self.tokenizer.identity:
            raise ValueError("Tokenizer requires a stable nonempty identity")
        self.normalizers = (
            normalizers if normalizers is not None else NormalizerRegistry()
        )
        self.options: dict[str, Any] = {
            **numbers,
            **flags,
            "semantic_mode": semantic_mode,
            "oversized_block_policy": oversized_block_policy,
        }

    def config(self) -> Mapping[str, Any]:
        """Identify all output choices for existing pipeline/cache machinery."""
        return {
            **self.options,
            **(
                {"semantic_refiner": self.semantic_refiner.config()}
                if self.options["semantic_mode"] == "llm"
                and self.semantic_refiner is not None
                else {}
            ),
            "tokenizer": self.tokenizer.identity,
            "normalizers": self.normalizers.identity,
            "engine": "structure:v5",
            "embedding_text_version": EMBEDDING_TEXT_VERSION,
        }

    def _count(self, text: str) -> int:
        value = self.tokenizer.count(text)
        if type(value) is not int or value < 0:
            raise ChunkingError("Tokenizer count must be a nonnegative integer")
        return value

    def _base_metadata(self, piece: _Piece) -> dict[str, Any]:
        path = piece.path if self.options["preserve_heading_hierarchy"] else []
        return {
            "heading_path": [n.title or "" for n in path],
            "headings": [n.title or "" for n in path],
            "heading_levels": [n.level or 1 for n in path],
            "heading_ids": [n.id for n in path],
            "displayed_heading_ids": piece.displayed,
            "include_heading_path_in_embedding": self.options[
                "include_heading_path_in_embedding"
            ],
            "embedding_text_version": EMBEDDING_TEXT_VERSION,
        }

    def _budget(self, piece: _Piece) -> int:
        return self._count(build_embedding_text(piece.text, self._base_metadata(piece)))

    def _raw_render(
        self, normalized: NormalizedDocument, node: NormalizedNode
    ) -> list[_Atom]:
        result: list[_Atom] = []
        for span in node.spans:
            source = normalized.source[span.start_offset : span.end_offset]
            if normalized.source_format == "html":
                renderer = _HtmlRender(source, span.start_offset)
                if node.kind == "list_item":
                    nodes = {n.id: n for n in normalized.walk()}
                    ancestry = []
                    current = node
                    while current.parent_id in nodes:
                        parent = nodes[current.parent_id]
                        if parent.kind == "list":
                            attrs = parent.metadata.get("attrs", {})
                            try:
                                start = int(attrs.get("start", "1"))
                            except (TypeError, ValueError):
                                start = 1
                            items = [
                                c for c in parent.children if c.kind == "list_item"
                            ]
                            ordinal = next(
                                (i for i, c in enumerate(items) if c.id == current.id),
                                0,
                            )
                            ancestry.append(
                                {
                                    "ordered": parent.metadata.get("tag") == "ol",
                                    "next": start + ordinal,
                                }
                            )
                        current = parent
                    renderer.lists = list(reversed(ancestry))
                result.extend(renderer.finish())
            else:
                result.extend(
                    _Atom(
                        c,
                        (SourceSpan(span.start_offset + i, span.start_offset + i + 1),),
                    )
                    for i, c in enumerate(source)
                )
        return result

    def _render(
        self, normalized: NormalizedDocument, node: NormalizedNode
    ) -> list[_Atom]:
        atoms = self._raw_render(normalized, node)
        if normalized.source_format == "html" and node.kind == "code_block":
            prefix, suffix = self._fences(node)
            match = re.search(r'language-([^\s"\'>]+)', node.content)
            if match:
                prefix = (
                    prefix.rstrip("\n") + match[1] + "\n"
                    if not node.metadata.get("language")
                    else prefix
                )
            if "".join(a.text for a in atoms).endswith("\n"):
                suffix = suffix.removeprefix("\n")
            return [_Atom(prefix, ()), *atoms, _Atom(suffix, ())]
        return atoms

    @property
    def accepted_content_formats(self) -> frozenset[str]:
        """Capabilities follow the supplied instance-local normalizer registry."""
        return self.normalizers.formats

    def split(self, document: Document) -> Iterator[SplitPiece]:
        """Normalize once, then use the independently callable tree engine."""
        self.validate_input(document)
        if self.options["semantic_mode"] == "llm":
            raise NotImplementedError("LLM refinement requires asplit()")
        yield from self.split_normalized(self.normalizers.normalize(document))

    async def asplit(self, document: Document) -> AsyncIterator[SplitPiece]:
        """Normalize, optionally refine safe prose, then deterministically pack."""
        self.validate_input(document)
        async for pair in self.asplit_normalized(self.normalizers.normalize(document)):
            yield pair

    async def asplit_normalized(
        self, normalized: NormalizedDocument
    ) -> AsyncIterator[SplitPiece]:
        """Async tree API; per-operation state isolates concurrent documents."""
        normalized.validate()
        cuts: dict[str, tuple[int, ...]] = {}
        stats: dict[str, Any] = {
            "model_calls": 0,
            "model_cache_hits": 0,
            "model_fallbacks": (),
        }
        if self.options["semantic_mode"] == "llm":
            assert self.semantic_refiner is not None
            nodes = {node.id: node for node in normalized.walk()}
            requests: list[tuple[str, str, tuple[int, ...]]] = []
            for node in nodes.values():
                ancestor = nodes.get(node.parent_id or "")
                protected = False
                while ancestor is not None:
                    if ancestor.kind not in {"document", "section"}:
                        protected = True
                        break
                    ancestor = nodes.get(ancestor.parent_id or "")
                if (
                    protected
                    or normalized.source_format not in {"markdown", "rst"}
                    or node.kind != "paragraph"
                    or node.children
                ):
                    continue
                # Conservatively exclude all inline syntax and markup. Atom
                # alignment forbids partial HTML/entity/render transformations.
                if re.search(r"[`*_<>\[\]{}|\\]", node.content):
                    continue
                atoms = self._render(normalized, node)
                text = "".join(a.text for a in atoms)
                positions = {0}
                position = 0
                for atom in atoms:
                    position += len(atom.text)
                    positions.add(position)
                safe = set(self.tokenizer.boundaries(text))
                allowed = tuple(
                    m.end()
                    for m in re.finditer(r"[.!?](?:[\"')\]]*)\s+", text)
                    if m.end() < len(text) and m.end() in safe and m.end() in positions
                )
                requests.append((node.id, text, allowed))
            # Paragraphs are independent: request them together, bounded by
            # the refiner's own concurrency, instead of one round trip each.
            results = await asyncio.gather(
                *(
                    self.semantic_refiner.refine(
                        node_id=node_id, text=text, allowed=allowed
                    )
                    for node_id, text, allowed in requests
                )
            )
            for (node_id, _, _), result in zip(requests, results):
                stats["model_calls"] += result.calls
                stats["model_cache_hits"] += int(result.cache_hit)
                if result.fallback:
                    stats["model_fallbacks"] += (result.fallback,)
                if result.boundaries:
                    cuts[node_id] = result.boundaries
            # Only failures reaching the model: input_limit is a local choice.
            failed = Counter(
                reason
                for reason in stats["model_fallbacks"]
                if reason not in {"input_limit", "invalid_candidates"}
            )
            if failed:
                _LOGGER.warning(
                    "LLM boundary refinement fell back to heuristic cuts for %d of "
                    "%d paragraphs of %s (%s)%s",
                    sum(failed.values()),
                    len(requests),
                    normalized.document.metadata.get(
                        "source_uri", normalized.document.id
                    ),
                    ", ".join(f"{reason} x{n}" for reason, n in failed.items()),
                    "; the model returned no content: a reasoning model may be "
                    "spending max_tokens on thinking (disable it, e.g. "
                    "chat_template_kwargs.enable_thinking=False)"
                    if "empty_response" in failed
                    else "",
                )
        cut_token = _REFINEMENT.set(cuts)
        stats_token = _MODEL_DIAGNOSTICS.set(stats)
        try:
            # Finish all synchronous work before yielding: async generator
            # suspension cannot leak ContextVar state into the caller task.
            output = list(self.split_normalized(normalized, _refined=True))
        finally:
            _REFINEMENT.reset(cut_token)
            _MODEL_DIAGNOSTICS.reset(stats_token)
        for pair in output:
            yield pair

    def split_normalized(
        self, normalized: NormalizedDocument, *, _refined: bool = False
    ) -> Iterator[SplitPiece]:
        """Consume any valid registered normalized tree without format routing."""
        if self.options["semantic_mode"] == "llm" and not _refined:
            raise NotImplementedError("LLM refinement requires asplit_normalized()")
        started = perf_counter()
        normalized.validate()
        pieces: list[_Piece] = []
        nodes = {n.id: n for n in normalized.walk()}

        def visit(
            section: NormalizedNode,
            path: list[NormalizedNode],
            additions: Sequence[NormalizedNode] = (),
        ) -> None:
            own_path = [*path, section] if section.kind == "section" else path
            children = [*section.children, *additions]
            covered = sorted(s for c in children for s in c.spans)
            for outer in section.spans:
                cursor = outer.start_offset
                gaps = []
                for span in covered:
                    if (
                        span.end_offset <= cursor
                        or span.start_offset >= outer.end_offset
                    ):
                        continue
                    if cursor < span.start_offset:
                        gaps.append(
                            SourceSpan(cursor, min(span.start_offset, outer.end_offset))
                        )
                    cursor = max(cursor, span.end_offset)
                if cursor < outer.end_offset:
                    gaps.append(SourceSpan(cursor, outer.end_offset))
                for gap in gaps:
                    children.append(
                        NormalizedNode(
                            id=f"{section.id}:gap:{gap.start_offset}",
                            kind="raw_block",
                            content=normalized.source[
                                gap.start_offset : gap.end_offset
                            ],
                            spans=(gap,),
                            parent_id=section.id,
                        )
                    )
            children.sort(key=lambda c: c.spans[0].start_offset if c.spans else 0)

            def pack_run(run: list[NormalizedNode]) -> None:
                direct = [c for c in run if c.kind != "heading"]
                if not any(
                    "".join(a.text for a in self._render(normalized, c)).strip()
                    for c in direct
                ):
                    return
                atoms = [a for c in run for a in self._render(normalized, c)]
                whole = _Piece(
                    atoms,
                    section,
                    own_path,
                    displayed=[section.id]
                    if any(c.kind == "heading" for c in run)
                    else [],
                    sections=[section.id],
                )
                if self._budget(whole) <= self.options["max_tokens"] and not any(
                    self.options["semantic_mode"] == "llm" and c.id in _REFINEMENT.get()
                    for c in direct
                ):
                    pieces.append(whole)
                    return
                packed: _Piece | None = None
                for block in direct:
                    for part in self._block(normalized, block, section, own_path):
                        if not part.text:
                            continue
                        if (
                            packed is not None
                            and part.reason == packed.reason == "block_fits"
                        ):
                            combined = _Piece(
                                [*packed.atoms, _Atom("\n\n", ()), *part.atoms],
                                section,
                                own_path,
                                reason="block_fits",
                                sections=[section.id],
                            )
                            if self._budget(combined) <= self.options["target_tokens"]:
                                packed = combined
                                continue
                        if packed is not None:
                            pieces.append(packed)
                        packed = part
                if packed is not None:
                    pieces.append(packed)

            run: list[NormalizedNode] = []
            cursor = 0
            while cursor < len(children):
                child = children[cursor]
                cursor += 1
                if child.kind == "section":
                    pack_run(run)
                    run = []
                    trailing = []
                    while cursor < len(children) and ":gap:" in children[cursor].id:
                        trailing.append(children[cursor])
                        cursor += 1
                    visit(child, own_path, trailing)
                else:
                    run.append(child)
            pack_run(run)

        for root in normalized.roots:
            visit(root, [])
        if self.options["merge_small_sections"]:
            pieces = self._merge(pieces, nodes)
        if not pieces:
            for unit in normalized.units:
                if nodes[unit.node_id].kind == "heading":
                    continue
                semantic = normalized.source[
                    unit.span.start_offset : unit.span.end_offset
                ]
                if normalized.source_format == "html":
                    semantic = unescape(re.sub(r"<[^>]*>", "", semantic))
                if semantic.strip():
                    raise ChunkingError("Body source has no readable serialization")
            self._validate_output(normalized, [], perf_counter() - started)
            return
        # Attach actual unrendered source intervals, never fabricate node coverage.
        assigned = _coalesce(
            [s for p in pieces for a in [*p.context_primary, *p.atoms] for s in a.spans]
        )
        missing: list[tuple[SourceSpan, str]] = []
        assigned_ends = [s.end_offset for s in assigned]
        for unit in normalized.units:
            cursor = unit.span.start_offset
            for span_index in range(bisect_right(assigned_ends, cursor), len(assigned)):
                span = assigned[span_index]
                if span.start_offset >= unit.span.end_offset:
                    break
                if (
                    span.end_offset <= cursor
                    or span.start_offset >= unit.span.end_offset
                ):
                    continue
                if cursor < span.start_offset:
                    missing.append(
                        (SourceSpan(cursor, span.start_offset), unit.node_id)
                    )
                cursor = max(cursor, span.end_offset)
            if cursor < unit.span.end_offset:
                missing.append((SourceSpan(cursor, unit.span.end_offset), unit.node_id))
        # One entry per distinct source start; ties choose the earlier piece.
        locations: dict[int, int] = {}
        empty_pieces = []
        for i, piece in enumerate(pieces):
            spans = [s for atom in piece.atoms for s in atom.spans]
            if not spans:
                empty_pieces.append(i)
            for source_span in spans:
                locations[source_span.start_offset] = min(
                    i, locations.get(source_span.start_offset, i)
                )
        starts = sorted(locations)
        for span, owner in missing:
            at = bisect_left(starts, span.start_offset)
            nearby = starts[max(0, at - 1) : at + 1]
            candidates = [(abs(x - span.start_offset), locations[x]) for x in nearby]
            candidates.extend((0, i) for i in empty_pieces)
            _, nearest_index = min(candidates)
            nearest = pieces[nearest_index]
            if span.start_offset not in locations:
                insort(starts, span.start_offset)
            locations[span.start_offset] = min(
                nearest_index, locations.get(span.start_offset, nearest_index)
            )
            if nearest_index in empty_pieces:
                empty_pieces.remove(nearest_index)
            role = "hierarchy" if nodes[owner].kind == "heading" else "syntax"
            raw = normalized.source[span.start_offset : span.end_offset]
            semantic = raw
            if normalized.source_format == "html":
                semantic = unescape(re.sub(r"<[^>]*>", "", raw))
            if role != "hierarchy" and semantic.strip():
                raise ChunkingError("Non-whitespace body source was not serialized")
            invisible = _Atom("", (span,), role)
            position = next(
                (
                    i
                    for i, a in enumerate(nearest.atoms)
                    if a.spans and a.spans[0].start_offset > span.start_offset
                ),
                len(nearest.atoms),
            )
            nearest.atoms.insert(position, invisible)
        self._validate_pieces(normalized, pieces)
        output = []
        unit_ends = [u.span.end_offset for u in normalized.units]
        for index, piece in enumerate(pieces):
            metadata = self._metadata(normalized, piece, index, unit_ends)
            count = self._budget(piece)
            if count > self.options["max_tokens"] and not piece.overflow:
                raise ChunkingError("Final serialized embedding exceeds max_tokens")
            output.append((piece.text, metadata))
        self._validate_output(normalized, output, perf_counter() - started)
        yield from output

    def _validate_output(self, normalized, output, operation_seconds=0.0):
        from fastrag.ingestion.steps.chunking.validation import (
            validate_structure_chunks,
        )

        report = validate_structure_chunks(
            normalized,
            output,
            self.tokenizer,
            self.options["max_tokens"],
            self.options["max_overlap_tokens"],
            self.options["min_preferred_tokens"],
        )
        report = replace(
            report,
            operation_seconds=operation_seconds + report.validation_seconds,
            **_MODEL_DIAGNOSTICS.get(),
        )
        if self.diagnostics is not None:
            self.diagnostics(report)
        return report

    #: ``validate_output`` hydrates every replayed chunk.
    replay_needs_content = True

    async def validate_output(self, item, produced, context):
        """Independently validate fresh/cached output, hydrating projections."""
        from fastrag.ingestion.steps.base import _materialize_item

        from fastrag.exceptions import TransientCacheError
        from fastrag.ingestion.cache.base import _DocumentHeader

        started = perf_counter()
        # Projected siblings usually share a bucket. Loading it for each sibling
        # repeats both database I/O and deserialization of every persisted map.
        stores = {}
        for value in [item, *produced]:
            if isinstance(value, _DocumentHeader):
                store, buckets = stores.setdefault(
                    id(value._store), (value._store, set())
                )
                buckets.add(value._bucket)
        snapshots = {}
        for store_id, (store, buckets) in stores.items():
            # Single reads let the store coalesce the buckets of every
            # concurrently validated document into one round trip.
            ordered = sorted(buckets)
            loaded = await asyncio.gather(
                *(store.get_document_entries(bucket) for bucket in ordered)
            )
            for bucket, entries in zip(ordered, loaded):
                snapshots[(store_id, bucket)] = {
                    entry.id: entry for entry in entries or []
                }

        async def hydrate(value):
            if not isinstance(value, _DocumentHeader):
                return await _materialize_item(value)
            resolved = snapshots.get((id(value._store), value._bucket), {}).get(
                value.id
            )
            if resolved is None:
                raise TransientCacheError(
                    "document cache changed while materializing a header"
                )
            return resolved

        item = await hydrate(item)
        produced[:] = [
            replace(await hydrate(chunk), cached=chunk.cached) for chunk in produced
        ]
        self._validate_output(
            self.normalizers.normalize(item), produced, perf_counter() - started
        )

    def _validate_pieces(
        self, normalized: NormalizedDocument, pieces: list[_Piece]
    ) -> None:
        """Reject primary duplication/loss/order defects before streaming results."""
        expected = _coalesce([u.span for u in normalized.units])
        actual = [
            s for p in pieces for a in [*p.context_primary, *p.atoms] for s in a.spans
        ]
        ordered = sorted(actual)
        if any(a.end_offset > b.start_offset for a, b in zip(ordered, ordered[1:])):
            raise ChunkingError("Duplicate primary source retention")
        if _coalesce(ordered) != expected:
            raise ChunkingError("Incomplete primary source retention")
        body = [
            s
            for p in pieces
            for a in [*p.context_primary, *p.atoms]
            if a.text.strip() and a.role == "body"
            for s in a.spans
        ]
        if any(a.end_offset > b.start_offset for a, b in zip(body, body[1:])):
            raise ChunkingError("Primary body source is out of order")
        for piece in pieces:
            if self._budget(piece) > self.options["max_tokens"] and not piece.overflow:
                raise ChunkingError("Final serialized embedding exceeds max_tokens")
            if any(
                not a.text
                and a.role == "body"
                and normalized.source[s.start_offset : s.end_offset].strip()
                for a in piece.atoms
                for s in a.spans
            ):
                raise ChunkingError("Body source has no readable serialization")

    def _merge(
        self, pieces: list[_Piece], nodes: dict[str, NormalizedNode]
    ) -> list[_Piece]:
        result: list[_Piece] = []
        eligible = False
        for piece in pieces:
            small = (
                piece.reason == "section_fits"
                and self._budget(piece) < self.options["min_preferred_tokens"]
            )
            if result and eligible and small:
                prior = result[-1]
                prior_section = nodes[prior.sections[-1]]
                sibling = prior_section.parent_id == piece.section.parent_id
                parent = nodes.get(piece.section.parent_id or "")
                top_allowed = parent is not None and (
                    parent.kind != "document"
                    or self.options["allow_cross_top_level_merge"]
                )
                if sibling and top_allowed:
                    path = piece.path[:-1]
                    combined = _Piece(
                        [*prior.atoms, _Atom("\n\n", ()), *piece.atoms],
                        parent,
                        path,
                        reason="small_sibling_merge",
                        displayed=[*prior.displayed, *piece.displayed],
                        sections=[*prior.sections, *piece.sections],
                    )
                    if self._budget(combined) <= self.options["target_tokens"]:
                        result[-1] = combined
                        continue
            result.append(piece)
            eligible = small
        return result

    def _block(
        self,
        normalized: NormalizedDocument,
        node: NormalizedNode,
        section: NormalizedNode,
        path: list[NormalizedNode],
    ) -> list[_Piece]:
        atoms = self._render(normalized, node)
        piece = _Piece(
            atoms,
            section,
            path,
            kind=node.kind,
            reason="block_fits",
            sections=[section.id],
            structure=_structure_attributes(node),
        )
        boundaries = (
            _REFINEMENT.get().get(node.id, ())
            if self.options["semantic_mode"] == "llm"
            else ()
        )
        if boundaries:
            result = []
            cursor = offset = 0
            for index, atom in enumerate(atoms, 1):
                offset += len(atom.text)
                if offset in boundaries or index == len(atoms):
                    fragment = replace(
                        piece, atoms=atoms[cursor:index], reason="semantic_refinement"
                    )
                    if self._budget(fragment) <= self.options["max_tokens"]:
                        result.append(fragment)
                    else:
                        result.extend(self._split_atoms(fragment, prose=True))
                    cursor = index
            return result
        if node.kind == "code_block" and self.options["preserve_code_blocks"]:
            atoms = self._code_atoms(
                normalized, node, self._raw_render(normalized, node)
            )
            piece.atoms = atoms
            piece.prefix, piece.suffix = self._fences(node)
            if normalized.source_format == "html" and not node.metadata.get("language"):
                match = re.search(r'language-([^\s"\'>]+)', node.content)
                if match:
                    piece.prefix = piece.prefix.rstrip("\n") + match[1] + "\n"
        if self._budget(piece) <= self.options["max_tokens"]:
            return [piece]
        if self.options[
            "oversized_block_policy"
        ] == "preserve_and_flag" and node.kind not in {"paragraph", "raw_block"}:
            piece.overflow = True
            piece.reason = "atomic_overflow"
            return [piece]
        if node.kind == "table" and not self.options["preserve_table_rows"]:
            piece.lost = True
            return self._split_atoms(piece)
        if node.kind == "code_block" and not self.options["preserve_code_blocks"]:
            piece.lost = True
        if node.kind in {"table", "list"}:
            return self._structured(normalized, node, piece)
        return self._split_atoms(
            piece, prose=node.kind in {"paragraph", "raw_block", "blockquote"}
        )

    def _fences(self, node: NormalizedNode) -> tuple[str, str]:
        language = str(node.metadata.get("language") or "")
        # Pick a fence longer than any backtick run in the actual code.
        fence = "`" * max(
            3,
            max(
                (
                    len(m.group()) + 1
                    for m in re.finditer(r"`+", str(node.metadata.get("code", "")))
                ),
                default=3,
            ),
        )
        return f"{fence}{language}\n", f"\n{fence}"

    def _code_atoms(
        self, normalized: NormalizedDocument, node: NormalizedNode, atoms: list[_Atom]
    ) -> list[_Atom]:
        if normalized.source_format == "rst" and node.metadata.get("code"):
            code_lines = str(node.metadata["code"]).splitlines()
            source_lines = node.content.splitlines(keepends=True)
            first = next(
                (
                    i
                    for i, line in enumerate(source_lines)
                    if code_lines and line.strip() == code_lines[0].strip()
                ),
                None,
            )
            if first is not None and len(node.spans) == 1:
                margin = max(
                    0,
                    len(source_lines[first])
                    - len(source_lines[first].lstrip())
                    - (len(code_lines[0]) - len(code_lines[0].lstrip())),
                )
                base = node.spans[0].start_offset
                offset = base
                ranges = []
                for i, line in enumerate(source_lines):
                    if first <= i < first + len(code_lines):
                        ranges.append(
                            SourceSpan(
                                offset + min(margin, len(line) - len(line.lstrip(" "))),
                                offset + len(line),
                            )
                        )
                    offset += len(line)
                return [
                    _Atom(
                        a.text
                        if any(
                            r.start_offset <= span.start_offset
                            and span.end_offset <= r.end_offset
                            for r in ranges
                            for span in a.spans
                        )
                        else "",
                        a.spans,
                        "body"
                        if any(
                            r.start_offset <= span.start_offset
                            and span.end_offset <= r.end_offset
                            for r in ranges
                            for span in a.spans
                        )
                        else "syntax",
                    )
                    for a in atoms
                ]
        if normalized.source_format == "markdown":
            match = re.match(r" {0,3}(`{3,}|~{3,})[^\r\n]*\r?\n", node.content)
            if match:
                opening = match.end()
                closing = re.search(
                    r"(?m)^ {0,3}"
                    + re.escape(match[1][0])
                    + "{"
                    + str(len(match[1]))
                    + r",}[ \t]*\r?$",
                    node.content[opening:],
                )
                end = opening + closing.start() if closing else len(node.content)
                # Markdown code nodes have one precise enclosing source span.
                base = node.spans[0].start_offset
                return [
                    _Atom(
                        a.text
                        if all(
                            base + opening <= s.start_offset
                            and s.end_offset <= base + end
                            for s in a.spans
                        )
                        else "",
                        a.spans,
                        a.role
                        if all(
                            base + opening <= s.start_offset
                            and s.end_offset <= base + end
                            for s in a.spans
                        )
                        else "syntax",
                    )
                    for a in atoms
                ]
        return atoms

    def _trim_atoms(self, atoms: list[_Atom]) -> list[_Atom]:
        raw = "".join(a.text for a in atoms)
        start, stop = len(raw) - len(raw.lstrip()), len(raw.rstrip())
        offset = 0
        result = []
        for atom in atoms:
            a, b = max(0, start - offset), min(len(atom.text), stop - offset)
            text = atom.text[a : max(a, b)]
            result.append(_Atom(text, atom.spans, atom.role if text else "syntax"))
            offset += len(atom.text)
        return result

    def _repeat(self, kind: str, atoms: list[_Atom]) -> dict[str, Any]:
        raw = "".join(a.text for a in atoms)
        text = raw.strip()
        leading = len(raw) - len(raw.lstrip())
        mapping = []
        offset = 0
        for atom in atoms:
            start = max(0, min(len(text), offset - leading))
            end = max(start, min(len(text), offset + len(atom.text) - leading))
            if atom.spans:
                mapping.append(
                    {
                        "output_start": start,
                        "output_end": end,
                        "source_spans": [_span(s) for s in atom.spans],
                        "role": "repeated_context",
                    }
                )
            offset += len(atom.text)
        return {
            "kind": kind,
            "text": text,
            "source_spans": [
                _span(s) for s in _coalesce([s for a in atoms for s in a.spans])
            ],
            "serialization_map": mapping,
        }

    def _text_prefix(self, atoms: list[_Atom], length: int) -> list[_Atom]:
        result = []
        for atom in atoms:
            if length <= 0:
                break
            result.append(_Atom(atom.text[:length], atom.spans, atom.role))
            length -= len(atom.text)
        return result

    def _structured(
        self, normalized: NormalizedDocument, node: NormalizedNode, template: _Piece
    ) -> list[_Piece]:
        wanted = "table_row" if node.kind == "table" else "list_item"
        children = [
            c
            for c in node.walk()
            if c.kind == wanted and (node.kind == "table" or c.parent_id == node.id)
        ]
        # Conservative overlapping cell/row locations cannot be independent slices.
        precise = all(c.spans and len(c.spans) == 1 for c in children)
        if precise:
            intervals = [c.spans[0] for c in children]
            precise = all(
                a.end_offset <= b.start_offset for a, b in zip(intervals, intervals[1:])
            )
        if not children or not precise:
            template.lost = True
            template.reason = "conservative_structure_fallback"
            return self._split_atoms(template)
        headers = (
            [c for c in children if c.metadata.get("header")]
            if node.kind == "table"
            else []
        )
        # Header source and delimiter syntax travel with the first row; repeated
        # text is source-backed context in later fragments, never primary coverage.
        header_atoms = [a for h in headers for a in self._render(normalized, h)]
        if headers and normalized.source_format == "markdown":
            end = headers[-1].spans[-1].end_offset
            body_rows = [c for c in children if c not in headers]
            if body_rows:
                stop = body_rows[0].spans[0].start_offset
                header_atoms.extend(
                    _Atom(c, (SourceSpan(i, i + 1),), "syntax")
                    for i, c in enumerate(normalized.source[end:stop], end)
                )
        if node.kind == "table" and normalized.source_format == "html" and not headers:
            columns = len(children[0].metadata.get("cells", [])) if children else 0
            if columns:
                labels = "|" + "".join(f" Column {i + 1} |" for i in range(columns))
                schema = "|" + " --- |" * columns
                header_atoms = [_Atom(labels + "\n" + schema, (), "synthetic")]
        header_atoms = self._trim_atoms(header_atoms)
        if node.kind == "table":
            captions = [c for c in node.walk() if c.kind == "caption"]
            caption_atoms = [a for c in captions for a in self._render(normalized, c)]
            caption_atoms = self._trim_atoms(caption_atoms)
            if any(a.text for a in caption_atoms):
                header_atoms = [*caption_atoms, _Atom("\n", ()), *header_atoms]
        header_text = "".join(a.text for a in header_atoms)
        rows = [c for c in children if c not in headers]
        if not rows:
            template.lost = True
            return self._split_atoms(template)
        result: list[_Piece] = []
        current: _Piece | None = None
        for row in rows:
            row_atoms = self._render(normalized, row)
            prefix = (header_text + "\n") if header_text else ""
            candidate = _Piece(
                row_atoms,
                template.section,
                template.path,
                kind=node.kind,
                reason=f"{node.kind}_boundary",
                prefix=prefix,
                sections=template.sections,
                structure={
                    "node": _structure_attributes(node),
                    "row": _structure_attributes(row),
                },
            )
            if header_text:
                candidate.repeats = [self._repeat("table_header", header_atoms)]
            if self._budget(candidate) > self.options["max_tokens"]:
                if current is not None:
                    result.append(current)
                    current = None
                candidate.lost = True
                if node.kind == "list" and self.options["preserve_list_hierarchy"]:
                    raw_label = "".join(a.text for a in row_atoms)
                    leading = len(raw_label) - len(raw_label.lstrip("\r\n"))
                    label = raw_label[leading:].splitlines()[0]
                    if self._count(label) > self.options["max_overlap_tokens"]:
                        marker = re.match(r"\s*(?:[-+*]|\d+[.)])\s+", label)
                        label = marker.group(0) if marker else ""
                    label_atoms = self._text_prefix(row_atoms, leading + len(label))
                    repeat = self._repeat("list_parent", label_atoms)
                    # Marker-only context retains the separating space in display.
                    candidate.prefix = repeat["text"] + "\n" if repeat["text"] else ""
                    candidate.repeats = [repeat] if repeat["text"] else []
                result.extend(self._split_atoms(candidate))
                continue
            if current is not None:
                combined = _Piece(
                    [*current.atoms, _Atom("\n", ()), *candidate.atoms],
                    template.section,
                    template.path,
                    kind=node.kind,
                    reason=f"{node.kind}_boundary",
                    prefix=current.prefix,
                    repeats=current.repeats,
                    sections=template.sections,
                    structure={
                        "node": _structure_attributes(node),
                        "rows": [
                            *current.structure.get(
                                "rows", [current.structure.get("row", {})]
                            ),
                            _structure_attributes(row),
                        ],
                    },
                )
                if self._budget(combined) <= self.options["target_tokens"]:
                    current = combined
                    continue
                result.append(current)
            current = candidate
        if current is not None:
            result.append(current)
        # The header/caption/schema is primary exactly once, with its actual
        # prefix display mapping, and repeated context in subsequent pieces.
        if header_atoms:
            result[0].context_primary = header_atoms
        # Allocate remaining format delimiters by their actual intervals.
        covered = _coalesce(
            [s for p in result for a in [*p.context_primary, *p.atoms] for s in a.spans]
        )
        extra = [
            a
            for a in template.atoms
            if a.spans
            and not any(
                s.start_offset <= a.spans[0].start_offset
                and a.spans[0].end_offset <= s.end_offset
                for s in covered
            )
        ]
        # Source text not represented by rows/header is layout syntax only for
        # HTML. For other formats a conservative original-source fallback is safer.
        if normalized.source_format != "html" and any(a.text.strip() for a in extra):
            template.lost = True
            template.reason = "conservative_structure_fallback"
            return self._split_atoms(template)
        result[0].atoms.extend(_Atom("", a.spans, "syntax") for a in extra)
        return result

    def _split_atoms(self, template: _Piece, prose: bool = False) -> list[_Piece]:
        atoms = template.atoms
        text = "".join(a.text for a in atoms)
        safe = list(self.tokenizer.boundaries(text))
        if (
            not safe
            or safe[0] != 0
            or safe[-1] != len(text)
            or any(type(b) is not int for b in safe)
            or any(a >= b for a, b in zip(safe, safe[1:]))
        ):
            raise ChunkingError(
                "Tokenizer boundaries must be ordered Unicode offsets including 0 and length"
            )
        safe_set = set(safe)
        offsets = [0]
        for atom in atoms:
            offsets.append(offsets[-1] + len(atom.text))
        candidates = [i for i in range(1, len(atoms) + 1) if offsets[i] in safe_set]
        token_starts = _builtin_token_starts(self.tokenizer, text)
        sentence_ends = {m.end() for m in re.finditer(r"[.!?](?:[\"')\]]*)\s+", text)}
        line_ends = {m.end() for m in re.finditer(r"\n", text)}
        logical = (
            {m.start() for m in re.finditer(r"(?m)^(?:async )?(?:def |class )", text)}
            if template.kind == "code_block" and self.options["preserve_code_blocks"]
            else set()
        )
        paragraphs = {m.end() for m in re.finditer(r"\n\s*\n", text)}
        result: list[_Piece] = []
        cursor = 0
        while cursor < len(atoms):
            remaining = _Piece(
                atoms[cursor:],
                template.section,
                template.path,
                kind=template.kind,
                reason="adaptive_split",
                prefix=template.prefix,
                suffix=template.suffix,
                repeats=list(template.repeats),
                sections=template.sections,
                lost=template.lost,
                structure=template.structure,
            )
            overlap_atoms: list[_Atom] = []
            if prose and result and self.options["overlap_tokens"]:
                previous = "".join(a.text for a in result[-1].atoms)
                starts = [
                    0,
                    *(m.end() for m in re.finditer(r"[.!?](?:[\"')\]]*)\s+", previous)),
                ]
                for start in starts:
                    suffix = previous[start:]
                    if (
                        suffix.strip()
                        and re.search(r"[.!?][\"')\]]*\s*$", suffix)
                        and self._count(suffix) <= self.options["overlap_tokens"]
                    ):
                        offset = 0
                        overlap_atoms = []
                        for atom in result[-1].atoms:
                            if offset >= start:
                                overlap_atoms.append(atom)
                            offset += len(atom.text)
                        break
                if overlap_atoms:
                    remaining.prefix = template.prefix + "".join(
                        a.text for a in overlap_atoms
                    )
                    remaining.overlap = _coalesce(
                        [s for a in overlap_atoms for s in a.spans]
                    )
                    remaining.repeats.append(self._repeat("overlap", overlap_atoms))
            base_metadata = self._base_metadata(remaining)
            counts: dict[int, tuple[bool, int]] = {}

            def candidate(end: int) -> tuple[bool, int]:
                if end not in counts:
                    display = (
                        remaining.prefix
                        + text[offsets[cursor] : offsets[end]]
                        + remaining.suffix
                    ).strip()
                    counts[end] = (
                        bool(display),
                        self._count(build_embedding_text(display, base_metadata)),
                    )
                return counts[end]

            if candidate(len(atoms))[1] <= self.options["max_tokens"]:
                result.append(remaining)
                break
            chosen: tuple[int, _Piece] | None = None
            # Exact tests of full serializations support nonadditive/nonmonotonic
            # tokenizers. Structure boundaries win, then largest safe prefix.
            groups = [
                logical,
                paragraphs,
                sentence_ends
                if prose and self.options["semantic_mode"] in {"heuristic", "llm"}
                else set(),
                line_ends
                if template.kind != "code_block" or self.options["preserve_code_blocks"]
                else set(),
                None,
            ]
            # Whole-body token offsets propose a window, never prove a budget.
            # Context and truncation can change BPE counts: recount complete text.
            context_count = (
                self._count(
                    build_embedding_text(
                        (remaining.prefix + remaining.suffix).strip(), base_metadata
                    )
                )
                if token_starts is not None
                else 0
            )

            def trials(limit: int, exhaustive: bool = False) -> Iterator[int]:
                cap = len(atoms)
                bounded = token_starts is not None and not exhaustive
                if bounded:
                    start = bisect_right(token_starts, offsets[cursor]) - 1
                    token_end = min(
                        len(token_starts) - 1, start + max(1, limit - context_count)
                    )
                    cap = max(
                        cursor + 1, bisect_right(offsets, token_starts[token_end]) - 1
                    )
                for group in groups:
                    ends = [
                        end
                        for end in candidates[
                            bisect_right(candidates, cursor) : bisect_right(
                                candidates, cap
                            )
                        ]
                        if group is None or offsets[end] in group
                    ]
                    index = len(ends) - 1
                    while index >= 0:
                        yield ends[index]
                        # Shrinking skips trials, not counts; no monotonic inference.
                        index = (
                            min(index - 1, int(index * 0.9)) if bounded else index - 1
                        )

            limits = (self.options["target_tokens"], self.options["max_tokens"])
            searches = [(limit, False) for limit in limits]
            if token_starts is not None:
                # Rare mandatory-context pressure must not create a false infeasibility.
                searches.extend((limit, True) for limit in limits)
            for limit, exhaustive in searches:
                for end in trials(limit, exhaustive):
                    visible, count = candidate(end)
                    if not visible or count > limit:
                        continue
                    part = _Piece(
                        atoms[cursor:end],
                        template.section,
                        template.path,
                        kind=template.kind,
                        reason="adaptive_split",
                        prefix=remaining.prefix,
                        suffix=remaining.suffix,
                        repeats=remaining.repeats,
                        overlap=remaining.overlap,
                        sections=template.sections,
                        lost=template.lost,
                        structure=template.structure,
                    )
                    part.lost = template.lost or (
                        template.kind not in {"paragraph", "raw_block", "blockquote"}
                        and offsets[end] not in line_ends
                        and offsets[end] not in logical
                    )
                    chosen = end, part
                    break
                if chosen:
                    break
            if chosen is None and overlap_atoms:
                # Optional overlap can never prevent primary source progress.
                remaining.prefix = template.prefix
                remaining.overlap = []
                remaining.repeats = list(template.repeats)
                retry = _Piece(
                    atoms[cursor:],
                    template.section,
                    template.path,
                    kind=template.kind,
                    prefix=template.prefix,
                    suffix=template.suffix,
                    repeats=list(template.repeats),
                    sections=template.sections,
                    lost=template.lost,
                    structure=template.structure,
                )
                result.extend(self._split_atoms(retry, prose=False))
                break
            if chosen is None:
                raise ChunkingError(
                    "Mandatory context or indivisible serialized source atom cannot fit max_tokens"
                )
            cursor, part = chosen
            result.append(part)
        return result

    def _metadata(
        self,
        normalized: NormalizedDocument,
        piece: _Piece,
        index: int,
        unit_ends: list[int],
    ) -> dict[str, Any]:
        text = piece.text
        raw = piece.prefix + "".join(a.text for a in piece.atoms) + piece.suffix
        leading = len(raw) - len(raw.lstrip())
        offset = 0 if piece.context_primary else len(piece.prefix)
        mapping: list[dict[str, Any]] = []
        retained: list[dict[str, Any]] = []
        for atom in [*piece.context_primary, *piece.atoms]:
            if atom is piece.atoms[0] and piece.context_primary:
                offset = len(piece.prefix)
            start = max(0, min(len(text), offset - leading))
            end = max(start, min(len(text), offset + len(atom.text) - leading))
            if atom.spans:
                mapping.append(
                    {
                        "output_start": start,
                        "output_end": end,
                        "source_spans": [_span(s) for s in atom.spans],
                        "role": atom.role,
                    }
                )
            offset += len(atom.text)
        mapping = _compact_maps(mapping, normalized.source, text)
        for value in mapping:
            for span in value["source_spans"]:
                for unit_index in range(
                    bisect_right(unit_ends, span["start_offset"]), len(normalized.units)
                ):
                    unit = normalized.units[unit_index]
                    if unit.span.start_offset >= span["end_offset"]:
                        break
                    a = max(span["start_offset"], unit.span.start_offset)
                    b = min(span["end_offset"], unit.span.end_offset)
                    if a < b:
                        affine = value.get("mapping_kind") == "affine"
                        retained.append(
                            {
                                "unit_id": unit.id,
                                "node_id": unit.node_id,
                                "start_offset": a,
                                "end_offset": b,
                                "role": value["role"],
                                "output_start": value["output_start"]
                                + a
                                - span["start_offset"]
                                if affine
                                else value["output_start"],
                                "output_end": value["output_start"]
                                + b
                                - span["start_offset"]
                                if affine
                                else value["output_end"],
                            }
                        )
        repeats: list[dict[str, Any]] = []
        repeat_cursor = 0
        for repeat in piece.repeats:
            start = raw.find(repeat["text"], repeat_cursor, len(piece.prefix))
            if start < 0:
                raise ChunkingError("Repeated context does not match serialized prefix")
            repeat_cursor = start + len(repeat["text"])
            base = start - leading
            repeats.append(
                {
                    **repeat,
                    "output_start": max(0, base),
                    "output_end": min(len(text), base + len(repeat["text"])),
                    "serialization_map": [
                        {
                            **m,
                            "output_start": max(0, base + m["output_start"]),
                            "output_end": min(
                                len(text), max(0, base + m["output_end"])
                            ),
                        }
                        for m in repeat.get("serialization_map", [])
                    ],
                }
            )
        for repeat in repeats:
            repeat["serialization_map"] = _compact_maps(
                repeat["serialization_map"], normalized.source, text
            )
        return {
            **self._base_metadata(piece),
            "document_id": normalized.document.id,
            "position": index,
            "parent_section_id": piece.section.id,
            "source_section_ids": piece.sections,
            "chunk_type": piece.kind,
            "source_format": normalized.source_format,
            "tokenizer_identity": self.tokenizer.identity,
            "token_count": self._count(text),
            "embedding_token_count": self._budget(piece),
            "source_spans": [
                _span(s)
                for s in _coalesce(
                    [s for a in [*piece.context_primary, *piece.atoms] for s in a.spans]
                )
            ],
            "retained_intervals": retained,
            "serialization_map": mapping,
            "split_reason": piece.reason,
            "repeated_context": repeats,
            "overlap_spans": [_span(s) for s in piece.overlap],
            "atomic_structure_lost": piece.lost,
            "overflow": piece.overflow,
            "overflow_reason": "atomic_block_exceeds_max_tokens"
            if piece.overflow
            else None,
            "structure_metadata": piece.structure,
        }
