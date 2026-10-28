"""Independent validation of persisted structural output, without a database."""

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from html import unescape
import re
from typing import Any

from fastrag.exceptions import ChunkingError
from fastrag.models import Document
from fastrag.normalization import NormalizedDocument


@dataclass(frozen=True, slots=True)
class ChunkValidation:
    """Runtime diagnostics kept outside chunk metadata and identity."""

    chunk_count: int
    retained_characters: int
    embedding_tokens: int
    classification: str
    validation_seconds: float = 0.0
    token_counts: tuple[int, ...] = ()
    merged_chunks: int = 0
    split_chunks: int = 0
    fallback_chunks: int = 0
    oversized_chunks: int = 0
    excluded_characters: int = 0
    normalization_diagnostics: tuple[str, ...] = ()
    cached_chunks: int = 0
    small_chunks: int = 0
    model_calls: int = 0
    model_cache_hits: int = 0
    model_fallbacks: tuple[str, ...] = ()
    operation_seconds: float = 0.0


def _intervals(values: Sequence[dict[str, Any]], limit: int) -> list[tuple[int, int]]:
    result = []
    for value in values:
        a, b = value["start_offset"], value["end_offset"]
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= limit:
            raise ChunkingError("Invalid source interval")
        result.append((a, b))
    return result


def _union(values: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for a, b in sorted(values):
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(b, result[-1][1]))
        else:
            result.append((a, b))
    return result


def validate_structure_chunks(
    normalized: NormalizedDocument,
    pieces: Sequence[Any],
    tokenizer: Any,
    max_tokens: int,
    max_overlap_tokens: int = 50,
    min_preferred_tokens: int = 100,
) -> ChunkValidation:
    """Validate split tuples or cached Documents against authoritative source.

    Raises ChunkingError for invalid persisted metadata. Counts are recomputed;
    neither recorded budgets nor node IDs substitute for interval coverage.
    """
    from fastrag.ingestion.steps.chunking.base import structural_chunk_id
    from fastrag.ingestion.steps.chunking.structure import (
        EMBEDDING_TEXT_VERSION,
        build_embedding_text,
    )

    started = perf_counter()
    normalized.validate()
    nodes = {n.id: n for n in normalized.walk()}
    units = {u.id: u for u in normalized.units}
    if not pieces:
        body = []
        for unit in normalized.units:
            if nodes[unit.node_id].kind == "heading":
                continue
            text = normalized.source[unit.span.start_offset : unit.span.end_offset]
            if normalized.source_format == "html":
                text = unescape(re.sub(r"<[^>]*>", "", text))
            if text.strip():
                body.append(unit)
        if body:
            raise ChunkingError("Missing all structural body output")
        return ChunkValidation(
            0,
            0,
            0,
            "hierarchy_only"
            if any(n.kind == "heading" for n in nodes.values())
            else "empty",
            validation_seconds=perf_counter() - started,
            excluded_characters=sum(
                e.span.end_offset - e.span.start_offset for e in normalized.exclusions
            ),
            normalization_diagnostics=tuple(d.code for d in normalized.diagnostics),
        )
    actual: list[tuple[int, int]] = []
    body_order: list[tuple[int, int]] = []
    tokens = 0
    counts = []
    merged = split = fallback = oversized = cached = 0
    try:
        for index, piece in enumerate(pieces):
            content, m = (
                (piece.content, piece.metadata)
                if isinstance(piece, Document)
                else piece
            )
            if (
                m["embedding_text_version"] != EMBEDDING_TEXT_VERSION
                or m["document_id"] != normalized.document.id
                or m["position"] != index
            ):
                raise ChunkingError("Invalid structure version/document/position")
            if isinstance(piece, Document):
                if (
                    piece.source_id != normalized.document.source_id
                    or m["parent_id"] != normalized.document.id
                    or m["chunk_index"] != index
                ):
                    raise ChunkingError("Invalid chunk lineage")
                if piece.id != structural_chunk_id(
                    normalized.document.id, index, content, m, piece.step_config_hash
                ):
                    raise ChunkingError("Invalid structural chunk ID")
            sections = m["source_section_ids"]
            if not sections or any(
                s not in nodes or nodes[s].kind not in {"section", "document"}
                for s in [m["parent_section_id"], *sections]
            ):
                raise ChunkingError("Unknown source section")
            if len(set(sections)) != len(sections):
                raise ChunkingError("Duplicate source sections")
            chain = []
            current = nodes[m["parent_section_id"]]
            while current:
                if current.kind == "section":
                    chain.append(current.id)
                current = nodes.get(current.parent_id)
            headings = m["heading_ids"]
            if headings and headings != list(reversed(chain)):
                raise ChunkingError("Heading path differs from section ancestry")
            if not len(headings) == len(m["heading_path"]) == len(m["heading_levels"]):
                raise ChunkingError("Inconsistent heading arrays")
            for ident, title, level in zip(
                headings, m["heading_path"], m["heading_levels"]
            ):
                node = nodes.get(ident)
                if (
                    node is None
                    or node.kind not in {"section", "heading"}
                    or node.title != title
                    or node.level != level
                ):
                    raise ChunkingError("Invalid heading path")
            if any(
                i not in nodes or nodes[i].kind not in {"section", "heading"}
                for i in m["displayed_heading_ids"]
            ):
                raise ChunkingError("Unknown displayed heading")

            def coordinates(value):
                a, b = value["output_start"], value["output_end"]
                if (
                    type(a) is not int
                    or type(b) is not int
                    or not 0 <= a <= b <= len(content)
                ):
                    raise ChunkingError("Invalid display coordinates")
                return a, b

            def validate_mapping(value):
                a, b = coordinates(value)
                spans = _intervals(value["source_spans"], len(normalized.source))
                kind = value.get("mapping_kind")
                if kind is not None:
                    if kind != "affine" or len(spans) != 1:
                        raise ChunkingError("Invalid serialization mapping kind")
                    x, y = spans[0]
                    if b - a != y - x or content[a:b] != normalized.source[x:y]:
                        raise ChunkingError("Invalid affine source/display mapping")
                return a, b, spans

            primary = []
            for value in m["retained_intervals"]:
                coordinates(value)
                span = _intervals([value], len(normalized.source))[0]
                unit = units.get(value["unit_id"])
                if (
                    unit is None
                    or value["node_id"] != unit.node_id
                    or not unit.span.start_offset
                    <= span[0]
                    < span[1]
                    <= unit.span.end_offset
                ):
                    raise ChunkingError("Invalid coverage-unit intersection")
                primary.append(span)
            maps = []
            for value in m["serialization_map"]:
                a, b, spans = validate_mapping(value)
                maps.extend(spans)
                if value["role"] == "body":
                    if a == b and any(normalized.source[x:y].strip() for x, y in spans):
                        raise ChunkingError("Invisible primary body")
                    if content[a:b].strip():
                        body_order.extend(spans)
            ordered_maps = sorted(maps)
            if any(a[1] > b[0] for a, b in zip(ordered_maps, ordered_maps[1:])):
                raise ChunkingError("Duplicate primary serialization mapping")
            if _union(primary) != _union(maps) or _union(primary) != _intervals(
                m["source_spans"], len(normalized.source)
            ):
                raise ChunkingError("Source/map/coverage disagreement")
            # Index already independently checked maps by their display identity.
            map_index = {}
            affine_index = {}
            for value in m["serialization_map"]:
                if value.get("mapping_kind") == "affine":
                    span = value["source_spans"][0]
                    affine_index.setdefault(value["role"], []).append(
                        (
                            span["start_offset"],
                            span["end_offset"],
                            value["output_start"],
                        )
                    )
                key = (value["output_start"], value["output_end"], value["role"])
                map_index.setdefault(key, []).extend(
                    (s["start_offset"], s["end_offset"]) for s in value["source_spans"]
                )
            for key, spans in map_index.items():
                spans.sort()
                map_index[key] = ([a for a, _ in spans], spans)
            for role, runs in affine_index.items():
                runs.sort()
                affine_index[role] = ([a for a, _, _ in runs], runs)
            for value in m["retained_intervals"]:
                key = (value["output_start"], value["output_end"], value["role"])
                starts, spans = map_index.get(key, ([], []))
                at = bisect_right(starts, value["start_offset"]) - 1
                exact = at >= 0 and value["end_offset"] <= spans[at][1]
                starts, runs = affine_index.get(value["role"], ([], []))
                at = bisect_right(starts, value["start_offset"]) - 1
                affine = False
                if at >= 0:
                    source_start, source_end, output_start = runs[at]
                    affine = (
                        value["end_offset"] <= source_end
                        and value["output_start"]
                        == output_start + value["start_offset"] - source_start
                        and value["output_end"]
                        == output_start + value["end_offset"] - source_start
                    )
                    # An affine run requires interior agreement even when a
                    # forged coverage entry copies its whole display bounds.
                    if value["end_offset"] <= source_end:
                        exact = False
                if not (exact or affine):
                    raise ChunkingError("Coverage/display mapping disagreement")
            actual.extend(primary)
            count = tokenizer.count(build_embedding_text(content, m))
            if (
                type(count) is not int
                or count < 0
                or count != m["embedding_token_count"]
                or tokenizer.count(content) != m["token_count"]
            ):
                raise ChunkingError("Incorrect independently recomputed token counts")
            if count > max_tokens and not (
                m["overflow"]
                and m["overflow_reason"] == "atomic_block_exceeds_max_tokens"
            ):
                raise ChunkingError("Unflagged embedding overflow")
            if m["overflow"] and count <= max_tokens:
                raise ChunkingError("Incorrect overflow classification")
            tokens += count
            counts.append(count)
            merged += len(sections) > 1
            split += m["split_reason"] not in {
                "section_fits",
                "block_fits",
                "small_sibling_merge",
            }
            fallback += bool(m["atomic_structure_lost"])
            oversized += bool(m["overflow"])
            cached += isinstance(piece, Document) and piece.cached
            overlap = []
            for repeat in m["repeated_context"]:
                a, b = coordinates(repeat)
                if content[a:b] != repeat["text"]:
                    raise ChunkingError("Repeated context differs from display")
                spans = _intervals(repeat["source_spans"], len(normalized.source))
                retained_source = _union(
                    [(u.span.start_offset, u.span.end_offset) for u in normalized.units]
                )
                if any(
                    not any(p <= a < b <= q for p, q in retained_source)
                    for a, b in spans
                ):
                    raise ChunkingError("Repeated context claims excluded source")
                repeat_maps = []
                for value in repeat["serialization_map"]:
                    x, y, mapped_spans = validate_mapping(value)
                    repeat_maps.extend(mapped_spans)
                    if not a <= x <= y <= b:
                        raise ChunkingError("Repeated mapping outside context")
                    if any(
                        not any(p <= s <= e <= q for p, q in spans)
                        for s, e in _intervals(
                            value["source_spans"], len(normalized.source)
                        )
                    ):
                        raise ChunkingError("Repeated source mapping outside context")
                if _union(repeat_maps) != _union(spans):
                    raise ChunkingError("Repeated source/map disagreement")
                if repeat["kind"] == "overlap":
                    overlap.extend(spans)
                    if tokenizer.count(repeat["text"]) > max_overlap_tokens:
                        raise ChunkingError("Overlap exceeds ceiling")
            if _union(overlap) != _union(
                _intervals(m["overlap_spans"], len(normalized.source))
            ):
                raise ChunkingError("Overlap provenance disagreement")
        ordered = sorted(actual)
        if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])) or _union(
            actual
        ) != _union(
            [(u.span.start_offset, u.span.end_offset) for u in normalized.units]
        ):
            raise ChunkingError("Missing or duplicate primary coverage")
        if any(a[1] > b[0] for a, b in zip(body_order, body_order[1:])):
            raise ChunkingError("Primary body source out of order")
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise ChunkingError(f"Malformed structural metadata: {error}") from error
    return ChunkValidation(
        len(pieces),
        sum(b - a for a, b in actual),
        tokens,
        "content",
        perf_counter() - started,
        tuple(counts),
        merged,
        split,
        fallback,
        oversized,
        sum(e.span.end_offset - e.span.start_offset for e in normalized.exclusions),
        tuple(d.code for d in normalized.diagnostics),
        cached,
        sum(n < min_preferred_tokens for n in counts),
    )
