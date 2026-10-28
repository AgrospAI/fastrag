"""Lossless source/display runs remain independently verifiable after transport."""

import copy
import json
from dataclasses import replace

import pytest
from helpers import generated_source

from fastrag.exceptions import ChunkingError, VectorStoreError
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext
from fastrag.ingestion.steps.chunking import (
    StructureAwareChunker,
    validate_structure_chunks,
)
from fastrag.models import Document
from fastrag.vector_store.milvus import MilvusVectorStore


def document(source, fmt="markdown"):
    return Document(
        "transport-parent", "local", source, metadata={"content_format": fmt}
    )


def check(doc, engine, pieces):
    return validate_structure_chunks(
        engine.normalizers.normalize(doc), pieces, engine.tokenizer, 600
    )


def test_long_identity_maps_are_compact_and_detect_interior_tampering():
    doc = document(
        "# Guide\n\n" + "Ordinary documentation words with useful details. " * 220
    )
    engine = StructureAwareChunker(cache=False)
    pieces = json.loads(json.dumps(list(engine.split(doc))))
    check(doc, engine, pieces)
    assert max(len(json.dumps(meta).encode()) for _, meta in pieces) < 16000
    assert max(len(meta["serialization_map"]) for _, meta in pieces) < 15
    assert max(len(meta["retained_intervals"]) for _, meta in pieces) < 15
    run = next(
        m
        for m in pieces[0][1]["serialization_map"]
        if m.get("mapping_kind") == "affine"
    )
    assert run["output_end"] - run["output_start"] > 10
    for fault in ("display", "source", "coverage", "duplicate"):
        bad = copy.deepcopy(pieces)
        meta = bad[0][1]
        if fault == "display":
            m = next(
                m
                for m in meta["serialization_map"]
                if m.get("mapping_kind") == "affine"
            )
            m["output_end"] -= 1
        elif fault == "source":
            m = next(
                m
                for m in meta["serialization_map"]
                if m.get("mapping_kind") == "affine"
            )
            m["source_spans"][0]["start_offset"] += 1
        elif fault == "coverage":
            m = next(
                m
                for m in meta["retained_intervals"]
                if m["output_end"] > m["output_start"] + 2
            )
            m["output_start"] += 1
        else:
            meta["serialization_map"].append(
                copy.deepcopy(meta["serialization_map"][0])
            )
        with pytest.raises(ChunkingError):
            check(doc, engine, bad)


@pytest.mark.parametrize("fmt", ["markdown", "rst", "html"])
def test_rich_transport_preserves_entities_nonaffine_and_unit_intersections(fmt):
    source = generated_source(8, fmt)
    if fmt == "html":
        source += "<p>Before &amp; after &#x1f600; entity.</p>"
    doc = document(source, fmt)
    engine = StructureAwareChunker(cache=False, target_tokens=100, max_tokens=600)
    pieces = json.loads(json.dumps(list(engine.split(doc))))
    check(doc, engine, pieces)
    assert max(len(json.dumps(meta).encode()) for _, meta in pieces) < 65536
    for text, meta in pieces:
        for m in meta["serialization_map"]:
            if m.get("mapping_kind") == "affine":
                s = m["source_spans"][0]
                assert (
                    text[m["output_start"] : m["output_end"]]
                    == source[s["start_offset"] : s["end_offset"]]
                )
    if fmt == "html":
        entity = next(
            m
            for _, meta in pieces
            for m in meta["serialization_map"]
            if any(
                source[s["start_offset"] : s["end_offset"]] == "&amp;"
                for s in m["source_spans"]
            )
        )
        assert "mapping_kind" not in entity


async def test_metadata_cap_checked_before_batch_with_exact_utf8_bytes(monkeypatch):
    store = MilvusVectorStore(url="http://unused", collection="transport_test")
    doc = replace(document("ok"), embedding=[1.0], metadata={"huge": "界" * 24000})
    calls = []
    monkeypatch.setattr(store, "_require_client", lambda: calls.append(True))
    with pytest.raises(
        VectorStoreError, match=r"metadata.*72011.*65536.*transport_test"
    ):
        await store.upsert([doc])
    assert not calls


@pytest.mark.parametrize("fmt", ["markdown", "html", "code", "shortcode"])
async def test_rich_long_transport_postgres_replay_and_milvus_search(
    cache, vector_store, fmt, monkeypatch
):
    source = (
        generated_source(8, fmt)
        + ("\n\n" if fmt == "markdown" else "<p>")
        + "Useful long documentation prose. " * 400
        + ("" if fmt == "markdown" else "</p>")
    )
    if fmt in {"code", "shortcode"}:
        source = (
            "# Huge code\n\n```python\n"
            + ("value = '" + "Useful code documentation " * 40 + "'\n") * 70
            + "```\n"
        )
        if fmt == "shortcode":
            source = (
                "# Huge code\n\n```python\n"
                + "value = 'Useful code documentation'\n" * 1800
                + "```\n"
            )
    doc = document(source, "markdown" if fmt in {"code", "shortcode"} else fmt)
    engine = StructureAwareChunker(cache=cache)
    context = ExecutionContext("transport", Policy(on_error="raise"), cache=cache)

    async def stream():
        yield doc

    current_config = engine.config
    monkeypatch.setattr(
        engine, "config", lambda: {**current_config(), "engine": "structure:v4"}
    )
    assert context.policy.timeout == 30.0
    old = [c async for c in engine.evaluate_items(stream(), context)]
    monkeypatch.setattr(engine, "config", current_config)
    fresh = [c async for c in engine.evaluate_items(stream(), context)]
    assert engine.config()["engine"] == "structure:v5"
    assert all(not c.cached for c in fresh)
    assert [c.content for c in old] == [c.content for c in fresh]
    assert {c.id for c in old}.isdisjoint(c.id for c in fresh)
    from helpers import RecordingEmbedder

    embedder = RecordingEmbedder(cache=cache)
    _ = [c async for c in embedder.evaluate_items(_stream(old), context)]
    new_vectors = [c async for c in embedder.evaluate_items(_stream(fresh), context)]
    assert all(not c.cached for c in new_vectors)
    warm = [c async for c in engine.evaluate_items(stream(), context)]
    assert warm == [replace(c, cached=True) for c in fresh]
    assert all(len(json.dumps(c.metadata).encode()) < 65536 for c in fresh)
    embedded = [replace(c, embedding=[1.0, 0.0, 0.0, 0.0]) for c in fresh]
    await vector_store.ensure_ready(4)
    await vector_store.upsert(embedded)
    hits = await vector_store.search([1.0, 0.0, 0.0, 0.0], k=len(fresh))
    hits.sort(key=lambda c: c.metadata["position"])
    assert [c.id for c in hits] == [c.id for c in fresh]
    assert [
        (c.content, {k: c.metadata[k] for k in original.metadata})
        for c, original in zip(hits, fresh)
    ] == [(c.content, dict(c.metadata)) for c in fresh]
    check(doc, engine, [(c.content, c.metadata) for c in hits])
    from fastrag.retrieval import SectionIndex

    assert all(
        SectionIndex([engine.normalizers.normalize(doc)]).resolve(c) for c in hits
    )


def test_repeated_affine_runs_keep_final_coordinates_and_reject_tampering():
    source = (
        "# Table\n\n| Long header | Value |\n| --- | --- |\n"
        + "| Useful data repeated in long rows | 42 |\n" * 100
    )
    doc = document(source)
    engine = StructureAwareChunker(
        cache=False, target_tokens=80, max_tokens=100, min_preferred_tokens=40
    )
    pieces = json.loads(json.dumps(list(engine.split(doc))))
    validate_structure_chunks(
        engine.normalizers.normalize(doc), pieces, engine.tokenizer, 100
    )
    repeats = [
        (i, repeat)
        for i, (_, meta) in enumerate(pieces)
        for repeat in meta["repeated_context"]
        if repeat["source_spans"]
    ]
    assert repeats
    i, repeat = repeats[0]
    maps = repeat["serialization_map"]
    assert len(maps) < 6
    assert any(m.get("mapping_kind") == "affine" for m in maps)
    for field in ("output_end", "mapping_kind", "source_spans"):
        bad = copy.deepcopy(pieces)
        m = next(
            m
            for m in bad[i][1]["repeated_context"][0]["serialization_map"]
            if m.get("mapping_kind") == "affine"
        )
        if field == "output_end":
            m[field] -= 1
        elif field == "mapping_kind":
            m[field] = "invented"
        else:
            m[field][0]["start_offset"] += 1
        with pytest.raises(ChunkingError):
            validate_structure_chunks(
                engine.normalizers.normalize(doc), bad, engine.tokenizer, 100
            )


def test_affine_run_across_units_retains_partial_unit_output_bounds():
    doc = document(
        "# Guide\n\nFirst paragraph with text.\n\nSecond paragraph with text.\n"
    )
    engine = StructureAwareChunker(cache=False)
    pieces = list(engine.split(doc))
    check(doc, engine, pieces)
    saw_partial = False
    for _, meta in pieces:
        for run in meta["serialization_map"]:
            if run.get("mapping_kind") != "affine":
                continue
            span = run["source_spans"][0]
            intersections = [
                v
                for v in meta["retained_intervals"]
                if span["start_offset"]
                <= v["start_offset"]
                < v["end_offset"]
                <= span["end_offset"]
            ]
            for v in intersections:
                assert (
                    v["output_start"]
                    == run["output_start"] + v["start_offset"] - span["start_offset"]
                )
                assert (
                    v["output_end"]
                    == run["output_start"] + v["end_offset"] - span["start_offset"]
                )
            saw_partial |= len(intersections) > 1
    assert saw_partial


async def _stream(items):
    for item in items:
        yield item


@pytest.mark.parametrize("byte_count", [65536, 65537])
async def test_metadata_exact_json_limit_boundary(monkeypatch, byte_count):
    from pymilvus.client.entity_helper import convert_to_json

    store = MilvusVectorStore(url="http://unused")
    metadata = {"x": "界" * 21000 + "x" * (byte_count - 63008)}
    assert len(convert_to_json(metadata)) == byte_count
    doc = replace(document("body"), metadata=metadata, embedding=[1.0])
    calls = []

    class Client:
        async def upsert(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(store, "_require_client", lambda: Client())
    if byte_count > 65536:
        with pytest.raises(VectorStoreError, match="65537 bytes"):
            await store.upsert([doc])
        assert not calls
    else:
        await store.upsert([doc])
        assert calls[0]["data"][0]["metadata"] == metadata


@pytest.mark.parametrize("kind", ["prose", "code"])
def test_huge_node_interpretation_is_referenced_without_parent_text_copies(kind):
    body = (
        ("Ordinary documentation content with details " * 30 + ".\n") * 60
        if kind == "prose"
        else ("value = '" + "Useful code documentation " * 40 + "'\n") * 70
    )
    source = "# Huge\n\n" + (
        body if kind == "prose" else "```python\n" + body + "```\n"
    )
    assert len(body.encode()) > 60000
    doc = document(source)
    engine = StructureAwareChunker(cache=False)
    normalized = engine.normalizers.normalize(doc)
    pieces = list(engine.split_normalized(normalized))
    check(doc, engine, pieces)
    assert len(pieces) > 2
    assert max(len(json.dumps(m).encode()) for _, m in pieces) < 16000
    nodes = {n.id: n for n in normalized.walk()}
    referenced = [m["structure_metadata"] for _, m in pieces if m["structure_metadata"]]
    assert referenced
    for projected in referenced:
        assert projected["node_id"] in nodes
        assert (
            not {"inline", "text", "code", "links", "images", "references"}
            & projected.keys()
        )
        if kind == "code":
            assert projected["language"] == "python"
            assert len(nodes[projected["node_id"]].metadata["code"]) == len(body)
        else:
            assert nodes[projected["node_id"]].metadata["inline"]


def test_structure_projection_keeps_table_attributes_and_parent_node_metadata():
    source = (
        "<h1>Table</h1><table>"
        + '<tr><td rowspan="2" colspan="3">Useful row content with details</td><td>Value</td></tr>'
        * 60
        + "</table>"
    )
    doc = document(source, "html")
    engine = StructureAwareChunker(
        cache=False, target_tokens=100, max_tokens=150, min_preferred_tokens=50
    )
    normalized = engine.normalizers.normalize(doc)
    pieces = list(engine.split_normalized(normalized))
    validate_structure_chunks(normalized, pieces, engine.tokenizer, 150)
    nodes = {n.id: n for n in normalized.walk()}
    rows = []
    for _, meta in pieces:
        structure = meta["structure_metadata"]
        rows.extend(
            structure.get("rows", [structure["row"]] if "row" in structure else [])
        )
    assert rows
    for row in rows:
        assert row["cells"][0]["attrs"]["rowspan"] == "2"
        assert row["cells"][0]["attrs"]["colspan"] == "3"
        assert "text" not in row["cells"][0]
        assert (
            nodes[row["node_id"]].metadata["cells"][0]["text"]
            == "Useful row content with details"
        )
