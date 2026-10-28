"""Chunk inspection preserves text, lineage, counters and pipeline failures."""

import builtins
import io
import json

import pytest
from helpers import EventPipeline

from fastrag.ingestion.events import DocumentChunked, ItemFailed, StepFailed
from fastrag.ingestion.logger import ChunkLogger, Logger
from fastrag.models import Document


def fixture_event():
    text = (
        "# Héading [bold]literal[/bold]\n```python\n"
        + "print('🍏')\n" * 120
        + "```\nEND"
    )
    parent = Document(
        "parent", "source", text, metadata={"source_uri": "https://example.org"}
    )
    chunks = [
        Document(
            "chunk",
            "source",
            text,
            step_config_hash="hash",
            cached=True,
            metadata={
                "headings": ["Héading"],
                "nested": {"values": [1, False]},
                "source_uri": "https://example.org",
            },
        ),
        Document("empty", "source", ""),
    ]
    return DocumentChunked(parent, chunks, step="HeadingChunker", event_name="Chunking")


async def test_jsonl_complete_records_and_summary():
    event = fixture_event()
    empty_parent = Document("empty-parent", "other-source", "")
    stream = io.StringIO()
    logger = ChunkLogger(
        EventPipeline([event, DocumentChunked(empty_parent)]),
        format="jsonl",
        stream=stream,
        configuration={"cache": False},
    )
    assert isinstance(logger, Logger)
    assert not stream.getvalue()
    await logger.run()
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert records[0] == {"type": "configuration", "cache": False}
    chunks = [record for record in records if record["type"] == "chunk"]
    assert chunks[0]["text"] == event.chunks[0].content
    assert chunks[0]["metadata"] == event.chunks[0].metadata
    assert chunks[0]["parent_document_id"] == "parent"
    assert chunks[0]["position"] == "1/2"
    assert chunks[0]["step_config_hash"] == "hash"
    assert chunks[0]["cached"] is True
    summary = records[-1]
    assert summary["sources"] == 2
    assert summary["documents"] == 2
    assert summary["chunks"] == logger.chunk_count == 2
    assert summary["cached_chunks"] == 1
    text = event.chunks[0].content
    assert summary["total_characters"] == len(text)
    assert summary["total_bytes"] == len(text.encode("utf-8"))
    assert summary["total_words"] == len(text.split())
    assert summary["total_lines"] == len(text.splitlines())
    assert summary["min_characters"] == 0
    assert summary["max_characters"] == len(text)
    assert summary["mean_characters"] == len(text) / 2
    assert summary["status"] == "completed"
    await logger.run()
    assert logger.chunk_count == 2  # A reused logger resets counters.


async def test_rich_literal_content_and_full_metadata(monkeypatch):
    pytest.importorskip("rich")
    monkeypatch.setenv("COLUMNS", "80")
    stream = io.StringIO()
    await ChunkLogger(EventPipeline([fixture_event()]), stream=stream).run()
    output = stream.getvalue()
    for label in (
        "Configuration",
        "Document",
        "Chunk 1",
        "Chunk 2",
        "Metadata (complete)",
        "Content (complete)",
        "Summary",
    ):
        assert label in output
    assert "[bold]literal[/bold]" in output
    assert "```python" in output
    assert output.count("print('🍏')") == 120
    assert "END" in output
    assert '"nested"' in output and '"values"' in output and "false" in output
    assert "\x1b" not in output  # Redirected logs have no ANSI escapes.


@pytest.mark.parametrize("fatal", [False, True])
async def test_failures_and_partial_summary(fatal):
    error = RuntimeError("chunking failed")
    events = [fixture_event(), ItemFailed(step="chunker", uri="fixture", error=error)]
    if fatal:
        events.append(StepFailed(step="chunker", error=error))
    stream = io.StringIO()
    logger = ChunkLogger(
        EventPipeline(events, error if fatal else None), stream=stream, format="jsonl"
    )
    if fatal:
        with pytest.raises(RuntimeError, match="chunking failed"):
            await logger.run()
    else:
        await logger.run()
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert any(
        record["type"] == "error" and "chunking failed" in record["error"]
        for record in records
    )
    assert records[-1]["chunks"] == 2
    assert records[-1]["failed_items"] == 1
    assert records[-1]["failed_steps"] == int(fatal)
    assert records[-1]["status"] == (
        "interrupted" if fatal else "completed with errors"
    )


async def test_rich_optional_dependency(monkeypatch):
    original_import = builtins.__import__

    def without_rich(name, *args, **kwargs):
        if name.startswith("rich"):
            raise ImportError("no Rich installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_rich)
    with pytest.raises(ImportError, match=r"fastrag-ai\[logging\]"):
        await ChunkLogger(EventPipeline([]), stream=io.StringIO()).run()
    stream = io.StringIO()
    await ChunkLogger(EventPipeline([]), format="jsonl", stream=stream).run()
    assert json.loads(stream.getvalue().splitlines()[-1])["chunks"] == 0


def test_invalid_format():
    with pytest.raises(ValueError, match="format"):
        ChunkLogger(EventPipeline([]), format="unknown")
