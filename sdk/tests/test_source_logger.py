"""Lossless source inspection, duplicate fetch events and failure behavior."""

import base64
import io
import json

import pytest
from helpers import EventPipeline

from fastrag.ingestion import Pipeline
from fastrag.ingestion.events import DocumentFetched, ItemFailed, StepFailed
from fastrag.ingestion.logger import SourceLogger
from fastrag.ingestion.models import Source
from fastrag.ingestion.steps.fetching import LocalFileFetcher, Sources


@pytest.mark.parametrize(
    "content", [b"", "Héllo [bold]literal[/bold]".encode(), b"\xff\x00\x80"]
)
async def test_lossless_content_and_deduplication(content):
    source = Source(
        "source",
        "file:///test",
        content,
        "application/octet-stream",
        "hash",
        metadata={"nested": [False, 1]},
        cached=True,
    )
    stream = io.StringIO()
    logger = SourceLogger(
        EventPipeline([DocumentFetched(source), DocumentFetched(source)]),
        format="jsonl",
        stream=stream,
    )
    await logger.run()
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(records) == 3
    assert base64.b64decode(records[1]["content_base64"]) == content
    assert records[1]["metadata"] == source.metadata
    assert records[-1]["sources"] == logger.source_count == 1
    assert records[-1]["total_bytes"] == len(content)
    assert records[-1]["cached_items"] == 1
    await logger.run()
    assert logger.source_count == 1


async def test_real_sources_aggregate(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("[bold]literal[/bold]\n```python\nprint('🍏')\n```")
    stream = io.StringIO()
    logger = SourceLogger(
        Pipeline(Sources([LocalFileFetcher(patterns=[str(path)])])), stream=stream
    )
    await logger.run()
    assert logger.source_count == 1
    assert "[bold]literal[/bold]" in stream.getvalue()
    assert "```python" in stream.getvalue()


async def test_binary_rich_and_partial_failure():
    source = Source("s", "file:///binary", b"\xff", "application/pdf", "hash")
    stream = io.StringIO()
    await SourceLogger(EventPipeline([DocumentFetched(source)]), stream=stream).run()
    assert "Content (base64)" in stream.getvalue()
    assert "/w==" in stream.getvalue()
    error = RuntimeError("fetch failed")
    stream = io.StringIO()
    logger = SourceLogger(
        EventPipeline(
            [
                DocumentFetched(source),
                ItemFailed(error=error),
                StepFailed("fetch", error),
            ],
            error,
        ),
        format="jsonl",
        stream=stream,
        stages=("Fetching",),
    )
    with pytest.raises(RuntimeError, match="fetch failed"):
        await logger.run()
    summary = json.loads(stream.getvalue().splitlines()[-1])
    assert summary["status"] == "interrupted"
    assert summary["failed_items"] == summary["failed_steps"] == 1
