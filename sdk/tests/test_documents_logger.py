"""Document inspection at real pipeline checkpoints and later event stages."""

import io
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from helpers import EventPipeline

from fastrag.ingestion import Pipeline
from fastrag.ingestion.events import DocumentEmbedded, DocumentParsed, DocumentSaved
from fastrag.ingestion.logger import DocumentsLogger
from fastrag.ingestion.steps.chunking import HeadingChunker
from fastrag.ingestion.steps.fetching import LocalFileFetcher
from fastrag.ingestion.steps.parsing import DefaultParser
from fastrag.models import Document


def records(stream):
    return [json.loads(line) for line in stream.getvalue().splitlines()]


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("entered", [False, True])
async def test_final_stage_only(tmp_path, chunked, entered):
    path = tmp_path / "document.md"
    path.write_text("# First\n\nHéllo [bold]literal[/bold].\n\n# Second\n\nWorld.")
    parsed = DefaultParser()(LocalFileFetcher(patterns=[str(path)]))
    root = HeadingChunker()(parsed) if chunked else parsed
    pipeline = Pipeline(root)
    stream = io.StringIO()
    logger = DocumentsLogger(pipeline, stream=stream, format="jsonl")
    if entered:
        async with pipeline:
            await logger.run()
    else:
        await logger.run()
    output = records(stream)
    items = [r for r in output if r["type"] in ("chunk", "document")]
    assert len(items) == (2 if chunked else 1)
    assert {r["event_name"] for r in items} == {"Chunking" if chunked else "Parsing"}
    assert output[-1]["documents"] == logger.document_count == len(items)
    assert output[-1]["sources"] == 1
    assert output[-1]["total_characters"] == sum(len(r["text"]) for r in items)
    await logger.run()
    assert logger.document_count == len(items)


async def test_embedding_indexing_and_stage_selection():
    document = Document("doc", "source", "Héllo", embedding=[0.1, 0.2], cached=True)
    events = [
        DocumentParsed(replace(document, embedding=None), event_name="Parsing"),
        DocumentEmbedded(document, event_name="Embedding"),
        DocumentSaved(document, event_name="Indexing"),
    ]
    pipeline = EventPipeline(events)
    pipeline.root = SimpleNamespace(event_name="Indexing")
    stream = io.StringIO()
    await DocumentsLogger(pipeline, format="jsonl", stream=stream).run()
    assert records(stream)[1]["embedding"] == [0.1, 0.2]
    assert records(stream)[1]["event_name"] == "Indexing"
    assert records(stream)[-1]["cached_items"] == 1
    stream = io.StringIO()
    await DocumentsLogger(
        pipeline, format="jsonl", stream=stream, stages=("Parsing", "Embedding")
    ).run()
    assert records(stream)[-1]["documents"] == 2
