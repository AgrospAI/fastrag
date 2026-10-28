"""Content previews preserve short text, full statistics and source bytes."""

import base64
import io
import json

import pytest
from helpers import EventPipeline

from fastrag.ingestion.events import DocumentChunked, DocumentFetched, DocumentParsed
from fastrag.ingestion.logger import ChunkLogger, DocumentsLogger, SourceLogger
from fastrag.ingestion.models import Source
from fastrag.models import Document


@pytest.mark.parametrize("logger_class", [DocumentsLogger, SourceLogger, ChunkLogger])
@pytest.mark.parametrize("format", ["rich", "jsonl"])
async def test_content_preview(logger_class, format):
    text = "first α\nsecond\nHIDDEN MIDDLE\nfourth\nlast 🍏\n"
    document = Document("document", "source", text, metadata={"complete": True})
    source = Source("source", "file:///source", text.encode(), "text/plain", "hash")
    event = (
        DocumentFetched(source)
        if logger_class is SourceLogger
        else DocumentChunked(document, [document])
        if logger_class is ChunkLogger
        else DocumentParsed(document)
    )
    stream = io.StringIO()
    await logger_class(
        EventPipeline([event]), format=format, stream=stream, content_lines=2
    ).run()
    output = stream.getvalue()
    assert "HIDDEN MIDDLE" not in output
    assert "first α" in output and "last 🍏" in output
    assert "1 lines omitted" in output
    assert document.content == text and source.content == text.encode()
    if format == "jsonl":
        records = [json.loads(line) for line in output.splitlines()]
        item = next(record for record in records if "text" in record)
        assert (
            item["text"]
            == "first α\nsecond\n[... 1 lines omitted ...]\nfourth\nlast 🍏\n"
        )
        assert item["content_truncated"] is True
        assert item["omitted_lines"] == 1
        assert item["length"] == records[-1]["total_characters"] == len(text)
        assert records[-1]["total_bytes"] == len(text.encode())
        if logger_class is SourceLogger:
            assert base64.b64decode(item["content_base64"]) == text.encode()
    else:
        assert "Content (preview)" in output


@pytest.mark.parametrize(
    "text", ["", "one", "one\n", "one\r\ntwo\r\n", "one\ntwo\nthree\nfour"]
)
async def test_short_content_is_unchanged(text):
    stream = io.StringIO()
    await DocumentsLogger(
        EventPipeline([DocumentParsed(Document("doc", "source", text))]),
        format="jsonl",
        stream=stream,
        content_lines=2,
    ).run()
    item = json.loads(stream.getvalue().splitlines()[1])
    assert item["text"] == text
    assert "content_truncated" not in item


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2"])
def test_invalid_content_lines(value):
    with pytest.raises(ValueError, match="content_lines"):
        DocumentsLogger(EventPipeline([]), content_lines=value)


@pytest.mark.parametrize("format", ["rich", "jsonl"])
async def test_character_preview_bounds_minified_html_without_losing_bytes(format):
    text = "<html>α" + "x" * 4000 + "HIDDEN TAIL</html>"
    source = Source("source", "file:///source", text.encode(), "text/html", "hash")
    stream = io.StringIO()
    await SourceLogger(
        EventPipeline([DocumentFetched(source)]),
        format=format,
        stream=stream,
        content_lines=5,
        content_chars=100,
        show_binary=True,
    ).run()
    output = stream.getvalue()
    assert "HIDDEN TAIL" not in output
    assert "characters omitted" in output
    assert source.content == text.encode()
    if format == "jsonl":
        item = json.loads(output.splitlines()[1])
        assert item["text"].startswith(text[:100] + "\n[...")
        assert item["omitted_characters"] == len(text) - 100
        assert item["length"] == len(text)
        assert base64.b64decode(item["content_base64"]) == source.content
    else:
        assert "Content (preview)" in output


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2"])
def test_invalid_content_chars(value):
    with pytest.raises(ValueError, match="content_chars"):
        SourceLogger(EventPipeline([]), content_chars=value)
