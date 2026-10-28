"""YouTube watch leaves and ``enrich_youtube`` titles in source URL summaries."""

import io

import httpx
import pytest
from helpers import EventPipeline

from fastrag.ingestion.events import DocumentFetched, ItemFailed
from fastrag.ingestion.logger import SourceListLogger, SourceTreeLogger
from fastrag.ingestion.models import Source

WATCH = "https://www.youtube.com/watch?v="


def fetched(uri, **metadata):
    return DocumentFetched(
        Source(uri, uri, b"", "audio/mp4", "hash", metadata=metadata)
    )


def events():
    return [
        fetched(f"{WATCH}aaaaaaaaaaa", title="Known\x1b[31m"),
        fetched(f"{WATCH}bbbbbbbbbbb"),
        fetched(f"{WATCH}gone_______"),
        fetched("https://example.org/page", title="Page"),
        ItemFailed(uri="https://youtu.be/bbbbbbbbbbb"),
    ]


@pytest.fixture
def oembed(monkeypatch):
    requested = []

    def handler(request):
        video_id = request.url.params["url"].removeprefix(WATCH)
        requested.append(video_id)
        if video_id == "gone_______":
            return httpx.Response(404)
        return httpx.Response(200, json={"title": f"Title of {video_id}"})

    client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **options: client(transport=httpx.MockTransport(handler), **options),
    )
    return requested


async def test_tree_shows_watch_video_ids_without_lookups(oembed):
    stream = io.StringIO()
    await SourceTreeLogger(EventPipeline(events()), stream=stream).run()
    assert stream.getvalue().splitlines() == [
        "example.org 1",
        "└── /page 1",
        "www.youtube.com 3",
        "└── /watch 3",
        "    ├── ?v=aaaaaaaaaaa 1",
        "    ├── ?v=bbbbbbbbbbb 1",
        "    └── ?v=gone_______ 1",
        "youtu.be 1",
        "└── /bbbbbbbbbbb 1",
        "",
        "4 new 0 cached 1 error",
    ]
    assert oembed == []


async def test_tree_titles(oembed):
    stream = io.StringIO()
    await SourceTreeLogger(
        EventPipeline(events()), stream=stream, enrich_youtube=True
    ).run()
    assert stream.getvalue().splitlines() == [
        "example.org 1",
        "└── /page 1",
        "www.youtube.com 3",
        "└── /watch 3",
        "    ├── ?v=aaaaaaaaaaa (Known\\x1b[31m) 1",
        "    ├── ?v=bbbbbbbbbbb (Title of bbbbbbbbbbb) 1",
        "    └── ?v=gone_______ 1",
        "youtu.be 1",
        "└── /bbbbbbbbbbb (Title of bbbbbbbbbbb) 1",
        "",
        "4 new 0 cached 1 error",
    ]
    # Metadata titles skip the lookup; equivalent URLs share one request.
    assert sorted(oembed) == ["bbbbbbbbbbb", "gone_______"]


async def test_list_titles(oembed):
    stream = io.StringIO()
    await SourceListLogger(
        EventPipeline(events()), stream=stream, enrich_youtube=True
    ).run()
    assert stream.getvalue().splitlines() == [
        "https://example.org/page",
        "",
        f"{WATCH}aaaaaaaaaaa (Known\\x1b[31m)",
        f"{WATCH}bbbbbbbbbbb (Title of bbbbbbbbbbb)",
        f"{WATCH}gone_______",
        "",
        "https://youtu.be/bbbbbbbbbbb (Title of bbbbbbbbbbb)",
        "",
        "4 new 0 cached 1 error",
    ]


async def test_list_colors_only_the_url(oembed):
    stream = io.StringIO()
    await SourceListLogger(
        EventPipeline([fetched(f"{WATCH}aaaaaaaaaaa", title="Known")]),
        stream=stream,
        color=True,
        enrich_youtube=True,
    ).run()
    assert stream.getvalue().splitlines()[0] == (
        f"\x1b[32m{WATCH}aaaaaaaaaaa\x1b[0m (Known)"
    )


@pytest.mark.parametrize("logger_class", [SourceTreeLogger, SourceListLogger])
async def test_interrupted_runs_skip_lookups(oembed, logger_class):
    error = RuntimeError("pipeline failed")
    stream = io.StringIO()
    with pytest.raises(RuntimeError, match="pipeline failed"):
        await logger_class(
            EventPipeline(events(), error), stream=stream, enrich_youtube=True
        ).run()
    assert oembed == []
    assert "aaaaaaaaaaa (Known\\x1b[31m)" in stream.getvalue()
    assert "Title of" not in stream.getvalue()
