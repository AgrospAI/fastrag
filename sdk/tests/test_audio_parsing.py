"""AudioParser sends multipart transcription requests and routes audio sources."""

import httpx
import pytest

from fastrag.exceptions import ParseError
from fastrag.ingestion.models import Source
from fastrag.ingestion.steps.parsing import (
    AudioParser,
    ParserRouter,
    audio_filter,
)

CONFIG = dict(
    model="openai/whisper-large-v3-turbo",
    api_key="secret",
    base_url="https://litellm.test/v1",
)


def source(uri="https://example.test/audios/es.wav", content_type="audio/x-wav"):
    return Source(
        id="audio-source",
        uri=uri,
        content=b"RIFFfake",
        content_type=content_type,
        content_hash="audio-hash",
    )


@pytest.fixture
def requests(monkeypatch):
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(200, json={"text": " Hola mundo. "})

    client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(handle), **kw),
    )
    return seen


async def collect(parser, item):
    return [d async for d in parser.transform(item, None)]


async def test_transcribes_with_multipart_request(requests):
    parser = AudioParser(**CONFIG, language="es")
    [document] = await collect(parser, source())
    assert document.content == "Hola mundo."
    assert document.metadata["source_uri"] == "https://example.test/audios/es.wav"
    request = requests[0]
    assert str(request.url) == "https://litellm.test/v1/audio/transcriptions"
    assert request.headers["authorization"] == "Bearer secret"
    body = request.content
    assert b'name="model"' in body and b"openai/whisper-large-v3-turbo" in body
    assert b'name="language"' in body and b"es" in body
    assert b'filename="es.wav"' in body and b"RIFFfake" in body


async def test_config_hash_excludes_credentials_includes_language():
    base = AudioParser(**CONFIG)
    assert base.config_hash == AudioParser(**{**CONFIG, "api_key": "other"}).config_hash
    assert base.config_hash != AudioParser(**CONFIG, language="es").config_hash


async def test_rejects_non_audio_and_wraps_http_errors(monkeypatch):
    with pytest.raises(ParseError):
        AudioParser(**CONFIG).validate_input(source(content_type="text/html"))

    client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: client(
            transport=httpx.MockTransport(lambda r: httpx.Response(500)), **kw
        ),
    )
    with pytest.raises(ParseError, match="Audio transcription failed"):
        await collect(AudioParser(**CONFIG), source())


def test_audio_filter_matches_audio_only():
    assert audio_filter.matches(source())
    assert audio_filter.matches(source(content_type="audio/mpeg"))
    assert not audio_filter.matches(source(content_type="text/html"))


async def test_parser_router_routes_audio(requests):
    route = ParserRouter().route(audio_filter, AudioParser(**CONFIG))
    assert isinstance(route._select(source()), AudioParser)
