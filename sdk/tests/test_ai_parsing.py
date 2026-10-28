"""AI parsers select original HTML; only model POSTs may access the network."""

import json

import httpx
import pytest

from fastrag.ingestion.models import Source
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext
from fastrag.ingestion.steps.parsing import DecisionsParser, OpenAIParser


@pytest.mark.parametrize("parser_type", [OpenAIParser, DecisionsParser])
async def test_parser_only_reads_supplied_content(parser_type, monkeypatch):
    calls = []

    def respond(request):
        assert request.method == "POST"
        assert request.url.host == "model.test"
        calls.append(request)
        body = json.loads(request.content)
        if parser_type is DecisionsParser:
            questions = {q["name"]: q for q in body["questions"]}
        else:
            questions = json.loads(
                body["messages"][0]["content"].split("Questions: ", 1)[1]
            )
        assert set(questions) == {"scope"}
        if parser_type is DecisionsParser:
            return httpx.Response(
                200,
                json={
                    "answers": [
                        {"type": "choice", "name": "scope", "choice": "scope_1"}
                    ]
                },
            )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": '{"scope":"scope_1"}'}}]}
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(respond), **kw),
    )
    parser = parser_type(
        model="fixture", api_key="secret", base_url="https://model.test"
    )
    item = Source(
        id="page",
        uri="file:///offline/page.html",
        content=b'<html><title>Research</title><nav>noise</nav><main><h1>Useful research</h1><p>Keep every original detail.</p><a href="https://never-fetch.test">Reference</a><script src="https://never-fetch.test/app.js"></script></main><footer>noise</footer></html>',
        content_type="text/html",
        content_hash="fixture",
    )
    docs = [
        doc
        async for doc in parser.transform(
            item, ExecutionContext(pipeline_id="test", policy=Policy())
        )
    ]
    assert len(calls) == 1
    assert "Keep every original detail." in docs[0].content
    assert "noise" not in docs[0].content
    assert "https://never-fetch.test" in docs[0].content
    assert docs[0].metadata["title"] == "Research"
    assert item.content.startswith(b"<html>")


@pytest.mark.parametrize("parser_type", [OpenAIParser, DecisionsParser])
@pytest.mark.parametrize("status", [401, 429, 503])
async def test_parser_fallback_only_handles_transient_model_failures(
    parser_type, status, monkeypatch
):
    from fastrag.exceptions import ParseError

    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body["model"])
        if body["model"] == "primary":
            return httpx.Response(status, text="private provider failure")
        if parser_type is DecisionsParser:
            return httpx.Response(
                200,
                json={
                    "answers": [
                        {"type": "choice", "name": "scope", "choice": "scope_0"}
                    ]
                },
            )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": '{"scope":"scope_0"}'}}]}
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(respond), **kw),
    )
    args = dict(api_key="secret", base_url="https://model.test")
    fallback = parser_type(model="fallback", **args)
    parser = parser_type(model="primary", fallback=fallback, **args)
    item = Source(
        id="fixture",
        uri="https://never-fetch.test",
        content=b"<main>Original source text.</main>",
        content_type="text/html",
        content_hash="fixture",
    )
    context = ExecutionContext(pipeline_id="test", policy=Policy())
    if status == 401:
        with pytest.raises(ParseError) as error:
            _ = [doc async for doc in parser.transform(item, context)]
        assert "private provider failure" not in str(error.value)
        assert requests == ["primary"]
    else:
        docs = [doc async for doc in parser.transform(item, context)]
        assert docs[0].content == "Original source text."
        assert requests == ["primary", "fallback"]


def test_parser_identity_tracks_selection_settings_without_credentials():
    args = dict(model="fixture", base_url="https://model.test")
    parser = DecisionsParser(api_key="secret", **args)
    assert parser.config_hash == DecisionsParser(api_key="other", **args).config_hash
    assert (
        parser.config_hash
        != DecisionsParser(api_key="secret", goal="Research only", **args).config_hash
    )
    assert "secret" not in json.dumps(parser.config())


@pytest.mark.parametrize(
    ("api_format", "path"), [("decisions", "/decisions"), ("systemone", "/systemone")]
)
async def test_decisions_parser_api_format_selects_route_and_body(
    api_format, path, monkeypatch
):
    seen = []

    def respond(request):
        body = json.loads(request.content)
        seen.append((request.url.path, body))
        if api_format == "systemone":
            answers = {"scope": {"choice": "scope_0"}}
        else:
            answers = [{"type": "choice", "name": "scope", "choice": "scope_0"}]
        return httpx.Response(200, json={"answers": answers})

    client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(respond), **kw),
    )
    parser = DecisionsParser(
        model="fixture",
        api_key="secret",
        base_url="https://model.test/v1/",
        api_format=api_format,
    )
    item = Source(
        id="page",
        uri="file:///offline/page.html",
        content=b"<main>Original source text.</main>",
        content_type="text/html",
        content_hash="fixture",
    )
    [
        doc
        async for doc in parser.transform(
            item, ExecutionContext(pipeline_id="test", policy=Policy())
        )
    ]
    route, body = seen[0]
    assert route == f"/v1{path}"
    assert ("state" in body) == (api_format == "systemone")
    assert ("input" in body) == (api_format == "decisions")
    assert parser.config()["api_format"] == api_format


def test_decisions_api_format_validated_and_part_of_config_hash():
    from fastrag.exceptions import ConfigurationError

    args = dict(model="m", api_key="k", base_url="https://model.test")
    with pytest.raises(ConfigurationError):
        DecisionsParser(api_format="openai", **args)
    assert (
        DecisionsParser(api_format="decisions", **args).config_hash
        != DecisionsParser(api_format="systemone", **args).config_hash
    )
