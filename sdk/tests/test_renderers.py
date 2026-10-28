"""Real local Chromium pages with deterministic HTTP model responses, no paid calls."""

import json
from typing import Any

import httpx
import pytest
from bs4 import BeautifulSoup

from fastrag.exceptions import ConfigurationError, FetchError, TransientFetchError
from fastrag.ingestion.models import Source
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext
from fastrag.ingestion.steps.fetching import ChatRenderer, DecisionsRenderer
from fastrag.ingestion.steps.fetching.http import FetchedPage


@pytest.fixture(params=[ChatRenderer, DecisionsRenderer])
def model_renderer(request, monkeypatch):
    """Mock only the model endpoint, leaving browser and source HTTP real."""
    renderer_type = request.param
    client_type = httpx.AsyncClient
    calls = []
    mode = {"value": "normal"}

    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        if renderer_type is DecisionsRenderer:
            assert request.url.path == "/v1/decisions"
            assert set(body) >= {"model", "input", "questions"}
            state = json.loads(body["input"])
            assert all(q["type"] == "choice" for q in body["questions"])
            questions = {
                q["name"]: {
                    "criteria": {c["value"]: c["description"] for c in q["choices"]}
                }
                for q in body["questions"]
            }
        else:
            assert request.url.path == "/v1/chat/completions"
            state = json.loads(body["messages"][1]["content"])
            questions = json.loads(
                body["messages"][0]["content"].split("Questions: ", 1)[1]
            )
        actions = questions["action"]["criteria"]
        scopes = {}
        if set(actions) in ({"static", "dynamic"}, {"hydrated", "interact"}):
            # Escalation gates: always hydrate, then always explore.
            gate = "dynamic" if "dynamic" in actions else "interact"
            calls.pop()  # gates are not exploration steps
            if renderer_type is DecisionsRenderer:
                return httpx.Response(
                    200,
                    json={
                        "answers": [
                            {"type": "choice", "name": "action", "choice": gate}
                        ]
                    },
                )
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": json.dumps({"action": gate})}}]
                },
            )
        assert set(questions) == {"action"}
        if renderer_type is DecisionsRenderer:
            assert all(
                isinstance(v, str) for v in [*actions.values(), *scopes.values()]
            )
            actions = {
                k: json.loads(v) if v.startswith("{") else v for k, v in actions.items()
            }
            scopes = {k: json.loads(v) for k, v in scopes.items()}
        # Link/form candidates must never be offered to either model.
        assert not any(
            isinstance(v, dict)
            and v.get("name")
            in {"Load another document", "Load other page", "Submit search"}
            for v in actions.values()
        )
        scope = "body"
        controls = [key for key, value in actions.items() if isinstance(value, dict)]
        if mode["value"] == "tabs":
            controls = [
                key for key in controls if actions[key].get("expanded") != "true"
            ]
        action = (
            controls[0]
            if controls and not (mode["value"] == "guard" and state["history"])
            else "done"
        )
        if mode["value"] == "invalid":
            action = "invented_control"
        if mode["value"] == "loop":
            action = "wait"
        if mode["value"] == "premature_done":
            action = "done"
        if mode["value"] == "error":
            return httpx.Response(500, text="sensitive provider response")
        if renderer_type is DecisionsRenderer:
            return httpx.Response(
                200,
                json={
                    "answers": [
                        {"type": "choice", "name": "action", "choice": action},
                        {"type": "choice", "name": "scope", "choice": scope},
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"action": action, "scope": scope})
                        }
                    }
                ]
            },
        )

    def client(**kwargs):
        return client_type(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    renderer = renderer_type(
        model="fixture-model",
        api_key="secret",
        base_url="https://model.test/v1",
        settle_seconds=0.2,
        max_steps=5,
    )
    return renderer, calls, mode, client_type


def always_explore(monkeypatch, *renderers):
    """Skip the model/browser gates so tests exercise ``_explore`` directly."""

    async def choose(page, instructions, choices):
        return "dynamic" if "dynamic" in choices else "interact"

    async def hydrate(page):
        return page

    for renderer in renderers:
        monkeypatch.setattr(renderer, "_choose", choose)
        monkeypatch.setattr(renderer.dynamic_renderer, "render", hydrate)


async def render_page(model_renderer, http_server, filename):
    renderer, _, _, client_type = model_renderer
    uri = f"{http_server}/ai-parser/{filename}"
    async with client_type() as client:
        response = await client.get(uri)
    try:
        return await renderer.render(
            FetchedPage(
                uri=uri,
                content=response.content,
                content_type="text/html",
                etag=None,
                last_modified=None,
                status_code=200,
            )
        )
    finally:
        await renderer.aclose()


@pytest.mark.parametrize("filename", ["dynamic.html", "dynamic-anchor.html"])
async def test_js_load_more_fetches_every_batch(model_renderer, http_server, filename):
    page = await render_page(model_renderer, http_server, filename)
    text = BeautifulSoup(page.content, "html.parser").get_text(" ", strip=True)
    for expected in [
        "Soil sensors improve irrigation decisions.",
        "Shared livestock data supports animal welfare research.",
        "Harvest forecasts reduce food waste.",
    ]:
        assert text.count(expected) == 1
    assert len(model_renderer[1]) == 3
    assert page.content_type == "text/html"
    assert page.uri.endswith(filename)
    assert "Forbidden linked page content" not in text


@pytest.mark.parametrize("filename", ["dynamic.html", "dynamic-anchor.html"])
async def test_completion_cannot_skip_load_more(model_renderer, http_server, filename):
    model_renderer[2]["value"] = "premature_done"
    page = await render_page(model_renderer, http_server, filename)
    assert b"Harvest forecasts reduce food waste." in page.content
    assert len(model_renderer[1]) == 3


@pytest.mark.parametrize("mode", ["normal", "premature_done"])
async def test_docusaurus_load_more_waits_for_hydration(
    model_renderer, http_server, mode
):
    model_renderer[2]["value"] = mode
    page = await render_page(model_renderer, http_server, "hydration.html")
    assert b"Plant protection integrated reports" in page.content
    assert len(model_renderer[1]) == 2


async def test_stalled_load_more_cannot_complete(model_renderer, http_server):
    model_renderer[2]["value"] = "premature_done"
    with pytest.raises(FetchError):
        await render_page(
            model_renderer, http_server, "../agrospai-homepage.html?stalled"
        )


async def test_navigation_buttons_and_popups_cannot_leave_page(
    model_renderer, http_server
):
    model_renderer[2]["value"] = "guard"
    page = await render_page(model_renderer, http_server, "navigation.html")
    assert "Retain this source content." in page.content.decode()
    assert "Forbidden linked page content" not in page.content.decode()
    calls = model_renderer[1]
    state = (
        json.loads(calls[-1]["input"])
        if "input" in calls[-1]
        else json.loads(calls[-1]["messages"][1]["content"])
    )
    assert state["url"].endswith("navigation.html")


@pytest.mark.parametrize("mode", ["invalid", "loop", "error"])
async def test_invalid_decisions_and_incomplete_exploration_fail(
    model_renderer, http_server, mode
):
    model_renderer[2]["value"] = mode
    with pytest.raises(FetchError) as caught:
        await render_page(model_renderer, http_server, "important.html")
    assert "important.html" in str(caught.value)
    assert "secret" not in str(caught.value)
    assert "sensitive provider response" not in str(caught.value)
    assert isinstance(caught.value, TransientFetchError) == (mode == "error")


@pytest.mark.parametrize("status", [400, 401, 429, 503, 529])
async def test_provider_failures_are_retried_only_when_transient(monkeypatch, status):
    from fastrag.ingestion.steps.base import _with_retries

    renderer = DecisionsRenderer(
        model="fixture", api_key="secret", base_url="https://model.test"
    )
    calls = 0

    async def explore(source):
        nonlocal calls
        calls += 1
        if calls == 1:
            response = httpx.Response(
                status, request=httpx.Request("POST", "https://model.test")
            )
            response.raise_for_status()
        return "Retained content", "Title", []

    always_explore(monkeypatch, renderer)
    monkeypatch.setattr(renderer, "_explore", explore)
    source = Source(
        id="fixture",
        uri="https://source.test",
        content=b"content",
        content_type="text/html",
        content_hash="fixture",
    )

    async def parse():
        return await renderer.render(
            FetchedPage(
                uri=source.uri,
                content=source.content,
                content_type=source.content_type,
                etag=None,
                last_modified=None,
                status_code=200,
            )
        )

    if status in {400, 401}:
        with pytest.raises(FetchError) as caught:
            await _with_retries(parse, Policy(backoff=0))
        assert not isinstance(caught.value, TransientFetchError)
        assert calls == 1
    else:
        docs = await _with_retries(parse, Policy(backoff=0))
        assert docs.content == b"Retained content"
        assert calls == 2


def test_cache_identity_excludes_credentials_and_tracks_extraction_settings():
    args: dict[str, Any] = dict(
        model="fixture", api_key="secret", base_url="https://model.test/v1"
    )
    renderer = DecisionsRenderer(**args)
    other_args = args.copy()
    other_args["api_key"] = "different"
    assert "secret" not in json.dumps(renderer.config())
    assert json.dumps(renderer.config(), sort_keys=True) == json.dumps(
        DecisionsRenderer(**other_args).config(), sort_keys=True
    )
    assert json.dumps(renderer.config(), sort_keys=True) != json.dumps(
        ChatRenderer(**args).config(), sort_keys=True
    )
    assert json.dumps(renderer.config(), sort_keys=True) != json.dumps(
        DecisionsRenderer(**args, goal="Only methods").config(), sort_keys=True
    )
    with pytest.raises(ConfigurationError):
        DecisionsRenderer(**args, max_steps=0)


@pytest.mark.parametrize("status", [401, 429, 529])
async def test_fallback_preserves_content_and_only_handles_transient_failures(
    monkeypatch, status
):
    fallback = ChatRenderer(
        model="chat", api_key="fallback-secret", base_url="https://chat.test"
    )
    renderer = DecisionsRenderer(
        model="decisions",
        api_key="secret",
        base_url="https://model.test",
        fallback=fallback,
    )
    calls = []

    async def primary(source):
        response = httpx.Response(
            status, request=httpx.Request("POST", "https://model.test")
        )
        response.raise_for_status()

    async def secondary(source):
        calls.append(source.uri)
        return "Original DOM content", "Title", []

    always_explore(monkeypatch, renderer, fallback)
    monkeypatch.setattr(renderer, "_explore", primary)
    monkeypatch.setattr(fallback, "_explore", secondary)
    source = Source(
        id="fixture",
        uri="https://source.test",
        content=b"content",
        content_type="text/html",
        content_hash="fixture",
    )
    if status == 401:
        with pytest.raises(FetchError):
            await renderer.render(
                FetchedPage(
                    uri=source.uri,
                    content=source.content,
                    content_type=source.content_type,
                    etag=None,
                    last_modified=None,
                    status_code=200,
                )
            )
        assert not calls
    else:
        docs = await renderer.render(
            FetchedPage(
                uri=source.uri,
                content=source.content,
                content_type=source.content_type,
                etag=None,
                last_modified=None,
                status_code=200,
            )
        )
        assert docs.content == b"Original DOM content"
        assert calls == [source.uri]
    assert "secret" not in json.dumps(renderer.config())
    assert (
        renderer.config()
        != DecisionsRenderer(
            model="decisions", api_key="secret", base_url="https://model.test"
        ).config()
    )


async def test_replaced_tabs_and_details_retain_all_source_states(
    model_renderer, http_server
):
    model_renderer[2]["value"] = "tabs"
    page = await render_page(model_renderer, http_server, "tabs.html")
    soup = BeautifulSoup(page.content, "html.parser")
    text = soup.get_text(" ", strip=True)
    assert "First panel contains irrigation research." in text
    assert "Second panel contains livestock research." in text
    assert "Sampling methods validate sensor readings." in text
    assert len(soup.find_all("h1", string="Research tabs")) == 1
    assert len(model_renderer[1]) == 3


@pytest.mark.parametrize("fetcher_type", ["file", "web", "sitemap"])
async def test_fetchers_hash_rendered_sources(
    model_renderer, http_server, fetcher_type
):
    from fastrag.ingestion._hashing import content_hash
    from fastrag.ingestion.steps.fetching import FileSource, SitemapCrawler, WebCrawler

    renderer, _, _, client_type = model_renderer
    uri = f"{http_server}/ai-parser/dynamic.html"
    if fetcher_type == "file":
        fetcher = FileSource(url=uri, renderer=renderer)
    elif fetcher_type == "web":
        fetcher = WebCrawler(url=[uri], max_depth=0, renderer=renderer)
    else:
        fetcher = SitemapCrawler(url=f"{http_server}/sitemap.xml", renderer=renderer)
    # Model HTTP is mocked by the fixture; the fetcher's HTTP is a real local request.
    fetcher._transport._client = client_type()
    context = ExecutionContext(pipeline_id="render-test", policy=Policy())
    try:
        sources = [source async for source in fetcher.evaluate_target(uri, context)]
        source = sources[0]
        assert b"Harvest forecasts reduce food waste." in source.content
        assert source.content_hash == content_hash(source.content)
        assert source.etag is None and source.last_modified is None
        assert fetcher.config()["renderer"] == renderer.config()
    finally:
        await fetcher.aclose()


async def test_dynamic_freshness_does_not_trust_html_etag(monkeypatch):
    from dataclasses import replace
    from unittest.mock import AsyncMock

    from fastrag.ingestion._hashing import content_hash
    from fastrag.ingestion.steps.fetching import FileSource, Renderer

    class DataRenderer(Renderer):
        def config(self):
            return {"renderer": "fixture"}

        async def render(self, page):
            return replace(page, content=b"Changed JS data")

    fetcher = FileSource(url="https://origin.test", renderer=DataRenderer())
    page = FetchedPage(
        uri=fetcher.url,
        content=b"Unchanged HTML",
        content_type="text/html",
        etag="same",
        last_modified=None,
        status_code=200,
    )
    monkeypatch.setattr(fetcher._transport, "fetch", AsyncMock(return_value=page))
    validators = AsyncMock(return_value={"etag": "same"})
    monkeypatch.setattr(fetcher._transport, "validators", validators)
    assert await fetcher.is_source_modified(
        fetcher.url, "same", None, content_hash(b"Old JS data")
    )
    validators.assert_not_awaited()


async def test_playwright_renderer_hydrates_script_and_hidden_diagram():
    from fastrag.ingestion.steps.fetching import PlaywrightRenderer

    page = FetchedPage(
        uri="file:///tmp/dynamic.html",
        content=b"""<main><div id="content"></div><script>
    document.getElementById('content').innerHTML = '<div hidden role="tabpanel"><svg width="300" height="100"><text x="10" y="30">Dynamically created node</text></svg></div>';
    </script></main>""",
        content_type="text/html",
        etag="original",
        last_modified=None,
        status_code=200,
    )
    rendered = await PlaywrightRenderer(wait_selector="#content svg").render(page)
    assert b"Dynamically created node" in rendered.content
    assert b"data-fastrag-png" in rendered.content
    assert rendered.etag is None
    assert b"data-fastrag-png" not in page.content


@pytest.mark.parametrize("renderer_type", [ChatRenderer, DecisionsRenderer])
async def test_renderers_leave_binary_responses_unchanged(renderer_type):
    renderer = renderer_type(
        model="fixture", api_key="secret", base_url="https://model.test"
    )
    page = FetchedPage(
        uri="https://source.test/report.pdf",
        content=b"%PDF-fixture",
        content_type="application/pdf",
        etag="original",
        last_modified=None,
        status_code=200,
    )
    assert await renderer.render(page) is page


async def test_web_crawler_discovers_links_from_hydrated_content():
    from fastrag.ingestion.steps.fetching import PlaywrightRenderer, WebCrawler

    requests = []

    def respond(request):
        requests.append(request.url.path)
        if request.url.path in {"", "/"}:
            html = '<main id="content"></main><script>document.querySelector("main").innerHTML = \'<a href="/child">Newly loaded document</a>\';</script>'
        else:
            html = "<main>Child document.</main>"
        return httpx.Response(200, text=html, headers={"content-type": "text/html"})

    crawler = WebCrawler(
        url="https://origin.test", max_depth=1, renderer=PlaywrightRenderer()
    )
    crawler._transport._client = httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    )
    try:
        sources = [
            s
            async for s in crawler.evaluate(
                ExecutionContext(pipeline_id="links", policy=Policy())
            )
        ]
    finally:
        await crawler.aclose()
    assert {s.uri for s in sources} == {
        "https://origin.test",
        "https://origin.test/child",
    }
    assert requests == ["/", "/child"]


async def test_ai_renderer_serves_supplied_bytes_at_a_bare_origin(model_renderer):
    renderer = model_renderer[0]
    page = FetchedPage(
        uri="https://origin.test",
        content=b"<main>Retain this supplied content.</main>",
        content_type="text/html",
        etag=None,
        last_modified=None,
        status_code=200,
    )
    rendered = await renderer.render(page)
    assert b"Retain this supplied content." in rendered.content


@pytest.mark.parametrize(
    ("answers", "hydrates", "explores"),
    [
        (["static"], False, False),
        (["dynamic", "hydrated"], True, False),
        (["dynamic", "interact"], True, True),
    ],
)
async def test_rendering_escalates_only_as_far_as_the_model_requires(
    monkeypatch, answers, hydrates, explores
):
    renderer = DecisionsRenderer(
        model="fixture", api_key="secret", base_url="https://model.test"
    )
    seen = {"hydrate": 0, "explore": 0}
    queue = iter(answers)

    async def choose(page, instructions, choices):
        return next(queue)

    async def hydrate(page):
        seen["hydrate"] += 1
        return page

    async def explore(source):
        seen["explore"] += 1
        return "<html>explored</html>", "Title", []

    monkeypatch.setattr(renderer, "_choose", choose)
    monkeypatch.setattr(renderer.dynamic_renderer, "render", hydrate)
    monkeypatch.setattr(renderer, "_explore", explore)
    page = FetchedPage(
        uri="https://source.test",
        content=b"<html>raw</html>",
        content_type="text/html",
        etag="x",
        last_modified=None,
        status_code=200,
    )
    rendered = await renderer.render(page)
    assert (seen["hydrate"] > 0, seen["explore"] > 0) == (hydrates, explores)
    assert rendered.etag is None
    assert (b"explored" in rendered.content) == explores
    await renderer.aclose()
