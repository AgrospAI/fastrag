"""Regression tests for hidden panels, JavaScript diagrams and vision inputs."""

import base64
import json

import httpx
import pytest

from fastrag.exceptions import ParseError
from fastrag.ingestion.models import Source
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext
from fastrag.ingestion.steps.parsing import (
    DocusaurusParser,
    HtmlParser,
    OpenAIVisionParser,
    SvgParser,
    docusaurus_filter,
)


def source(content, content_type="text/html", uri="https://docs.example/page"):
    return Source(
        id=uri,
        uri=uri,
        content=content.encode() if isinstance(content, str) else content,
        content_type=content_type,
        content_hash="fixture",
    )


async def parse(parser, item):
    return [
        doc
        async for doc in parser.transform(
            item, ExecutionContext(pipeline_id="test", policy=Policy())
        )
    ]


@pytest.mark.parametrize("parser", [HtmlParser, DocusaurusParser])
async def test_hidden_tabs_and_svg_labels(parser):
    item = source("""<div class="markdown"><h1>Build</h1>
    <div role="tabpanel"><p>Faster and lighter build. Ideal when you know the target node.</p></div>
    <div role="tabpanel" hidden style="display:none!important" aria-hidden="true">
    <p>The registry stores a <em>manifest index</em> that points to each variant; the node chooses its own.</p></div>
    <div class="docusaurus-mermaid-container"><svg><title>Publishing</title><text>Build</text>
    <foreignObject><div>Fix digest</div></foreignObject></svg></div><svg><path d="M0 0"/></svg>
    </div>""")
    text = (await parse(parser(), item))[0].content
    assert "Faster and lighter build. Ideal when you know the target node." in text
    assert (
        "The registry stores a *manifest index* that points to each variant; the node chooses its own."
        in text
    )
    assert "Publishing" in text and "Fix digest" in text
    assert "<svg" not in text
    assert (
        "Fix digest"
        not in (await parse(parser(strip_selectors=["svg"]), item))[0].content
    )


async def test_docusaurus_filter_and_svg_parser():
    item = source(
        '<meta name="generator" content="Docusaurus"><div class="markdown"><h1>Article</h1></div><p>Chrome</p>'
    )
    assert docusaurus_filter.matches(item)
    assert not docusaurus_filter.matches(source("<p>About Docusaurus</p>"))
    assert "Chrome" not in (await parse(DocusaurusParser(), item))[0].content
    svg = source("<svg><text>Node A</text><text>Node B</text></svg>", "image/svg+xml")
    text = (await parse(SvgParser(), svg))[0].content
    assert "Node A" in text and "Node B" in text


async def test_vision_embedded_responsive_data_and_standalone_images(monkeypatch):
    fetched, payloads = [], []

    def handle(request):
        if request.method == "GET":
            fetched.append(str(request.url))
            return httpx.Response(
                200, content=b"png-fixture", headers={"content-type": "image/png"}
            )
        payload = json.loads(request.content)
        payloads.append(payload)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "Explorer shows blockchain transactions."}}
                ]
            },
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(handle), **kw),
    )
    vision = OpenAIVisionParser(
        model="vision-test", api_key="secret", base_url="https://vision.example/v1"
    )
    img = '<img src="small.png" srcset="small.png 100w, /large.png 1000w">'
    inline = base64.b64encode(b"inline-png").decode()
    item = source(
        f'<div class="markdown">{img}{img}<img src="data:image/png;base64,{inline}"></div>'
    )
    text = (await parse(DocusaurusParser(vision=vision), item))[0].content
    assert fetched == ["https://docs.example/large.png"]
    assert text.count("Explorer shows blockchain transactions.") == 3
    assert len(payloads) == 2
    assert payloads[0]["model"] == "vision-test"
    assert (
        payloads[0]["messages"][0]["content"][1]["image_url"]["url"]
        == "data:image/png;base64," + base64.b64encode(b"png-fixture").decode()
    )
    assert "data:image" not in text
    assert "secret" not in json.dumps(vision.config())
    image = source(b"png", "image/png")
    assert "transactions" in (await parse(vision, image))[0].content
    assert DocusaurusParser(vision=vision).config_hash != DocusaurusParser().config_hash


async def test_vision_failure_is_not_silently_dropped(monkeypatch):
    client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: client(
            transport=httpx.MockTransport(lambda req: httpx.Response(500)), **kw
        ),
    )
    vision = OpenAIVisionParser(
        model="vision", api_key="secret", base_url="https://vision.example/v1"
    )
    with pytest.raises(ParseError):
        await vision.describe(b"png", "image/png")
