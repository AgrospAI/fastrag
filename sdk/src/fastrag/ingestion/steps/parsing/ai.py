"""Model-guided content selection from supplied HTML, without fetching."""

from __future__ import annotations
import asyncio
from collections.abc import AsyncIterator, Mapping
from copy import deepcopy
from typing import Any, Literal
import httpx
from fastrag.exceptions import ConfigurationError, ParseError, TransientParseError
from fastrag.ingestion.cache import CacheStore
from fastrag.ingestion.models import Source
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext
from fastrag.ingestion.steps.parsing.base import Parser
from fastrag.ingestion.steps.parsing.html import BeautifulSoup, CONVERSION, convert
from fastrag.ingestion.steps._decisions import (
    ApiFormat,
    ModelDecisions,
    SCOPE_INSTRUCTIONS,
    validate_api_format,
)
from fastrag.models import ContentFormat, Document


class _AIParser(Parser, ModelDecisions):
    """Select original content using model choices; never execute page scripts."""

    content_format = ContentFormat.MARKDOWN
    protocol = "openai"

    accepted_media_types = frozenset({"text/html", "application/xhtml+xml"})

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        goal: str = "Extract all useful page content for a searchable knowledge base.",
        timeout: float = 600,
        max_snapshot_chars: int = 24000,
        request_timeout: float = 120,
        extra_body: Mapping[str, Any] | None = None,
        fallback: _AIParser | None = None,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
    ) -> None:
        """Configure content selection and provider fallback, without rendering settings."""
        if BeautifulSoup is None or convert is None:
            raise ConfigurationError("AI parsers need fastrag-ai[html]")
        if min(timeout, request_timeout, max_snapshot_chars) <= 0:
            raise ConfigurationError("Parser budgets must be positive")
        super().__init__(cache=cache, policy=policy)
        self.model, self.api_key, self.base_url = model, api_key, base_url
        self.goal, self.timeout, self.request_timeout = goal, timeout, request_timeout
        self.max_snapshot_chars = max_snapshot_chars
        self.extra_body = deepcopy(dict(extra_body or {}))
        self.fallback = fallback
        if self.extra_body.keys() & {
            "model",
            "state",
            "input",
            "questions",
            "messages",
        }:
            raise ConfigurationError("extra_body cannot override model decision inputs")

    def config(self) -> Mapping[str, Any]:
        """Return content-selection settings without credentials."""
        return {
            "protocol": self.protocol,
            **({"api_format": self.api_format} if self.protocol == "decisions" else {}),
            "parsing_version": 3,
            "model": self.model,
            "base_url": self.base_url,
            "goal": self.goal,
            "timeout": self.timeout,
            "max_snapshot_chars": self.max_snapshot_chars,
            "request_timeout": self.request_timeout,
            "extra_body": self.extra_body,
            "fallback": self.fallback.config() if self.fallback else None,
            "instructions": SCOPE_INSTRUCTIONS,
        }

    async def _parse(self, source: Source) -> tuple[str, str]:
        soup = BeautifulSoup(source.content, "html.parser")
        regions = [
            soup.body or soup,
            *soup.select('main, [role="main"], article, section'),
        ]
        regions = [region for region in regions if region.get_text(strip=True)]
        if not regions:
            return "", ""
        # Bound the provider choice set, retaining the whole-document option.
        regions = regions[:255]
        scopes = {
            f"scope_{i}": {
                "tag": (region.name or "body").upper(),
                "characters": len(region.get_text()),
                "excerpt": _clip(
                    region.get_text(" ", strip=True),
                    max(1, self.max_snapshot_chars // len(regions)),
                ),
            }
            for i, region in enumerate(regions)
        }
        state = {
            "url": source.uri,
            "goal": self.goal,
            "snapshot": _clip(soup.get_text(" ", strip=True), self.max_snapshot_chars),
        }
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            _, scope = await self._decide(client, state, {}, scopes)
        region = regions[int(scope.removeprefix("scope_"))]
        title = soup.title.get_text(strip=True) if soup.title else ""
        return self._markdown(str(region), source.uri), title

    @staticmethod
    def _markdown(html: str, uri: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        from fastrag.ingestion.steps.parsing._content import reveal_content

        reveal_content(soup)
        for node in soup.select(
            'script, style, noscript, nav, header, footer, aside, [role="navigation"], [role="banner"], [role="contentinfo"], button, summary, [role="button"], [role="tab"]'
        ):
            node.decompose()
        from fastrag.ingestion.steps.parsing.html import HtmlParser

        for svg in soup.find_all("svg"):
            labels = [
                node.get_text(" ", strip=True)
                for node in svg.find_all(["text", "title", "desc", "foreignObject"])
            ]
            svg.replace_with(
                " ".join(dict.fromkeys(label for label in labels if label))
            )
        HtmlParser._absolutize(soup, uri)
        return convert(str(soup), options=CONVERSION).content.strip()

    async def transform(
        self, item: Source, context: ExecutionContext
    ) -> AsyncIterator[Document]:
        """Yield useful Markdown from supplied source bytes without browser or page requests."""
        self.validate_input(item)
        if not item.content.strip():
            return
        try:
            async with asyncio.timeout(self.timeout):
                content, title = await self._parse(item)
            if content:
                yield self.build_document(item, content, {"title": title})
        except ConfigurationError:
            raise
        except Exception as error:
            # Provider bodies and credentials must not leak through exception text.
            transient = isinstance(error, (httpx.TransportError, TimeoutError)) or (
                isinstance(error, httpx.HTTPStatusError)
                and (
                    error.response.status_code == 429
                    or error.response.status_code >= 500
                )
            )
            error_type = TransientParseError if transient else ParseError
            if transient and self.fallback is not None:
                async for document in self.fallback.transform(item, context):
                    yield document
                return
            raise error_type(
                f"AI parsing failed for {item.uri}: {_describe(error, self.timeout)}"
            ) from error


#: First code point of the CJK blocks (radicals, kana, ideographs, hangul…),
#: where text runs about one token per character instead of about four.
_WIDE_CHARACTERS = 0x2E80


def _clip(text: str, budget: int) -> str:
    """Clip ``text`` to ``budget`` characters, counting a CJK character as four.

    ``max_snapshot_chars`` is sized for alphabetic text; counted naively, a
    Chinese page sends about four times the tokens and overflows the model's
    context, failing every attempt with an HTTP 500. Alphabetic text clips
    exactly as plain slicing would.
    """
    if len(text) <= budget // 4 or text.isascii():
        return text[:budget]
    cost = 0
    for index, character in enumerate(text):
        cost += 4 if ord(character) >= _WIDE_CHARACTERS else 1
        if cost > budget:
            return text[:index]
    return text


def _describe(error: Exception, timeout: float) -> str:
    """The failure kind, with the status for an HTTP error, never a provider body."""
    if isinstance(error, httpx.HTTPStatusError):
        response = error.response
        return f"HTTP {response.status_code} {response.reason_phrase}".strip()
    if isinstance(error, TimeoutError):
        return f"TimeoutError (no answer within {timeout:g}s)"
    return type(error).__name__


class OpenAIParser(_AIParser):
    """Select useful source content with an OpenAI-compatible model."""


class DecisionsParser(_AIParser):
    """Select useful source content with typed Decisions/JEV choices.

    ``api_format`` picks the wire format: ``"decisions"`` (OpenAI Decisions,
    ``/decisions``) or ``"systemone"`` (TypeSafe System One, ``/systemone``).
    """

    protocol = "decisions"

    def __init__(self, *, api_format: ApiFormat = "decisions", **kwargs: Any) -> None:
        """Configure the parser; other arguments match ``OpenAIParser``."""
        self.api_format = validate_api_format(api_format)
        super().__init__(**kwargs)
