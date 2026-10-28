"""Image descriptions through an OpenAI-compatible vision endpoint."""

from __future__ import annotations

import base64
from copy import deepcopy
from collections.abc import AsyncIterator, Mapping
from typing import Any, Literal

import httpx

from fastrag.exceptions import ParseError
from fastrag.ingestion.cache import CacheStore
from fastrag.ingestion.models import Source
from fastrag.ingestion.policy import Policy
from fastrag.models import Document
from fastrag.ingestion.steps.base import ExecutionContext
from fastrag.ingestion.steps.parsing.base import Parser


class OpenAIVisionParser(Parser):
    """Describe standalone images, or enrich images embedded in HTML parsers.

    SVGs are rasterized with the optional ``dynamic`` extra. Credentials are
    excluded from cache identity; model, endpoint, prompt and extra body are included.
    ``extra_body`` adds JSON request fields and overrides matching defaults.
    """

    accepted_media_types = frozenset({"image/*"})

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        prompt: str = (
            "Describe this image for a searchable documentation index. Transcribe "
            "meaningful text and explain diagram nodes, arrow directions and relationships. "
            "For screenshots describe the interface and visible values. Use concise English "
            "and only information visible in the image."
        ),
        timeout: float = 120,
        extra_body: Mapping[str, Any] | None = None,
        max_tokens: int = 2048,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
    ) -> None:
        """Configure a vision-capable chat model and its OpenAI base URL."""
        super().__init__(cache=cache, policy=policy)
        self.model, self.api_key, self.base_url = model, api_key, base_url
        self.prompt, self.timeout, self.max_tokens = prompt, timeout, max_tokens
        self.extra_body = deepcopy(dict(extra_body or {}))

    def config(self) -> Mapping[str, Any]:
        """Return description-affecting settings without credentials."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            **({"extra_body": self.extra_body} if self.extra_body else {}),
        }

    async def describe(self, content: bytes, content_type: str) -> str:
        """Send image bytes as a data URL and return the generated description."""
        if content_type.split(";", 1)[0] == "image/svg+xml":
            from fastrag.ingestion.steps.parsing.dynamic import rasterize_svg

            content = await rasterize_svg(content, self.timeout)
            content_type = "image/png"
        data = base64.b64encode(content).decode("ascii")
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.base_url.rstrip("/") + "/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "temperature": 0,
                        "max_tokens": self.max_tokens,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": self.prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{content_type};base64,{data}"
                                        },
                                    },
                                ],
                            }
                        ],
                        **self.extra_body,
                    },
                )
                response.raise_for_status()
                text = response.json()["choices"][0]["message"]["content"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("empty vision response")
                return text.strip()
        except Exception as error:
            raise ParseError(
                f"Vision description failed for model {self.model}: {type(error).__name__}"
            ) from error

    async def transform(
        self, item: Source, context: ExecutionContext
    ) -> AsyncIterator[Document]:
        """Yield one searchable description for an image source."""
        self.validate_input(item)
        if item.content:
            yield self.build_document(
                item, await self.describe(item.content, item.content_type)
            )
