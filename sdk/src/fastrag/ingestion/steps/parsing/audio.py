"""Speech transcription through an OpenAI-compatible ``/audio/transcriptions`` endpoint."""

from __future__ import annotations

import mimetypes
from collections.abc import AsyncIterator, Mapping
from copy import deepcopy
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

from fastrag.exceptions import ParseError, TransientParseError
from fastrag.ingestion.cache import CacheStore
from fastrag.ingestion.filters import MediaTypeFilter
from fastrag.ingestion.models import Source
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext
from fastrag.ingestion.steps.parsing.base import Parser
from fastrag.models import Document

#: Audio media types :class:`AudioParser` is meant to be routed to, including
#: the aliases different platforms' ``mimetypes`` tables report.
AUDIO_MEDIA_TYPES: frozenset[str] = frozenset(
    {
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/vnd.wave",
        "audio/mpeg",
        "audio/mp3",
        "audio/mp4",
        "audio/x-m4a",
        "audio/m4a",
        "audio/aac",
        "audio/ogg",
        "audio/opus",
        "audio/flac",
        "audio/x-flac",
        "audio/webm",
    }
)

#: The filter a :class:`~fastrag.ingestion.steps.parsing.router.ParserRouter`
#: route to :class:`AudioParser` matches against.
#:
#: .. code-block:: python
#:
#:     parser = ParserRouter(cache=cache).route(audio_filter, AudioParser(...))
audio_filter = MediaTypeFilter(include_types=list(AUDIO_MEDIA_TYPES))


class AudioParser(Parser):
    """Transcribe audio sources to text with an OpenAI-compatible Whisper endpoint.

    ``base_url`` is the API root (e.g. ``https://host/v1``); the request goes to
    ``{base_url}/audio/transcriptions`` as multipart form data. Credentials are
    excluded from cache identity; model, endpoint, language, prompt and extra
    form fields are included.
    """

    accepted_media_types = frozenset({"audio/*", "video/webm", "video/mp4"})

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        language: str | None = None,
        prompt: str | None = None,
        timeout: float = 600,
        extra_body: Mapping[str, Any] | None = None,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
    ) -> None:
        """Configure a transcription model and its OpenAI base URL."""
        super().__init__(cache=cache, policy=policy)
        self.model, self.api_key, self.base_url = model, api_key, base_url
        self.language, self.prompt, self.timeout = language, prompt, timeout
        self.extra_body = deepcopy(dict(extra_body or {}))

    def config(self) -> Mapping[str, Any]:
        """Return transcription-affecting settings without credentials."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            **({"language": self.language} if self.language else {}),
            **({"prompt": self.prompt} if self.prompt else {}),
            **({"extra_body": self.extra_body} if self.extra_body else {}),
        }

    async def transcribe(self, content: bytes, filename: str, content_type: str) -> str:
        """POST audio bytes to the transcription endpoint and return the text."""
        data: dict[str, Any] = {"model": self.model, "response_format": "json"}
        if self.language:
            data["language"] = self.language
        if self.prompt:
            data["prompt"] = self.prompt
        data.update({k: str(v) for k, v in self.extra_body.items()})
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.base_url.rstrip("/") + "/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    data=data,
                    files={"file": (filename, content, content_type)},
                )
                response.raise_for_status()
                text = response.json()["text"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("empty transcription")
                return text.strip()
        except Exception as error:
            # A dropped connection, a timeout or a 429/5xx is worth a retry.
            status = (
                error.response.status_code
                if isinstance(error, httpx.HTTPStatusError)
                else None
            )
            transient = isinstance(error, (httpx.TransportError, TimeoutError)) or (
                status is not None and (status == 429 or status >= 500)
            )
            detail = f"HTTP {status}" if status is not None else type(error).__name__
            raise (TransientParseError if transient else ParseError)(
                f"Audio transcription failed for model {self.model}: {detail}"
            ) from error

    async def transform(
        self, item: Source, context: ExecutionContext
    ) -> AsyncIterator[Document]:
        """Yield one transcript document for an audio source."""
        self.validate_input(item)
        if not item.content:
            return
        filename = urlsplit(item.uri).path.rsplit("/", 1)[-1] or "audio"
        content_type = (
            (item.content_type or "").split(";", 1)[0].strip()
            if item.content_type and item.content_type != "application/octet-stream"
            else mimetypes.guess_type(filename)[0] or "application/octet-stream"
        )
        yield self.build_document(
            item,
            await self.transcribe(item.content, filename, content_type),
            {"transcription_model": self.model},
        )
