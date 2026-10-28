"""``OpenAIChatModel``: OpenAI-compatible chat completions, via langchain-openai.

A :class:`~fastrag.query.llm.base.ChatModel` whose ``generate() -> Completion``
collapses into ``invoke() -> str`` — token counts are not part of this
contract. Any endpoint speaking the OpenAI protocol (Ollama, vLLM, llama.cpp)
works by pointing ``base_url`` at it.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from fastrag.exceptions import (
    ChatError,
    ConfigurationError,
    ContextLengthError,
    TransientChatError,
)
from fastrag.query.llm.base import ChatModel, Message

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

#: The chat class, or ``None`` when the optional "openai" extra is missing;
#: :meth:`OpenAIChatModel.__init__` refuses to run without it.
_ChatOpenAI: Any

try:
    from langchain_openai import ChatOpenAI as _ChatOpenAIImpl

    _ChatOpenAI = _ChatOpenAIImpl
except ImportError:  # pragma: no cover - exercised by installs without the extra
    _ChatOpenAI = None

MISSING_EXTRA = (
    'OpenAIChatModel needs the "openai" extra: pip install "fastrag-ai[openai]"'
)


class OpenAIChatModel(ChatModel):
    """Generate with any OpenAI-compatible endpoint.

    Choose this for OpenAI's models or for a self-hosted server that speaks
    the OpenAI protocol (vLLM, Ollama, text-generation-inference). Point
    ``base_url`` at the self-hosted endpoint; the only things that change are
    the model name and the credential the endpoint expects.

    .. code-block:: python

        import os

        from fastrag.query.llm import OpenAIChatModel

        llm = OpenAIChatModel(
            model=os.environ["FASTRAG_CHAT_MODEL"],
            api_key=os.environ["FASTRAG_CHAT_API_KEY"],
            base_url=os.environ["FASTRAG_CHAT_BASE_URL"],
        )

    Args:
        model: Model name, e.g. ``gpt-4o-mini`` or ``llama3.2:3b``.
        api_key: API key. Self-hosted endpoints usually accept any non-empty
            string.
        base_url: Endpoint base URL. ``None`` uses the OpenAI API.
        temperature: Sampling temperature passed to the provider.
        max_tokens: Cap on generated tokens. ``None`` leaves the provider's
            own default in place.
        extra_body: Additional JSON request fields; matching fields override defaults.
        timeout: Seconds per provider call. CPU-bound self-hosted models are
            far slower than a hosted API — raise this before raising retries.

    Raises:
        fastrag.exceptions.ConfigurationError: The "openai" extra is not
            installed, or ``max_tokens < 1``.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float = 60.0,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        """Validate configuration and build the underlying langchain client."""
        client = _ChatOpenAI
        if client is None:
            raise ConfigurationError(MISSING_EXTRA)
        if max_tokens is not None and max_tokens < 1:
            raise ConfigurationError("OpenAIChatModel.max_tokens must be at least 1")

        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_body = deepcopy(dict(extra_body or {}))

        kwargs: dict[str, Any] = {
            "model": model,
            "api_key": api_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Retries are the caller's job; two retry layers would multiply
            # the wait on a dead endpoint.
            "max_retries": 0,
            "timeout": timeout,
        }
        if base_url is not None:
            kwargs["base_url"] = base_url
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        self._client: ChatOpenAI = client(**kwargs)

    @property
    def model_id(self) -> str:
        """The configured model name."""
        return self.model

    async def invoke(self, messages: Sequence[Message]) -> str:
        """Generate the whole answer at once, as plain text.

        No provider exception escapes: everything is translated into the
        Fastrag hierarchy, with timeouts, rate limits, and server errors
        marked transient so a caller wrapping this in retries gets correct
        behaviour for free.
        """
        try:
            async with asyncio.timeout(self.timeout):
                response = await self._client.ainvoke(_as_tuples(messages))
        except Exception as error:
            raise _wrap(error) from error
        return str(response.text)

    async def stream(self, messages: Sequence[Message]) -> AsyncIterator[str]:
        """Generate the answer as text deltas.

        Not retried: tokens already yielded cannot be un-yielded, so a retry
        mid-stream would emit two overlapping partial answers. A stall still
        raises after ``timeout`` seconds with no delta.
        """
        try:
            async with asyncio.timeout(self.timeout):
                async for chunk in self._client.astream(_as_tuples(messages)):
                    if chunk.text:
                        yield str(chunk.text)
        except Exception as error:
            raise _wrap(error) from error


def _as_tuples(messages: Sequence[Message]) -> list[tuple[str, str]]:
    """Fastrag's own ``Message`` to langchain's ``(role, content)`` shape."""
    return [(message.role, message.content) for message in messages]


def _wrap(error: Exception) -> ChatError:
    """Translate a provider failure, marking transient causes retryable."""
    message = f"chat completion failed: {error}"
    if _is_context_length(error):
        return ContextLengthError(message)
    if _is_transient(error):
        return TransientChatError(message)
    return ChatError(message)


def _is_context_length(error: Exception) -> bool:
    """Best-effort detection of a context-length failure.

    Detection is by provider error code (``context_length_exceeded``) with a
    message-substring fallback, and is best-effort by nature — a provider
    that reports it as a plain 400 lands on ``ChatError`` (``MODEL_ERROR``)
    instead. It is **not** a transient error: retrying an over-long prompt
    produces the same failure.
    """
    code = getattr(error, "code", None)
    if isinstance(code, str) and "context_length" in code:
        return True
    message = str(error).lower()
    return "context_length_exceeded" in message or "maximum context length" in message


def _is_transient(error: Exception) -> bool:
    """Return True for a timeout, rate limit, or server error; a 4xx is not."""
    if isinstance(error, TimeoutError):
        return True
    # openai's APITimeoutError and APIConnectionError are not builtin
    # TimeoutErrors, but both are connection-level failures worth retrying.
    if type(error).__name__ in {"APITimeoutError", "APIConnectionError"}:
        return True
    status = getattr(error, "status_code", None)
    return isinstance(status, int) and (status == 429 or 500 <= status <= 599)
