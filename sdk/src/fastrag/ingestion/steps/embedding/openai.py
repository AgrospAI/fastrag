"""``OpenAIEmbedder``: OpenAI-compatible embeddings, via langchain-openai.

An :class:`~fastrag.ingestion.steps.embedding.base.Embedder` that delegates to
OpenAI's embeddings API (or any OpenAI-compatible endpoint) — options and
transient-error classification as usual for an :class:`~fastrag.ingestion.steps.embedding.base.Embedder`.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from fastrag.exceptions import (
    ConfigurationError,
    EmbeddingError,
    TransientEmbeddingError,
)
from fastrag.ingestion.cache import CacheStore
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext, _materialize_item
from fastrag.ingestion.steps.embedding.base import Embedder
from fastrag.models import Document

if TYPE_CHECKING:
    from langchain_openai import OpenAIEmbeddings

try:
    from langchain_openai import OpenAIEmbeddings as _OpenAIEmbeddingsImpl

    _OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by installs without the extra
    _OPENAI_AVAILABLE = False

#: The embeddings class, or ``None`` when the optional "openai" extra is
#: missing; :meth:`OpenAIEmbedder.__init__` refuses to run without it.
_OpenAIEmbeddings: Any = _OpenAIEmbeddingsImpl if _OPENAI_AVAILABLE else None

MISSING_EXTRA = (
    'OpenAIEmbedder needs the "openai" extra: pip install "fastrag-ai[openai]"'
)

#: Output widths known without a network call. ``dimension`` resolves eagerly
#: at construction because a vector store names its collection from it; any
#: other model must pass ``dimensions=`` explicitly.
KNOWN_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(Embedder):
    """Embed with any OpenAI-compatible endpoint.

    Choose this for OpenAI's models or for a self-hosted server that speaks
    the OpenAI protocol (vLLM, Ollama, text-generation-inference). Point
    ``base_url`` at the self-hosted endpoint; the only things that change are
    the model name and, for a model Fastrag has never heard of, ``dimensions``.

    .. code-block:: python

        from fastrag.ingestion.steps.embedding import OpenAIEmbedder

        embedder = OpenAIEmbedder(
            model="text-embedding-3-small",
            api_key=api_key,
            cache=cache,
        )

    Args:
        model: Model name, e.g. ``text-embedding-3-small`` or
            ``qwen3-embedding:8b``.
        api_key: API key. Self-hosted endpoints usually accept any non-empty
            string.
        base_url: Endpoint base URL. ``None`` uses the OpenAI API. When set,
            context-length checking is disabled: langchain then posts strings
            rather than tiktoken token-id arrays, which is what an
            Ollama/vLLM-style server expects.
        dimensions: Pinned output width. ``None`` infers the width of a known
            OpenAI model and raises for anything else — a self-hosted model
            must pin it explicitly. Pinning also forces a model that can
            shorten its output (``text-embedding-3-*``) to one stable width,
            which is what keeps a collection's vectors comparable. Explicit
            values are sent to the provider and require provider/model support;
            inferred widths do not add a dimensions request field. Returned
            counts and widths are validated before indexing or querying.
        batch_size: Texts per provider request — langchain calls this
            ``chunk_size``. Distinct from ``policy.batch_size`` (how many
            documents this step's own batcher groups into one call to
            :meth:`embed`); this instead bounds a single such call, exactly
            as in the old service.
        extra_body: Additional JSON request fields; matching fields override defaults.
        timeout: Seconds per provider call.

    Raises:
        fastrag.exceptions.ConfigurationError: The "openai" extra is not
            installed, dimensions is not a positive integer, ``batch_size < 1``, or the
            dimension of ``model`` cannot be inferred and ``dimensions`` was
            not given.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 256,
        timeout: float = 60.0,
        extra_body: Mapping[str, Any] | None = None,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
    ) -> None:
        """Validate configuration, resolve the dimension, and build the client."""
        client = _OpenAIEmbeddings
        if client is None:
            raise ConfigurationError(MISSING_EXTRA)
        if dimensions is not None and (
            not isinstance(dimensions, int)
            or isinstance(dimensions, bool)
            or dimensions < 1
        ):
            raise ConfigurationError(
                "OpenAIEmbedder.dimensions must be a positive integer"
            )
        if batch_size < 1:
            raise ConfigurationError("OpenAIEmbedder.batch_size must be at least 1")

        dimension = dimensions or KNOWN_DIMENSIONS.get(model)
        if dimension is None:
            raise ConfigurationError(
                f"cannot infer the dimension of model {model!r}; pass dimensions="
            )

        super().__init__(cache=cache, policy=policy)
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.timeout = timeout
        self.extra_body = deepcopy(dict(extra_body or {}))
        self._dimension = dimension

        kwargs: dict[str, Any] = {
            "model": model,
            "api_key": api_key,
            "chunk_size": batch_size,  # langchain's name for the batch
            # Retries are Policy's job; two retry layers would multiply
            # the wait on a dead endpoint.
            "max_retries": 0,
            "timeout": timeout,
        }
        if dimensions is not None:
            kwargs["dimensions"] = dimensions
        if base_url is not None:
            kwargs["base_url"] = base_url
            # With the check on, langchain tokenizes with tiktoken and posts
            # token-id arrays; a vLLM/Ollama-style server rejects those, and
            # tiktoken has no encoding for the model anyway.
            kwargs["check_embedding_ctx_length"] = False
        if self.extra_body:
            kwargs["model_kwargs"] = {"extra_body": self.extra_body}
        self._client: OpenAIEmbeddings = client(**kwargs)

    def config(self) -> Mapping[str, Any]:
        """This embedder's configuration, for cache-identity hashing."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "dimensions": self.dimensions,
            **(
                {"dimensions_request_version": 1} if self.dimensions is not None else {}
            ),
            **({"extra_body": self.extra_body} if self.extra_body else {}),
        }

    @property
    def model_id(self) -> str:
        """The model, suffixed with the pinned dimension when there is one."""
        if self.dimensions is None:
            return self.model
        return f"{self.model}@{self.dimensions}"

    @property
    def dimension(self) -> int:
        """The width of every returned vector, resolved at construction."""
        return self._dimension

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts.

        No provider exception escapes: everything is translated into the
        Fastrag hierarchy, with timeouts, rate limits, and server errors
        marked transient so ``Policy`` can retry them.
        """
        try:
            async with asyncio.timeout(self.timeout):
                vectors = await self._client.aembed_documents(list(texts))
        except Exception as error:
            raise _wrap(error) from error
        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"embedding vector count {len(vectors)} does not match input count {len(texts)}"
            )
        for index, vector in enumerate(vectors):
            self._validate_vector(vector, str(index))
        return vectors

    def _validate_vector(self, vector: Sequence[float] | None, label: str) -> None:
        width = len(vector) if vector is not None else None
        if width != self.dimension:
            raise EmbeddingError(
                f"embedding vector {label} width {width} does not match "
                f"declared dimension {self.dimension} for model {self.model!r}; "
                "check dimensions and provider/model support"
            )

    async def _on_produced(
        self, item: Document, produced: list[Document], context: ExecutionContext
    ) -> None:
        """Validate replayed vectors before events or downstream indexing."""
        for index, document in enumerate(produced):
            document = await _materialize_item(document)
            self._validate_vector(document.embedding, document.id)
            produced[index] = document
        await super()._on_produced(item, produced, context)


def _wrap(error: Exception) -> EmbeddingError:
    """Translate a provider failure, marking transient causes retryable."""
    message = f"embedding failed: {error}"
    if _is_transient(error):
        return TransientEmbeddingError(message)
    return EmbeddingError(message)


def _is_transient(error: Exception) -> bool:
    """A timeout, rate limit, or server error is worth a retry; a 4xx is not."""
    if isinstance(error, TimeoutError):
        return True
    # openai's APITimeoutError and APIConnectionError are not builtin
    # TimeoutErrors, but both are connection-level failures worth retrying.
    if type(error).__name__ in {"APITimeoutError", "APIConnectionError"}:
        return True
    status = getattr(error, "status_code", None)
    return isinstance(status, int) and (status == 429 or 500 <= status <= 599)
