"""``Model2VecEmbedder``: local static embeddings, computed in this process.

A local, CPU-backed :class:`~fastrag.ingestion.steps.embedding.base.Embedder`:
loads a small static model into memory and encodes in-process, so indexing
needs no embedding API key.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from fastrag.exceptions import ConfigurationError, EmbeddingError
from fastrag.ingestion.cache import CacheStore
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.embedding.base import Embedder

if TYPE_CHECKING:
    from model2vec import StaticModel

try:
    from model2vec import StaticModel as _StaticModelImpl

    _MODEL2VEC_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by installs without the extra
    _MODEL2VEC_AVAILABLE = False

#: The model class, or ``None`` when the optional "model2vec" extra is
#: missing; :meth:`Model2VecEmbedder.__init__` refuses to run without it.
_StaticModel: Any = _StaticModelImpl if _MODEL2VEC_AVAILABLE else None

MISSING_EXTRA = (
    'Model2VecEmbedder needs the "model2vec" extra: pip install "fastrag-ai[model2vec]"'
)

DEFAULT_MODEL = "minishlab/potion-base-8M"


class Model2VecEmbedder(Embedder):
    """Embed with a static model, computed in this process.

    Choose this over
    :class:`~fastrag.ingestion.steps.embedding.openai.OpenAIEmbedder` when the
    run must not depend on a server: a CI pipeline, a laptop with no endpoint,
    a demo. The model is a lookup table of token vectors averaged per text —
    no attention, no GPU, no server — which is why it is fast and why it is
    weaker than a transformer encoder on paraphrase.

    .. code-block:: python

        from fastrag.ingestion.steps.embedding import Model2VecEmbedder

        embedder = Model2VecEmbedder(model="minishlab/potion-base-8M", cache=cache)

    Args:
        model: HuggingFace model id or a local directory of weights.
            Defaults to ``minishlab/potion-base-8M``: ~30 MB, 256 dimensions.
        dimensions: Truncate the model's output to this width. ``None`` keeps
            the model's full width. Changing it also changes :attr:`model_id`
            (and therefore the document-cache identity), so a truncated model
            never shares a cached vector with the full one.
        normalize: Whether to L2-normalize vectors. ``None`` keeps the model's
            own default (potion models ship normalized).
        batch_size: Texts per model2vec ``encode`` call. Distinct from
            ``policy.batch_size`` (how many documents this step's batcher
            groups into one call to :meth:`embed`) — this instead bounds
            memory *inside* one such call, exactly as in the old service.
        token: HuggingFace access token, for gated models.

    The first construction downloads ~30 MB into ``~/.cache/huggingface``;
    every later one reads the cache, and ``HF_HUB_OFFLINE=1`` forbids the
    network outright. Pass a local directory as ``model`` for a fully
    air-gapped install.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        dimensions: int | None = None,
        normalize: bool | None = None,
        batch_size: int = 1024,
        token: str | None = None,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
    ) -> None:
        """Configure and load the model, resolving its dimension immediately.

        Raises:
            fastrag.exceptions.ConfigurationError: The "model2vec" extra is
                not installed, ``dimensions < 1``, ``batch_size < 1``, or the
                model failed to load.
        """
        static_model = _StaticModel
        if static_model is None:
            raise ConfigurationError(MISSING_EXTRA)
        if dimensions is not None and dimensions < 1:
            raise ConfigurationError("Model2VecEmbedder.dimensions must be at least 1")
        if batch_size < 1:
            raise ConfigurationError("Model2VecEmbedder.batch_size must be at least 1")

        super().__init__(cache=cache, policy=policy)
        self.model = model
        self.dimensions = dimensions
        self.normalize = normalize
        self.batch_size = batch_size
        self.token = token

        try:
            self._model: StaticModel = static_model.from_pretrained(
                model,
                token=token,
                normalize=normalize,
                dimensionality=dimensions,
                # model2vec defaults this to True, which re-downloads the
                # weights on every construction. The cached copy is the whole
                # point: after the first load this embedder touches no network.
                force_download=False,
            )
        except Exception as error:
            raise ConfigurationError(
                f"could not load model2vec model {model!r}: {error}"
            ) from error
        self._dimension = int(self._model.dim)

    def config(self) -> Mapping[str, Any]:
        """This embedder's configuration, for cache-identity hashing."""
        return {
            "model": self.model,
            "dimensions": self.dimensions,
            "normalize": self.normalize,
        }

    @property
    def model_id(self) -> str:
        """The model, suffixed with the pinned dimension when there is one."""
        if self.dimensions is None:
            return self.model
        return f"{self.model}@{self.dimensions}"

    @property
    def dimension(self) -> int:
        """The width of every returned vector, read from the loaded model."""
        return self._dimension

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts.

        The work happens in a worker thread: ``encode`` is synchronous and
        CPU-bound, and offloading it keeps other batches' concurrent work
        (cache lookups, other steps) running while this one encodes.
        """
        if not texts:
            return []
        try:
            vectors = await asyncio.to_thread(
                self._model.encode,
                list(texts),
                batch_size=self.batch_size,
                show_progress_bar=False,
                # model2vec forks worker processes above 10 000 texts; forking
                # from a worker thread inside an asyncio program is a bad
                # trade for a batch that encodes in milliseconds.
                use_multiprocessing=False,
            )
        except Exception as error:
            # No TransientEmbeddingError from this backend: a local matrix
            # lookup has no transient failure mode — if it raises, it raises
            # identically next time. Marking it retryable would make a
            # permanent bug take three times as long to fail.
            raise EmbeddingError(f"embedding failed: {error}") from error
        return vectors.tolist()
