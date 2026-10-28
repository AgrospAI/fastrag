"""Embedding-distance sentence boundaries over exact source slices."""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any, Literal

from fastrag.exceptions import ChunkingError, ConfigurationError
from fastrag.ingestion.cache import CacheStore
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext
from fastrag.ingestion.steps.chunking.base import Chunker, SplitPiece
from fastrag.ingestion.steps.embedding.base import Embedder
from fastrag.models import Document

_DEFAULTS = {
    "percentile": 95.0,
    "standard_deviation": 3.0,
    "interquartile": 1.5,
    "gradient": 95.0,
}


def _percentile(values: list[float], amount: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * amount / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class EmbeddingSemanticChunker(Chunker):
    """Select sentence cuts with an existing asynchronous SDK Embedder.

    Threshold modes mirror LangChain Experimental statistics. Requested counts
    instead select the strongest cuts exactly, with earlier offsets winning ties.
    Minimum size is in source characters and may reduce that count. This is a
    sentence splitter, with no syntax awareness or hard token budget. Both direct
    and pipeline outputs preserve all original whitespace and source order.
    """

    accepted_content_formats = frozenset(["text", "markdown"])

    def __init__(
        self,
        *,
        embeddings: Embedder,
        buffer_size: int = 1,
        breakpoint_threshold_type: Literal[
            "percentile", "standard_deviation", "interquartile", "gradient"
        ] = "percentile",
        breakpoint_threshold_amount: float | None = None,
        number_of_chunks: int | None = None,
        sentence_split_regex: str = r"(?<=[.?!])\s+",
        min_chunk_size: int | None = None,
        add_start_index: bool = False,
        batch_size: int = 64,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
    ) -> None:
        """Configure source sentence windows and deterministic distance selection."""
        for name, value, minimum in [
            ("buffer_size", buffer_size, 0),
            ("batch_size", batch_size, 1),
            ("number_of_chunks", number_of_chunks, 1),
            ("min_chunk_size", min_chunk_size, 1),
        ]:
            if value is None and name in {"number_of_chunks", "min_chunk_size"}:
                continue
            if type(value) is not int or value < minimum:
                raise ConfigurationError(f"{name} must be an integer >= {minimum}")
        if breakpoint_threshold_type not in _DEFAULTS:
            raise ConfigurationError("Unknown embedding breakpoint threshold type")
        amount = (
            _DEFAULTS[breakpoint_threshold_type]
            if breakpoint_threshold_amount is None
            else breakpoint_threshold_amount
        )
        if (
            isinstance(amount, bool)
            or not isinstance(amount, (int, float))
            or not math.isfinite(amount)
            or amount < 0
            or (
                breakpoint_threshold_type in {"percentile", "gradient"} and amount > 100
            )
        ):
            raise ConfigurationError("Invalid embedding breakpoint threshold amount")
        if type(add_start_index) is not bool:
            raise ConfigurationError("add_start_index must be a boolean")
        try:
            regex = re.compile(sentence_split_regex)
        except (re.error, TypeError) as error:
            raise ConfigurationError("Invalid sentence_split_regex") from error
        if regex.groups or regex.search("") is not None:
            raise ConfigurationError(
                "sentence_split_regex must not capture or match empty text"
            )
        if type(embeddings.dimension) is not int or embeddings.dimension < 1:
            raise ConfigurationError("Embedder dimension must be a positive integer")
        super().__init__(cache=cache, policy=policy)
        self.embeddings = embeddings
        self.buffer_size = buffer_size
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = float(amount)
        self.number_of_chunks = number_of_chunks
        self.sentence_split_regex = sentence_split_regex
        self.min_chunk_size = min_chunk_size
        self.add_start_index = add_start_index
        self.batch_size = batch_size
        self._regex = regex

    def config(self) -> Mapping[str, Any]:
        """Return strategy settings and opaque nonsecret provider identity."""
        return {
            "algorithm": "embedding-semantic:v1",
            "model": self.embeddings.model_id,
            "dimension": self.embeddings.dimension,
            "provider_config_hash": self.embeddings.config_hash,
            "buffer_size": self.buffer_size,
            "breakpoint_threshold_type": self.breakpoint_threshold_type,
            "breakpoint_threshold_amount": self.breakpoint_threshold_amount,
            "number_of_chunks": self.number_of_chunks,
            "sentence_split_regex": self.sentence_split_regex,
            "min_chunk_size": self.min_chunk_size,
            "add_start_index": self.add_start_index,
            "batch_size": self.batch_size,
        }

    def split(self, document: Document) -> Iterator[SplitPiece]:
        """Require asynchronous model evaluation before yielding any pieces."""
        self.validate_input(document)
        raise NotImplementedError(
            "EmbeddingSemanticChunker uses asplit(); consume it async"
        )

    def _offsets(self, text: str) -> list[int]:
        offsets = [0]
        for match in self._regex.finditer(text):
            if match.start() == match.end():
                raise ConfigurationError(
                    "sentence_split_regex matched zero-width source text"
                )
            if (
                match.start() > offsets[-1]
                and text[offsets[-1] : match.start()].strip()
            ):
                offsets.append(match.start())
        if len(offsets) > 1 and not text[offsets[-1] :].strip():
            offsets.pop()
        return [*offsets, len(text)]

    def _vectors(self, raw: list[list[float]], count: int) -> list[list[float]]:
        if len(raw) != count:
            raise ChunkingError(
                "Embedder returned the wrong number of sentence vectors"
            )
        result = []
        for vector in raw:
            if len(vector) != self.embeddings.dimension or any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                for v in vector
            ):
                raise ChunkingError("Invalid sentence embedding dimension or component")
            scale = max(abs(v) for v in vector)
            if scale == 0:
                raise ChunkingError(
                    "Zero sentence embedding has undefined cosine distance"
                )
            scaled = [v / scale for v in vector]
            norm = math.sqrt(math.fsum(v * v for v in scaled))
            result.append([v / norm for v in scaled])
        return result

    def _cuts(self, distances: list[float]) -> list[int]:
        if self.number_of_chunks is not None:
            return sorted(
                sorted(range(len(distances)), key=lambda i: (-distances[i], i))[
                    : self.number_of_chunks - 1
                ]
            )
        scores = distances
        mode = self.breakpoint_threshold_type
        amount = self.breakpoint_threshold_amount
        if mode == "gradient":
            if len(distances) < 2:
                return []
            scores = [
                distances[1] - distances[0],
                *[
                    (distances[i + 1] - distances[i - 1]) / 2
                    for i in range(1, len(distances) - 1)
                ],
                distances[-1] - distances[-2],
            ]
        if mode in {"percentile", "gradient"}:
            threshold = _percentile(scores, amount)
        elif mode == "standard_deviation":
            threshold = statistics.fmean(scores) + amount * statistics.pstdev(scores)
        else:
            threshold = statistics.fmean(scores) + amount * (
                _percentile(scores, 75) - _percentile(scores, 25)
            )
        return [i for i, score in enumerate(scores) if score > threshold]

    async def asplit(self, document: Document) -> AsyncIterator[SplitPiece]:
        """Embed neighboring source sentences, then slice at validated offsets."""
        self.validate_input(document)
        text = document.content
        if not text.strip():
            return
        offsets = self._offsets(text)
        count = len(offsets) - 1
        cuts = []
        if count > 1 and not (
            self.number_of_chunks is None
            and self.breakpoint_threshold_type == "gradient"
            and count == 2
        ):
            vectors = []
            for start in range(0, count, self.batch_size):
                windows = [
                    text[
                        offsets[max(0, i - self.buffer_size)] : offsets[
                            min(count, i + self.buffer_size + 1)
                        ]
                    ]
                    for i in range(start, min(count, start + self.batch_size))
                ]
                vectors.extend(
                    self._vectors(await self.embeddings.embed(windows), len(windows))
                )
            distances = [
                1
                - max(
                    -1.0,
                    min(
                        1.0, math.fsum(a * b for a, b in zip(left, right, strict=True))
                    ),
                )
                for left, right in zip(vectors[:-1], vectors[1:], strict=True)
            ]
            cuts = [offsets[i + 1] for i in self._cuts(distances)]
        boundaries = [0]
        minimum = self.min_chunk_size or 0
        for cut in cuts:
            if cut - boundaries[-1] >= minimum:
                boundaries.append(cut)
        if len(boundaries) > 1 and len(text) - boundaries[-1] < minimum:
            boundaries.pop()
        boundaries.append(len(text))
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            piece = text[start:end]
            yield (
                (piece, {"start_index": start, "end_index": end})
                if self.add_start_index
                else piece
            )

    async def transform(
        self, item: Document, context: ExecutionContext
    ) -> AsyncIterator[Document]:
        """Use base-owned metadata/identity while retaining exact source whitespace."""
        chunks = []
        async for piece in self.asplit(item):
            text, extra = piece if isinstance(piece, tuple) else (piece, None)
            chunks.append(self._build_chunk(item, len(chunks), text, extra))
        for chunk in chunks:
            yield chunk

    async def validate_output(self, item, produced, context) -> None:
        """Restore source order after stores replay chunks in arbitrary ID order."""
        indices = [chunk.metadata.get("chunk_index") for chunk in produced]
        if any(type(index) is not int for index in indices) or sorted(indices) != list(
            range(len(produced))
        ):
            raise ChunkingError("Invalid embedding-semantic chunk indices")
        produced.sort(key=lambda chunk: chunk.metadata["chunk_index"])
