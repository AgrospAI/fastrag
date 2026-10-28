# Customizing

Every interface in {doc}`concepts` is public and designed for subclassing.
Pass a class instance where the built-in would have gone. Format normalization
also offers an instance-local registry keyed by explicit document representation;
{doc}`structure` includes a complete public-model adapter recipe. Byte parsing,
normalization and chunk packing remain separate extension points.

## A custom step

```python
from collections.abc import Iterator

from fastrag.ingestion.steps.chunking import Chunker
from fastrag.models import Document


class ParagraphChunker(Chunker):
    """Split on blank lines, packing up to `max_chars` per chunk."""

    def __init__(self, *, max_chars: int = 400, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_chars = max_chars

    def split(self, document: Document) -> Iterator[str]:
        buffer = ""
        for paragraph in document.content.split("\n\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if buffer and len(buffer) + len(paragraph) > self.max_chars:
                yield buffer
                buffer = paragraph
            else:
                buffer = f"{buffer}\n\n{paragraph}".strip()
        if buffer:
            yield buffer
```

A `Chunker` subclass implements `split(document) -> Iterator[str]` or yields
`(text, metadata)` pairs to retain custom provenance. Declare
`accepted_content_formats` for representations you support; the base and routed
chunkers reject incompatible input before execution. `asplit` supports optional
asynchronous processing, and `validate_output` checks fresh and cached output.
The base
turns that into fully identified `Document`s (stamping `id`, `source_id`,
`step_config_hash`, and the reserved `parent_id`/`chunk_index` metadata) and
drives caching, retries, and event emission. The same shape applies to the
other step kinds:

- A `Parser` subclass implements `transform` and builds output with
  `self.build_document(source, content, metadata=None)`.
- An `Embedder` subclass implements `model_id`, `dimension`, `embed(texts)`,
  and `config()`.
- A `SourceStep` subclass implements `evaluate(context)`.

`config()` covers only what changes an item's bytes — never secrets, and
never what decides _which_ items exist — because it feeds the cache key
({doc}`caching`).

## A custom service

```python
from collections.abc import Sequence

from fastrag.ingestion.steps.embedding import Embedder


class LocalEmbedder(Embedder):
    def __init__(self, model) -> None:
        self._model = model

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return (await self._model.encode(list(texts))).tolist()

    @property
    def dimension(self) -> int:
        return self._model.dimension

    @property
    def model_id(self) -> str:
        return self._model.name

    def config(self) -> dict:
        return {"model": self._model.name}
```

A service has no base-class machinery to inherit — it is a plain client for
something outside the process, so the contract is entirely in the abstract
methods. `dimension` and `model_id` must be cheap and synchronous. The same
shape applies to `VectorStore` (implement `ensure_ready`, `upsert`,
`existing_ids`, `search`), `ChatModel` (`invoke`, `stream`), and `Reranker`
(`rerank`).

## A custom retriever

```python
from fastrag.models import Document
from fastrag.retrieval import Retriever


class UnionRetriever(Retriever):
    """Merge several retrievers' hits, deduplicated by document id, best score wins."""

    def __init__(self, retrievers: list[Retriever]) -> None:
        self.retrievers = retrievers

    async def retrieve(self, query: str, *, k: int = 5) -> list[Document]:
        best: dict[str, Document] = {}
        for retriever in self.retrievers:
            for hit in await retriever.retrieve(query, k=k):
                key = hit.id or hit.content
                score = hit.metadata.get("score", 0.0)
                if key not in best or score > best[key].metadata.get("score", 0.0):
                    best[key] = hit
        return sorted(
            best.values(),
            key=lambda d: d.metadata.get("score", 0.0),
            reverse=True,
        )
```

`Retriever` is the one pipeline-kind interface meant to be subclassed
directly — `QueryEngine` and `Benchmark` are composition roots you configure
rather than extend. Anything implementing `retrieve(query, *, k) ->
list[Document]` slots into a `QueryEngine` exactly like a built-in one.

## A custom metric

```python
import re

from fastrag.benchmark.metrics import Metric


class CitesSource(Metric):
    """Fraction of the answer's sentences that carry a [n] citation."""

    name = "cites_source"
    needs_answer = True

    async def score(self, sample, outcome) -> float:
        return 1.0 if re.search(r"\[\d+\]", outcome.answer) else 0.0
```

Pass it in `metrics=[...]` alongside the built-ins ({doc}`benchmarking`).
Because metrics are objects, no registry, string name, or plugin loader is
involved — `needs_answer = True` is enough for `Benchmark` to refuse this
metric against a bare `Retriever` before running a single sample.

## See also

{doc}`concepts` for which kind a symbol belongs to before you extend it;
{doc}`ingestion`, {doc}`retrieval`, and {doc}`benchmarking` for the
built-ins each recipe above has as siblings.
