# Retrieval

Turn a question into the documents that answer it, by searching a collection
you already indexed. `fastrag.retrieval` sits on top of a `VectorStore`
opened by you: a `VectorRetriever` embeds the query with the same model the
collection was indexed with and searches `k` nearest neighbours; a
`RerankingRetriever` over-fetches on similarity and narrows on relevance with
a `Reranker`. `examples/10_retrieving.py` runs this end to end:

```{literalinclude} /../../examples/10_retrieving.py
:language: python
```

A retrieved `Document` is the shared `fastrag.models.Document`: its text is
`doc.content`, and its provenance for a citation lives in
`doc.metadata["source_uri"]` (stamped by the parser before chunking and
embedded). The relevance a store assigned it is in the reserved metadata keys
`score` and `rank`, which `VectorStore.search` stamps.

## Retrievers

`Retriever.retrieve(query, *, k=...)` returns `list[Document]`, best first.

- **`VectorRetriever`** — dense similarity search. `embedder.embed` runs on
  one query as one call, never as a pipeline step.
- **`RerankingRetriever`** — composes a `vector_store`, an `embedder`, and a
  `Reranker` (e.g. `CohereReranker`). Retrieves `k` candidates, then reranks
  down to `top_k` with a cross-encoder that scores each chunk against the
  query directly.

A `Retriever` is what {doc}`querying`'s `QueryEngine` consumes — pass any
retriever in and the engine knows retrieval only by its contract, not by its
strategy.

## See also

{doc}`querying` for retrieval plus generation; {doc}`indexing` for how the
collection is written; `examples/10_retrieving.py` and
`examples/11_reranking.py` for both retrievers end to end.

## BM25, dense retrieval and RRF

`BM25Retriever(store=...)` takes any `LexicalStore`. Both
`OpenSearchVectorStore` and `MilvusVectorStore(enable_bm25=True)` implement it
alongside `VectorStore`, so one open store can serve both branches. With a
vector store that has no BM25, such as `ChromaVectorStore`, pass a separate
`LexicalStore` to `BM25Retriever` instead. The retrievers and fusion strategy are SDK abstractions;
switching stores leaves the retrieval and reranking composition unchanged:

```python
from fastrag.retrieval import (
    BM25Retriever,
    VectorRetriever,
    HybridRetriever,
    ReciprocalRankFusion,
    RerankingRetriever,
)

hybrid = HybridRetriever(
    retrievers=[
        BM25Retriever(store=store),
        VectorRetriever(vector_store=store, embedder=embedder),
    ],
    fusion=ReciprocalRankFusion(rank_constant=60),
    candidate_k=40,
)
retriever = RerankingRetriever(
    retriever=hybrid,
    reranker=reranker,
    candidate_k=40,
)
documents = await retriever.retrieve("What is AgrospAI?", k=5)
```

Branches run concurrently. Fusion deduplicates by document id and sums
`1 / (rank_constant + one_based_rank)` for each ranking. It uses positions,
not backend scores, which are on different scales. Implement `RankFusion.fuse`
to replace RRF, or supply any `Retriever` as a branch. Each stage returns
`Document` objects with copied `score` and zero-based `rank` metadata.

`RerankingRetriever` continues to accept the original `vector_store` /
`embedder` constructor pair. With `retriever=...` it wraps any retrieval
strategy; `candidate_k` sets the over-fetch size independently of the final
`k` consumed by `QueryEngine`.

## Multilingual sites

A site that serves each page in several languages gets every page indexed
once per language, and a multilingual embedder and reranker score its
translations alike. The top results then repeat one passage in several
languages. `LanguageAwareRetriever` wraps the ranking and fixes both:

```python
from fastrag.retrieval import LanguageAwareRetriever, LinguaLanguageDetector

retriever = LanguageAwareRetriever(
    retriever=RerankingRetriever(retriever=hybrid, reranker=reranker),
    detector=LinguaLanguageDetector(["en", "es", "ca", "zh"]),
    candidate_k=40,
)
```

It fetches `candidate_k` candidates and detects the language of the question
and of each document (`metadata["language"]`, when set, wins). Documents whose
`source_uri` differs only by a leading locale segment (`/docs/a`, `/en/docs/a`)
are one page, and only one translation is kept: the question's, when it was
retrieved. Documents in other languages then have their `score` scaled by
`other_language_weight` (0.8), so they yield to results of similar relevance
in the question's language, but not to much weaker ones. Each returned
document carries `metadata["language"]`, which the default prompt shows next
to its title. `LinguaLanguageDetector` needs the `language` extra; naming only
the corpus's languages keeps short questions reliable.

OpenSearch filters are JSON-encoded Query DSL objects. Milvus filters are
native expression strings. Chroma filters are JSON-encoded `where` objects.
`BM25Retriever(filter=...)` passes the filter
through to lexical search. OpenSearch stores arbitrary metadata in `_source`
without indexing it; filter on explicit fields such as `source_uri` and
`source_id`.

Backend APIs: [OpenSearch k-NN](https://docs.opensearch.org/latest/query-dsl/specialized/k-nn/index/)
and [Milvus full-text search](https://milvus.io/docs/full-text-search.md).

## Resolve original sections

For structural hits, keep normalized parent documents in `SectionIndex` and
call `resolve(hit, ancestors=True)` to obtain original retained source sections
with document-scoped provenance. Merged chunks can resolve several sections;
missing parents raise unless you select `missing="skip"`. This database-independent
API does not alter vectors or prompts and needs application-owned parent
persistence and context budgeting. See {doc}`structure` for a runnable recipe.
