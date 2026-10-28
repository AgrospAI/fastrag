"""Sweep chunk sizes, score retrieval with typed metrics, compare every run.

End to end on the new stack: build and sync one collection per chunk size
through the lazy pipeline + ``MilvusVectorStore``, retrieve with a
``VectorRetriever``, score it with ``fastrag.benchmark``'s typed retrieval
metrics, save each run, and put every run in one table.

    docker compose up -d
    uv run python examples/16_benchmarking.py

Because ``examples/05_indexing.py`` and ``examples/10_retrieving.py``
already ran against this stack, the syncs on the default path are cached —
only chunking and embedding repeat per new ``chunk_size`` in the sweep.

Retrieval metrics only on the default path: deterministic, free, fast, no
chat model. Pass ``--judge`` to also run one generation pass through the
Compose Ollama model, demonstrating the ``Faithfulness`` / ``TokenOverlap``
path — this is minutes, not seconds, on CPU inference, and a 3B judge is a
demonstration of the plumbing, not a trustworthy score:

    uv run python examples/16_benchmarking.py --judge

Connection, model, chunk sizes, dataset and results accept environment overrides.
"""

import asyncio
import sys
import os

from _runtime import ExampleConfig, CheckedPipeline, config_from_env
from pathlib import Path

from fastrag.benchmark import Benchmark, Comparison, QADataset
from fastrag.benchmark.metrics import (
    MRR,
    NDCG,
    Faithfulness,
    HitRate,
    Latency,
    Precision,
    Recall,
    TokenOverlap,
)
from fastrag.ingestion.cache import PostgresCache
from fastrag.ingestion.steps.chunking import RecursiveCharacterTextSplitter
from fastrag.ingestion.steps.embedding import Model2VecEmbedder
from fastrag.ingestion.steps.fetching import SitemapCrawler
from fastrag.ingestion.steps.parsing import (
    DefaultParser,
    DocusaurusParser,
    HtmlParser,
    ParserRouter,
    docusaurus_filter,
    html_filter,
)
from fastrag.query import QueryEngine
from fastrag.query.llm import OpenAIChatModel
from fastrag.retrieval import VectorRetriever
from fastrag.vector_store import MilvusVectorStore


def benchmark_settings():
    sizes = tuple(
        int(value)
        for value in os.environ.get("FASTRAG_BENCHMARK_SIZES", "500,800,1200").split(
            ","
        )
    )
    if not sizes or any(size < 2 for size in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError("benchmark sizes must be distinct integers >= 2")
    results = Path(
        os.environ.get(
            "FASTRAG_BENCHMARK_RESULTS", str(Path(__file__).parent / "results")
        )
    )
    dataset = Path(
        os.environ.get(
            "FASTRAG_BENCHMARK_DATASET",
            str(Path(__file__).parent / "eval" / "agrospai.json"),
        )
    )
    return sizes, results, dataset


def require_success(result):
    if (
        result.status.successful < 1
        or result.status.errored
        or result.status.incomplete
    ):
        raise RuntimeError(f"Benchmark did not complete successfully: {result.status}")


async def build_index(
    chunk_size: int,
    *,
    cfg: ExampleConfig,
    cache: PostgresCache,
    vector_store: MilvusVectorStore,
    embedder: Model2VecEmbedder,
) -> None:
    """Build one explicitly selected collection through the original lazy graph."""
    sources = SitemapCrawler(
        url=cfg.sitemap_url,
        max_urls=cfg.max_urls,
        renderer=cfg.make_renderer(),
        cache=cache,
        policy=cfg.policy(),
    )
    parser = (
        ParserRouter(cache=cache, policy=cfg.policy())
        .route(html_filter & docusaurus_filter, DocusaurusParser())
        .route(html_filter, HtmlParser())
        .default(DefaultParser())
    )
    chunker = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=min(50, chunk_size // 4),
        cache=cache,
        policy=cfg.policy(),
    )
    embedded = embedder(chunker(parser(sources)))
    await CheckedPipeline(vector_store.save(embedded)).run()


async def run_sweep(
    *,
    cache: PostgresCache,
    cfg: ExampleConfig,
    embedder: Model2VecEmbedder,
) -> None:
    """Sync one collection per chunk size, benchmark retrieval, compare."""
    sizes, results_dir, dataset_path = benchmark_settings()
    results_dir.mkdir(parents=True, exist_ok=True)
    # docs:benchmarking-run-start
    dataset = QADataset.load(dataset_path)

    benchmark = Benchmark(
        dataset=dataset,
        # key="source_uri": examples/eval/agrospai.json's ground truth is
        # page URLs, not chunk ids — chunk ids depend on the chunker
        # configuration, so they change with every chunk_size in the sweep,
        # and id-keyed ground truth would score 0.0 for most runs and look
        # like a real finding.
        metrics=[
            Recall(k=5, key="source_uri"),
            Precision(k=5, key="source_uri"),
            MRR(k=5, key="source_uri"),
            NDCG(k=5, key="source_uri"),
            HitRate(k=5, key="source_uri"),
            Latency(),
            # TokenUsage and Cost are absent on purpose: no generation runs
            # in this phase, so both would return None for every sample and
            # mark the whole run "incomplete".
        ],
        max_concurrent=4,
    )

    results = []
    for size in sizes:
        # Store identity, not a save fingerprint, isolates each configuration.
        async with MilvusVectorStore(
            url=cfg.milvus_uri, collection=f"{cfg.collection}_recursive_{size}"
        ) as vector_store:
            await build_index(
                size, cfg=cfg, cache=cache, vector_store=vector_store, embedder=embedder
            )
            print(f"collection={vector_store.collection} chunk_size={size} synced")
            retriever = VectorRetriever(vector_store=vector_store, embedder=embedder)
            result = await benchmark.run(retriever, name=f"recursive-{size}")
            require_success(result)
            result.save(results_dir / f"recursive-{size}.json")
            results.append(result)
    # docs:benchmarking-run-end

    print()
    for result in results:
        status = result.status
        print(
            f"{result.name}  {status.successful} successful / {status.errored} "
            f"errored / {status.incomplete} incomplete   {result.duration_s:.1f}s"
        )

    print()
    comparison = Comparison(results)
    print(comparison.to_markdown())
    print(f"\nbest ndcg@5: {comparison.best('ndcg@5').name}")
    names = ",".join(f"recursive-{size}" for size in sizes)
    print(f"wrote {results_dir}/{{{names}}}.json")


async def run_judge(
    *,
    cache: PostgresCache,
    vector_store: MilvusVectorStore,
    cfg: ExampleConfig,
    embedder: Model2VecEmbedder,
) -> None:
    """Generate and judge the configured middle configuration with the real chat path."""
    sizes, results_dir, dataset_path = benchmark_settings()
    size = sizes[len(sizes) // 2]
    print(f"\n--judge: generation and faithfulness on recursive-{size}")
    await build_index(
        size, cfg=cfg, cache=cache, vector_store=vector_store, embedder=embedder
    )

    chat = OpenAIChatModel(
        model=cfg.chat_model,
        api_key=os.environ.get("FASTRAG_CHAT_API_KEY", "ollama"),
        base_url=cfg.chat_base_url,
        temperature=0.0,
        max_tokens=400,
        # CPU inference is slow enough that the 60s default times out a real
        # answer — same as the old 06_querier.py.
        timeout=cfg.timeout,
    )
    retriever = VectorRetriever(vector_store=vector_store, embedder=embedder)
    engine = QueryEngine(retriever=retriever, llm=chat)

    benchmark = Benchmark(
        dataset=QADataset.load(dataset_path),
        metrics=[TokenOverlap(), Faithfulness(judge=chat), Latency()],
        # QueryEngine returns plaintext; TokenUsage cannot report counts here.
        # One request at a time: this Ollama container serves one CPU
        # inference at once, so anything higher just queues.
        max_concurrent=1,
    )
    result = await benchmark.run(engine, name=f"generation-{size}")
    require_success(result)
    result.save(results_dir / f"generation-{size}.json")

    print(f"\n{result.name}  {result.status}")
    for name, dist in result.summary.items():
        print(f"  {name}: mean={dist.mean:.3f}  missing={dist.missing}")
    print(f"wrote {results_dir}/generation-{size}.json")


async def main() -> None:
    cfg = config_from_env()
    sizes, _, _ = benchmark_settings()
    embedder = Model2VecEmbedder(model=cfg.embedding_model, policy=cfg.policy())
    async with PostgresCache(url=cfg.database_url, schema=cfg.cache_schema) as cache:
        await run_sweep(cache=cache, cfg=cfg, embedder=embedder)
        if "--judge" in sys.argv[1:]:
            size = sizes[len(sizes) // 2]
            async with MilvusVectorStore(
                url=cfg.milvus_uri, collection=f"{cfg.collection}_recursive_{size}"
            ) as vector_store:
                await run_judge(
                    cache=cache, vector_store=vector_store, cfg=cfg, embedder=embedder
                )


if __name__ == "__main__":
    asyncio.run(main())
