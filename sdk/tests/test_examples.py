"""The runnable ``examples/`` against isolated local services and fixture providers."""

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
EXAMPLES = ROOT / "examples"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def runtime_module():
    return load_module("example_runtime", EXAMPLES / "_runtime.py")


def smoke_module():
    return load_module("example_smoke", ROOT / "scripts" / "example_smoke.py")


def test_importing_examples_does_not_run_main():
    code = """
import asyncio, runpy, sys
from pathlib import Path
sys.path.insert(0, str(Path('examples').resolve()))
def forbidden(*args, **kwargs):
    raise AssertionError('import executed an async main')
asyncio.run = forbidden
paths = sorted(Path('examples').glob('[0-9]*.py')) + sorted(Path('examples/docling').glob('*.py'))
assert paths
for path in paths:
    runpy.run_path(str(path), run_name='imported_example')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_config_reads_environment_on_each_call(monkeypatch):
    runtime = runtime_module()
    first = runtime.config_from_env()
    monkeypatch.setenv("FASTRAG_COLLECTION", "owned_fixture")
    monkeypatch.setenv("FASTRAG_MAX_URLS", "2")
    monkeypatch.setenv("FASTRAG_QUESTION", "fixture question")
    second = runtime.config_from_env()
    assert second.collection == "owned_fixture"
    assert second.max_urls == 2
    assert second.question == "fixture question"
    assert second is not first
    assert "api_key" not in repr(second)


@pytest.mark.parametrize("value", ["0", "-1", "no-number"])
def test_invalid_bounds_fail_before_execution(monkeypatch, value):
    monkeypatch.setenv("FASTRAG_MAX_URLS", value)
    with pytest.raises(ValueError):
        runtime_module().config_from_env()


async def test_empty_stream_fails():
    runtime = runtime_module()

    async def empty():
        if False:
            yield None

    with pytest.raises(RuntimeError, match="no output"):
        async for _ in runtime.require_output(empty()):
            pass


async def test_indexing_logging_and_retrieval_examples(
    http_server, cache, vector_store, embedder
):
    env = os.environ.copy()
    env.update(
        FASTRAG_SITEMAP_URL=f"{http_server}/sitemap.xml",
        FASTRAG_MAX_URLS="2",
        FASTRAG_DATABASE_URL=cache._url,  # noqa: SLF001
        FASTRAG_CACHE_SCHEMA=cache._schema_name,  # noqa: SLF001
        FASTRAG_MILVUS_URI=vector_store.url,
        FASTRAG_COLLECTION=vector_store.collection,
        FASTRAG_LOCAL_EMBEDDING_MODEL=embedder.model,
        FASTRAG_QUESTION="Page A documentation",
    )
    for script, expected in (
        ("05_indexing.py", "Document:"),
        ("06_logging.py", "Indexing"),
        ("07_advanced_logging.py", "Document saved:"),
        ("10_retrieving.py", "Source URI:"),
    ):
        process = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, str(EXAMPLES / script)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert process.returncode == 0, process.stderr
        assert expected in process.stdout
    rows = await vector_store._client.query(  # noqa: SLF001
        collection_name=vector_store.collection,
        filter="",
        limit=100,
        output_fields=["vector", "metadata"],
        consistency_level="Strong",
    )
    assert rows
    assert all(len(row["vector"]) == embedder.dimension for row in rows)
    assert all(row["metadata"]["source_uri"].startswith(http_server) for row in rows)


async def test_checked_pipeline_requires_destination_events():
    from fastrag.ingestion.steps.base import SourceStep

    class EmptySource(SourceStep):
        def config(self):
            return {}

        async def evaluate(self, context):
            if False:
                yield None

    runtime = runtime_module()
    pipeline = runtime.CheckedPipeline(EmptySource(cache=False))
    with pytest.raises(RuntimeError, match="no output at its destination"):
        async for _ in pipeline:
            pass


async def test_serving_cli_and_benchmarking_examples(
    tmp_path, http_server, cache, vector_store, embedder, chat_store
):
    smoke = smoke_module()
    with smoke.FixtureServer(tmp_path / "site") as provider:
        env = smoke.example_env(provider, prefix="unused")
        env.update(
            FASTRAG_SITEMAP_URL=f"{http_server}/sitemap.xml",
            FASTRAG_DATABASE_URL=cache._url,
            FASTRAG_CACHE_SCHEMA=cache._schema_name,
            FASTRAG_CHAT_SCHEMA=chat_store._schema_name,
            FASTRAG_MILVUS_URI=vector_store.url,
            FASTRAG_COLLECTION=vector_store.collection,
            FASTRAG_LOCAL_EMBEDDING_MODEL=embedder.model,
            FASTRAG_MAX_URLS="2",
            FASTRAG_BENCHMARK_SIZES="150,300",
            FASTRAG_BENCHMARK_RESULTS=str(tmp_path / "results"),
            FASTRAG_CLI_TOKEN_FILE=str(tmp_path / "session"),
        )
        dataset = tmp_path / "dataset.json"
        dataset.write_text(
            json.dumps(
                {
                    "samples": [
                        {
                            "question": "Page A",
                            "expected_answers": [smoke.ANSWER],
                            "relevant_ids": [f"{http_server}/page-a.html"],
                        }
                    ]
                }
            )
        )
        env["FASTRAG_BENCHMARK_DATASET"] = str(dataset)
        await asyncio.to_thread(smoke.run_script, "examples/05_indexing.py", env)
        await asyncio.to_thread(smoke.check_server_cli, env, tmp_path)
        try:
            await asyncio.to_thread(
                smoke.run_script, "examples/16_benchmarking.py", env, args=("--judge",)
            )
            identifiers = []
            for size in (150, 300):
                payload = json.loads(
                    (tmp_path / "results" / f"recursive-{size}.json").read_text()
                )
                assert payload["status"] == {
                    "successful": 1,
                    "errored": 0,
                    "incomplete": 0,
                }
                rows = await vector_store._client.query(
                    collection_name=f"{vector_store.collection}_recursive_{size}",
                    filter="",
                    limit=100,
                    output_fields=["id", "metadata"],
                    consistency_level="Strong",
                )
                assert rows
                identifiers.append({row["id"] for row in rows})
                assert all(
                    row["metadata"]["source_uri"].startswith(http_server)
                    for row in rows
                )
            # An unchanged short source can legitimately share its content-derived ID.
            # Each collection must also retain only its own configuration's distinct chunks.
            assert identifiers[0] - identifiers[1]
            assert identifiers[1] - identifiers[0]
            generation = json.loads(
                (tmp_path / "results" / "generation-300.json").read_text()
            )
            assert generation["status"] == {
                "successful": 1,
                "errored": 0,
                "incomplete": 0,
            }
        finally:
            for size in (150, 300):
                name = f"{vector_store.collection}_recursive_{size}"
                if await vector_store._client.has_collection(name):
                    await vector_store._client.drop_collection(name)
        assert provider.counts["stream"] >= 2
        assert (tmp_path / "session").exists()


async def test_docling_xlsx_examples(tmp_path):
    smoke = smoke_module()
    with smoke.FixtureServer(tmp_path / "site") as provider:
        env = smoke.example_env(provider, prefix="unused")
        parsed = await asyncio.to_thread(
            smoke.run_script, "examples/docling/parse_xlsx.py", env
        )
        assert "Fixture vision" in parsed
        assert provider.counts["vision"] >= 1
        chunks = await asyncio.to_thread(
            smoke.run_script, "examples/docling/chunk_xlsx.py", env
        )
        assert "RecursiveCharacterTextSplitter" in chunks
        assert "DoclingChunker" in chunks
        assert "Apples" in chunks


async def test_openai_compatible_embedding_provider_is_called(tmp_path):
    from fastrag.ingestion.steps.embedding import OpenAIEmbedder

    smoke = smoke_module()
    with smoke.FixtureServer(tmp_path / "site") as provider:
        embedder = OpenAIEmbedder(
            model="fixture",
            api_key="fixture",
            base_url=f"{provider.url}/v1",
            dimensions=8,
            cache=False,
        )
        vectors = await embedder.embed(["AgrospAI", "crop data"])
        assert len(vectors) == 2
        assert all(len(vector) == 8 for vector in vectors)
        assert provider.counts["embeddings"] == 1
