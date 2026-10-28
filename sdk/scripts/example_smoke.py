"""Run original examples with owned local fixtures and deterministic protocol APIs.

The APIs test protocol plumbing only. PostgreSQL/Milvus/Model2Vec/Docling/browser
remain real. Run `python scripts/test_services.py start` first. No default or
externally supplied collections/schemas are deleted by this workflow.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
from uuid import uuid4

import asyncpg
import httpx
from pymilvus import AsyncMilvusClient

ROOT = Path(__file__).resolve().parents[1]
NUMBERED = tuple(
    f"examples/{path.name}"
    for path in sorted((ROOT / "examples").glob("[0-9][0-9]_*.py"))
)
SCRIPTS = NUMBERED + (
    "examples/docling/parse_xlsx.py",
    "examples/docling/chunk_xlsx.py",
    "agrospai-docs-chatbot/fetching.py",
    "agrospai-docs-chatbot/querying.py",
)
DATABASE_URL = "postgresql://postgres@127.0.0.1:55433/fastrag_sdk_test"
MILVUS_URI = "http://127.0.0.1:19531"
ANSWER = "AgrospAI fixture documentation supports agricultural data sharing [1]."


class FixtureServer:
    """Loopback-only site, pictured workbook and compatible deterministic APIs."""

    def __init__(self, directory: Path):
        """Allocate an ephemeral loopback listener without starting its thread."""
        self.directory = directory
        self.counts: Counter[str] = Counter()
        self.embedding_requests: list[dict] = []
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            partial(FixtureHandler, fixture=self, directory=str(directory)),
        )
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        """Generate fixtures and start serving them."""
        self.directory.mkdir(parents=True, exist_ok=True)
        paths = ("/docs/overview.html", "/en/docs/guide.html", "/page.html")
        for index, path in enumerate(paths):
            target = self.directory / path.lstrip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                f"""<!doctype html><html><head><title>AgrospAI fixture {index}</title></head><body><main><div class="markdown"><h1>AgrospAI documentation</h1><p>AgrospAI supports agricultural data sharing. The fixture documents describe registration and crop information.</p><h2>Registration</h2><p>Farmers register through the agricultural data space portal.</p><a href="/page.html">Overview</a></div></main></body></html>"""
            )
        (self.directory / "index.html").write_text(
            '<html><body><h1>Data sharing homepage</h1><a href="/page.html">Overview</a><a href="/docs/overview.html">Docs</a></body></html>'
        )
        (self.directory / "sitemap.xml").write_text(
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            + "".join(f"<url><loc>{self.url}{path}</loc></url>" for path in paths)
            + "</urlset>"
        )
        (self.directory / "dataset.json").write_text(
            json.dumps(
                {
                    "samples": [
                        {
                            "question": "What is AgrospAI?",
                            "relevant_ids": [f"{self.url}/docs/overview.html"],
                            "expected_answers": [ANSWER],
                        }
                    ]
                }
            )
        )
        self._workbook()
        self.thread.start()
        return self

    def _workbook(self):
        from openpyxl import Workbook
        from openpyxl.drawing.image import Image as SheetImage
        from PIL import Image, ImageDraw

        picture = Image.new("RGB", (360, 160), "white")
        ImageDraw.Draw(picture).text(
            (15, 40), "AgrospAI crop diagram\nApples: 42", fill="black"
        )
        image_path = self.directory / "picture.png"
        picture.save(image_path)
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Harvest"
        sheet.append(["Crop", "Quantity"])
        sheet.append(["Apples", 42])
        sheet.append(["Pears", 17])
        sheet.add_image(SheetImage(str(image_path)), "D2")
        workbook.save(self.directory / "pictured.xlsx")

    def __exit__(self, *args):
        """Stop and join the owned HTTP listener."""
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("Fixture server failed to stop")


class FixtureHandler(SimpleHTTPRequestHandler):
    """Serve static fixtures and deterministic protocol responses."""

    def __init__(self, *args, fixture: FixtureServer, **kwargs):
        """Attach the owned fixture request counters."""
        self.fixture = fixture
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):  # noqa: A002
        """Keep request payloads and credentials out of logs."""
        pass

    def send_json(self, payload):
        """Write a complete JSON protocol response."""
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        """Implement fixture chat, vision, embedding and rerank protocols."""
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/chat/completions"):
            vision = any(
                isinstance(m.get("content"), list)
                and any(p.get("type") == "image_url" for p in m["content"])
                for m in request.get("messages", [])
            )
            answer = (
                "Fixture vision: AgrospAI crop diagram. Apples: 42."
                if vision
                else ANSWER
            )
            if not vision and any(
                "Respond with only a " in str(m.get("content", ""))
                for m in request.get("messages", [])
            ):
                answer = "1.0"
            key = "vision" if vision else "stream" if request.get("stream") else "chat"
            self.fixture.counts[key] += 1
            common = {
                "id": "fixture-completion",
                "created": 1,
                "model": request.get("model", "fixture"),
            }
            if request.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for piece in (answer[:25], answer[25:]):
                    chunk = {
                        **common,
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": piece},
                                "finish_reason": None,
                            }
                        ],
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                final = {
                    **common,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                self.wfile.write(
                    f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode()
                )
            else:
                self.send_json(
                    {
                        **common,
                        "object": "chat.completion",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": answer},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 12,
                            "total_tokens": 22,
                        },
                    }
                )
        elif self.path.endswith("/embeddings"):
            self.fixture.counts["embeddings"] += 1
            self.fixture.embedding_requests.append(request)
            inputs = request["input"]
            if isinstance(inputs, str) or inputs and isinstance(inputs[0], int):
                inputs = [inputs]
            dimensions = int(request.get("dimensions", 8))
            self.send_json(
                {
                    "object": "list",
                    "model": request.get("model", "fixture"),
                    "data": [
                        {
                            "object": "embedding",
                            "index": i,
                            "embedding": [1.0] + [0.0] * (dimensions - 1),
                        }
                        for i, _ in enumerate(inputs)
                    ],
                    "usage": {"prompt_tokens": 10, "total_tokens": 10},
                }
            )
        elif self.path.endswith("/rerank"):
            self.fixture.counts["rerank"] += 1
            count = min(
                len(request["documents"]),
                int(request.get("top_n", len(request["documents"]))),
            )
            self.send_json(
                {
                    "id": "fixture-rerank",
                    "results": [
                        {"index": i, "relevance_score": 1.0 / (i + 1)}
                        for i in range(count)
                    ],
                    "meta": {
                        "api_version": {"version": "1"},
                        "billed_units": {"search_units": 1},
                    },
                }
            )
        else:
            self.send_error(404)


def example_env(provider: FixtureServer, prefix: str) -> dict[str, str]:
    """Explicitly select local inputs, providers and owned service destinations."""
    env = os.environ.copy()
    # Explicit fixture credentials prevent adjacent dotenv files from selecting real providers.
    env.update(
        FASTRAG_DATABASE_URL=DATABASE_URL,
        FASTRAG_MILVUS_URI=MILVUS_URI,
        FASTRAG_CACHE_SCHEMA=f"{prefix}_cache",
        FASTRAG_CHAT_SCHEMA=f"{prefix}_chat",
        FASTRAG_COLLECTION=f"{prefix}_documents",
        FASTRAG_SITEMAP_URL=f"{provider.url}/sitemap.xml",
        FASTRAG_WEB_URL=f"{provider.url}/",
        FASTRAG_MAX_URLS="3",
        FASTRAG_QUESTION="What is AgrospAI?",
        FASTRAG_ROUTE_PATTERN=r"https?://[^/]+/(?:[^/]+/)?docs/.*",
        FASTRAG_CLI_TOKEN_FILE=str(provider.directory.parent / "session"),
        FASTRAG_TIMEOUT="60",
        FASTRAG_CHAT_BASE_URL=f"{provider.url}/v1",
        FASTRAG_CHAT_MODEL="fixture-chat",
        FASTRAG_CHAT_API_KEY="fixture",
        FASTRAG_VISION_BASE_URL=f"{provider.url}/v1",
        FASTRAG_VISION_MODEL="fixture-vision",
        FASTRAG_VISION_API_KEY="fixture",
        FASTRAG_EMBEDDING_BASE_URL=f"{provider.url}/v1",
        FASTRAG_EMBEDDING_API_KEY="fixture",
        FASTRAG_EMBEDDING_PROVIDER="model2vec",
        FASTRAG_EMBEDDING_MODEL="minishlab/potion-base-8M",
        FASTRAG_LOCAL_EMBEDDING_MODEL="minishlab/potion-base-8M",
        FASTRAG_RERANKER_BASE_URL=f"{provider.url}/v1",
        FASTRAG_RERANKER_MODEL="fixture-rerank",
        FASTRAG_RERANKER_API_KEY="fixture",
        FASTRAG_XLSX_URL=f"{provider.url}/pictured.xlsx",
        FASTRAG_CHUNKS_TO_SHOW="3",
        FASTRAG_BENCHMARK_SIZES="150,300",
        FASTRAG_BENCHMARK_DATASET=str(provider.directory / "dataset.json"),
        FASTRAG_BENCHMARK_RESULTS=str(provider.directory.parent / "results"),
        FASTRAG_INCLUDE="*docs/*",
        FASTRAG_RENDERER="http",
    )
    # Never carry through production service authentication into local fixture runs.
    for name in (
        "FASTRAG_MILVUS_USERNAME",
        "FASTRAG_MILVUS_PASSWORD",
        "FASTRAG_SESSION_TOKEN",
    ):
        env[name] = ""
    return env


def run_script(
    script: str, env: dict[str, str], *, args=(), input_text=None, timeout=120
) -> str:
    """Invoke an original script with a finite subprocess timeout and output guard."""
    process = subprocess.run(
        [sys.executable, str(ROOT / script), *args],
        cwd=ROOT,
        env=env,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if process.returncode:
        raise RuntimeError(
            f"{script} exited {process.returncode}\n{process.stdout}\n{process.stderr}"
        )
    if not process.stdout.strip():
        raise RuntimeError(f"{script} produced empty output")
    return process.stdout


def free_port():
    """Select an available loopback port for the original uvicorn server."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def check_server_cli(env: dict[str, str], directory: Path, *, run_cli=True):
    """Exercise health, authenticated SSE, actual CLI and graceful termination."""
    env = env.copy()
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    env.update(
        FASTRAG_SERVE_PORT=str(port),
        FASTRAG_SERVE_HOST="127.0.0.1",
        FASTRAG_SERVE_SECONDS="20",
        FASTRAG_BACKEND_URL=base,
    )
    env.setdefault("FASTRAG_CLI_TOKEN_FILE", str(directory / "session"))
    log_path = directory / "server.log"
    with log_path.open("w+") as log:
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "examples/14_serving.py")],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=log,
        )
        try:
            with httpx.Client(timeout=30) as client:
                deadline = time.monotonic() + 40
                while True:
                    if process.poll() is not None:
                        log.seek(0)
                        raise RuntimeError(f"Server exited before health: {log.read()}")
                    try:
                        response = client.get(f"{base}/healthz")
                        if response.status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Server did not become healthy")
                    time.sleep(0.2)
                session = client.post(f"{base}/v1/sessions", json={})
                session.raise_for_status()
                token = session.json()["session_id"]
                response = client.post(
                    f"{base}/v1/chats",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"content": env.get("FASTRAG_QUESTION", "What is AgrospAI?")},
                )
                response.raise_for_status()
                deltas = [
                    json.loads(line[6:])["delta"]["text"]
                    for line in response.text.splitlines()
                    if line.startswith("data: ") and '"delta"' in line
                ]
                if (
                    "event: message_end" not in response.text
                    or "event: sources" not in response.text
                    or "".join(deltas) != ANSWER
                ):
                    raise RuntimeError(
                        f"Authenticated SSE did not complete: {response.text}"
                    )
            if run_cli:
                output = run_script(
                    "examples/15_asking.py",
                    env,
                    input_text="What is AgrospAI?\n/quit\n",
                    timeout=50,
                )
                if ANSWER not in output:
                    raise RuntimeError(f"CLI did not consume answer: {output}")
        finally:
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                raise RuntimeError("Server failed graceful shutdown") from None
        if code != 0:
            log.seek(0)
            raise RuntimeError(f"Server shutdown exit {code}: {log.read()}")


async def assert_unowned_destinations_absent(
    env: dict[str, str], collections: set[str], schemas: set[str]
):
    """Reject improbable UUID collisions before any script or destructive cleanup."""
    client = AsyncMilvusClient(uri=env["FASTRAG_MILVUS_URI"])
    try:
        existing = set(await client.list_collections())
        if existing.intersection(collections):
            raise RuntimeError("Smoke collection ownership collision")
    finally:
        await client.close()
    connection = await asyncpg.connect(env["FASTRAG_DATABASE_URL"])
    try:
        existing_schemas = await connection.fetch(
            "SELECT nspname FROM pg_namespace WHERE nspname = ANY($1::text[])",
            list(schemas),
        )
        if existing_schemas:
            raise RuntimeError("Smoke schema ownership collision")
    finally:
        await connection.close()


async def cleanup(env: dict[str, str], collections: set[str], schemas: set[str]):
    """Remove only the exact UUID-owned destinations recorded by this run."""
    client = AsyncMilvusClient(uri=env["FASTRAG_MILVUS_URI"])
    try:
        for name in sorted(collections):
            if await client.has_collection(name):
                await client.drop_collection(name)
    finally:
        await client.close()
    connection = await asyncpg.connect(env["FASTRAG_DATABASE_URL"])
    try:
        for schema in sorted(schemas):
            # Identifiers originate solely from our UUID prefix, never external config.
            await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    finally:
        await connection.close()


def main():
    """Run selected original mains and teardown owned resources on every exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=SCRIPTS,
        action="append",
        help="Repeat to select original paths; prerequisites run automatically",
    )
    parser.add_argument("--renderer", choices=("http", "browser"), default="http")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Also run original16 generation/faithfulness phase against the deterministic fixture",
    )
    args = parser.parse_args()
    selected = set(args.only or SCRIPTS)
    prefix = f"fastrag_smoke_{uuid4().hex[:16]}"
    with tempfile.TemporaryDirectory(prefix="fastrag-smoke-") as temporary:
        directory = Path(temporary)
        with FixtureServer(directory / "site") as provider:
            env = example_env(provider, prefix)
            env["FASTRAG_RENDERER"] = args.renderer
            collections = {
                env["FASTRAG_COLLECTION"],
                f"{prefix}_agrospai",
                f"{prefix}_api_embeddings",
            } | {f"{env['FASTRAG_COLLECTION']}_recursive_{size}" for size in (150, 300)}
            schemas = {
                env["FASTRAG_CACHE_SCHEMA"],
                env["FASTRAG_CHAT_SCHEMA"],
                f"{prefix}_agro_cache",
                f"{prefix}_api_cache",
            }
            asyncio.run(assert_unowned_destinations_absent(env, collections, schemas))
            print(
                f"Owned schemas={sorted(schemas)} collections={sorted(collections)}",
                flush=True,
            )
            completed = []
            try:
                # Retrieval/serving mains require an actual index; never silently skip seeding.
                needs_index = any(path in selected for path in NUMBERED[9:15])
                if needs_index and "examples/05_indexing.py" not in selected:
                    run_script("examples/05_indexing.py", env)
                for script in SCRIPTS:
                    if script not in selected or script in (
                        "examples/14_serving.py",
                        "examples/15_asking.py",
                    ):
                        continue
                    actual_env = env.copy()
                    if script.startswith("agrospai"):
                        actual_env.update(
                            FASTRAG_COLLECTION=f"{prefix}_agrospai",
                            FASTRAG_CACHE_SCHEMA=f"{prefix}_agro_cache",
                        )
                        if (
                            script.endswith("querying.py")
                            and "agrospai-docs-chatbot/fetching.py" not in completed
                        ):
                            run_script("agrospai-docs-chatbot/fetching.py", actual_env)
                    script_args = (
                        ("--judge",)
                        if args.judge and script == "examples/16_benchmarking.py"
                        else ()
                    )
                    output = run_script(script, actual_env, args=script_args)
                    if script == "agrospai-docs-chatbot/fetching.py":
                        run_script(script, actual_env)  # actual warm replay
                        api_env = actual_env | {
                            "FASTRAG_COLLECTION": f"{prefix}_api_embeddings",
                            "FASTRAG_CACHE_SCHEMA": f"{prefix}_api_cache",
                        }
                        run_script(
                            script,
                            api_env,
                            args=(
                                "--embedding-provider",
                                "openai",
                                "--embedding-model",
                                "fixture",
                                "--embedding-dimensions",
                                "8",
                            ),
                        )
                        if not provider.counts["embeddings"]:
                            raise RuntimeError(
                                "Actual fetching did not call compatible embedding API"
                            )
                    if script.endswith("parse_xlsx.py") and (
                        not provider.counts["vision"] or "Fixture vision" not in output
                    ):
                        raise RuntimeError(
                            "Pictured XLSX enrichment did not call compatible vision API"
                        )
                    print(f"PASS {script}", flush=True)
                    completed.append(script)
                if selected.intersection(
                    {"examples/14_serving.py", "examples/15_asking.py"}
                ):
                    check_server_cli(
                        env, directory, run_cli="examples/15_asking.py" in selected
                    )
                    for script in ("examples/14_serving.py", "examples/15_asking.py"):
                        if script in selected:
                            print(f"PASS {script}", flush=True)
                            completed.append(script)
                print(
                    f"{len(completed)} original scripts completed; protocol calls={dict(provider.counts)}",
                    flush=True,
                )
            finally:
                asyncio.run(cleanup(env, collections, schemas))
                print(
                    "Owned destinations removed; fixture/token/results cleaned on context exit",
                    flush=True,
                )


if __name__ == "__main__":
    main()
