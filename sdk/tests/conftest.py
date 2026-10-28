"""Shared fixtures for the ``fastrag.ingestion`` end-to-end test suite.

By design this suite favours few
end-to-end tests exercising the real classes against real infrastructure,
not many unit tests pinned to internal structure. Every test in ``tests/``
gets its infrastructure from the four fixtures defined here — a locally
served fixture site, an isolated Postgres cache schema, an isolated Milvus
collection, and a shared embedder — and none of them reaches around these to
build its own.

No test in this suite touches the public internet: ``http_server`` is the
only origin any fetcher is ever pointed at.
"""

from __future__ import annotations

import functools
import http.server
import os
import shutil
import threading
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest
import pytest_asyncio

from fastrag.exceptions import CacheError, ChatStoreError
from fastrag.ingestion.cache import PostgresCache
from fastrag.ingestion.steps.embedding import Model2VecEmbedder
from fastrag.query.chat import PostgresChatStore
from fastrag.vector_store import MilvusVectorStore

#: The static site templates/pages served by ``http_server``.
FIXTURE_SITE = Path(__file__).parent / "fixtures" / "site"

#: Same variable names and same defaults ``.env.example`` documents for
#: ``examples/`` — the published-port coordinates of the Postgres/Milvus a
#: contributor already has up via ``docker compose up -d``. `task e2e`'s
#: `e2e-tests` service overrides both to the in-network service names/ports
#: (see ``docker-compose.yaml``); `task test` on the host sets neither, so
#: these defaults keep it working exactly as before.
DATABASE_URL = os.environ.get(
    "FASTRAG_DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/fastrag"
)
MILVUS_URL = os.environ.get("FASTRAG_MILVUS_URI", "http://localhost:19530")
EMBEDDING_MODEL = os.environ.get("FASTRAG_EMBEDDING_MODEL", "minishlab/potion-base-8M")

_UNREACHABLE = (
    "{service} is unreachable at {url} — start the project's infrastructure "
    "with `docker compose up -d` before running the test suite."
)


class _QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """``SimpleHTTPRequestHandler`` that keeps pytest's output free of access logs."""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Suppress the default per-request stderr line."""


@pytest.fixture(scope="session")
def http_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Serve ``tests/fixtures/site/`` over HTTP on an ephemeral localhost port.

    The fixture tree is copied into a session-scoped temp directory, and
    ``sitemap.xml`` is rendered from ``sitemap.xml.tmpl`` once the ephemeral
    port is known — the sitemap is the one file whose entries must be
    absolute, origin-qualified URLs, since :class:`~fastrag.ingestion.steps.fetching.sitemap.SitemapCrawler`
    fetches whatever a ``<loc>`` gives it verbatim. Every other page uses
    relative links, so they work unmodified regardless of the port.

    Yields the server's base URL (``http://127.0.0.1:PORT``, no trailing
    slash). Torn down at session end.
    """
    site_dir = tmp_path_factory.mktemp("site")
    for item in FIXTURE_SITE.iterdir():
        if item.name == "sitemap.xml.tmpl":
            continue
        if item.is_dir():
            shutil.copytree(item, site_dir / item.name)
        else:
            shutil.copy(item, site_dir / item.name)

    handler = functools.partial(_QuietHTTPRequestHandler, directory=str(site_dir))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"

    template = (FIXTURE_SITE / "sitemap.xml.tmpl").read_text()
    (site_dir / "sitemap.xml").write_text(template.replace("__BASE_URL__", base_url))

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield base_url
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


async def _drop_schema(schema: str) -> None:
    """Drop ``schema`` via a throwaway connection, independent of ``PostgresCache``.

    ``PostgresCache`` has no method for removing its own schema (only the
    rows inside it — see ``clear``/``collect_garbage``), so teardown needs a
    raw connection.
    """
    connection = await asyncpg.connect(DATABASE_URL)
    try:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    finally:
        await connection.close()


@pytest_asyncio.fixture
async def cache() -> AsyncIterator[PostgresCache]:
    """An isolated :class:`PostgresCache` on a per-run schema, dropped at teardown.

    Never writes to ``fastrag_cache``, whose warm rows ``examples/``
    depends on — every test gets ``schema=f"fastrag_test_{uuid4().hex}"``,
    unique per test function so tests are independent of execution order and
    safe to run repeatedly.

    Fails loudly, naming ``docker compose up -d``, if Postgres is
    unreachable — it does not silently skip.
    """
    schema = f"fastrag_test_{uuid4().hex}"
    store = PostgresCache(url=DATABASE_URL, schema=schema)
    try:
        await store.__aenter__()
    except CacheError as error:
        raise RuntimeError(
            _UNREACHABLE.format(service="Postgres", url=DATABASE_URL)
        ) from error

    try:
        yield store
    finally:
        await store.__aexit__(None, None, None)
        await _drop_schema(schema)


@pytest_asyncio.fixture
async def chat_store() -> AsyncIterator[PostgresChatStore]:
    """An isolated :class:`PostgresChatStore` on a per-run schema, dropped at teardown.

    Same shape as the ``cache`` fixture above: every test gets
    ``schema=f"fastrag_test_{uuid4().hex}"``, unique per test function, so
    tests are independent of execution order and safe to run repeatedly.

    Fails loudly, naming ``docker compose up -d``, if Postgres is
    unreachable — it does not silently skip.
    """
    schema = f"fastrag_test_{uuid4().hex}"
    store = PostgresChatStore(database_url=DATABASE_URL, schema=schema)
    try:
        await store.__aenter__()
    except ChatStoreError as error:
        raise RuntimeError(
            _UNREACHABLE.format(service="Postgres", url=DATABASE_URL)
        ) from error

    try:
        yield store
    finally:
        await store.__aexit__(None, None, None)
        await _drop_schema(schema)


@pytest_asyncio.fixture
async def vector_store() -> AsyncIterator[MilvusVectorStore]:
    """An isolated :class:`MilvusVectorStore` on a per-run collection, dropped at teardown.

    ``collection=f"fastrag_test_{uuid4().hex}"``, unique per test function.
    Construction and ``__aenter__`` never touch the network (the pymilvus
    client is lazy — see ``MilvusVectorStore``'s own docstring), so
    connectivity is checked explicitly here via a lightweight
    ``list_collections`` call, reaching into the underlying pymilvus client
    directly rather than ``ensure_ready`` — calling ``ensure_ready`` would
    create the collection with a throwaway dimension before the real
    embedder's width is known, and the schema cannot be changed afterward.

    Fails loudly, naming ``docker compose up -d``, if Milvus is unreachable
    — it does not silently skip.
    """
    collection = f"fastrag_test_{uuid4().hex}"
    store = MilvusVectorStore(url=MILVUS_URL, collection=collection)
    await store.__aenter__()

    try:
        await store._client.list_collections()  # noqa: SLF001 - connectivity probe
    except Exception as error:
        await store.__aexit__(None, None, None)
        raise RuntimeError(
            _UNREACHABLE.format(service="Milvus", url=MILVUS_URL)
        ) from error

    try:
        yield store
    finally:
        try:
            if await store._client.has_collection(collection_name=collection):  # noqa: SLF001
                await store._client.drop_collection(collection_name=collection)  # noqa: SLF001
        finally:
            await store.__aexit__(None, None, None)


@pytest.fixture(scope="session")
def embedder() -> Model2VecEmbedder:
    """A small :class:`Model2VecEmbedder`, the same model ``examples/`` uses.

    Session-scoped: loading the model is the only expensive part (once
    ``minishlab/potion-base-8M`` is in the local Hugging Face cache,
    construction is local deserialization, not a network call), and the
    instance itself carries no cache/schema state of its own — every test
    supplies its own isolated cache via ``Pipeline(root, cache=cache)``,
    which this embedder's default ``cache=None`` inherits.
    """
    return Model2VecEmbedder(model=EMBEDDING_MODEL)
