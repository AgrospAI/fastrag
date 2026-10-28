"""Serve the FastRAG API on :8000, built from ``pipeline.py``.

Two ways to run this module:

- **Production** (the image's default ``CMD``): ``python -m app.serve`` runs
  ``main()``, which opens the vector store and chat store, serves, and closes
  them cleanly on shutdown — the same shape as ``examples/14_serving.py``.
- **Development** (hot reload): ``uvicorn app.serve:build_app_sync --factory
  --reload`` — see ``Dockerfile``'s ``dev`` target. uvicorn's reloader kills
  and reimports this module on every source change under a bind-mounted
  ``app/`` and ``src/``, so ``build_app_sync`` opens the async resources with
  a throwaway event loop and deliberately never closes them: the whole
  process is about to be replaced anyway, and the OS reclaims the sockets.
  Not used in production, where ``main()``'s ``async with`` gives a proper
  graceful shutdown instead.
"""

from __future__ import annotations

import asyncio
import sys
import threading

import uvicorn
from fastapi import FastAPI

from fastrag.query.chat import PostgresChatStore
from fastrag.query.engine import QueryEngine
from fastrag.serve import create_app

from app import pipeline as pl
from app.config import MissingEnvironmentVariable, Settings, load_settings

HOST = "0.0.0.0"
PORT = 8000


def _build_engine_and_config(
    vector_store: pl.MilvusVectorStore, settings: Settings
) -> tuple[QueryEngine, pl.ServeConfig]:
    retriever = pl.build_retriever(vector_store)
    engine = QueryEngine(
        retriever=retriever, llm=pl.build_chat_model(settings), prompt=pl.PROMPT
    )
    return engine, pl.build_serve_config(settings)


async def main() -> None:
    """Read the environment, build the app, and serve it until shutdown."""
    settings = load_settings()

    async with (
        pl.build_vector_store(settings) as vector_store,
        PostgresChatStore(database_url=settings.database_url) as chat_store,
    ):
        engine, config = _build_engine_and_config(vector_store, settings)
        app = create_app(engine=engine, chat_store=chat_store, config=config)

        print(f"serving on http://{HOST}:{PORT}")
        server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=PORT))
        await server.serve()


def build_app_sync(settings: Settings | None = None) -> FastAPI:
    """Dev-only synchronous factory for ``uvicorn --reload --factory``.

    uvicorn's ``--factory`` calls this from inside the event loop its own
    reloader subprocess is already running (``Config.load()`` runs
    synchronously from within ``Server.serve()``), so a plain
    ``asyncio.run()`` here would raise "cannot be called from a running
    event loop". Instead this opens the vector store and chat store on a
    *second* event loop, run to completion on a throwaway thread, and
    returns the built app without ever closing them — see the module
    docstring for why that is fine here and nowhere else.
    """
    settings = settings or load_settings()

    async def _build() -> FastAPI:
        vector_store = pl.build_vector_store(settings)
        chat_store = PostgresChatStore(database_url=settings.database_url)
        await vector_store.__aenter__()
        await chat_store.__aenter__()
        engine, config = _build_engine_and_config(vector_store, settings)
        return create_app(engine=engine, chat_store=chat_store, config=config)

    result: dict[str, FastAPI | BaseException] = {}

    def _run() -> None:
        loop = asyncio.new_event_loop()
        try:
            result["app"] = loop.run_until_complete(_build())
        except BaseException as error:  # re-raised on the caller's thread
            result["app"] = error

    thread = threading.Thread(target=_run)
    thread.start()
    thread.join()

    built = result["app"]
    if isinstance(built, BaseException):
        raise built
    return built


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except MissingEnvironmentVariable as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
