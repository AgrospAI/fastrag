"""Serve the grounded-answer API on :8100.

The serving app is a thin adapter over the same ``QueryEngine`` /
``ChatStore`` the other examples build — nothing is fetched, parsed, or
indexed here; run ``05_indexing.py`` first to populate the Milvus
collection, then start this server, then POST a question to it:

    docker compose up -d
    uv run python examples/05_indexing.py
    uv run python examples/14_serving.py            # terminal 1
    curl -sf http://127.0.0.1:8100/healthz
    curl -sN -X POST http://127.0.0.1:8100/v1/chats \\
      -H 'content-type: application/json' -d '{"content":"What is AgrospAI?"}'
"""

import asyncio
import os

from _runtime import config_from_env

import uvicorn

from fastrag.ingestion.steps.embedding import Model2VecEmbedder
from fastrag.query import PromptTemplate
from fastrag.query.chat import PostgresChatStore
from fastrag.query.engine import QueryEngine
from fastrag.query.llm import OpenAIChatModel
from fastrag.retrieval import VectorRetriever
from fastrag.serve import ServeConfig, create_app
from fastrag.vector_store import MilvusVectorStore


async def main():
    cfg = config_from_env()
    host = os.environ.get("FASTRAG_SERVE_HOST", "127.0.0.1")
    port = int(os.environ.get("FASTRAG_SERVE_PORT", "8100"))
    lifetime = float(os.environ.get("FASTRAG_SERVE_SECONDS", "0"))
    if not 1 <= port <= 65535 or lifetime < 0:
        raise ValueError("port must be 1–65535 and lifetime must be nonnegative")
    async with (
        MilvusVectorStore(
            url=cfg.milvus_uri, collection=cfg.collection
        ) as vector_store,
        PostgresChatStore(
            database_url=cfg.database_url,
            schema=os.environ.get("FASTRAG_CHAT_SCHEMA", "fastrag_chat"),
        ) as chat_store,
    ):
        prompt_template = PromptTemplate(
            system="You are an expert on data spaces. Cite sources as [n].",
            template="Documents:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        )

        retriever = VectorRetriever(
            vector_store=vector_store,
            embedder=Model2VecEmbedder(model=cfg.embedding_model),
        )

        engine = QueryEngine(
            retriever=retriever,
            llm=OpenAIChatModel(
                model=cfg.chat_model,
                api_key=os.environ.get("FASTRAG_CHAT_API_KEY", "ollama"),
                base_url=cfg.chat_base_url,
                timeout=cfg.timeout,
            ),
            prompt=prompt_template,
        )

        # docs:serving-create-app-start
        app = create_app(
            engine=engine,
            chat_store=chat_store,
            # ServeConfig rejects a "*" wildcard (credentialed CORS cannot
            # use Access-Control-Allow-Origin: *). For local use, allow the
            # loopback origin the API is served from; a deployed instance
            # names its real origins here.
            config=ServeConfig(
                allowed_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
                # The terminal client (15_asking.py) mints its session token
                # via POST /v1/sessions, which only exists when bearer tokens
                # are enabled — the old 10_serving.py example set this for
                # exactly that reason.
                bearer_tokens=True,
            ),
        )
        # docs:serving-create-app-end

        print(f"Serving on http://{host}:{port}")
        # Serve from inside the async context managers so the stores stay
        # open for the server's lifetime. uvicorn.run() creates its own event
        # loop and would clash with asyncio.run(main()); Server.serve() runs
        # on the loop we are already on.
        config = uvicorn.Config(app, host=host, port=port, timeout_graceful_shutdown=10)
        server = uvicorn.Server(config)

        async def stop_after_lifetime():
            await asyncio.sleep(lifetime)
            server.should_exit = True

        timer = asyncio.create_task(stop_after_lifetime()) if lifetime else None
        try:
            await server.serve()
        finally:
            if timer:
                timer.cancel()
                await asyncio.gather(timer, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
