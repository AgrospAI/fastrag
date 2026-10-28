"""The editable FastRAG pipeline definition.

Sources, parser, chunker, embedder, vector store, retriever, chat model,
prompt and serving config. Both ``ingest.py`` and ``serve.py`` build from
these functions and constants
instead of each declaring their own — change the chunker here, and both
entry points pick it up on their next run. Nothing here reads ``os.environ``:
the two entry points read it (via ``config.Settings``) and pass the result
in, so this file stays a plain, testable declaration.

Change the chunker by editing ``CHUNK_SIZE``/``CHUNK_OVERLAP`` below, or
swap ``RecursiveCharacterTextSplitter`` for another
``fastrag.ingestion.steps.chunking`` class. Add a second source by turning
``SOURCE`` into a ``Sources([...])`` — see ``examples/08_sources.py``.
"""

from __future__ import annotations

from fastrag.ingestion.cache import CacheStore
from fastrag.ingestion.steps.chunking import RecursiveCharacterTextSplitter
from fastrag.ingestion.steps.embedding import Model2VecEmbedder
from fastrag.ingestion.steps.fetching import Fetcher, SitemapCrawler
from fastrag.ingestion.steps.parsing import (
    DefaultParser,
    DocusaurusParser,
    HtmlParser,
    ParserRouter,
    docusaurus_filter,
    html_filter,
)
from fastrag.query import PromptTemplate
from fastrag.query.llm import OpenAIChatModel
from fastrag.retrieval import VectorRetriever
from fastrag.serve import ServeConfig
from fastrag.vector_store import MilvusVectorStore

from app.config import Settings

# -- Sources ---------------------------------------------------------------
# The reference site: AgrospAI's own sitemap, already what examples/08_sources.py
# scrapes. max_urls caps a live third-party site rather than crawling the
# whole domain — raise it once the pipeline is one you trust.
SITEMAP_URL = "https://agrospai.udl.cat/en/sitemap.xml"
MAX_URLS = 200

# -- Chunking ----------------------------------------------------------------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# -- Embedding ----------------------------------------------------------------
# A small local model: no extra credential, runs on the CPU. Swap for
# Model2VecEmbedder(model="...") or OpenAIEmbedder(...) to use a different one
# — both entry points call build_embedder(), so one edit changes both.
EMBEDDING_MODEL = "minishlab/potion-base-8M"

# -- Answering ----------------------------------------------------------------
PROMPT = PromptTemplate(
    system=(
        "You are a friendly assistant answering questions about AgrospAI. Cite sources as [n]."
    ),
    template="Documents:\n{context}\n\nQuestion: {question}\n\nAnswer:",
)


def build_source(cache: CacheStore) -> Fetcher:
    """The (single, for now) source ingestion reads from."""
    return SitemapCrawler(url=SITEMAP_URL, max_urls=MAX_URLS, cache=cache)


def build_parser(cache: CacheStore) -> ParserRouter:
    """Route each fetched page to the right parser — HTML by default."""
    return (
        ParserRouter(cache=cache)
        .route(html_filter & docusaurus_filter, DocusaurusParser())
        .route(html_filter, HtmlParser())
        .default(DefaultParser())
    )


def build_chunker(cache: CacheStore) -> RecursiveCharacterTextSplitter:
    """Split parsed documents into embeddable chunks."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, cache=cache
    )


def build_embedder(cache: CacheStore | None = None) -> Model2VecEmbedder:
    """Embed chunks.

    ``cache`` is passed at ingest time only — a query-time embedder (built
    for the retriever) re-embeds each question fresh.
    """
    return Model2VecEmbedder(model=EMBEDDING_MODEL, cache=cache)


def build_vector_store(settings: Settings) -> MilvusVectorStore:
    """The vector store both ingestion and retrieval read and write."""
    return MilvusVectorStore(url=settings.milvus_uri)


def build_retriever(vector_store: MilvusVectorStore) -> VectorRetriever:
    """The retriever the serving app queries — plain vector search."""
    return VectorRetriever(vector_store=vector_store, embedder=build_embedder())


def build_chat_model(settings: Settings) -> OpenAIChatModel:
    """The chat model that turns retrieved context into an answer.

    Any OpenAI-protocol endpoint — Ollama, vLLM, llama.cpp, or OpenAI
    itself — via ``chat_base_url``.
    """
    return OpenAIChatModel(
        model=settings.chat_model,
        api_key=settings.chat_api_key,
        base_url=settings.chat_base_url,
    )


def build_serve_config(settings: Settings) -> ServeConfig:
    """The serving app's configuration, from the environment."""
    return ServeConfig(
        allowed_origins=settings.allowed_origins,
        metrics_token=settings.metrics_token,
        bearer_tokens=True,
    )
