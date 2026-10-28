# Quickstart

Index one site and ask it a question — the whole pipeline, no API key
required for the embeddings. It needs the `model2vec`, `milvus`, and `openai`
extras (`pip install "fastrag-ai[model2vec,milvus,openai]"`), a Milvus instance
to write vectors to, a PostgreSQL database for the step cache
({doc}`install`), and any OpenAI-compatible chat endpoint — a local
[Ollama](https://ollama.com), or OpenAI itself.

```python
import asyncio
import os

from fastrag.ingestion import Pipeline
from fastrag.ingestion.cache import PostgresCache
from fastrag.ingestion.logger import ConsoleLogger
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
from fastrag.query import PromptTemplate, QueryEngine
from fastrag.query.llm import OpenAIChatModel
from fastrag.retrieval import VectorRetriever
from fastrag.vector_store import MilvusVectorStore

DATABASE_URL = os.environ.get(
    "FASTRAG_DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/fastrag"
)
MILVUS_URI = os.environ.get("FASTRAG_MILVUS_URI", "http://localhost:19530")


async def main() -> None:
    async with (
        PostgresCache(url=DATABASE_URL) as cache,
        MilvusVectorStore(url=MILVUS_URI) as vector_store,
    ):
        sources = SitemapCrawler(url="https://example.com/sitemap.xml", cache=cache)
        parser = (
            ParserRouter(cache=cache)
            .route(html_filter & docusaurus_filter, DocusaurusParser())
            .route(html_filter, HtmlParser())
            .default(DefaultParser())
        )
        chunker = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, cache=cache
        )
        embedder = Model2VecEmbedder(
            model=os.environ.get("FASTRAG_EMBEDDING_MODEL", "minishlab/potion-base-8M"),
            cache=cache,
        )

        saved = vector_store.save(embedder(chunker(parser(sources))))
        await ConsoleLogger(Pipeline(saved)).run()

        retriever = VectorRetriever(vector_store=vector_store, embedder=embedder)
        engine = QueryEngine(
            retriever=retriever,
            llm=OpenAIChatModel(
                model=os.environ.get("FASTRAG_CHAT_MODEL", "llama3.2:3b"),
                api_key=os.environ.get("FASTRAG_CHAT_API_KEY", "ollama"),
                base_url=os.environ.get(
                    "FASTRAG_CHAT_BASE_URL", "http://localhost:11434/v1"
                ),
            ),
            prompt=PromptTemplate(
                system="You are an expert on data spaces. Cite sources as [n].",
                template="Documents:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            ),
        )

        answer = await engine.ask("What is this site about?")
        print(answer.text)
        print([source.metadata.get("source_uri") for source in answer.sources])


asyncio.run(main())
```

Run it again after the source changes — the `ConsoleLogger` table shows new
vs cached counts, and nothing duplicated, because chunk identity is
deterministic ({doc}`ingestion`, {doc}`caching`).

`Model2VecEmbedder` runs a small model in-process, so indexing needs no
embedding API key; swap it for `OpenAIEmbedder` when recall matters more than
a zero-dependency setup ({doc}`ingestion`). The chunker defaults here to
`RecursiveCharacterTextSplitter`, which needs the `chunkers` extra — see the
note in {doc}`install`.

`examples/` in the repository has runnable versions of each stage separately
— fetch alone, then parsing, then chunking — building up to this same
pipeline; see {doc}`examples`.

## See also

{doc}`ingestion` for the fetch/parse/chunk stages this glosses over;
{doc}`querying` for `ask` versus `stream` and how the prompt is built.
