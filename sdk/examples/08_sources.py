import asyncio

from _runtime import config_from_env

from _runtime import CheckedPipeline as Pipeline
from fastrag.ingestion.cache import PostgresCache
from fastrag.ingestion.logger import ConsoleLogger
from fastrag.ingestion.steps.chunking import RecursiveCharacterTextSplitter
from fastrag.ingestion.steps.embedding import Model2VecEmbedder
from fastrag.ingestion.steps.fetching import SitemapCrawler, Sources, WebCrawler
from fastrag.vector_store import MilvusVectorStore
from fastrag.ingestion.steps.parsing import (
    DefaultParser,
    DocusaurusParser,
    HtmlParser,
    ParserRouter,
    docusaurus_filter,
    html_filter,
)


async def main():
    cfg = config_from_env()
    async with (
        PostgresCache(url=cfg.database_url, schema=cfg.cache_schema) as cache,
        MilvusVectorStore(
            url=cfg.milvus_uri, collection=cfg.collection
        ) as vector_store,
    ):
        sources = Sources(
            sources=[
                SitemapCrawler(
                    url=cfg.sitemap_url,
                    max_urls=cfg.max_urls,
                    renderer=cfg.make_renderer(),
                    policy=cfg.policy(),
                ),
                WebCrawler(
                    url=cfg.web_url,
                    max_urls=cfg.max_urls,
                    renderer=cfg.make_renderer(),
                    policy=cfg.policy(),
                ),
            ],
            cache=cache,
        )
        parser = (
            ParserRouter(cache=cache, policy=cfg.policy())
            .route(html_filter & docusaurus_filter, DocusaurusParser())
            .route(html_filter, HtmlParser())
            .default(DefaultParser())
        )
        chunker = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, cache=cache, policy=cfg.policy()
        )
        embedder = Model2VecEmbedder(
            model=cfg.embedding_model, cache=cache, policy=cfg.policy()
        )

        parsed_documents = parser(sources)
        chunked_documents = chunker(parsed_documents)
        embedded_documents = embedder(chunked_documents)
        saved_documents = vector_store.save(embedded_documents)

        pipeline = Pipeline(saved_documents, policy=cfg.policy())

        await ConsoleLogger(pipeline).run()


if __name__ == "__main__":
    asyncio.run(main())
