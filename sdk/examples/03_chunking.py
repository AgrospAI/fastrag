import asyncio

from _runtime import config_from_env, require_output

from fastrag.ingestion.cache import PostgresCache
from fastrag.ingestion.steps.chunking import RecursiveCharacterTextSplitter
from fastrag.ingestion.steps.fetching import SitemapCrawler
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
    async with PostgresCache(url=cfg.database_url, schema=cfg.cache_schema) as cache:
        sources = SitemapCrawler(
            url=cfg.sitemap_url,
            max_urls=cfg.max_urls,
            renderer=cfg.make_renderer(),
            policy=cfg.policy(),
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

        parsed_documents = parser(sources)
        chunked_documents = chunker(parsed_documents)

        async for document in require_output(chunked_documents):
            print(f"Document: {document.id}" + (" (cached)" if document.cached else ""))
            print(f"Content: {document.content[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
