import asyncio

from _runtime import config_from_env, require_output

from fastrag.ingestion.cache import PostgresCache
from fastrag.ingestion.steps.fetching import SitemapCrawler


async def main():
    cfg = config_from_env()
    # docs:caching-open-start
    async with PostgresCache(url=cfg.database_url, schema=cfg.cache_schema) as cache:
        sources = SitemapCrawler(
            url=cfg.sitemap_url,
            max_urls=cfg.max_urls,
            renderer=cfg.make_renderer(),
            policy=cfg.policy(),
            cache=cache,
        )
        # docs:caching-open-end

        async for source in require_output(sources):
            print(f"Source: {source.uri}" + (" (cached)" if source.cached else ""))


if __name__ == "__main__":
    asyncio.run(main())
