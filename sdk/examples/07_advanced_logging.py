import asyncio

from _runtime import config_from_env

from _runtime import CheckedPipeline as Pipeline
from fastrag.ingestion.cache import PostgresCache
from fastrag.ingestion.events import (
    DocumentChunked,
    DocumentEmbedded,
    DocumentFetched,
    DocumentParsed,
    DocumentSaved,
    ItemFailed,
    StepCompleted,
    StepFailed,
    StepStarted,
)
from fastrag.ingestion.steps.chunking import RecursiveCharacterTextSplitter
from fastrag.ingestion.steps.embedding import Model2VecEmbedder
from fastrag.ingestion.steps.fetching import SitemapCrawler
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
        embedder = Model2VecEmbedder(
            model=cfg.embedding_model, cache=cache, policy=cfg.policy()
        )

        parsed_documents = parser(sources)
        chunked_documents = chunker(parsed_documents)
        embedded_documents = embedder(chunked_documents)
        saved_documents = vector_store.save(embedded_documents)

        pipeline = Pipeline(saved_documents, policy=cfg.policy())

        async for event in pipeline:
            match event:
                case StepStarted(step_name):
                    print(f"Step started: {step_name}")
                case StepCompleted(step_name):
                    print(f"Step completed: {step_name}")
                case StepFailed(step_name, error):
                    print(f"Step failed: {step_name}, error: {error}")
                case DocumentFetched(document):
                    print(
                        f"Document fetched: {document.id}"
                        + (" (cached)" if document.cached else "")
                    )
                case DocumentParsed(document):
                    print(
                        f"Document parsed: {document.id}"
                        + (" (cached)" if document.cached else "")
                    )
                case DocumentChunked(document):
                    print(
                        f"Document chunked: {document.id}"
                        + (" (cached)" if document.cached else "")
                    )
                case DocumentEmbedded(document):
                    print(
                        f"Document embedded: {document.id}"
                        + (" (cached)" if document.cached else "")
                    )
                case DocumentSaved(document):
                    print(
                        f"Document saved: {document.id}"
                        + (" (cached)" if document.cached else "")
                    )
                case ItemFailed(step, uri, error):
                    print(f"Item failed in step {step}: {uri}, error: {error}")


if __name__ == "__main__":
    asyncio.run(main())
