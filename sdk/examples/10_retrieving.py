import asyncio

from _runtime import config_from_env

from fastrag.ingestion.steps.embedding import Model2VecEmbedder
from fastrag.vector_store import MilvusVectorStore
from fastrag.retrieval import VectorRetriever


async def main():
    cfg = config_from_env()
    async with MilvusVectorStore(
        url=cfg.milvus_uri, collection=cfg.collection
    ) as vector_store:
        retriever = VectorRetriever(
            vector_store=vector_store,
            embedder=Model2VecEmbedder(model=cfg.embedding_model),
        )
        documents = await retriever.retrieve(cfg.question, k=5)
        if not documents:
            raise RuntimeError("No matching documents; run 05_indexing.py first")
        print(f"Top 5 documents for query '{cfg.question}':")
        for doc in documents:
            source_uri = doc.metadata.get("source_uri", "Unknown")
            print(f"Source URI: {source_uri}")
            print(f"Content: {doc.content}")
            print("-" * 80)


if __name__ == "__main__":
    asyncio.run(main())
