import asyncio
import os

from _runtime import config_from_env

from fastrag.ingestion.steps.embedding import Model2VecEmbedder
from fastrag.retrieval import CohereReranker, RerankingRetriever
from fastrag.vector_store import MilvusVectorStore


# Run a local Cohere-compatible server exposing /rerank with this model.


async def main():
    cfg = config_from_env()
    async with MilvusVectorStore(
        url=cfg.milvus_uri, collection=cfg.collection
    ) as vector_store:
        reranker = CohereReranker(
            model=cfg.rerank_model,
            base_url=cfg.rerank_base_url,
            api_key=os.environ.get("FASTRAG_RERANKER_API_KEY", "local"),
            timeout=cfg.timeout,
        )
        retriever = RerankingRetriever(
            vector_store=vector_store,
            embedder=Model2VecEmbedder(model=cfg.embedding_model),
            reranker=reranker,
        )
        try:
            documents = await retriever.retrieve(cfg.question, k=200, top_k=5)
            if not documents:
                raise RuntimeError("No matching documents; run 05_indexing.py first")
            print(f"Top 5 documents for query '{cfg.question}':")
            for doc in documents:
                source_uri = doc.metadata.get("source_uri", "Unknown")
                print(f"Source URI: {source_uri}")
                print(f"Content: {doc.content}")
                print("-" * 80)
        finally:
            await reranker.close()


if __name__ == "__main__":
    asyncio.run(main())
