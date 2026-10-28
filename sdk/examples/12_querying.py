import asyncio
import os

from _runtime import config_from_env

from fastrag.ingestion.steps.embedding import Model2VecEmbedder
from fastrag.query import PromptTemplate
from fastrag.query.llm import OpenAIChatModel
from fastrag.retrieval import VectorRetriever
from fastrag.vector_store import MilvusVectorStore


async def main():
    cfg = config_from_env()
    async with MilvusVectorStore(
        url=cfg.milvus_uri, collection=cfg.collection
    ) as vector_store:
        prompt_template = PromptTemplate(
            system="You are an expert on data spaces. Cite sources as [n].",
            template="Documents:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        )

        retriever = VectorRetriever(
            vector_store=vector_store,
            embedder=Model2VecEmbedder(model=cfg.embedding_model),
        )

        context = await retriever.retrieve(cfg.question, k=5)

        if not context:
            raise RuntimeError("No matching documents; run 05_indexing.py first")
        prompt = prompt_template.render(context=context, question=cfg.question)

        llm = OpenAIChatModel(
            model=cfg.chat_model,
            api_key=os.environ.get("FASTRAG_CHAT_API_KEY", "ollama"),
            base_url=cfg.chat_base_url,
            timeout=cfg.timeout,
        )

        answer = await llm.invoke(prompt)

        if not answer.strip():
            raise RuntimeError("Provider returned an empty answer")
        print(answer)


if __name__ == "__main__":
    asyncio.run(main())
