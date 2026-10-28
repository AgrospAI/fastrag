"""Embedding: documents in, the same documents with vectors attached out."""

from fastrag.ingestion.steps.embedding.base import Embedder
from fastrag.ingestion.steps.embedding.model2vec import Model2VecEmbedder
from fastrag.ingestion.steps.embedding.openai import OpenAIEmbedder

__all__ = ["Embedder", "Model2VecEmbedder", "OpenAIEmbedder"]
