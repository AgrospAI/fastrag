"""Chunking: parsed documents in, embeddable chunks out.

``DoclingChunker`` is an opt-in chunker for a
:attr:`~fastrag.models.ContentFormat.DOCLING`-format document (task 011's
``DoclingParser(output_format="docling")``): it chunks the real
``DoclingDocument`` structure with one of Docling's own chunkers, instead of
re-splitting flattened text — never a ``ChunkerRouter``
default, the same reasoning
:class:`~fastrag.ingestion.steps.parsing.docling.DoclingParser` documents for
itself.
"""

from fastrag.ingestion.steps.chunking.base import Chunker
from fastrag.ingestion.steps.chunking.character import CharacterTextSplitter
from fastrag.ingestion.steps.chunking.docling import DoclingChunker
from fastrag.ingestion.steps.chunking.embedding_semantic import EmbeddingSemanticChunker
from fastrag.ingestion.steps.chunking.heading import HeadingChunker
from fastrag.ingestion.steps.chunking.langchain_semantic import LangchainSemanticChunker
from fastrag.ingestion.steps.chunking.recursive import RecursiveCharacterTextSplitter
from fastrag.ingestion.steps.chunking.router import ChunkerRouter

from fastrag.ingestion.steps.chunking.refinement import (
    ChatBoundaryRefiner,
    RefinementResult,
)
from fastrag.ingestion.steps.chunking.semantic import SemanticChunker
from fastrag.ingestion.steps.chunking.structure import (
    StructureAwareChunker,
    TiktokenTokenizer,
    Tokenizer,
    Utf8ByteTokenizer,
    build_embedding_text,
)

__all__ = [
    "ChatBoundaryRefiner",
    "RefinementResult",
    "ChunkValidation",
    "validate_structure_chunks",
    "CharacterTextSplitter",
    "Chunker",
    "ChunkerRouter",
    "DoclingChunker",
    "HeadingChunker",
    "LangchainSemanticChunker",
    "EmbeddingSemanticChunker",
    "RecursiveCharacterTextSplitter",
    "SemanticChunker",
    "StructureAwareChunker",
    "Tokenizer",
    "TiktokenTokenizer",
    "Utf8ByteTokenizer",
    "build_embedding_text",
]

from fastrag.ingestion.steps.chunking.validation import (
    ChunkValidation,
    validate_structure_chunks,
)
