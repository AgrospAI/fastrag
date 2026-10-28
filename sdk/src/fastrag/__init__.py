"""fastrag — SDK for building RAG chatbots from dynamic sources."""

from importlib.metadata import PackageNotFoundError, version

from fastrag.ingestion import Pipeline, Policy, Source
from fastrag.models import ContentFormat, Document

# pyproject.toml is the only place the version is written down.
try:
    __version__ = version("fastrag-ai")
except PackageNotFoundError:  # imported from a source tree that was never installed
    __version__ = "0.0.0"

__all__ = [
    "ContentFormat",
    "Document",
    "Pipeline",
    "Policy",
    "Source",
    "__version__",
]
