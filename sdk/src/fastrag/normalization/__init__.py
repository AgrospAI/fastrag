"""Extensible source-preserving normalization separate from ingestion parsers."""

from __future__ import annotations

from fastrag.models import Document
from fastrag.normalization.html import HtmlNormalizer
from fastrag.normalization.markdown import MarkdownNormalizer
from fastrag.normalization.models import (
    CoverageUnit,
    Diagnostic,
    Exclusion,
    NormalizedDocument,
    NormalizedNode,
    Normalizer,
    SourceSpan,
    _Node,
    finish,
)
from fastrag.normalization.rst import RstNormalizer


class TextNormalizer:
    """Conservative whole-source text block without MIME sniffing."""

    identity = "text:original:v1"

    def normalize(self, document: Document) -> NormalizedDocument:
        """Retain the entire text unchanged."""
        return finish(
            document,
            "text",
            [_Node("paragraph", 0, len(document.content))] if document.content else [],
        )


class NormalizerRegistry:
    """Instance-local format routing with lazily constructed built-in adapters."""

    def __init__(self, *, builtins: bool = True) -> None:
        """Configure deterministic normalization options."""
        self._adapters: dict[str, Normalizer] = {}
        self._factories = (
            {
                "text": TextNormalizer,
                "markdown": MarkdownNormalizer,
                "rst": RstNormalizer,
                "html": HtmlNormalizer,
            }
            if builtins
            else {}
        )

    def register(
        self, content_format: str, adapter: Normalizer, *, replace: bool = False
    ) -> None:
        """Register a custom adapter; duplicates require explicit replacement."""
        if not replace and (
            content_format in self._adapters or content_format in self._factories
        ):
            raise ValueError(f"Format {content_format!r} already registered")
        self._factories.pop(content_format, None)
        self._adapters[content_format] = adapter

    @property
    def formats(self) -> frozenset[str]:
        """Registered representations, without loading optional dependencies."""
        return frozenset(self._adapters.keys() | self._factories.keys())

    def adapter(self, content_format: str) -> Normalizer:
        """Resolve one adapter, with actionable dependency/unknown-format errors."""
        if content_format not in self._adapters:
            if content_format not in self._factories:
                raise ValueError(
                    f"Unsupported normalization format {content_format!r}; registered: {sorted(self._adapters.keys() | self._factories.keys())}"
                )
            self._adapters[content_format] = self._factories[content_format]()
        return self._adapters[content_format]

    @property
    def identity(self) -> str:
        """Identify every route without requiring unselected optional dependencies."""
        from importlib.metadata import PackageNotFoundError, version

        def installed(package: str) -> str:
            try:
                return version(package)
            except PackageNotFoundError:
                return "unavailable"

        lazy = {
            "text": TextNormalizer.identity,
            "html": HtmlNormalizer().identity,
            "markdown": f"markdown-it:{installed('markdown-it-py')}:commonmark:tables=True:v1",
            "rst": f"docutils:{installed('docutils')}:no-includes:no-raw:no-transforms:v2",
        }
        identities = [
            (fmt, self._adapters[fmt].identity if fmt in self._adapters else lazy[fmt])
            for fmt in sorted(self._adapters.keys() | self._factories.keys())
        ]
        return "normalization-registry:v1:" + repr(identities)

    def normalize(self, document: Document) -> NormalizedDocument:
        """Route by explicit output representation, defaulting legacy docs to text."""
        fmt = str(document.metadata.get("content_format", "text"))
        result = self.adapter(fmt).normalize(document)
        result.validate()
        return result


__all__ = [
    "CoverageUnit",
    "Diagnostic",
    "Exclusion",
    "HtmlNormalizer",
    "MarkdownNormalizer",
    "NormalizedDocument",
    "NormalizedNode",
    "Normalizer",
    "NormalizerRegistry",
    "RstNormalizer",
    "SourceSpan",
    "TextNormalizer",
]
