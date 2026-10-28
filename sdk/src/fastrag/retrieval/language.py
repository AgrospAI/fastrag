"""Retrieval that prefers documents in the language of the question.

A multilingual site holds each page once per language, and a multilingual
embedder and reranker score a page's translations about equally. So the top
results for a question are often one passage repeated in several languages,
and the answer cites a translation in a language the reader did not write in.
:class:`LanguageAwareRetriever` keeps one translation of each page, the one in
the question's language when it was retrieved, and ranks documents in other
languages below documents of similar relevance in the question's.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import replace
from math import isfinite
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from fastrag.exceptions import ConfigurationError
from fastrag.models import Document
from fastrag.retrieval.base import Retriever

try:
    from lingua import IsoCode639_1 as _IsoCode639_1
    from lingua import LanguageDetectorBuilder as _LanguageDetectorBuilder

    _LINGUA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by installs without the extra
    _LINGUA_AVAILABLE = False

#: Placeholders when the optional "language" extra is missing.
#: :class:`LinguaLanguageDetector` refuses to build until it exists.
IsoCode639_1: Any = _IsoCode639_1 if _LINGUA_AVAILABLE else None
LanguageDetectorBuilder: Any = _LanguageDetectorBuilder if _LINGUA_AVAILABLE else None

MISSING_EXTRA = (
    'LinguaLanguageDetector needs the "language" extra: '
    'pip install "fastrag-ai[language]"'
)


class LanguageDetector(ABC):
    """Names the language each text is written in.

    Attributes:
        languages: The ISO 639-1 codes :meth:`detect` can return. A URL path
            segment naming one of them is read as a locale (see
            :class:`LanguageAwareRetriever`).
    """

    languages: frozenset[str]

    @abstractmethod
    def detect(self, texts: Sequence[str]) -> list[str | None]:
        """Return the ISO 639-1 code of each text's language, or ``None`` if unsure.

        Blocking: :class:`LanguageAwareRetriever` runs it in a worker thread.
        """


class LinguaLanguageDetector(LanguageDetector):
    """Detects languages offline with Lingua, choosing among a corpus's own.

    Needs the "language" extra (``lingua-language-detector``). Naming only the
    languages the corpus is written in keeps short questions reliable: a
    short Catalan question is told apart from Spanish far more often when
    those are among a handful of candidates rather than 75.

    .. code-block:: python

        detector = LinguaLanguageDetector(["en", "es", "ca", "zh"])
        detector.detect(["How can I obtain a credential?"])  # ["en"]

    Args:
        languages: ISO 639-1 codes of the languages to choose among, at
            least two.
        minimum_relative_distance: How far ahead of the runner-up the most
            likely language must be, from ``0.0`` to ``0.99``; text that does
            not clear it gets ``None``. ``0.0`` always names a language.
    """

    def __init__(
        self, languages: Iterable[str], *, minimum_relative_distance: float = 0.0
    ) -> None:
        """Build the detector, loading each language's models on first use."""
        if LanguageDetectorBuilder is None:
            raise ConfigurationError(MISSING_EXTRA)
        self.languages = frozenset(code.strip().lower() for code in languages)
        if len(self.languages) < 2:
            raise ConfigurationError("LinguaLanguageDetector needs two languages")
        try:
            codes = [IsoCode639_1.from_str(code) for code in sorted(self.languages)]
        except ValueError as error:
            raise ConfigurationError(f"Unknown ISO 639-1 code: {error}") from error
        self._detector = (
            LanguageDetectorBuilder.from_iso_codes_639_1(*codes)
            .with_minimum_relative_distance(minimum_relative_distance)
            .build()
        )

    def detect(self, texts: Sequence[str]) -> list[str | None]:
        """Detect every text's language, in parallel."""
        detected = self._detector.detect_languages_in_parallel_of(list(texts))
        return [
            language.iso_code_639_1.name.lower() if language is not None else None
            for language in detected
        ]


class LanguageAwareRetriever(Retriever):
    """Prefer documents in the question's language, one translation per page.

    Over-fetches ``candidate_k`` documents from ``retriever``, best first,
    and detects the language of the question and of each document (a
    document's own ``metadata["language"]`` wins over detection). Then:

    1. Translations collapse. Documents whose ``source_uri`` differs only by
       a leading locale path segment naming one of ``detector.languages``
       (``/docs/a``, ``/en/docs/a``, ``/ca/docs/a``) are copies of one page,
       and only one URL's documents are kept: one written in the question's
       language, preferably under its locale, if any was retrieved, else the
       best-ranked one. The same file served under every locale collapses
       too.
    2. Documents written in another language have their ``score`` scaled by
       ``other_language_weight`` and the rest are re-sorted, so they yield to
       documents of similar relevance in the question's language but still
       outrank much weaker ones. Documents without a numeric ``score`` keep
       ``retriever``'s order.

    The first ``k`` are returned with their ``rank`` restated and their
    ``metadata["language"]`` set where known, so a prompt can show it. A
    question whose language is not detected skips step 2.

    .. code-block:: python

        retriever = LanguageAwareRetriever(
            retriever=RerankingRetriever(retriever=hybrid, reranker=reranker),
            detector=LinguaLanguageDetector(["en", "es", "ca", "zh"]),
        )

    Args:
        retriever: Ranks the candidates. A
            :class:`~fastrag.retrieval.reranking_retriever.RerankingRetriever`
            returns comparable scores across languages.
        detector: Detects the question's and the documents' languages.
        candidate_k: How many documents to fetch from ``retriever`` before
            choosing ``k``.
        other_language_weight: Factor, in ``(0, 1]``, applied to the score of
            a document written in another language than the question. A
            negative score is divided by it instead, so it always demotes.
    """

    def __init__(
        self,
        *,
        retriever: Retriever,
        detector: LanguageDetector,
        candidate_k: int = 40,
        other_language_weight: float = 0.8,
    ) -> None:
        """Store the collaborators; no I/O happens until :meth:`retrieve`."""
        if candidate_k < 1:
            raise ConfigurationError("candidate_k must be at least 1")
        if not 0 < other_language_weight <= 1:
            raise ConfigurationError("other_language_weight must be in (0, 1]")
        self.retriever = retriever
        self.detector = detector
        self.candidate_k = candidate_k
        self.other_language_weight = other_language_weight

    async def retrieve(self, query: str, *, k: int = 5) -> list[Document]:
        """Return the ``k`` best documents, preferring the question's language."""
        if k < 1:
            raise ConfigurationError("retrieval k must be at least 1")
        hits = await self.retriever.retrieve(query, k=max(k, self.candidate_k))
        if not hits:
            return []
        language, *detected = await asyncio.to_thread(
            self.detector.detect, [query, *(hit.content for hit in hits)]
        )
        hits = [
            _with_language(hit, detected_language)
            for hit, detected_language in zip(hits, detected, strict=True)
        ]
        kept = self._one_translation_per_page(hits, language)
        if language is not None and all(_score(hit) is not None for hit in kept):
            kept.sort(key=lambda hit: self._weighted_score(hit, language), reverse=True)
        return [
            replace(hit, metadata={**hit.metadata, "rank": rank})
            for rank, hit in enumerate(kept[:k])
        ]

    def _one_translation_per_page(
        self, hits: list[Document], language: str | None
    ) -> list[Document]:
        """Keep, of each page's URLs, the documents of the one ``_preference`` picks."""
        urls: dict[str, list[str]] = {}
        for hit in hits:
            if uri := hit.metadata.get("source_uri"):
                page, _ = _page(uri, self.detector.languages)
                if uri not in urls.setdefault(page, []):
                    urls[page].append(uri)
        chosen = {
            min(candidates, key=lambda uri: self._preference(uri, hits, language))
            for candidates in urls.values()
        }
        return [
            hit
            for hit in hits
            if not hit.metadata.get("source_uri")
            or hit.metadata["source_uri"] in chosen
        ]

    def _preference(
        self, uri: str, hits: list[Document], language: str | None
    ) -> tuple[bool, bool, int]:
        """Sort key, best first: in ``language``, under its locale, best-ranked."""
        rank, hit = next(
            (rank, hit)
            for rank, hit in enumerate(hits)
            if hit.metadata.get("source_uri") == uri
        )
        _, locale = _page(uri, self.detector.languages)
        return (
            language is None or hit.metadata.get("language") != language,
            language is None or locale != language,
            rank,
        )

    def _weighted_score(self, hit: Document, language: str) -> float:
        score = _score(hit)
        assert score is not None  # checked by the caller
        if hit.metadata.get("language") in (None, language):
            return score
        weight = self.other_language_weight
        return score * weight if score >= 0 else score / weight


def _with_language(hit: Document, language: str | None) -> Document:
    """``hit`` with ``metadata["language"]`` set, unless it already has one."""
    if language is None or hit.metadata.get("language"):
        return hit
    return replace(hit, metadata={**hit.metadata, "language": language})


def _score(hit: Document) -> float | None:
    score = hit.metadata.get("score")
    if (
        isinstance(score, (int, float))
        and not isinstance(score, bool)
        and isfinite(score)
    ):
        return float(score)
    return None


def _page(uri: str, languages: frozenset[str]) -> tuple[str, str | None]:
    """``uri`` without a leading locale path segment, and that locale's language.

    ``https://a.org/en/docs/x`` -> ``("https://a.org/docs/x", "en")`` when
    ``"en"`` is in ``languages``; ``zh-CN`` and ``pt_BR`` segments count as
    their primary language. Any other ``uri`` is returned unchanged, with
    ``None``.
    """
    parts = urlsplit(uri)
    segments = parts.path.split("/")
    if len(segments) < 2:
        return uri, None
    locale = segments[1].replace("_", "-").split("-")[0].lower()
    if locale not in languages:
        return uri, None
    return urlunsplit(parts._replace(path="/" + "/".join(segments[2:]))), locale
