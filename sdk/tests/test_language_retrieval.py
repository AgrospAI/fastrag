"""LanguageAwareRetriever: one translation per page, the question's language first."""

import pytest

from fastrag.exceptions import ConfigurationError
from fastrag.models import Document
from fastrag.retrieval import (
    LanguageAwareRetriever,
    LanguageDetector,
    LinguaLanguageDetector,
    Retriever,
)

SITE = "https://site.org"


def doc(id, uri, score, written=None, **metadata):
    """A chunk whose content names the language it is ``written`` in, for ``PrefixDetector``."""
    return Document(
        id=id,
        source_id=uri or id,
        content=f"{written}: chunk {id}",
        metadata={
            **({"source_uri": uri} if uri else {}),
            **({"score": score} if score is not None else {}),
            **metadata,
        },
    )


class PrefixDetector(LanguageDetector):
    """Reads the language off a ``"xx: ..."`` prefix; ``None`` for ``"None: ..."``."""

    languages = frozenset({"en", "es", "ca", "zh"})

    def detect(self, texts):
        return [
            code if (code := text.split(":")[0]) in self.languages else None
            for text in texts
        ]


class StaticRetriever(Retriever):
    def __init__(self, documents):
        self.documents = documents
        self.requested = []

    async def retrieve(self, query, *, k=5):
        self.requested.append((query, k))
        return self.documents[:k]


def retriever(documents, **options):
    return LanguageAwareRetriever(
        retriever=StaticRetriever(documents), detector=PrefixDetector(), **options
    )


async def test_keeps_the_translation_in_the_question_language():
    inner = [
        doc("es1", f"{SITE}/docs/a", 0.99, "es"),
        doc("zh1", f"{SITE}/zh/docs/a", 0.98, "zh"),
        doc("en1", f"{SITE}/en/docs/a", 0.97, "en"),
        doc("ca1", f"{SITE}/ca/docs/a", 0.96, "ca"),
        doc("en2", f"{SITE}/en/docs/a", 0.95, "en"),
        doc("es2", f"{SITE}/docs/b", 0.50, "es"),
    ]
    found = await retriever(inner, candidate_k=30).retrieve("en: question", k=5)
    assert [d.id for d in found] == ["en1", "en2", "es2"]
    assert [d.metadata["rank"] for d in found] == [0, 1, 2]
    assert [d.metadata["language"] for d in found] == ["en", "en", "es"]


async def test_one_copy_of_a_file_served_under_every_locale():
    inner = [
        doc(f"{locale}pdf", f"{SITE}{locale}/files/guide.pdf", 0.9, "en")
        for locale in ("/ca", "/zh", "/en", "")
    ]
    found = await retriever(inner).retrieve("en: question")
    assert [d.metadata["source_uri"] for d in found] == [f"{SITE}/en/files/guide.pdf"]
    # Without its language under a locale, the best-ranked copy.
    found = await retriever(inner).retrieve("es: pregunta")
    assert [d.metadata["source_uri"] for d in found] == [f"{SITE}/ca/files/guide.pdf"]


async def test_other_languages_yield_to_similar_scores_only():
    inner = [
        doc("video", "https://video.org/v", 0.999, "es"),
        doc("page", f"{SITE}/en/docs/a", 0.95, "en"),
        doc("weak", f"{SITE}/en/docs/b", 0.30, "en"),
        doc("unknown", f"{SITE}/en/docs/c", 0.20, None),
    ]
    found = await retriever(inner).retrieve("en: question")
    assert [d.id for d in found] == ["page", "video", "weak", "unknown"]
    assert "language" not in found[-1].metadata
    found = await retriever(inner, other_language_weight=1).retrieve("en: q")
    assert [d.id for d in found] == ["video", "page", "weak", "unknown"]


async def test_negative_scores_are_demoted_too():
    inner = [
        doc("es", f"{SITE}/docs/a", -1.0, "es"),
        doc("en", f"{SITE}/en/docs/b", -1.1, "en"),
    ]
    found = await retriever(inner).retrieve("en: question")
    assert [d.id for d in found] == ["en", "es"]


async def test_undetected_question_or_missing_scores_keep_the_order():
    inner = [
        doc("es", f"{SITE}/docs/a", 0.9, "es"),
        doc("ca", f"{SITE}/ca/docs/a", 0.8, "ca"),
        doc("en", f"{SITE}/en/docs/b", 0.7, "en"),
        doc("none", None, 0.6, "zh"),
    ]
    found = await retriever(inner).retrieve("None: ?")
    assert [d.id for d in found] == ["es", "en", "none"]
    unscored = [
        doc("es", f"{SITE}/docs/b", None, "es"),
        doc("en", f"{SITE}/en/x", 0.1, "en"),
    ]
    found = await retriever(unscored).retrieve("en: q")
    assert [d.id for d in found] == ["es", "en"]


async def test_metadata_language_wins_and_k_bounds_the_result():
    inner = [
        doc("tagged", f"{SITE}/docs/a", 0.9, "es", language="en"),
        doc("other", f"{SITE}/docs/b", 0.8, "es"),
    ]
    wrapped = retriever(inner, candidate_k=7)
    found = await wrapped.retrieve("en: question", k=1)
    assert [(d.id, d.metadata["language"]) for d in found] == [("tagged", "en")]
    assert wrapped.retriever.requested == [("en: question", 7)]
    assert await retriever([]).retrieve("en: q") == []


@pytest.mark.parametrize(
    "options",
    [{"candidate_k": 0}, {"other_language_weight": 0}, {"other_language_weight": 1.5}],
)
def test_invalid_configuration(options):
    with pytest.raises(ConfigurationError):
        retriever([], **options)


async def test_invalid_k():
    with pytest.raises(ConfigurationError):
        await retriever([]).retrieve("q", k=0)


def test_lingua_detector_tells_close_languages_apart():
    pytest.importorskip("lingua")
    detector = LinguaLanguageDetector(["en", "es", "ca", "zh"])
    assert detector.languages == {"en", "es", "ca", "zh"}
    assert detector.detect(
        [
            "How can I obtain a participant credential?",
            "¿Cómo puedo obtener una credencial de participante?",
            "Com puc obtenir una credencial de participant?",
            "如何获取参与者凭证？",
            "",
        ]
    ) == ["en", "es", "ca", "zh", None]


@pytest.mark.parametrize("languages", [["en"], ["en", "xx"]])
def test_lingua_detector_rejects_bad_languages(languages):
    pytest.importorskip("lingua")
    with pytest.raises(ConfigurationError):
        LinguaLanguageDetector(languages)
