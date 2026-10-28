"""Independent fixed-vector semantic boundaries and exact source slicing."""

import asyncio
import math

import pytest
from helpers import FixedEmbedder, doc

from fastrag.exceptions import ChunkingError, ConfigurationError, EmbeddingError
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext
from fastrag.ingestion.steps.chunking import EmbeddingSemanticChunker


async def pieces(chunker, source=None):
    return [p async for p in chunker.asplit(source or doc())]


async def test_topic_cut_exact_whitespace_and_pipeline_identity():
    text = "  Apples grow.\n\tPears grow.  Stars shine.\r\nPlanets orbit.  "
    source = doc(text)
    emb = FixedEmbedder([[1, 0], [1, 0], [0, 1], [0, 1]])
    chunker = EmbeddingSemanticChunker(
        embeddings=emb, buffer_size=0, add_start_index=True
    )
    context = ExecutionContext("test", Policy())
    chunks = [c async for c in chunker.transform(source, context)]
    cut = text.index("  Stars")
    assert [c.content for c in chunks] == [text[:cut], text[cut:]]
    assert "".join(c.content for c in chunks) == text
    assert [(c.metadata["start_index"], c.metadata["end_index"]) for c in chunks] == [
        (0, cut),
        (cut, len(text)),
    ]
    assert all(
        c.metadata["parent_id"] == source.id and c.metadata["title"] == "Guide"
        for c in chunks
    )
    assert [c.metadata["chunk_index"] for c in chunks] == [0, 1]
    assert all(c.step_config_hash == chunker.config_hash for c in chunks)


async def test_buffer_windows_and_batching_use_exact_source():
    source = doc("A.\nB.\tC.  D.")
    emb = FixedEmbedder([[1, 0]] * 4)
    chunker = EmbeddingSemanticChunker(embeddings=emb, buffer_size=1, batch_size=2)
    assert await pieces(chunker, source) == [source.content]
    assert emb.calls == [["A.\nB.", "A.\nB.\tC."], ["\nB.\tC.  D.", "\tC.  D."]]


@pytest.mark.parametrize(
    "mode,amount",
    [
        ("percentile", 50),
        ("standard_deviation", 0),
        ("interquartile", 0),
        ("gradient", 50),
    ],
)
async def test_statistical_thresholds(mode, amount):
    # Distances [0, 1, 0], gradient [1, 0, -1].
    emb = FixedEmbedder([[1, 0], [1, 0], [0, 1], [0, 1]])
    chunker = EmbeddingSemanticChunker(
        embeddings=emb,
        buffer_size=0,
        breakpoint_threshold_type=mode,
        breakpoint_threshold_amount=amount,
    )
    expected = (
        ["Apples grow.", " Pears grow. Stars shine. Planets orbit."]
        if mode == "gradient"
        else ["Apples grow. Pears grow.", " Stars shine. Planets orbit."]
    )
    assert await pieces(chunker) == expected


async def test_requested_count_ties_minimum_tail_and_clamping():
    emb = FixedEmbedder([[1, 0]] * 4)
    chunker = EmbeddingSemanticChunker(
        embeddings=emb, number_of_chunks=3, min_chunk_size=6
    )
    assert await pieces(chunker, doc("A. B. C. D.")) == ["A. B. C. D."]
    for count, expected in [
        (1, ["A. B. C. D."]),
        (3, ["A.", " B.", " C. D."]),
        (99, ["A.", " B.", " C.", " D."]),
    ]:
        assert (
            await pieces(
                EmbeddingSemanticChunker(
                    embeddings=FixedEmbedder([[1, 0]] * 4), number_of_chunks=count
                ),
                doc("A. B. C. D."),
            )
            == expected
        )


@pytest.mark.parametrize("text", ["", " \n", "One sentence", "One.\n"])
async def test_few_sentences_bypass_provider(text):
    emb = FixedEmbedder(AssertionError("must not call"))
    assert await pieces(EmbeddingSemanticChunker(embeddings=emb), doc(text)) == (
        [text] if text.strip() else []
    )
    assert not emb.calls


async def test_two_sentence_gradient_and_equal_thresholds():
    for mode in ["percentile", "gradient", "standard_deviation", "interquartile"]:
        emb = FixedEmbedder([[1, 0], [0, 1]])
        assert await pieces(
            EmbeddingSemanticChunker(embeddings=emb, breakpoint_threshold_type=mode),
            doc("A. B."),
        ) == ["A. B."]
    assert await pieces(
        EmbeddingSemanticChunker(embeddings=FixedEmbedder([[1, 0]] * 4))
    ) == [doc().content]


@pytest.mark.parametrize(
    "vectors",
    [
        [],
        [[1, 0]],
        [[0, 0]] * 4,
        [[math.nan, 1]] * 4,
        [[math.inf, 0]] * 4,
        [[1]] * 4,
        [[True, 0]] * 4,
        [["a", 0]] * 4,
    ],
)
async def test_invalid_vectors_raise(vectors):
    with pytest.raises(ChunkingError):
        await pieces(EmbeddingSemanticChunker(embeddings=FixedEmbedder(vectors)))


@pytest.mark.parametrize("scale", [1e-300, 1e300])
async def test_cosine_is_scale_stable(scale):
    emb = FixedEmbedder([[scale, 0], [scale, 0], [0, scale], [0, scale]])
    assert await pieces(EmbeddingSemanticChunker(embeddings=emb)) == [
        "Apples grow. Pears grow.",
        " Stars shine. Planets orbit.",
    ]


@pytest.mark.parametrize(
    "options",
    [
        {"buffer_size": -1},
        {"buffer_size": True},
        {"batch_size": 0},
        {"number_of_chunks": False},
        {"min_chunk_size": 0},
        {"breakpoint_threshold_type": "bad"},
        {"breakpoint_threshold_amount": math.nan},
        {"breakpoint_threshold_amount": 101},
        {
            "breakpoint_threshold_type": "interquartile",
            "breakpoint_threshold_amount": -1,
        },
        {"sentence_split_regex": "("},
        {"sentence_split_regex": "(x)"},
        {"sentence_split_regex": ""},
        {"add_start_index": 1},
    ],
)
def test_bad_configuration(options):
    with pytest.raises(ConfigurationError):
        EmbeddingSemanticChunker(embeddings=FixedEmbedder([]), **options)


async def test_actual_zero_width_regex_rejected_before_embedding():
    emb = FixedEmbedder([])
    with pytest.raises(ConfigurationError):
        await pieces(
            EmbeddingSemanticChunker(embeddings=emb, sentence_split_regex=r"(?=A)")
        )
    assert not emb.calls


@pytest.mark.parametrize("error", [EmbeddingError("failed"), asyncio.CancelledError()])
async def test_provider_failures_propagate(error):
    with pytest.raises(type(error)):
        await pieces(EmbeddingSemanticChunker(embeddings=FixedEmbedder(error)))


def test_sync_and_cache_identity():
    emb = FixedEmbedder([])
    base = EmbeddingSemanticChunker(embeddings=emb)
    with pytest.raises(NotImplementedError):
        list(base.split(doc()))
    assert not emb.calls
    for options in [
        {"buffer_size": 0},
        {"batch_size": 1},
        {"number_of_chunks": 2},
        {"min_chunk_size": 5},
        {"add_start_index": True},
        {"sentence_split_regex": r"\n+"},
        {"breakpoint_threshold_type": "gradient"},
        {"breakpoint_threshold_amount": 50},
    ]:
        assert (
            base.config_hash
            != EmbeddingSemanticChunker(embeddings=emb, **options).config_hash
        )
    assert (
        base.config_hash
        != EmbeddingSemanticChunker(
            embeddings=FixedEmbedder([], identity="other endpoint")
        ).config_hash
    )
    assert "other endpoint" not in str(
        EmbeddingSemanticChunker(
            embeddings=FixedEmbedder([], identity="other endpoint")
        ).config()
    )


@pytest.mark.parametrize(
    "mode,amount,expected",
    [
        ("interquartile", 0.5, ["A. B. C. D.", " E."]),
        ("standard_deviation", 1, ["A. B. C. D.", " E."]),
        ("standard_deviation", 3, ["A. B. C. D. E."]),
        ("percentile", 100, ["A. B. C. D. E."]),
        ("percentile", 0, ["A. B. C.", " D.", " E."]),
        ("gradient", 50, ["A. B. C.", " D.", " E."]),
    ],
)
async def test_nonzero_statistical_contract(mode, amount, expected):
    # Independent cosine distances [0, 0, .6, 1]. Mean=.4, IQR=.7.
    root = math.sqrt(0.84)
    emb = FixedEmbedder([[1, 0], [1, 0], [1, 0], [0.4, root], [-root, 0.4]])
    chunker = EmbeddingSemanticChunker(
        embeddings=emb,
        buffer_size=0,
        breakpoint_threshold_type=mode,
        breakpoint_threshold_amount=amount,
    )
    assert await pieces(chunker, doc("A. B. C. D. E.")) == expected


async def test_minimum_merges_short_final_tail_backwards():
    emb = FixedEmbedder([[1, 0], [0, 1], [0, 1]])
    chunker = EmbeddingSemanticChunker(
        embeddings=emb, number_of_chunks=3, min_chunk_size=5
    )
    assert await pieces(chunker, doc("Long first. Long second. X.")) == [
        "Long first.",
        " Long second. X.",
    ]


async def test_unicode_custom_separator_and_no_spurious_blank_sentence():
    emb = FixedEmbedder([[1, 0], [0, 1]])
    chunker = EmbeddingSemanticChunker(
        embeddings=emb, sentence_split_regex=r"\n+", number_of_chunks=2
    )
    source = doc("\nÉén\n\n二\n")
    assert await pieces(chunker, source) == ["\nÉén", "\n\n二\n"]


async def test_repeated_execution_identical_ids_and_provider_width_errors():
    emb = FixedEmbedder([[1, 0], [1, 0], [0, 1], [0, 1]] * 2)
    chunker = EmbeddingSemanticChunker(embeddings=emb)
    context = ExecutionContext("test", Policy())
    first = [c async for c in chunker.transform(doc(), context)]
    second = [c async for c in chunker.transform(doc(), context)]
    assert first == second
    with pytest.raises(ChunkingError):
        await pieces(
            EmbeddingSemanticChunker(
                embeddings=FixedEmbedder([[1, 0], [1], [0, 1], [0, 1]])
            )
        )


async def test_resolved_output_order_and_invalid_cache_indices():
    from dataclasses import replace

    emb = FixedEmbedder([[1, 0], [1, 0], [0, 1], [0, 1]])
    chunker = EmbeddingSemanticChunker(embeddings=emb)
    context = ExecutionContext("test", Policy())
    output = [c async for c in chunker.transform(doc(), context)]
    output.reverse()
    await chunker.validate_output(doc(), output, context)
    assert "".join(c.content for c in output) == doc().content
    for value in [0, True, "1", -1, None]:
        broken = [
            output[0],
            replace(output[1], metadata={**output[1].metadata, "chunk_index": value}),
        ]
        with pytest.raises(ChunkingError):
            await chunker.validate_output(doc(), broken, context)
