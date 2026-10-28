"""Verify reproducibility and transparent metrics, without asserting a winner."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def evaluation_module():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_structure.py"
    spec = importlib.util.spec_from_file_location("evaluate_structure", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_retrieval_comparison_is_reproducible_and_metrics_are_recomputed(
    evaluation_module,
):
    module = evaluation_module
    first = await module.evaluate()
    assert first == await module.evaluate()
    assert set(first["arms"]) == {"fixed400_50", "structural400_600", "refined400_600"}
    assert len(first["questions"]) == 18
    assert len(first["corpus_sha256"]) == len(first["questions_sha256"]) == 64
    assert first["embedding"]["model"] == "local:lexical-count:v1"
    for arm in first["arms"].values():
        assert arm["chunks"] > 3
        assert set(arm["categories"]) == {
            "answer",
            "section",
            "cross_paragraph",
            "table",
            "code",
            "multisection",
        }
        assert [q["query"] for q in arm["questions"]] == [
            q["query"] for q in first["questions"]
        ]
        for row in arm["questions"]:
            correct = [
                text
                for hit, text in zip(row["hits"], row["hit_texts"], strict=True)
                if hit.split(":")[1] == row["document_id"]
            ]
            oracle = all(
                answer.lower() in "\n".join(correct).lower()
                for answer in row["answers"]
            )
            assert row["answer_complete_at_3"] == oracle
            assert row["chunk_context_bytes"] == len(
                "\n".join(row["hit_texts"]).encode()
            )
            assert 0 <= row["reciprocal_rank"] <= 1
            assert 0 <= row["section_recall_at_3"] <= 1
        for metric, aggregate in arm["metrics"].items():
            assert aggregate == pytest.approx(
                sum(row[metric] for row in arm["questions"]) / 18
            )
        assert (
            arm["metrics"]["parent_context_bytes"]
            > arm["metrics"]["chunk_context_bytes"]
        )


async def test_fixture_embedder_exact_vectors_and_zero_query(evaluation_module):
    model = evaluation_module.LexicalFixtureEmbedder(["apple", "pear"])
    vectors = await model.embed(["PEAR pear apple", "unseen", "apple"])
    assert vectors[0] == pytest.approx([1 / 5**0.5, 2 / 5**0.5])
    assert vectors[1:] == [[0.0, 0.0], [1.0, 0.0]]
    assert model.dimension == 2


@pytest.mark.parametrize(
    "size,counts",
    [
        (0, []),
        (1, [1]),
        (400, [400]),
        (401, [400, 51]),
        (750, [400, 400]),
        (751, [400, 400, 51]),
    ],
)
def test_fixed_baseline_stops_after_complete_coverage(evaluation_module, size, counts):
    pieces = evaluation_module.fixed_windows("x" * size)
    assert [len(text) for text, _ in pieces] == counts
    occupied = {
        point for _, m in pieces for point in range(m["window_start"], m["window_end"])
    }
    assert occupied == set(range(size))
    assert all(
        b[1]["window_start"] == a[1]["window_end"] - 50
        for a, b in zip(pieces, pieces[1:])
    )
