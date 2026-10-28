"""Reproduce a small offline retrieval comparison, without quality assumptions.

Run: .venv/bin/python scripts/evaluate_structure.py --output agents/T08C-evaluation.json
The transparent lexical Embedder is a fixture model, not a pretrained model.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from fastrag.ingestion.steps.chunking import (
    ChatBoundaryRefiner,
    StructureAwareChunker,
    Utf8ByteTokenizer,
    build_embedding_text,
)
from fastrag.ingestion.steps.embedding import Embedder
from fastrag.models import Document
from fastrag.normalization import NormalizerRegistry
from fastrag.query.llm import ChatModel
from fastrag.retrieval import SectionIndex


def corpus() -> tuple[list[Document], list[dict[str, Any]]]:
    """Return versioned fixtures with answer spans in six retrieval categories."""
    documents, questions = [], []
    for crop, dose, wait, port in [
        ("pear", "17", "31", "8201"),
        ("apple", "23", "43", "8202"),
        ("plum", "29", "47", "8203"),
    ]:
        sections = {
            "Irrigation": f"For {crop} irrigation the measured dose is {dose} litres. "
            + "Sensors record moisture daily. " * 18,
            "Schedule": f"The {crop} schedule begins at sunrise. "
            + "The operator verifies the controller before starting. " * 9
            + f"\n\nAfter {wait} minutes the {crop} schedule enters the cooling phase. "
            + "Cooling protects the orchard canopy. " * 10,
            "Rates": f"| Crop | Dose | Wait |\n| --- | --- | --- |\n| {crop} | {dose} | {wait} |\n"
            + "| reference | 11 | 13 |\n" * 25,
            "Controller": f"```python\ndef configure_{crop}():\n    port = {port}\n    return port\n"
            + "    # monitor controller status\n" * 20
            + "```",
            "Safety": f"The {crop} emergency stop uses the amber switch. "
            + "Operators inspect safety equipment before each shift. " * 15,
        }
        text = f"# {crop.title()} Manual\n\n" + "\n\n".join(
            f"## {title}\n\n{body}" for title, body in sections.items()
        )
        documents.append(
            Document(crop, "fixture", text, metadata={"content_format": "markdown"})
        )
        cases = [
            (
                "answer",
                f"{crop} irrigation measured dose litres",
                [f"{dose} litres"],
                ["Irrigation"],
            ),
            (
                "section",
                f"{crop} safety emergency stop switch",
                ["amber switch"],
                ["Safety"],
            ),
            (
                "cross_paragraph",
                f"{crop} schedule sunrise cooling minutes",
                ["sunrise", f"{wait} minutes"],
                ["Schedule"],
            ),
            (
                "table",
                f"{crop} rates dose wait",
                [f"| {crop} | {dose} | {wait} |"],
                ["Rates"],
            ),
            (
                "code",
                f"configure_{crop} controller port",
                [f"configure_{crop}", f"port = {port}"],
                ["Controller"],
            ),
            (
                "multisection",
                f"{crop} irrigation safety dose emergency switch",
                [f"{dose} litres", "amber switch"],
                ["Irrigation", "Safety"],
            ),
        ]
        for category, query, answers, titles in cases:
            questions.append(
                {
                    "id": f"{crop}-{category}",
                    "category": category,
                    "document_id": crop,
                    "query": query,
                    "answers": answers,
                    "sections": titles,
                }
            )
    return documents, questions


def terms(text: str) -> list[str]:
    """Apply the model's fixed lowercase alphanumeric tokenization."""
    return re.findall(r"[a-z0-9_]+", text.lower())


class LexicalFixtureEmbedder(Embedder):
    """Transparent dense term-count vectors with fixed shared vocabulary."""

    model_id = "local:lexical-count:v1"

    def __init__(self, vocabulary: list[str]):
        """Use one vocabulary for all arms and questions."""
        super().__init__(cache=False)
        self.vocabulary = vocabulary

    @property
    def dimension(self) -> int:
        """Return the fixed vector width."""
        return len(self.vocabulary)

    def config(self):
        """Describe every output-affecting model setting."""
        return {
            "model": self.model_id,
            "vocabulary": self.vocabulary,
            "normalization": "l2",
            "terms": "lowercase [a-z0-9_]+",
        }

    async def embed(self, texts):
        """Encode fixture texts without providers or hidden corpus training."""
        result = []
        for text in texts:
            counts = Counter(terms(text))
            vector = [float(counts[word]) for word in self.vocabulary]
            norm = math.sqrt(sum(value * value for value in vector)) or 1
            result.append([value / norm for value in vector])
        return result


class FirstSentenceChat(ChatModel):
    """Local controlled refinement; no semantic intelligence is claimed."""

    model_id = "local:first-allowed:v1"

    async def invoke(self, messages):
        """Return the first supplied safe boundary, never generated prose."""
        request = json.loads(messages[-1].content)
        return json.dumps({"boundaries": request["allowed"][:1]})

    async def stream(self, messages):
        """Expose the required ChatModel streaming interface."""
        yield await self.invoke(messages)


def digest(value: Any) -> str:
    """Hash canonical fixture/configuration JSON for reproducibility."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def fixed_windows(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Return literal 400-byte/50-overlap windows for this ASCII fixture."""
    if not text.isascii():
        raise ValueError("The fixed fixture baseline requires ASCII")
    result = []
    start = 0
    while start < len(text):
        end = min(start + 400, len(text))
        result.append((text[start:end], {"window_start": start, "window_end": end}))
        if end == len(text):
            break
        start += 350
    return result


async def evaluate() -> dict[str, Any]:
    """Compute all arm/question evidence using identical model and rank settings."""
    documents, questions = corpus()
    registry = NormalizerRegistry()
    normalized = {doc.id: registry.normalize(doc) for doc in documents}
    parents = SectionIndex(normalized.values())
    embedder = LexicalFixtureEmbedder(
        sorted(
            set(
                terms(
                    "\n".join(d.content for d in documents)
                    + "\n"
                    + "\n".join(q["query"] for q in questions)
                )
            )
        )
    )
    question_vectors = await embedder.embed([q["query"] for q in questions])
    structural = StructureAwareChunker(tokenizer=Utf8ByteTokenizer(), cache=False)
    refined = StructureAwareChunker(
        tokenizer=Utf8ByteTokenizer(),
        cache=False,
        semantic_mode="llm",
        semantic_refiner=ChatBoundaryRefiner(
            llm=FirstSentenceChat(),
            identity="local:first-allowed:v1",
            allow_external_transmission=True,
        ),
    )
    arms = {}
    for name, engine in [
        ("fixed400_50", None),
        ("structural400_600", structural),
        ("refined400_600", refined),
    ]:
        chunks = []
        for doc in documents:
            if engine is None:
                pairs = fixed_windows(doc.content)
            else:
                pairs = []
                async for pair in engine.asplit(doc):
                    if isinstance(pair, str):
                        raise TypeError("Structural output must include metadata")
                    body, metadata = pair
                    pairs.append((body, dict(metadata)))
            chunks.extend(
                Document(
                    f"{name}:{doc.id}:{i}",
                    doc.source_id,
                    body,
                    metadata={**meta, "document_id": doc.id},
                )
                for i, (body, meta) in enumerate(pairs)
            )
        vectors = await embedder.embed(
            [build_embedding_text(c.content, c.metadata) for c in chunks]
        )
        rows = []
        for question, query_vector in zip(questions, question_vectors, strict=True):
            ranking = sorted(
                range(len(chunks)),
                key=lambda i: (
                    -sum(a * b for a, b in zip(query_vector, vectors[i], strict=True)),
                    chunks[i].id,
                ),
            )
            hits = [chunks[i] for i in ranking[:3]]
            text = "\n".join(
                hit.content
                for hit in hits
                if hit.metadata["document_id"] == question["document_id"]
            )
            expanded = {}
            titles = set()
            for hit in hits:
                doc_id = hit.metadata["document_id"]
                if engine is None:
                    expanded[(doc_id, "document")] = normalized[doc_id].document.content
                    if doc_id == question["document_id"]:
                        for node in normalized[doc_id].walk():
                            if node.kind == "section" and any(
                                s.start_offset < hit.metadata["window_end"]
                                and s.end_offset > hit.metadata["window_start"]
                                for s in node.spans
                            ):
                                titles.add(node.title)
                else:
                    for parent in parents.resolve(hit, ancestors=True):
                        expanded[(doc_id, parent.section_id)] = parent.content
                    if doc_id == question["document_id"]:
                        titles.update(
                            p.heading_path[-1]
                            for p in parents.resolve(hit)
                            if p.heading_path
                        )
            parent_text = "\n".join(
                content
                for (doc_id, _), content in expanded.items()
                if doc_id == question["document_id"]
            )

            def complete(context):
                return all(
                    answer.lower() in context.lower() for answer in question["answers"]
                )

            first_rank = next(
                (
                    rank
                    for rank, i in enumerate(ranking, 1)
                    if chunks[i].metadata["document_id"] == question["document_id"]
                    and complete(chunks[i].content)
                ),
                None,
            )
            rows.append(
                {
                    **question,
                    "hits": [h.id for h in hits],
                    "hit_texts": [h.content for h in hits],
                    "answer_complete_at_1": complete(hits[0].content)
                    and hits[0].metadata["document_id"] == question["document_id"],
                    "answer_complete_at_3": complete(text)
                    and any(
                        h.metadata["document_id"] == question["document_id"]
                        for h in hits
                    ),
                    "section_recall_at_3": len(
                        titles.intersection(question["sections"])
                    )
                    / len(question["sections"]),
                    "reciprocal_rank": 1 / first_rank if first_rank else 0,
                    "parent_answer_complete_at_3": complete(parent_text),
                    "parent_context_bytes": sum(
                        len(content.encode()) for content in expanded.values()
                    ),
                    "chunk_context_bytes": sum(
                        len(hit.content.encode()) for hit in hits
                    )
                    + 2,
                }
            )
        metrics = [
            "answer_complete_at_1",
            "answer_complete_at_3",
            "section_recall_at_3",
            "reciprocal_rank",
            "parent_answer_complete_at_3",
            "parent_context_bytes",
            "chunk_context_bytes",
        ]

        def aggregate(items):
            return {
                metric: sum(row[metric] for row in items) / len(items)
                for metric in metrics
            }

        arms[name] = {
            "configuration": {"window_bytes": 400, "overlap_bytes": 50}
            if engine is None
            else engine.config(),
            "chunks": len(chunks),
            "mean_embedding_bytes": sum(
                len(build_embedding_text(c.content, c.metadata).encode())
                for c in chunks
            )
            / len(chunks),
            "metrics": aggregate(rows),
            "categories": {
                category: aggregate(
                    [row for row in rows if row["category"] == category]
                )
                for category in sorted({q["category"] for q in questions})
            },
            "questions": rows,
        }
    return {
        "version": "structure-retrieval-fixture:v1",
        "corpus_sha256": digest([(d.id, d.content) for d in documents]),
        "questions_sha256": digest(questions),
        "documents": [{"id": d.id, "content": d.content} for d in documents],
        "questions": questions,
        "embedding": embedder.config(),
        "ranking": {
            "similarity": "cosine",
            "top_k": 3,
            "tie_break": "chunk id lexical",
            "mrr": "first single chunk containing all answer terms in correct document",
        },
        "limitations": [
            "Synthetic ASCII fixture and lexical model; no real embedding/answer-generation quality claim.",
            "Answer completeness tests exact answer substrings; code/table syntax is intentionally required.",
            "Section coverage uses source intersection for baseline windows and source section IDs for structural chunks.",
            "Baseline parents expand documents; structural parents expand sections plus ancestors. Cost includes overlapping ancestor text, deduplicated by section ID only.",
            "Local refinement selects first allowed sentence, not a production LLM.",
        ],
        "arms": arms,
    }


def main() -> None:
    """Write a deterministic, reviewable JSON artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("agents/T08C-evaluation.json")
    )
    options = parser.parse_args()
    result = asyncio.run(evaluate())
    options.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {name: arm["metrics"] for name, arm in result["arms"].items()}, indent=2
        )
    )


if __name__ == "__main__":
    main()
