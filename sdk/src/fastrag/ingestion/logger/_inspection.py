"""Shared rendering and accounting for lossless item inspection."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TextIO

from fastrag.exceptions import describe_error
from fastrag.ingestion.events import ItemFailed, PipelineEvent, StepFailed
from fastrag.ingestion.logger.base import Logger
from fastrag.ingestion.pipeline import Pipeline


_BINARY_PREVIEW_CHARS = 64


class _InspectionLogger(Logger):
    label = "Item inspection"

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        format: Literal["rich", "jsonl"] = "rich",
        stream: TextIO | None = None,
        configuration: Mapping[str, Any] | None = None,
        stages: Sequence[str] | None = None,
        content_lines: int | None = None,
        content_chars: int | None = None,
        metadata_only: bool = False,
        show_binary: bool = False,
    ) -> None:
        super().__init__(pipeline)
        if format not in ("rich", "jsonl"):
            raise ValueError("format must be 'rich' or 'jsonl'")
        if content_lines is not None and (
            isinstance(content_lines, bool)
            or not isinstance(content_lines, int)
            or content_lines < 1
        ):
            raise ValueError("content_lines must be a positive integer or None")
        if content_chars is not None and (
            isinstance(content_chars, bool)
            or not isinstance(content_chars, int)
            or content_chars < 1
        ):
            raise ValueError("content_chars must be a positive integer or None")
        self.content_chars = content_chars
        self.metadata_only = metadata_only
        self.show_binary = show_binary
        self.content_lines = content_lines
        self.format = format
        self.stream = stream if stream is not None else sys.stdout
        self.configuration = dict(configuration or {})
        self.stages = None if stages is None else frozenset(stages)
        self.document_count = self.chunk_count = self.source_count = 0

    def _summary_documents(self, parents: int) -> int:
        return self.document_count

    def _records(self, event: PipelineEvent) -> list[dict[str, Any]]:
        raise NotImplementedError

    def _write(self, record: dict[str, Any], console: Any) -> None:
        if self.metadata_only:
            record = {
                k: v
                for k, v in record.items()
                if k not in ("text", "content_base64", "embedding")
            }
        elif (
            not self.show_binary
            and isinstance(record.get("content_base64"), str)
            and len(record["content_base64"]) > _BINARY_PREVIEW_CHARS
        ):
            full = record["content_base64"]
            record = {
                **record,
                "content_base64": full[:_BINARY_PREVIEW_CHARS]
                + f"... [{len(full) - _BINARY_PREVIEW_CHARS} base64 characters omitted]",
                "content_truncated": True,
            }
        if self.content_lines is not None and isinstance(record.get("text"), str):
            lines = record["text"].splitlines(keepends=True)
            omitted = len(lines) - 2 * self.content_lines
            if omitted > 0:
                first = "".join(lines[: self.content_lines])
                last = "".join(lines[-self.content_lines :])
                record = {
                    **record,
                    "text": first + f"[... {omitted} lines omitted ...]\n" + last,
                    "content_truncated": True,
                    "omitted_lines": omitted,
                }
        if self.content_chars is not None and isinstance(record.get("text"), str):
            omitted = len(record["text"]) - self.content_chars
            if omitted > 0:
                record = {
                    **record,
                    "text": record["text"][: self.content_chars]
                    + f"\n[... {omitted} characters omitted ...]",
                    "content_truncated": True,
                    "omitted_characters": omitted,
                }
        if self.format == "jsonl":
            print(
                json.dumps(record, ensure_ascii=False, default=str),
                file=self.stream,
                flush=True,
            )
            return

        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        kind = record["type"]
        fields = Table.grid(padding=(0, 2), expand=True)
        fields.add_column(style="bold cyan", overflow="fold")
        fields.add_column(ratio=1, overflow="fold")
        labels = {
            "id": "ID",
            "source_id": "Source ID",
            "source_uri": "Source URI",
            "url": "URL",
            "parent_document_id": "Parent document ID",
            "step_config_hash": "Step config hash",
            "length": "Characters",
            "length_bytes": "UTF-8 bytes",
        }
        for key, value in record.items():
            if key in ("type", "metadata", "text", "content_base64", "embedding"):
                continue
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, ensure_ascii=False, default=str, indent=2)
            fields.add_row(
                Text(labels.get(key, key.replace("_", " ").capitalize())),
                Text(str(value)),
            )
        sections: list[Any] = [fields]
        if "metadata" in record:
            sections.append(
                Panel(
                    Text(
                        json.dumps(
                            record["metadata"],
                            ensure_ascii=False,
                            default=str,
                            indent=2,
                        )
                    ),
                    title="Metadata (complete)",
                    border_style="dim",
                )
            )
        if "metadata" in record and not self.metadata_only:
            sections.extend(
                [
                    Panel(
                        Text(record.get("text") or record.get("content_base64", "")),
                        title=(
                            "Content (preview)"
                            if record.get("content_truncated")
                            else "Content (complete)"
                            if record.get("text") is not None
                            else "Content (base64)"
                        ),
                        border_style="green",
                    ),
                ]
            )
        if record.get("embedding") is not None:
            sections.append(
                Panel(
                    Text(json.dumps(record["embedding"])), title="Embedding (complete)"
                )
            )
        titles = {
            "configuration": f"{self.label} · Configuration",
            "document": f"Document {self.document_count}",
            "chunk": f"Chunk {self.chunk_count}",
            "source": f"Source {self.source_count}",
            "document_result": "Document · Chunking result",
            "error": "Pipeline error",
            "summary": f"{self.label} · Summary",
        }
        console.print(
            Panel(
                Group(*sections),
                title=titles[kind],
                border_style="red" if kind == "error" else "cyan",
            )
        )
        console.print()

    async def run(self) -> None:
        """Inspect items and write a partial summary before propagating failures."""
        console = None
        if self.format == "rich":
            try:
                from rich.console import Console
            except ImportError as exc:
                raise ImportError(
                    "Rich inspection requires fastrag-ai[logging]; "
                    "install it or use format='jsonl'."
                ) from exc
            console = Console(file=self.stream, markup=False, highlight=False)

        self.document_count = self.chunk_count = self.source_count = 0
        sources: set[str] = set()
        seen_sources: set[str] = set()
        parents = cached = cached_chunks = errors = failed_steps = 0
        characters = byte_count = words = lines = items = 0
        smallest: int | None = None
        largest = 0
        started = time.monotonic()
        status = "completed"
        self._write({**self.configuration, "type": "configuration"}, console)
        try:
            async for event in self.pipeline:
                if isinstance(event, (ItemFailed, StepFailed)):
                    errors += int(isinstance(event, ItemFailed))
                    failed_steps += int(isinstance(event, StepFailed))
                    self._write(
                        {
                            "type": "error",
                            "step": event.step,
                            "uri": getattr(event, "uri", ""),
                            "source_id": getattr(event, "source_id", ""),
                            "event_name": event.event_name,
                            "error": describe_error(event.error),
                        },
                        console,
                    )
                    continue
                if (
                    self.stages is not None
                    and getattr(event, "event_name", "") not in self.stages
                ):
                    continue
                for record in self._records(event):
                    kind = record["type"]
                    if kind == "source":
                        # Sources aggregates repeat their leaf fetcher's event.
                        if record["id"] in seen_sources:
                            continue
                        seen_sources.add(record["id"])
                        self.source_count += 1
                    sources.add(record["source_id"])
                    if kind == "document_result":
                        parents += 1
                    else:
                        self.document_count += int(kind in ("document", "chunk"))
                        self.chunk_count += int(kind == "chunk")
                        items += 1
                        length = record.get("length", 0)
                        characters += length
                        byte_count += record["length_bytes"]
                        words += record.get("word_count", 0)
                        lines += record.get("line_count", 0)
                        cached += int(record["cached"])
                        cached_chunks += int(kind == "chunk" and record["cached"])
                        smallest = length if smallest is None else min(smallest, length)
                        largest = max(largest, length)
                    self._write(record, console)
        except BaseException:
            status = "interrupted"
            raise
        finally:
            self._write(
                {
                    "type": "summary",
                    "status": status
                    if status != "completed" or not (errors or failed_steps)
                    else "completed with errors",
                    "sources": len(sources),
                    "documents": self._summary_documents(parents),
                    "parent_documents": parents,
                    "chunks": self.chunk_count,
                    "cached_items": cached,
                    "cached_chunks": cached_chunks,
                    "failed_items": errors,
                    "failed_steps": failed_steps,
                    "total_characters": characters,
                    "total_bytes": byte_count,
                    "total_words": words,
                    "total_lines": lines,
                    "min_characters": smallest or 0,
                    "max_characters": largest,
                    "mean_characters": round(characters / items, 2) if items else 0,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                },
                console,
            )
