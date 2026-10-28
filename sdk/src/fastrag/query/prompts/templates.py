"""The ``PromptTemplate``: retrieved documents plus a question to messages."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from fastrag.exceptions import ConfigurationError
from fastrag.models import Document
from fastrag.query.llm.base import Message

_PLACEHOLDER = re.compile(r"\{(context|question)\}")


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """Renders retrieved documents and a question into messages for a ``ChatModel``.

    Args:
        template: The user-message body. Must contain both ``{context}`` and
            ``{question}``; any other text is copied through verbatim.
        system: Optional system message, sent unchanged ahead of the rendered
            template.
    """

    template: str
    system: str | None = None

    def __post_init__(self) -> None:
        """Validate the template.

        Raises:
            fastrag.exceptions.ConfigurationError: ``template`` is missing
                ``{context}`` or ``{question}``. Checked at construction, not
                at render time, because the failure this prevents is
                invisible: a template without ``{context}`` silently
                produces a system that retrieves documents, discards them,
                and answers from the model's own knowledge —
                indistinguishable from working RAG until someone checks the
                citations.
        """
        if "{context}" not in self.template or "{question}" not in self.template:
            raise ConfigurationError(
                "PromptTemplate.template must contain both {context} and {question}"
            )

    def render(self, *, context: Sequence[Document], question: str) -> list[Message]:
        """Render ``context`` and ``question`` into the messages to send.

        ``context`` is numbered by source, as :func:`group_sources` groups
        it — ``[n]`` is every chunk of the page in ``groups[n - 1]`` — so a
        "cite as [n]" system instruction cites each page by one number.
        """
        values = {
            "context": _format_context(context),
            "question": question,
        }
        # A callable replacement, not a string one: `re.sub` never
        # reinterprets a function's return value as backreferences, so
        # document text that happens to contain a literal backslash cannot
        # corrupt the render.
        body = _PLACEHOLDER.sub(lambda match: values[match.group(1)], self.template)

        messages: list[Message] = []
        if self.system is not None:
            messages.append(Message(role="system", content=self.system))
        messages.append(Message(role="user", content=body))
        return messages


def group_sources(context: Sequence[Document]) -> list[list[Document]]:
    """Group retrieved chunks into the sources a prompt cites as ``[n]``.

    Chunks sharing a :func:`page_url` (one page, and each page of a PDF
    apart) form one group, placed where the page's best-ranked chunk was
    retrieved and holding its chunks in page order (``chunk_index``). A chunk
    with no URL is a group of its own. ``groups[n - 1]`` is what ``[n]``
    cites: the serving layer maps an answer's citations back to pages through
    this same function.
    """
    groups: dict[str | int, list[Document]] = {}
    for rank, document in enumerate(context):
        key = page_url(document) or rank
        groups.setdefault(key, []).append(document)
    return [_in_page_order(group) for group in groups.values()]


def page_url(document: Document) -> str | None:
    """The link to the page ``document`` came from, if known.

    A PDF parser records each page's link, with its ``#page=N`` fragment, as
    ``source_url``; any other document links to its ``source_uri``.
    """
    metadata = document.metadata
    return metadata.get("source_url") or metadata.get("source_uri") or None


def _in_page_order(chunks: list[Document]) -> list[Document]:
    """``chunks`` by ``chunk_index``, or as retrieved when one lacks it."""
    indexes = [chunk.metadata.get("chunk_index") for chunk in chunks]
    if not all(isinstance(index, int) for index in indexes):
        return chunks
    return sorted(chunks, key=lambda chunk: chunk.metadata["chunk_index"])


def _format_context(context: Sequence[Document]) -> str:
    """Render each source as a numbered block with its provenance, best first.

    The header names the page's URL, ``title`` and ``language`` (an ISO 639-1
    code, as :class:`~fastrag.retrieval.LanguageAwareRetriever` sets it), when
    known. A page's chunks follow its header in page order, with ``…``
    marking text between two of them that was not retrieved.
    """
    blocks = []
    for position, chunks in enumerate(group_sources(context), start=1):
        metadata = chunks[0].metadata
        uri = page_url(chunks[0]) or "unknown source"
        header = f"[{position}] {uri}"
        if (title := metadata.get("title")) and title != uri:
            header += f"\nTitle: {title}"
        if language := metadata.get("language"):
            header += f"\nLanguage: {language}"
        body = chunks[0].content
        for previous, chunk in zip(chunks, chunks[1:], strict=False):
            body += "\n\n" if _adjacent(previous, chunk) else "\n\n…\n\n"
            body += chunk.content
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


def _adjacent(previous: Document, chunk: Document) -> bool:
    """Whether ``chunk`` directly follows ``previous`` in their page."""
    before = previous.metadata.get("chunk_index")
    return isinstance(before, int) and chunk.metadata.get("chunk_index") == before + 1


DEFAULT_RAG_PROMPT = PromptTemplate(
    system=(
        "Answer the question in its own language, using only the documents "
        "provided. If they do not contain the answer, say so.\n\n"
        "Prefer documents written in the language of the question. Use a "
        "document in another language only for what none of those say. When "
        "several documents say the same thing, such as one page in different "
        "languages, use only the one in the language of the question.\n\n"
        # One closing line, not inline citations: told to cite each document
        # once, inline, Qwen3.8-27B still repeated a number in 34-42 of 48
        # answers however the rule was worded (first or last use, with or
        # without examples), mostly a citation per paragraph plus a closing
        # [1][2][3]. This wording repeated none.
        "Do not cite documents inside the text. End the answer with one line "
        "that lists the number of every document you used, in square brackets "
        "and each only once, like [1][3]. Never write a document's number "
        "anywhere else in the answer."
    ),
    template="Documents:\n{context}\n\nQuestion: {question}\n\nAnswer:",
)
