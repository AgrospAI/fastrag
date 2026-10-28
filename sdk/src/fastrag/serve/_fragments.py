"""URL text fragments that point a source's link at the passages an answer used.

A link to ``url#:~:text=start,end`` makes the browser scroll to the text from
``start`` through ``end`` and highlight it. Both terms have to match the
page's rendered text, case aside, and each has to sit inside one block
(a paragraph, heading or list item) and start and end on word boundaries. So
each term is taken from a single line of a chunk's Markdown, with the markup
removed, skipping text the parser added that is not on the page.

A browser that cannot match a term, or does not support text fragments, just
opens the page at the top.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterator, Sequence

from fastrag.models import Document
from fastrag.serve._schemas import TextFragment

#: Media types whose text the browser renders as a page it can search. A PDF
#: or a video's transcript is not.
_PAGE_TYPES = ("text/html", "application/xhtml+xml")

#: Words in each term. More words mean fewer false matches earlier on the
#: page, but also more chances for one word to differ from the page.
_TERM_WORDS = 5

#: A single line up to this many words is matched whole, with no end term.
_EXACT_WORDS = 2 * _TERM_WORDS

#: Lines shorter than this, such as an admonition's "note" label, match too
#: much of the page to locate anything.
_MIN_WORDS = 3

#: The same limits for CJK text, in characters. CJK has no spaces, and a
#: cut inside a word does not match, so a term ends after a clause, however
#: long, up to ``_MAX_CHARS``.
_MIN_CHARS = 6
_EXACT_CHARS = 40
_MAX_CHARS = 60

#: Paragraphs the parser writes itself, describing images and diagrams.
_GENERATED = ("Image description:", "Diagram:")

_CJK = re.compile("[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
_CJK_PUNCTUATION = "，。、；：！？"
#: Splits after each clause's punctuation, keeping it with the clause.
_CJK_CLAUSES = re.compile(f"(?<=[{_CJK_PUNCTUATION}])")
_FENCE = re.compile(r"^\s*(```|~~~)")
_SKIPPED_LINE = re.compile(r"^\s*(\||:::|[-*_]{3,}\s*$)")
_BLOCK_PREFIX = re.compile(
    r"^\s*(?:#{1,6}\s+|(?:>\s*)+|(?:[-*+]|\d+[.)])\s+(?:\[[ xX]\]\s+)?)*"
)
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\](?:\([^)]*\)|\[[^\]]*\])")
_AUTOLINK = re.compile(r"<([^<>\s]+[@:][^<>\s]*)>")
_TAG = re.compile(r"</?[a-zA-Z][^<>]*>")
_EMPHASIS = re.compile(r"(?<!\w)[*_]+|[*_]+(?!\w)")
_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|<>~])")
_INVISIBLE = re.compile("[\u00ad\u200b-\u200d\u2060\ufeff]")
_WORD = re.compile(r"\w")


def text_fragments(chunks: Sequence[Document]) -> list[TextFragment]:
    """Locate ``chunks``, all from one page and in page order, on that page.

    Consecutive chunks of one section, or adjacent in the page, are joined
    into a single fragment spanning all of them. None for a page that is not
    HTML.
    """
    if not all(
        str(chunk.metadata.get("content_type", "")).startswith(_PAGE_TYPES)
        for chunk in chunks
    ):
        return []
    fragments: list[TextFragment] = []
    for passage in _passages(chunks):
        fragment = _fragment(passage)
        if fragment is not None and fragment not in fragments:
            fragments.append(fragment)
    return fragments


def _passages(chunks: Sequence[Document]) -> Iterator[list[Document]]:
    """Group consecutive chunks that are one stretch of the page."""
    passage: list[Document] = []
    for chunk in chunks:
        if passage and not _continues(passage[-1], chunk):
            yield passage
            passage = []
        passage.append(chunk)
    if passage:
        yield passage


def _continues(previous: Document, chunk: Document) -> bool:
    """Whether ``chunk`` shares ``previous``'s section or directly follows it."""
    section = _section(chunk)
    if section is not None and section == _section(previous):
        return True
    before, index = (
        previous.metadata.get("chunk_index"),
        chunk.metadata.get("chunk_index"),
    )
    return isinstance(before, int) and index == before + 1


def _section(chunk: Document) -> object:
    """The section ``chunk`` belongs to, as its chunker recorded it, if any."""
    metadata = chunk.metadata
    if section_id := metadata.get("parent_section_id"):
        return section_id
    headings = metadata.get("heading_path") or metadata.get("headings")
    return tuple(headings) if headings else None


def _fragment(passage: Sequence[Document]) -> TextFragment | None:
    """The passage's first words through its last ones."""
    lines = [line for chunk in passage for line in _visible_lines(chunk.content)]
    if not lines:
        return None
    if len(lines) == 1 and _fits(lines[0]):
        return TextFragment(start=lines[0])
    start, end = _leading(lines[0]), _trailing(lines[-1])
    return TextFragment(start=start, end=end if end != start else None)


def _visible_lines(markdown: str) -> Iterator[str]:
    """``markdown``'s lines as the page renders them, long enough to locate."""
    in_fence = False
    in_generated = False
    for raw in markdown.splitlines():
        if _FENCE.match(raw):
            in_fence = not in_fence
            continue
        if not raw.strip():
            in_generated = False
            continue
        if in_fence or in_generated or _SKIPPED_LINE.match(raw):
            continue
        line = _plain(_BLOCK_PREFIX.sub("", raw))
        if line.startswith(_GENERATED):
            in_generated = True
            continue
        if _long_enough(line):
            yield line


def _plain(line: str) -> str:
    """``line`` with its inline Markdown removed, as the page shows it."""
    line = _IMAGE.sub("", line)
    line = _LINK.sub(r"\1", line)
    line = _AUTOLINK.sub(r"\1", line)
    line = _TAG.sub("", line)
    line = _EMPHASIS.sub("", line).replace("`", "")
    line = _INVISIBLE.sub("", _ESCAPE.sub(r"\1", html.unescape(line)))
    words = line.split()
    # Stray markup the parser left behind, such as a closing ":::", is
    # not on the page; real punctuation is attached to a word.
    while words and not _WORD.search(words[0]):
        words.pop(0)
    while words and not _WORD.search(words[-1]):
        words.pop()
    return " ".join(words)


def _is_cjk(line: str) -> bool:
    """Whether ``line`` is mostly written without spaces between words."""
    return len(_CJK.findall(line)) * 2 > len(line.replace(" ", ""))


def _long_enough(line: str) -> bool:
    if _is_cjk(line):
        return len(line) >= _MIN_CHARS
    return len(line.split()) >= _MIN_WORDS


def _fits(line: str) -> bool:
    """Whether ``line`` is short enough to match whole."""
    if _is_cjk(line):
        return len(line) <= _EXACT_CHARS
    return len(line.split()) <= _EXACT_WORDS


def _leading(line: str) -> str:
    """The first words of ``line``; CJK text up to the end of a clause."""
    if _is_cjk(line):
        return _clauses(_CJK_CLAUSES.split(line))[:_MAX_CHARS]
    return " ".join(line.split()[:_TERM_WORDS])


def _trailing(line: str) -> str:
    """The last words of ``line``; CJK text from the start of a clause."""
    if _is_cjk(line):
        clauses = _CJK_CLAUSES.split(line.rstrip(_CJK_PUNCTUATION))
        return _clauses(clauses[::-1], reverse=True)[-_MAX_CHARS:]
    return " ".join(line.split()[-_TERM_WORDS:])


def _clauses(clauses: Sequence[str], *, reverse: bool = False) -> str:
    """Join ``clauses`` in order until the text is long enough to locate."""
    text = ""
    for clause in clauses:
        text = clause + text if reverse else text + clause
        if len(text.strip(_CJK_PUNCTUATION)) >= _MIN_CHARS:
            break
    return text.strip(_CJK_PUNCTUATION)
