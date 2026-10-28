"""HTML structural normalization with precise source policy exclusions."""

from __future__ import annotations

from html import unescape
from html.parser import HTMLParser

from fastrag.models import Document
from fastrag.normalization.models import (
    Diagnostic,
    Exclusion,
    NormalizedDocument,
    SourceSpan,
    _Node,
    finish,
    line_offsets,
)


class _TreeParser(HTMLParser):
    def __init__(self, source: str) -> None:
        super().__init__(convert_charrefs=False)
        self.source = source
        self.offsets = line_offsets(source)
        self.root = _Node("html", 0, len(source))
        self.stack = [self.root]
        self.diagnostics: list[Diagnostic] = []

    def position(self) -> int:
        line, column = self.getpos()
        return self.offsets[line - 1] + column

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        start = self.position()
        # HTML optional end tags, constrained to the same enclosing list/table.
        barriers = {
            "li": {"ul", "ol"},
            "td": {"tr", "table"},
            "th": {"tr", "table"},
            "tr": {"table"},
            "p": {"div", "section", "article"},
        }
        closable = {"td", "th"} if tag in {"td", "th"} else {tag}
        if tag in barriers:
            for i in range(len(self.stack) - 1, 0, -1):
                if self.stack[i].kind in barriers[tag]:
                    break
                if self.stack[i].kind in closable:
                    self.close_to(i, start)
                    break
        if (
            tag
            in {
                "div",
                "section",
                "article",
                "ul",
                "ol",
                "table",
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
            }
            and self.stack[-1].kind == "p"
        ):
            self.close_to(len(self.stack) - 1, start)
        node = _Node(
            tag,
            start,
            start + len(self.get_starttag_text() or ""),
            metadata={"attrs": dict(attrs)},
        )
        self.stack[-1].children.append(node)
        if tag not in {
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "param",
            "source",
            "track",
            "wbr",
        }:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if self.stack[-1].kind == tag:
            self.close_to(
                len(self.stack) - 1,
                self.position() + len(self.get_starttag_text() or ""),
            )

    def close_to(self, index: int, end: int) -> None:
        for node in self.stack[index:]:
            node.end = end
        del self.stack[index:]

    def handle_endtag(self, tag: str) -> None:
        start = self.position()
        end = self.source.find(">", start)
        end = len(self.source) if end < 0 else end + 1
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].kind == tag:
                for node in self.stack[i + 1 :]:
                    node.end = start
                self.stack[i].end = end
                del self.stack[i:]
                return
        self.diagnostics.append(
            Diagnostic(
                "unmatched_end_tag", f"Unmatched </{tag}>", SourceSpan(start, end)
            )
        )

    def handle_data(self, data: str) -> None:
        start = self.position()
        self.stack[-1].children.append(_Node("#text", start, start + len(data)))

    def handle_entityref(self, name: str) -> None:
        self.handle_data(
            "&"
            + name
            + (
                ";"
                if self.source[
                    self.position() + len(name) + 1 : self.position() + len(name) + 2
                ]
                == ";"
                else ""
            )
        )

    def handle_charref(self, name: str) -> None:
        self.handle_data(
            "&#"
            + name
            + (
                ";"
                if self.source[
                    self.position() + len(name) + 2 : self.position() + len(name) + 3
                ]
                == ";"
                else ""
            )
        )


class HtmlNormalizer:
    """Recover HTML while retaining tabs and explicitly recording exclusions."""

    def __init__(
        self,
        *,
        exclude_tags: tuple[str, ...] = ("script", "style"),
        exclude_hidden: bool = False,
    ) -> None:
        """Configure deterministic normalization options."""
        self.exclude_tags = tuple(sorted(set(t.lower() for t in exclude_tags)))
        self.exclude_hidden = exclude_hidden

    @property
    def identity(self) -> str:
        """Return output policy and parser contract version."""
        return (
            f"html.parser:v1:exclude={self.exclude_tags}:hidden={self.exclude_hidden}"
        )

    def normalize(self, document: Document) -> NormalizedDocument:
        """Parse original source positions without rendering or executing HTML."""
        parser = _TreeParser(document.content)
        parser.feed(document.content)
        parser.close()
        if len(parser.stack) > 1:
            parser.diagnostics.append(
                Diagnostic("unclosed_tags", "Unclosed elements recovered at EOF")
            )
            parser.close_to(1, len(document.content))
        exclusions: list[Exclusion] = []
        mapping = {
            "p": "paragraph",
            "ul": "list",
            "ol": "list",
            "dl": "list",
            "li": "list_item",
            "dt": "list_item",
            "dd": "list_item",
            "pre": "code_block",
            "table": "table",
            "tr": "table_row",
            "blockquote": "blockquote",
            "figure": "figure",
            "figcaption": "caption",
            "caption": "caption",
            "img": "image",
            "hr": "horizontal_rule",
        }

        def text(node: _Node) -> str:
            if node.kind in self.exclude_tags:
                return ""
            if node.kind == "#text":
                return unescape(document.content[node.start : node.end])
            return "".join(text(c) for c in node.children)

        def walk(node: _Node):
            yield node
            for c in node.children:
                yield from walk(c)

        def convert(node: _Node) -> list[_Node]:
            attrs = node.metadata.get("attrs", {})
            hidden = (
                "hidden" in attrs
                or str(attrs.get("aria-hidden", "")).lower() == "true"
                or "display:none"
                in str(attrs.get("style", "")).replace(" ", "").lower()
            )
            if node.kind in self.exclude_tags or (self.exclude_hidden and hidden):
                exclusions.append(
                    Exclusion(
                        SourceSpan(node.start, node.end),
                        f"html:{node.kind}"
                        if node.kind in self.exclude_tags
                        else "html:hidden",
                    )
                )
                return []
            children = [c for child in node.children for c in convert(child)]
            kind = mapping.get(node.kind)
            if node.kind in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                return [
                    _Node(
                        "heading",
                        node.start,
                        node.end,
                        level=int(node.kind[1]),
                        title=text(node).strip(),
                        metadata={"tag": node.kind},
                    )
                ]
            if kind:
                meta = {
                    "tag": node.kind,
                    "attrs": attrs,
                    "text": text(node),
                    "links": [
                        {
                            "text": text(n),
                            "destination": n.metadata.get("attrs", {}).get("href", ""),
                        }
                        for n in walk(node)
                        if n.kind == "a"
                    ],
                }
                meta["images"] = [
                    {
                        "src": n.metadata.get("attrs", {}).get("src", ""),
                        "alt": n.metadata.get("attrs", {}).get("alt", ""),
                    }
                    for n in walk(node)
                    if n.kind == "img"
                ]
                if kind == "code_block":
                    meta.update(
                        code=text(node),
                        language=str(attrs.get("class", "")).removeprefix("language-"),
                    )
                if kind == "table_row":
                    meta["header"] = any(c.kind == "th" for c in node.children)
                    meta["cells"] = [
                        {"text": text(c), "attrs": c.metadata.get("attrs", {})}
                        for c in node.children
                        if c.kind in {"th", "td"}
                    ]
                if kind == "image":
                    meta.update(src=attrs.get("src", ""), alt=attrs.get("alt", ""))
                return [
                    _Node(
                        kind,
                        node.start,
                        node.end,
                        children if kind not in {"paragraph", "code_block"} else [],
                        metadata=meta,
                    )
                ]
            if node.kind == "#text":
                return (
                    [_Node("paragraph", node.start, node.end)]
                    if text(node).strip()
                    else []
                )
            if node.kind == "a":
                return [
                    _Node(
                        "paragraph",
                        node.start,
                        node.end,
                        metadata={
                            "links": [
                                {
                                    "text": text(node),
                                    "destination": attrs.get("href", ""),
                                }
                            ]
                        },
                    )
                ]
            return children

        return finish(
            document, "html", convert(parser.root), exclusions, parser.diagnostics
        )
