"""CommonMark with pipe tables, exact block source, and inline metadata."""

from __future__ import annotations

from importlib.metadata import version

from fastrag.models import Document
from fastrag.normalization.models import NormalizedDocument, _Node, finish, line_offsets


class MarkdownNormalizer:
    """Parse Markdown using markdown-it-py without rendering or execution."""

    def __init__(self, *, tables: bool = True) -> None:
        """Configure deterministic normalization options."""
        try:
            from markdown_it import MarkdownIt
        except ImportError as exc:
            raise ImportError(
                "Install fastrag-ai[normalization] for Markdown normalization"
            ) from exc
        self.tables = tables
        self._parser = MarkdownIt("commonmark")
        if tables:
            self._parser.enable("table")

    @property
    def identity(self) -> str:
        """Return parser version and dialect options."""
        return f"markdown-it:{version('markdown-it-py')}:commonmark:tables={self.tables}:v1"

    def normalize(self, document: Document) -> NormalizedDocument:
        """Build nested nodes from the parser token stream and original lines."""
        offsets = line_offsets(document.content)
        blocks: list[_Node] = []
        stack: list[_Node] = []
        mapping = {
            "heading": "heading",
            "paragraph": "paragraph",
            "bullet_list": "list",
            "ordered_list": "list",
            "list_item": "list_item",
            "blockquote": "blockquote",
            "table": "table",
            "tr": "table_row",
            "thead": "raw_block",
            "tbody": "raw_block",
            "th": "raw_block",
            "td": "raw_block",
            "fence": "code_block",
            "code_block": "code_block",
            "html_block": "raw_block",
            "hr": "horizontal_rule",
        }
        for token in self._parser.parse(document.content):
            if token.nesting == -1:
                if stack:
                    stack.pop()
                continue
            if token.type == "inline":
                if stack:
                    node = stack[-1]
                    node.metadata["inline"] = [
                        {"type": t.type, "content": t.content, "attrs": dict(t.attrs)}
                        for t in token.children or []
                    ]
                    if node.kind == "heading":
                        node.title = "".join(
                            t.content for t in token.children or [] if t.nesting == 0
                        )
                    for t in token.children or []:
                        if t.type == "image":
                            node.children.append(
                                _Node(
                                    "image",
                                    node.start,
                                    node.end,
                                    metadata={
                                        "src": t.attrGet("src"),
                                        "alt": t.content,
                                        "location_precision": "containing_block",
                                    },
                                )
                            )
                continue
            typ = token.type.removesuffix("_open")
            if typ not in mapping:
                continue
            if token.map:
                start, end = offsets[token.map[0]], offsets[token.map[1]]
            elif stack:
                start, end = stack[-1].start, stack[-1].end
            else:
                continue
            meta = {"markup": token.markup}
            if typ in {"fence", "code_block"}:
                meta.update(
                    language=token.info.split()[0] if token.info.strip() else "",
                    code=token.content,
                )
            if typ == "tr":
                meta["header"] = any(n.metadata.get("tag") == "thead" for n in stack)
            meta["tag"] = token.tag
            if token.map is None:
                meta["location_precision"] = "containing_block"
            node = _Node(
                mapping[typ],
                start,
                end,
                level=int(token.tag[1]) if typ == "heading" else None,
                metadata=meta,
            )
            (stack[-1].children if stack else blocks).append(node)
            if token.nesting == 1:
                stack.append(node)

        def enrich(node: _Node) -> None:
            for child in node.children:
                enrich(child)
            if node.kind == "table_row":
                node.metadata["cells"] = [
                    {
                        "text": "".join(
                            t["content"]
                            for t in c.metadata.get("inline", [])
                            if t["type"] in {"text", "code_inline"}
                        ),
                        "inline": c.metadata.get("inline", []),
                        "attrs": {},
                    }
                    for c in node.children
                ]

        for block in blocks:
            enrich(block)
        return finish(document, "markdown", blocks)
