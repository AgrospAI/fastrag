"""Safe Docutils structural adapter retaining unsupported directive source."""

from __future__ import annotations

from copy import copy
from importlib.metadata import version
from io import StringIO
from typing import Any, cast

from fastrag.models import Document
from fastrag.normalization.models import (
    Diagnostic,
    NormalizedDocument,
    _Node,
    finish,
    line_offsets,
)


class RstNormalizer:
    """Use the RST parser directly with local/remote file insertion disabled."""

    def __init__(self) -> None:
        """Configure deterministic normalization options."""
        try:
            import docutils
        except ImportError as exc:
            raise ImportError(
                "Install fastrag-ai[normalization] for RST normalization"
            ) from exc
        self._version = docutils.__version__

    @property
    def identity(self) -> str:
        """Identify parser and security policy."""
        return f"docutils:{version('docutils')}:no-includes:no-raw:no-transforms:v2"

    def normalize(self, document: Document) -> NormalizedDocument:
        """Map Docutils block tree and conservative source locations."""
        from docutils import frontend, nodes, utils
        from docutils.parsers.rst import Parser

        from docutils.parsers.rst.directives import admonitions, body, images, tables

        # Use known built-in classes directly, never the process-global custom
        # directive registry, whose callbacks may execute arbitrary host code.
        safe = {
            name.lower(): getattr(admonitions, name)
            for name in (
                "Admonition",
                "Attention",
                "Caution",
                "Danger",
                "Error",
                "Hint",
                "Important",
                "Note",
                "Tip",
                "Warning",
            )
        }
        safe.update(
            {
                "code": body.CodeBlock,
                "code-block": body.CodeBlock,
                "parsed-literal": body.ParsedLiteral,
                "image": images.Image,
                "figure": images.Figure,
                "table": tables.RSTTable,
                "list-table": tables.ListTable,
                "csv-table": tables.CSVTable,
            }
        )

        diagnostics: list[Diagnostic] = []

        class SafeDirectives:
            def __init__(self: Any, state_machine, debug=False):
                super().__init__(state_machine, debug)
                # Docutils otherwise uses global state classes for every
                # section/list/admonition nested parse (and a shared cache).
                self.nested_sm_kwargs["state_classes"] = local_state_classes
                self.nested_sm_cache = []
                if hasattr(self, "explicit"):
                    self.explicit = copy(self.explicit)
                    self.explicit.constructs = [
                        (
                            SafeDirectives.directive
                            if method.__name__ == "directive"
                            else method,
                            pattern,
                        )
                        for method, pattern in self.explicit.constructs
                    ]

            def directive(self: Any, match, **option_presets):
                name = match.group(1).lower()
                # A directive may assign node.line *after* consuming its body.
                # Capture parser coordinates at entry/exit, not the returned hint.
                start_line = self.state_machine.abs_line_number() - 1
                cls = safe.get(name)
                result, blank_finish = (
                    self.unknown_directive(name)
                    if cls is None
                    else self.run_directive(cls, match, name, option_presets)
                )
                end_line = self.state_machine.abs_line_number()
                messages = [n for n in result if isinstance(n, nodes.system_message)]
                for message in messages:
                    diagnostics.append(Diagnostic("rst_parser", message.astext()))
                semantic = [
                    n for n in result if not isinstance(n, nodes.system_message)
                ]
                if not semantic:
                    fallback = nodes.container()
                    cast(Any, fallback)._fastrag_raw_directive = True
                    semantic = [fallback]
                for node in semantic:
                    cast(Any, node)._fastrag_directive_lines = (start_line, end_line)
                return semantic, blank_finish

        parser = Parser()
        local_state_classes = tuple(
            type(cls.__name__, (SafeDirectives, cls), {})
            for cls in parser.state_classes
        )
        cast(Any, parser).state_classes = local_state_classes
        settings = frontend.get_default_settings(parser)
        settings.file_insertion_enabled = False
        settings.raw_enabled = False
        settings.warning_stream = StringIO()
        settings.halt_level = 6
        settings.report_level = 5
        tree = utils.new_document("<normalized>", settings=settings)
        parser.parse(document.content, tree)
        offsets = line_offsets(document.content)
        mapping = {
            "paragraph": "paragraph",
            "bullet_list": "list",
            "enumerated_list": "list",
            "definition_list": "list",
            "field_list": "list",
            "definition_list_item": "list_item",
            "field": "list_item",
            "list_item": "list_item",
            "literal_block": "code_block",
            "doctest_block": "code_block",
            "table": "table",
            "row": "table_row",
            "block_quote": "blockquote",
            "image": "image",
            "figure": "figure",
            "caption": "caption",
            "transition": "horizontal_rule",
            "system_message": "raw_block",
        }

        def location(child, low: int, high: int) -> int:
            directive_lines = getattr(child, "_fastrag_directive_lines", None)
            if directive_lines is not None:
                return min(
                    high, max(low, offsets[min(directive_lines[0], len(offsets) - 1)])
                )
            lines = [n.line for n in child.findall(nodes.Element) if n.line]
            index = (child.line or (min(lines) if lines else 1)) - 1
            if isinstance(child, (nodes.title, nodes.section)):
                title = (
                    child
                    if isinstance(child, nodes.title)
                    else next(iter(child.findall(nodes.title)), None)
                )
                if title is not None:
                    # Docutils reports underline line, or title line for overlined titles.
                    for candidate in range(
                        max(0, index - 2), min(index + 1, len(offsets) - 1)
                    ):
                        if (
                            document.content[
                                offsets[candidate] : offsets[candidate + 1]
                            ].strip()
                            == title.rawsource.strip()
                        ):
                            index = candidate
                            if candidate > 0:
                                previous = document.content[
                                    offsets[candidate - 1] : offsets[candidate]
                                ].strip()
                                if (
                                    previous
                                    and len(set(previous)) == 1
                                    and not previous[0].isalnum()
                                    and candidate + 1 < len(offsets) - 1
                                    and previous
                                    == document.content[
                                        offsets[candidate + 1] : offsets[candidate + 2]
                                    ].strip()
                                    and (
                                        candidate == 1
                                        or not document.content[
                                            offsets[candidate - 2] : offsets[
                                                candidate - 1
                                            ]
                                        ].strip()
                                    )
                                ):
                                    index -= 1
                            break
            elif isinstance(child, nodes.table):
                index = max(0, index - 1)
            return min(high, max(low, offsets[min(index, len(offsets) - 1)]))

        def convert(parent, depth: int, low: int, high: int) -> list[_Node]:
            children = []
            for child in parent.children:
                if not isinstance(child, nodes.Element):
                    continue
                if isinstance(child, nodes.system_message) and not any(
                    child.findall(nodes.literal_block)
                ):
                    # Informational diagnostics (e.g. duplicate title targets)
                    # have no independent source body and must not shorten it.
                    diagnostics.append(Diagnostic("rst_parser", child.astext()))
                    continue
                children.append(child)
            output: list[_Node] = []
            starts = [location(c, low, high) for c in children]
            for i, child in enumerate(children):
                start = starts[i]
                end = next((s for s in starts[i + 1 :] if s > start), high)
                directive_lines = getattr(child, "_fastrag_directive_lines", None)
                if directive_lines is not None:
                    end = min(end, offsets[min(directive_lines[1], len(offsets) - 1)])
                if isinstance(child, nodes.title):
                    # Only title/adornment lines are hierarchy; separator gaps
                    # belong to coverage rather than swallowing the next body.
                    first_line = offsets.index(start)
                    overline = (
                        document.content[start : offsets[first_line + 1]].strip()
                        != child.rawsource.strip()
                    )
                    end = min(
                        end,
                        offsets[
                            min(first_line + (3 if overline else 2), len(offsets) - 1)
                        ],
                    )
                elif directive_lines is not None or isinstance(
                    child, (nodes.literal_block, nodes.doctest_block)
                ):
                    # Keep original indentation/CRLF, but not separator blanks.
                    while end > start:
                        previous = max(
                            start, document.content.rfind("\n", start, end - 1) + 1
                        )
                        if document.content[previous:end].strip():
                            break
                        end = previous
                if child.rawsource and isinstance(child, nodes.paragraph):
                    # Search only within the parser-provided block bounds, preserving CRLF.
                    raw = (
                        child.rawsource.replace("\n", "\r\n")
                        if "\r\n" in document.content
                        else child.rawsource
                    )
                    found = document.content.find(raw, start, end)
                    if found >= 0:
                        start, end = found, found + len(raw)
                if isinstance(child, nodes.section):
                    output.extend(convert(child, depth + 1, start, end))
                    continue
                if isinstance(child, nodes.title):
                    output.append(
                        _Node(
                            "heading",
                            start,
                            end,
                            level=max(1, depth),
                            title=child.astext(),
                        )
                    )
                    continue
                kind = (
                    "raw_block"
                    if getattr(child, "_fastrag_raw_directive", False)
                    else "admonition"
                    if isinstance(child, nodes.Admonition)
                    else mapping.get(child.tagname)
                )
                if kind is None:
                    # Structural wrappers and inline elements keep their parent syntax.
                    nested = convert(child, depth, start, end)
                    if nested:
                        output.extend(nested)
                    elif child.rawsource:
                        output.append(
                            _Node(
                                "raw_block",
                                start,
                                end,
                                metadata={"rst_type": child.tagname},
                            )
                        )
                    continue
                meta = {
                    "rst_type": child.tagname,
                    "location_precision": "block"
                    if child.line or directive_lines
                    else "containing_block",
                }
                if not child.line and directive_lines is None:
                    diagnostics.append(
                        Diagnostic(
                            "conservative_location",
                            f"Docutils {child.tagname} lacks exact source location",
                        )
                    )
                if kind == "code_block":
                    meta.update(
                        code=child.astext(),
                        language=next(
                            (c for c in child.get("classes", []) if c != "code"), ""
                        ),
                    )
                if kind == "table_row":
                    meta["header"] = isinstance(child.parent, nodes.thead)
                    meta["cells"] = [
                        {
                            "text": c.astext(),
                            "rowspan": c.get("morerows", 0) + 1,
                            "colspan": c.get("morecols", 0) + 1,
                        }
                        for c in child.children
                        if isinstance(c, nodes.entry)
                    ]
                if kind == "image":
                    meta.update(src=child.get("uri", ""), alt=child.get("alt", ""))
                if isinstance(child, nodes.system_message):
                    diagnostics.append(Diagnostic("rst_parser", child.astext()))
                # Inline interpretation is metadata; original source remains authoritative.
                meta["references"] = [
                    {
                        "text": c.astext(),
                        "destination": c.get("refuri", c.get("refname", "")),
                    }
                    for c in child.findall(nodes.reference)
                ]
                output.append(
                    _Node(
                        kind,
                        start,
                        end,
                        convert(child, depth, start, end)
                        if kind not in {"paragraph", "code_block", "raw_block", "image"}
                        else [],
                        metadata=meta,
                    )
                )
            return output

        return finish(
            document,
            "rst",
            convert(tree, 0, 0, len(document.content)),
            diagnostics=diagnostics,
        )
