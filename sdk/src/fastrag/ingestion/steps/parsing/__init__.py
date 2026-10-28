"""Source parsing: raw bytes in, documents out.

HTML parsers preserve hidden content and SVG labels by default. JavaScript
rendering and vision descriptions are optional. ``ParserRouter`` dispatches
each source to a parser by filter, e.g. ``html_filter`` to ``HtmlParser``.
"""

from __future__ import annotations

from fastrag.ingestion.steps.parsing.ai import DecisionsParser, OpenAIParser
from fastrag.ingestion.steps.parsing.audio import AudioParser, audio_filter
from fastrag.ingestion.steps.parsing.chandra import ChandraParser
from fastrag.ingestion.steps.parsing.base import Parser
from fastrag.ingestion.steps.parsing.default import DefaultParser
from fastrag.ingestion.steps.parsing.docling import DoclingParser, office_filter
from fastrag.ingestion.steps.parsing.docling_serve import DoclingServeParser
from fastrag.ingestion.steps.parsing.docusaurus import (
    DocusaurusParser,
    docusaurus_filter,
)
from fastrag.ingestion.steps.parsing.html import HtmlParser, html_filter
from fastrag.ingestion.steps.parsing.mineru import MineruParser
from fastrag.ingestion.steps.parsing.pdf import PDFParser, pdf_filter
from fastrag.ingestion.steps.parsing.router import ParserRouter
from fastrag.ingestion.steps.parsing.vision import OpenAIVisionParser
from fastrag.ingestion.steps.parsing.svg import SvgParser
from fastrag.ingestion.steps.parsing.trafilatura import TrafilaturaParser

__all__ = [
    "AudioParser",
    "ChandraParser",
    "DecisionsParser",
    "OpenAIParser",
    "DefaultParser",
    "DoclingParser",
    "DoclingServeParser",
    "DocusaurusParser",
    "HtmlParser",
    "MineruParser",
    "PDFParser",
    "Parser",
    "ParserRouter",
    "OpenAIVisionParser",
    "SvgParser",
    "TrafilaturaParser",
    "audio_filter",
    "docusaurus_filter",
    "html_filter",
    "office_filter",
    "pdf_filter",
]
