import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _runtime import config_from_env, require_output

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    ConvertPipelineOptions,
    PictureDescriptionApiOptions,
)
from docling.document_converter import ExcelFormatOption

from fastrag.ingestion.steps.fetching import FileSource
from fastrag.ingestion.steps.parsing import DoclingParser

XLSX_URL = "https://agrospai.udl.cat/en/files/guia_de_diseno_del_caso_de_uso.xlsx"

# A local OpenAI-compatible vision model describes the images embedded in
# the spreadsheet; without it Docling can only emit the `<!-- image -->`
# placeholder, and the pictures contribute nothing to the parsed text.
# Start Ollama locally and download the vision model with `ollama pull llava`.
BASE_URL = "http://localhost:11434/v1"
API_KEY = "ollama"
VISION_MODEL = "llava"

image_prompt = """
Extract all useful information from this image for a Retrieval-Augmented Generation (RAG) system.

Your goal is faithful information extraction, NOT a brief visual description.

Instructions:

1. Transcribe all visible text exactly as written, preserving:
    * headings and labels
    * numbers, dates, times, percentages, prices, units, IDs, codes, URLs, and formulas
    * punctuation and capitalization where meaningful
    * table contents and column/row relationships
2. For tables, reproduce the data in a clear Markdown table whenever possible. Do not omit rows, columns, totals, headers, footnotes, or captions.
3. For charts, graphs, diagrams, and plots, extract:
    * title
    * axis labels and units
    * legend entries
    * category names
    * displayed values
    * annotations
    * trends or relationships explicitly supported by the visual
4. For forms, extract field names and their corresponding values, including unchecked/checked states when visible.
5. For screenshots, extract all visible UI text, messages, labels, buttons, filenames, paths, and other meaningful information.
6. For maps, extract visible place names, labels, legends, routes, and other textual information. Do not infer locations that are not explicitly shown.
7. Preserve the logical reading order and relationships between pieces of information.
8. If text is partially obscured, blurry, cropped, or illegible, mark it as [illegible] or [uncertain] rather than guessing.
9. Do not invent, infer, summarize, or paraphrase information that is not visibly supported by the image.
10. Include a short “Visual context” section only when it helps interpret the extracted information (for example, explaining that a table is a financial summary or that a diagram shows a process).
11. If the image contains no meaningful textual or structured information, briefly describe the visual content and explain why it may still be relevant for retrieval.

Return the result as clean, self-contained text suitable for embedding and retrieval. Be comprehensive rather than concise.
"""


def xlsx_options():
    cfg = config_from_env()
    description = PictureDescriptionApiOptions(
        url=f"{os.environ.get('FASTRAG_VISION_BASE_URL', BASE_URL).rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ.get('FASTRAG_VISION_API_KEY', API_KEY)}"
        },
        params={"model": os.environ.get("FASTRAG_VISION_MODEL", VISION_MODEL)},
        prompt=image_prompt,
        timeout=cfg.timeout,
        concurrency=1,
        picture_area_threshold=0.0,
    )
    options = ConvertPipelineOptions(
        do_picture_description=True,
        picture_description_options=description,
        enable_remote_services=True,
    )
    return {InputFormat.XLSX: ExcelFormatOption(pipeline_options=options)}


async def main():
    cfg = config_from_env()
    source = FileSource(
        url=os.environ.get("FASTRAG_XLSX_URL", XLSX_URL),
        cache=False,
        policy=cfg.policy(),
    )
    parser = DoclingParser(
        cache=False, format_options=xlsx_options(), policy=cfg.policy()
    )
    documents = parser(source)

    async for document in require_output(documents):
        source_uri = document.metadata["source_uri"]
        content_type = document.metadata["content_type"]
        print(f"Source: {source_uri} ({content_type})")
        print(f"Document: {document.id}" + (" (cached)" if document.cached else ""))
        print(f"content_format: {document.metadata['content_format']}")
        print("Metadata:")
        for key, value in document.metadata.items():
            print(f"  {key}: {value}")
        print("Content:")
        print(document.content)


if __name__ == "__main__":
    asyncio.run(main())
