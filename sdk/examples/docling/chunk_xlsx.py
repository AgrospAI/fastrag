import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _runtime import config_from_env, require_output

from fastrag.ingestion.steps.chunking import (
    DoclingChunker,
    RecursiveCharacterTextSplitter,
)
from fastrag.ingestion.steps.fetching import FileSource
from fastrag.ingestion.steps.parsing import DoclingParser

XLSX_URL = "https://agrospai.udl.cat/en/files/guia_de_diseno_del_caso_de_uso.xlsx"
CHUNKS_TO_SHOW = 3


async def main():
    cfg = config_from_env()
    url = os.environ.get("FASTRAG_XLSX_URL", XLSX_URL)
    show = int(os.environ.get("FASTRAG_CHUNKS_TO_SHOW", str(CHUNKS_TO_SHOW)))
    if show < 1:
        raise ValueError("CHUNKS_TO_SHOW must be positive")
    # No PostgresCache here — this example must run with no services running,
    # unlike its siblings, which need `docker compose up -d`.
    recursive_chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, cache=False, policy=cfg.policy()
    )(
        DoclingParser(cache=False, policy=cfg.policy())(
            FileSource(url=url, cache=False, policy=cfg.policy())
        )
    )

    print("--- DoclingParser() + RecursiveCharacterTextSplitter ---")
    count = 0
    async for chunk in require_output(recursive_chunks):
        if count >= show:
            continue
        headings = chunk.metadata.get("headings", "n/a")
        print(f"chunk_index={chunk.metadata['chunk_index']} headings={headings}")
        print(chunk.content)
        print()
        count += 1

    docling_chunks = DoclingChunker(cache=False, policy=cfg.policy())(
        DoclingParser(output_format="docling", cache=False, policy=cfg.policy())(
            FileSource(url=url, cache=False, policy=cfg.policy())
        )
    )

    print("--- DoclingParser(output_format='docling') + DoclingChunker ---")
    count = 0
    async for chunk in require_output(docling_chunks):
        if count >= show:
            continue
        print(
            f"chunk_index={chunk.metadata['chunk_index']} headings={chunk.metadata['headings']}"
        )
        print(chunk.content)
        print()
        count += 1


if __name__ == "__main__":
    asyncio.run(main())
