"""Generate the openapi.json file for the FastRAG API.

This allows the frontend to generate TypeScript types for having an end-to-end type-safe experience.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from fastrag.query.chat.base import ChatStore
from fastrag.query.engine import QueryEngine
from fastrag.serve import ServeConfig, create_app
from fastrag.serve._openapi import openapi_schema

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "openapi.json"


def main() -> None:
    """Build the app and write its OpenAPI document to :data:`OUTPUT_PATH`."""
    app = create_app(
        engine=cast(QueryEngine, object()),
        chat_store=cast(ChatStore, object()),
        config=ServeConfig(allowed_origins=[]),
    )
    schema = openapi_schema(app)

    OUTPUT_PATH.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
