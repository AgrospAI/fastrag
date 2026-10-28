"""Extra OpenAPI schema components FastAPI does not publish on its own.

FastAPI only adds a Pydantic model to ``components/schemas`` when some
route's declared request or response references it. The SSE frame payloads
never are — a streaming route's actual return type is a bare
``StreamingResponse`` with no declared body schema — so :func:`openapi_schema`
force-publishes them, purely so a document built from
:func:`~fastrag.serve.create_app` still names every shape ``streaming.md``
documents, for a TypeScript generator (or any other OpenAPI consumer) to
pick up as a real, reusable type.

Used only by ``scripts/generate_openapi.py`` (``task openapi``) — the app's
own in-process ``app.openapi()`` (what a running server would answer
``GET /openapi.json`` with, if docs were ever enabled) is untouched.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic.json_schema import models_json_schema

from fastrag.serve._schemas import (
    DeltaEvent,
    MessageEndEvent,
    MessageStartEvent,
    SourcesEvent,
    TextDelta,
)

#: The SSE event payloads named in ``streaming.md``, in event order. No
#: route response references these, so they would otherwise never appear in
#: ``components/schemas`` at all.
SSE_EVENT_SCHEMAS = (
    MessageStartEvent,
    SourcesEvent,
    DeltaEvent,
    TextDelta,
    MessageEndEvent,
)


def _drop_null_defaults(defs: dict[str, Any]) -> None:
    """Strip ``"default": null`` from every property, in place.

    FastAPI's own route-based schema generation never emits ``default`` on a
    response model's properties at all (a response has no concept of a
    caller-omitted field); :func:`~pydantic.json_schema.models_json_schema`,
    called directly here with no route to attach to, does. Left alone, that
    mismatch makes a TypeScript generator treat an
    ``x: Model | None = None`` field on one of these models as always
    present (``x: Model | null``) while the identical pattern on a
    route-attached model like ``MessageResponse`` comes out optional (``x?:
    Model | null``) — same shape, different generated type, for a reason
    that has nothing to do with the wire format. Every field affected here
    is already absent from its model's own ``required`` list (pydantic never
    requires a field with a default), so this only removes a misleading
    annotation, not a real constraint.
    """
    for definition in defs.values():
        for prop in definition.get("properties", {}).values():
            if prop.get("default", "not-null-sentinel") is None:
                del prop["default"]


def openapi_schema(app: FastAPI) -> dict[str, Any]:
    """``app``'s OpenAPI document, with the SSE event schemas added."""
    schema = app.openapi()
    _, definitions = models_json_schema(
        [(model, "serialization") for model in SSE_EVENT_SCHEMAS],
        ref_template="#/components/schemas/{model}",
    )
    defs = definitions.get("$defs", {})
    _drop_null_defaults(defs)
    schema.setdefault("components", {}).setdefault("schemas", {}).update(defs)
    return schema
