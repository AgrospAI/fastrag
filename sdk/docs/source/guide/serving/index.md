# Serving

Expose a `QueryEngine` plus a `ChatStore` as a FastAPI app that streams
grounded answers over SSE. `fastrag.serve.create_app(engine=, chat_store=,
config=)` is a thin adapter over a `Retriever`+`ChatModel` you already built —
it owns no database engine and no ORM session, and it is the only place in
the SDK a `PipelineStep` is piped to an HTTP `POST` rather than an `async
for`.

```{literalinclude} /../../examples/14_serving.py
:language: python
:start-after: "docs:serving-create-app-start"
:end-before: "docs:serving-create-app-end"
:dedent:
```

Start it with `task serve` (runs `examples/14_serving.py` on `:8100`), index
once first with `task index`-equivalent (`examples/05_indexing.py`), then:

```bash
curl -sf http://127.0.0.1:8100/healthz
curl -sN -X POST http://127.0.0.1:8100/v1/chats \
  -H 'content-type: application/json' -d '{"content":"What is AgrospAI?"}'
```

The fastest way to see this running for real, with a dashboard and the
chatbot widget in front of it rather than a bare `curl`, is the repository
root's composed stack (`docker compose -f compose/docker-compose.yaml up -d` from the repository root,
documented in its `README.md`) — `sdk/app/` is the same `create_app` call
shown above, built into the `backend` service that stack runs.

The serving docs split into five audiences below: the wire contract
(`{doc}`http-api``), how the stream is framed (`{doc}`streaming``), what
`GET /metrics` exports (`{doc}`metrics``), rate limiting and client
IP/country resolution (`{doc}`admission``), and what the app is configured
with (`{doc}`configuration``).

```{toctree}
:maxdepth: 1
:hidden:

http-api
streaming
metrics
admission
configuration
```
