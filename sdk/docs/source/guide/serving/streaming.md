# Streaming

Every generation answer is a `text/event-stream` response. The frame shapes
below are exactly the ones the app created by `fastrag.serve.create_app`
writes and the example client `examples/15_asking.py` parses.

Each event is two lines — `event: <name>` and `data: <json>` — separated by
blank lines. Keep-alive comments (`: keep-alive`) arrive between tokens on a
long generation.

## The event sequence

```text
event: message_start   data: {chat_id, user_message_id, message_id}
event: sources         data: {message_id, sources: [{url, title, site_name, favicon_url, positions, fragments}, ...]}
: keep-alive  ...
event: delta           data: {message_id, delta: {"type":"text","text":"..."}}
...
event: message_end     data: {message_id, finish_reason: "stop"}
```

- **`message_start`** — the server-chosen ids for the user message and the
  assistant message it will produce. A client never chooses an id.
- **`sources`** — the retrieved pages, one per `url`, as `{url, title,
  site_name, favicon_url, positions, fragments}` — sent before any `delta` so
  a client can display them before the answer streams. `title` falls back to
  the url. `site_name` and `favicon_url` are what the page declared in its
  `<head>` (`og:site_name`, `<link rel="icon">`) when it was fetched, or
  `null`. A source fetched alongside a page of the same site, such as a PDF
  the page links to, inherits them. `positions` holds the `n` the answer
  cites the page as `[n]`: the prompt numbers each page once, with all of its
  retrieved chunks. `fragments` locates those chunks on the page as `{start,
  end}` text fragment terms, one per passage, with chunks of one section
  joined. A link to `url#:~:text=start,end&text=…` (each term
  percent-encoded, `-` included) scrolls to the passages and highlights them.
  `fragments` is empty for a page the browser cannot search, such as a PDF.
- **`delta`** — one per token.
- **`message_end`** — carries the `finish_reason` (`stop`, `length`,
  `error`, or `interrupted` on a disconnect). When `finish_reason` is
  `"error"`, the frame also carries `error: {code, message}`, e.g.
  `event: message_end   data: {message_id, finish_reason: "error", error:
  {code: "TIMEOUT", message: "..."}}`. `code` is one of the `error_code`
  values a failed generation can actually produce: `TIMEOUT`,
  `CONTEXT_TOO_LONG`, `SERVICE_UNAVAILABLE`, `MODEL_ERROR`, or
  `INTERNAL_ERROR`. This is a different, narrower vocabulary than the
  request-level `code` in a `4xx`/`5xx` JSON error body ({doc}`http-api`) —
  a request that never reaches a generation (a validation failure, a cap
  reached, an unknown chat) is rejected before any `message_start` and never
  produces one of these.

A message is stored only once its generation finishes, so an interrupted
stream is recorded with `finish_reason: "interrupted"` rather than resumed.

## See also

{doc}`http-api` for the routes that produce these frames; {doc}`configuration`
for the fields that bound a generation; {doc}`/contributing/frontend-types`
for how these event shapes are published as named OpenAPI schemas
(`MessageStartEvent`, `SourcesEvent`, `DeltaEvent`, `MessageEndEvent`) and
generated into TypeScript.
