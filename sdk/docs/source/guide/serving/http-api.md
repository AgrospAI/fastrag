# HTTP API

The `/v1` chats API, served by `create_app`, is a thin adapter over a
`ChatStore` plus a `QueryEngine`. A client with a session cookie (or a
bearer token) creates chats and posts messages; every generation answer
comes back as an SSE stream ({doc}`streaming`).

## Sessions

`POST /v1/sessions` mints a session and returns its `session_id` in the
response body (`404` unless `ServeConfig.bearer_tokens` is enabled). The session id
is the bearer credential — sent as a cookie, or in the `Authorization:
Bearer <token>` header. A request without a valid session is `401
UNAUTHORIZED`.

## Chats and messages

| Method & path                        | Body                        | Returns                                                         |
| ------------------------------------ | --------------------------- | --------------------------------------------------------------- |
| `POST /v1/chats`                     | `{"content": "..."}`        | a fresh chat and the streamed answer to the opening message     |
| `GET /v1/chats`                      | —                           | the session's chats, newest first, paginated                    |
| `GET /v1/chats/{id}`                 | —                           | one chat and its messages                                       |
| `DELETE /v1/chats/{id}`              | —                           | removes the chat and its messages                               |
| `POST /v1/chats/{id}/messages`       | `{"content": "..."}`        | appends a user message and streams the answer                   |
| `POST /v1/chats/{id}/messages/{mid}` | `{"content": ...}` optional | regenerate (or edit) the message at `mid` and stream the answer |

Cap checks (`MAX_MESSAGES_PER_CHAT`, `MAX_CHATS_PER_SESSION`) and a `content`
over `max_content_chars` all surface as `422`; a missing chat or message as
`404`; lock contention during a generation as `409`; a rate limit exceeded
(when `ServeConfig.rate_limiting_enabled`, see {doc}`admission`) as `429`
with code `RATE_LIMIT`. Every JSON error body is `{"error": {"code",
"message"}}`; a request-level failure like these carries a free-form `code`
such as `INVALID_REQUEST` or `NOT_FOUND`, while a generation that starts and
then fails carries one of the `error_code` values (e.g. `MODEL_ERROR`,
`TIMEOUT`) documented in {doc}`streaming`.

## Health

`GET /healthz` returns `{"status": "ok"}` without touching the query engine.

## Metrics

`GET /metrics` exports a Prometheus text exposition of the app's own traffic
and generation series; see {doc}`metrics`.

## See also

{doc}`streaming` for the SSE frame shapes; {doc}`metrics` for `GET /metrics`;
{doc}`admission` for rate limiting and client IP/country resolution;
{doc}`configuration` for the fields that gate these routes;
`examples/15_asking.py` for a client that talks to this API;
{doc}`/contributing/frontend-types` for the generated OpenAPI document and
TypeScript types this API's contract is now published as.
