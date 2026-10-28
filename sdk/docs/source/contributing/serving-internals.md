# Serving internals

What a contributor needs to know about `fastrag.serve` beyond the wire
contract ({doc}`/guide/serving/http-api`): the layout, the lock design, and
the model. `fastrag.serve` is the port of the pre-queue-4 CLI serving layer
onto the new `QueryEngine` + `ChatStore` design; the wire contract is
unchanged, so a client written against the old app works unmodified.

## Layout

Only five modules are public; everything else is `_`-prefixed and internal.

- `app.py` — `create_app`, the app factory with no engine of its own.
- `config.py` — `ServeConfig`, frozen and validated.
- `cli.py` — `CLI`, the Typer client ({doc}`/guide/serving/http-api`).
- `cleanup.py` — `cleanup_expired`, the batched expiry sweep.
- `_auth.py`, `_errors.py`, `_generation.py`, `_guards.py`, `_locking.py`,
  `_logging.py`, `_metrics.py`, `_net.py`, `_openapi.py`, `_rate_limit.py`,
  `_routes.py`, `_schemas.py`, `_shield.py`, and `_sse.py` are internal to
  `serve`. `_openapi.py` force-publishes the SSE event payload schemas into
  the OpenAPI document's `components/schemas` — used only by
  `scripts/generate_openapi.py` (`task openapi`), see {doc}`frontend-types`.
  The
  ULID minting `_generation.py` uses for a stored assistant message id lives
  at `fastrag._ids`, shared with `fastrag.query.chat` (see
  {doc}`architecture`'s "shared private layer"). `_metrics.py` holds
  `ServeMetrics` (the app-owned `CollectorRegistry` and its nine series) and
  `MetricsMiddleware`; both are wired up by `app.py`, never imported by a
  caller directly — see {doc}`/guide/serving/metrics`. `_net.py` holds
  `resolve_client_ip` (shared by the rate limiter and the session writer, so
  the two can never disagree about who a request is from). The optional
  GeoIP lookup is grouped by feature in `geolocalization/`: `resolver.py`
  holds `CountryResolver` (built by `create_app`, closed by its lifespan)
  and `dependencies.py` the `get_country_resolver` dependency
  `get_or_create_session` receives it through; `_rate_limit.py` holds the
  sliding-window `RateLimiter` itself, deliberately free of any import from
  `_metrics.py`, `_errors.py`, or `_net.py` — `config.py` imports
  `_rate_limit.parse_rate` to validate its own rate-limit fields, and a
  `_rate_limit → _metrics → config` edge would make that an import cycle.
  See {doc}`/guide/serving/admission` for the user-facing behaviour both
  modules produce.

`create_app(engine=, chat_store=, config=)` — `engine` is a
`QueryEngine`. The app borrows the store's engine (via the
`PostgresChatStore.engine` property) for the advisory lock and `/healthz`;
it creates no engine or session itself.

## Design

- **The session id is the bearer secret.** There is no username or password;
  the session ULID _is_ the credential, stored as the plain-text primary key
  and used as both the bearer token and the cookie value. Session tokens are
  deliberately not hashed.
- **Generations are serialized per chat** with a session-level Postgres
  advisory lock, acquired before generate and released after; contention
  returns `409`. A store at a non-default schema needs a schema-qualified
  variant before `cleanup_expired` will sweep that schema.
- **Regenerate/edit appends rather than truncates.** `ChatStore` has no
  range-delete primitive, so `POST /v1/chats/{id}/messages/{mid}` validates
  the request fully and returns the same SSE stream, but the chat stays
  linear and grows — see `cli.py`'s module docstring. Restoring true
  truncate needs a `ChatStore` primitive first.
- **Rate limiting is a dependency, not a middleware.** `_guards.rate_limit(...)`
  is a FastAPI dependency (`dependencies=[Depends(rate_limit("generation"))]`
  on the gated routes), not a `BaseHTTPMiddleware` like
  `RequestGuardMiddleware`/`MetricsMiddleware`. It has to be: a dependency
  runs inside the router, after FastAPI's `ExceptionMiddleware` is already
  wrapping the call, so raising `ApiError` there is caught by the same
  registered handler every other request-time error goes through. A
  `BaseHTTPMiddleware` sits *outside* `ExceptionMiddleware` in Starlette's
  stack — an exception raised directly from its `dispatch` (as opposed to
  one raised inside the `call_next` it awaits) would bypass those handlers
  entirely and fall through to a generic 500, which is exactly why
  `RequestGuardMiddleware` above builds its `JSONResponse` by hand instead
  of raising. The one limit that isn't a route-level dependency,
  `rate_limit_new_session`, is checked inline in
  `_auth.get_or_create_session` for the same reason `enforce_content_length`
  isn't middleware either: it only applies to one branch of one function,
  not to a route as a whole.

## See also

{doc}`/guide/serving/index` for the user's door; {doc}`/guide/serving/admission`
for the rate limiter and IP/country resolution this section describes the
placement of; {doc}`architecture` for where the package sits in the tree.
