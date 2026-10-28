# Metrics

`GET /metrics` exports a Prometheus text exposition of the serving app's own
traffic and generation series — flat `prometheus_client` names, no
OpenTelemetry. The names are chosen to match
`dashboard/lib/metrics/prometheus.ts`'s `PROM_QUERIES` exactly, so the
dashboard's PromQL is correct against a stock `create_app` out of the box.

```bash
curl -sf http://127.0.0.1:8100/metrics | head -20
```

## The series

| Series                              | Type      | Meaning                                                          |
| ------------------------------------ | --------- | ----------------------------------------------------------------- |
| `http_requests_total`                | counter   | requests received, labelled by `method`, `route`, `status`; excludes `/metrics` itself |
| `http_requests_in_flight`            | gauge     | requests currently being handled                                  |
| `http_request_errors_total`          | counter   | requests that ended in a `5xx` response or an unhandled exception |
| `llm_time_to_first_token_seconds`    | histogram | time from the start of a generation to its first token reaching the client |
| `llm_time_to_last_token_seconds`     | histogram | time from the start of a generation to it reaching a terminal state (done, timed out, errored, or interrupted) |
| `llm_question_length_chars`          | histogram | length of each accepted user message                              |
| `llm_answer_length_chars`            | histogram | length of each persisted assistant message                        |
| `rejected_requests_total`            | counter   | requests rejected by rate limiting — every `429` the limiter raised |
| `requests_per_ip_total`              | counter   | requests seen by the rate limiter, labelled by `route`             |

`route` is the matched route's path template (e.g.
`/v1/chats/{chat_id}/messages`), not the raw URL, so a chat or message id
never becomes its own timeseries. A request rejected before routing — by the
request guard, or a genuine `404` — falls back to the raw path.

Both `llm_time_to_first_token_seconds` and `llm_time_to_last_token_seconds`
are measured against the same clock, started once per generation, so a
generation that streams tokens over several seconds always records a TTFT
strictly smaller than its TTLT.

`requests_per_ip_total` is, despite its name, not labelled by the client IP
itself — only by `route` — to keep its cardinality bounded; see
{doc}`admission` for the rate limiter that feeds both of these series and
for why they read zero until `ServeConfig.rate_limiting_enabled=True`.

## Configuration

`GET /metrics` is on by default; `ServeConfig.metrics_enabled=False` withholds
the route entirely (a `404`, not a `401`). `ServeConfig.metrics_token`
optionally guards it — a request without a matching
`Authorization: Bearer <token>` header gets `401` and no metric text. See
{doc}`configuration` for both fields.

## See also

{doc}`configuration` for the fields that gate this route; {doc}`http-api` and
{doc}`streaming` for the `/v1` routes these series describe; {doc}`admission`
for the rate limiter and IP/country resolution behind the last two series.
