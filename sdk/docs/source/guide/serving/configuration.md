# Configuration

Everything the serving app needs is one frozen, keyword-only
`ServeConfig(allowed_origins=..., ...)` handed to `create_app`. Because
credentials and endpoints are constructor arguments everywhere (see
{doc}`/guide/configuration`), a caller reads the environment and passes the
result in — `fastrag.serve` itself never reads `os.environ`.

## Required: `allowed_origins`

The one required field. The app configures its own `CORSMiddleware` from this
list, so callers never add a middleware themselves. A `"*"` wildcard is
rejected at construction — credentialed CORS cannot use
`Access-Control-Allow-Origin: *` — so a browser-targeted deployment must
enumerate its origins.

## Every field

| Field                   | Default      | Meaning                                                  |
| ----------------------- | ------------ | -------------------------------------------------------- |
| `allowed_origins`       | — (required) | CORS origins                                             |
| `cookie_secure`         | `True`       | `Secure` flag on the session cookie                      |
| `cookie_domain`         | `None`       | `Domain` for the session cookie                          |
| `session_ttl_days`      | —            | how long a session (and its chats) live without activity |
| `expiry_bump_min_days`  | —            | minimum headroom before an expiry bump is written        |
| `max_request_bytes`     | —            | reject a larger request body with `413`                  |
| `max_content_chars`     | —            | reject a longer `content` with `422`                     |
| `max_messages_per_chat` | —            | cap (422 beyond), checked by counting messages           |
| `max_chats_per_session` | —            | cap (422 beyond), checked by counting chats              |
| `generation_timeout_s`  | —            | end-to-end budget; a timeout lands as `TIMEOUT`          |
| `keepalive_interval_s`  | —            | seconds between SSE keep-alive comments and lock-connection pings |
| `log_level`             | —            | root logger level for the JSON log output                |
| `bearer_tokens`         | `False`      | enable the bearer-token path and `POST /v1/sessions`     |
| `metrics_enabled`       | `True`       | mount `GET /metrics` at all ({doc}`metrics`)              |
| `metrics_token`         | `None`       | bearer token guarding `GET /metrics`, unguarded if unset  |
| `trust_proxy_headers`   | `False`      | trust `X-Forwarded-For` for the client IP ({doc}`admission`) |
| `rate_limiting_enabled` | `False`      | enforce the three limits below at all ({doc}`admission`)  |
| `rate_limit_generation` | `"10/minute"` | limit on the generation endpoints, per client IP          |
| `rate_limit_read`       | `"60/minute"` | limit on the two `GET` list/read endpoints, per client IP |
| `rate_limit_new_session`| `"20/hour"`  | limit on minting a new session, per client IP              |
| `geoip_database_path`   | `None`       | path to a GeoLite2-Country `.mmdb` for `country` resolution |

Every field is validated in `__post_init__` — a bad value fails at the line
that constructed the config, not at startup.

## See also

{doc}`http-api` for the routes these fields gate; {doc}`metrics` for what
`GET /metrics` exports; {doc}`admission` for rate limiting and client
IP/country resolution; `examples/14_serving.py` for a complete `ServeConfig`
and `create_app` wiring.
