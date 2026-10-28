# Admission: rate limiting and client IP/country

Two related concerns that both need the *real* client IP: rejecting a
caller that sends too many requests, and recording who a session belongs to.
They are documented together because they share one resolution of that IP —
see {doc}`/contributing/serving-internals` for why splitting them would mean
resolving it twice.

## Client IP resolution, and why it defaults conservatively

By default, the client IP is `request.client.host` — the address of
whatever connected directly to this process. Behind a reverse proxy or load
balancer (the normal way this stack is deployed), that is the proxy's own
address, not the caller's; the caller's address is usually forwarded in an
`X-Forwarded-For` header instead.

`ServeConfig.trust_proxy_headers=True` switches to trusting that header — the
**leftmost** entry, which by convention is the original client (every proxy
in the chain appends its own address to the right). This is **off by
default** and should stay off unless a proxy in front of this app is known to
set that header itself: `X-Forwarded-For` is an ordinary request header, so
any client can send one, and trusting it unconditionally would let a caller
claim any IP it likes — defeating both the rate limiter below (spoof a fresh
IP, get a fresh quota) and the audit value of a session's recorded
`ip_address`. Turn it on only when the proxy in front of this app
overwrites (not merely forwards) the header before it reaches this process.

## Rate limiting

Three independently configurable limits, each a `"<count>/<unit>"` string
(`unit` is `second`, `minute`, `hour`, or `day`), all off by default via the
single `ServeConfig.rate_limiting_enabled` switch:

| Limit                     | Default       | Applies to                                                                 |
| ------------------------- | ------------- | --------------------------------------------------------------------------- |
| `rate_limit_generation`   | `"10/minute"` | `POST /v1/chats`, `POST /v1/chats/{id}/messages`, `POST /v1/chats/{id}/messages/{mid}` |
| `rate_limit_read`         | `"60/minute"` | `GET /v1/chats`, `GET /v1/chats/{id}`                                        |
| `rate_limit_new_session`  | `"20/hour"`   | only the branch of a request that mints a new session, not the endpoint as a whole |

`rate_limit_new_session` is narrower than it looks: `POST /v1/chats` and
`POST /v1/sessions` both *can* mint a session, but only do when the caller
presents no valid one — a returning caller with a live cookie or bearer
token never touches this limit, no matter how many chats they create (that
traffic is already bounded by `rate_limit_generation`).

Exceeding a limit is a `429` with the same `{"error": {"code", "message"}}`
envelope as every other error, `code: "RATE_LIMIT"` — a normal application
error, not a different response shape to special-case.

### Design: in-process, per-worker, no Redis

The limiter is a plain in-process sliding-window counter, keyed by client
IP. It does **not** coordinate across worker processes or machines: running
this app with `N` uvicorn workers gives roughly `limit × N` effective
capacity, not a globally enforced cap. For the great majority of
deployments — one process, or a handful behind a load balancer where an
approximate cap is good enough — this is simpler to operate than adding
Redis as a hard dependency of every deployment just to enforce a rate limit
exactly. A deployment that genuinely needs a global, cross-worker limit
should put one in front of this app (at the proxy/gateway layer) rather than
expect `fastrag.serve` to grow a distributed limiter.

## Country resolution (`ServeConfig.geoip_database_path`)

`sessions.country` is populated from a caller-supplied [MaxMind
GeoLite2-Country](https://dev.maxmind.com/geoip/geolite2-free-geolocation-data)
database (a free, account-gated download). `fastrag` does **not** vendor,
download, or ship this file — `geoip_database_path` points at wherever the
operator put it, and is `None` (country resolution off) by default.

Country resolution degrades silently rather than failing a request:

- `geoip_database_path` unset — `country` is always `NULL`.
- The file is missing, unreadable, or the `geoip` extra (`geoip2`) isn't
  installed — `country` is `NULL`, and at most one warning is logged (not one
  per request).
- The resolved IP is private or loopback, or genuinely absent from the
  database — `country` is `NULL`, no warning (this is the normal case for
  local development and internal traffic).

Install the extra to enable it at all:

```bash
uv pip install "fastrag-ai[serving,geoip]"
```

Country is resolved once, at session creation, from the same IP the
paragraph above resolves — never re-resolved or updated afterward, the same
way `ip_address` is written once and left alone.

### Getting the database

1. Sign up for a free GeoLite account at
   <https://www.maxmind.com/en/geolite2/signup>.
2. In the account portal, create a license key under
   [Manage License Keys](https://www.maxmind.com/en/accounts/current/license-key).
   Note the account ID shown alongside it.
3. Download the binary (`.mmdb`) edition `GeoLite2-Country`, either directly or
   with `geoipupdate`. Both are described below.

Direct download (HTTP Basic Auth with account ID and license key; the archive
is a gzipped tarball containing `GeoLite2-Country.mmdb`):

```bash
curl -L -u "$MAXMIND_ACCOUNT_ID:$MAXMIND_LICENSE_KEY" \
  -o GeoLite2-Country.tar.gz \
  'https://download.maxmind.com/geoip/databases/GeoLite2-Country/download?suffix=tar.gz'
tar -xzf GeoLite2-Country.tar.gz --strip-components=1 --wildcards '*/GeoLite2-Country.mmdb'
```

Or use MaxMind's [`geoipupdate`](https://github.com/maxmind/geoipupdate),
which is better for keeping the file current. Give it a `GeoIP.conf` (the portal
can generate a pre-filled one):

```text
AccountID YOUR_ACCOUNT_ID
LicenseKey YOUR_LICENSE_KEY
EditionIDs GeoLite2-Country
```

```bash
geoipupdate -f GeoIP.conf -d /path/to/geoip
```

Then point the config at the extracted file:

```python
ServeConfig(..., geoip_database_path=Path("/path/to/geoip/GeoLite2-Country.mmdb"))
```

MaxMind publishes new GeoLite releases regularly, so run `geoipupdate` on a
schedule (for example a cron job twice a week). The reader is opened once at
app startup, so restart the app to pick up a new file. The GeoLite EULA also
has terms you must follow, such as attribution and keeping the data current;
read it before you deploy.

## See also

{doc}`configuration` for every field; {doc}`metrics` for
`rejected_requests_total` and `requests_per_ip_total`, the two series this
page's rate limiter feeds; {doc}`/contributing/serving-internals` for how
the dependency-based enforcement avoids bypassing the app's exception
handlers.
