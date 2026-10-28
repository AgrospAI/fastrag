"""``GET /metrics``: a flat ``prometheus_client`` vocabulary, no OpenTelemetry.

``dashboard/lib/metrics/prometheus.ts``'s ``PROM_QUERIES`` is the contract —
these are exactly the series names it queries, spelled the same way. The
pre-queue-4 CLI served the OTel semantic-convention names (``gen_ai.client.*``,
``rag.*``) instead; that vocabulary and its OTel machinery
(``FastAPIInstrumentor``, an OTLP exporter, a meter/tracer provider) are
deliberately not carried over here — this SDK exists so a user can test
their own pipeline, and eight extra OTel dependencies is not what that user
is here for.

Each app gets its own :class:`~prometheus_client.CollectorRegistry` rather
than the process-global default one, so two apps in one process (the test
suite does this) never raise a duplicate-timeseries error.
"""

from __future__ import annotations

import hmac
from collections.abc import Awaitable, Callable

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from fastrag.serve.config import ServeConfig

#: Not part of ``/v1`` and not counted by ``http_requests_total`` — scraping
#: must not inflate the very traffic numbers it reports.
METRICS_ROUTE = "/metrics"

#: ``generate_latest`` always renders the classic Prometheus text exposition
#: (``_total``/``_bucket``/``_sum``/``_count`` suffixes, the format
#: ``dashboard/lib/metrics/prometheus.ts``'s PromQL assumes) regardless of
#: installed version, but newer ``prometheus_client`` releases changed its
#: own ``CONTENT_TYPE_LATEST`` constant to advertise ``version=1.0.0``
#: (an OpenMetrics-negotiation artifact, not a body format change). Setting
#: the header explicitly keeps it pinned to the classic, universally
#: understood ``version=0.0.4`` this body actually is.
_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


class ServeMetrics:
    """The app's Prometheus series, all bound to one owned registry.

    Constructed once per :func:`~fastrag.serve.create_app` call and stored at
    ``app.state.metrics``; :class:`MetricsMiddleware` and the generation loop
    in ``_generation.py`` are the only writers.
    """

    def __init__(self) -> None:
        """Register every series on a fresh, app-owned registry."""
        self.registry = CollectorRegistry()

        self.http_requests_total = Counter(
            "http_requests_total",
            "HTTP requests received, excluding /metrics itself.",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.http_requests_in_flight = Gauge(
            "http_requests_in_flight",
            "HTTP requests currently being handled.",
            registry=self.registry,
        )
        self.http_request_errors_total = Counter(
            "http_request_errors_total",
            "HTTP requests that ended in a 5xx response or an unhandled exception.",
            registry=self.registry,
        )
        self.llm_time_to_first_token_seconds = Histogram(
            "llm_time_to_first_token_seconds",
            "Time from the start of a generation to its first token reaching "
            "the client.",
            registry=self.registry,
        )
        self.llm_time_to_last_token_seconds = Histogram(
            "llm_time_to_last_token_seconds",
            "Time from the start of a generation to it reaching a terminal "
            "state (done, timed out, errored, or interrupted).",
            registry=self.registry,
        )
        self.llm_question_length_chars = Histogram(
            "llm_question_length_chars",
            "Length, in characters, of each accepted user message.",
            registry=self.registry,
        )
        self.llm_answer_length_chars = Histogram(
            "llm_answer_length_chars",
            "Length, in characters, of each persisted assistant message.",
            registry=self.registry,
        )
        self.rejected_requests_total = Counter(
            "rejected_requests_total",
            "Requests rejected by rate limiting (every 429 the limiter raised).",
            registry=self.registry,
        )
        self.requests_per_ip_total = Counter(
            "requests_per_ip_total",
            "Requests seen by the rate limiter, labelled by route. Not "
            "labelled by IP itself — that would make cardinality unbounded.",
            ["route"],
            registry=self.registry,
        )

    def render(self) -> bytes:
        """Render every series as Prometheus text exposition."""
        return generate_latest(self.registry)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Records ``http_requests_total`` / ``_in_flight`` / ``_errors_total``.

    Registered outward of :class:`~fastrag.serve._guards.RequestGuardMiddleware`
    (see ``app.py``'s middleware order), so a request the guard rejects with
    ``413``/``415`` before it ever reaches the router is still counted. The
    route label is the matched route's path template
    (``request.scope["route"]``, set once Starlette's router has matched)
    rather than the raw path, so a chat or message id in the URL never
    becomes its own timeseries; a request that never reaches the router
    (guard-rejected, or a genuine 404) falls back to the raw path instead.
    """

    def __init__(self, app, metrics: ServeMetrics) -> None:
        """Bind to the app's :class:`ServeMetrics`, one per registry."""
        super().__init__(app)
        self.metrics = metrics

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Count the request, excluding ``/metrics`` itself."""
        if request.url.path == METRICS_ROUTE:
            return await call_next(request)

        self.metrics.http_requests_in_flight.inc()
        try:
            try:
                response = await call_next(request)
            except Exception:
                self._record(request, status="500")
                self.metrics.http_request_errors_total.inc()
                raise
            self._record(request, status=str(response.status_code))
            if response.status_code >= 500:
                self.metrics.http_request_errors_total.inc()
            return response
        finally:
            self.metrics.http_requests_in_flight.dec()

    def _record(self, request: Request, *, status: str) -> None:
        route = request.scope.get("route")
        route_path = route.path if route is not None else request.url.path
        self.metrics.http_requests_total.labels(
            method=request.method, route=route_path, status=status
        ).inc()


def metrics_response(request: Request) -> Response:
    """Build the ``GET /metrics`` response: ``401`` when a token is set and missing/wrong.

    ``config.metrics_token`` compared with :func:`hmac.compare_digest`
    against the request's ``Authorization: Bearer <token>`` header — a
    mismatch or absent header is ``401`` with no metric text, exactly like
    every other authentication failure in ``fastrag.serve``.
    """
    config: ServeConfig = request.app.state.config
    if config.metrics_token is not None:
        authorization = request.headers.get("authorization", "")
        scheme, _, token = authorization.partition(" ")
        expected = f"Bearer {config.metrics_token}"
        if not hmac.compare_digest(f"{scheme} {token}" if scheme else "", expected):
            return Response(status_code=401)

    metrics: ServeMetrics = request.app.state.metrics
    return Response(content=metrics.render(), media_type=_CONTENT_TYPE)
