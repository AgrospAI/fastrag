"""One JSON object per line to stdout."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())


class JsonFormatter(logging.Formatter):
    """One JSON object per line.

    Never emits multi-line records so log lines survive line-oriented
    pipelines. ``trace_id``/``span_id`` are omitted rather than emitted empty
    — there is no tracing in this port.
    """

    def __init__(self, service_name: str) -> None:
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "service.name": self.service_name,
            "event.name": getattr(record, "event.name", record.name),
        }

        for key, value in record.__dict__.items():
            if key not in _RESERVED and key not in payload:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_serve_logging(service_name: str, level: str) -> None:
    """Install the JSON formatter on the root logger."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter(service_name))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
