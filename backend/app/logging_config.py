"""Structured JSON logging via stdlib `logging`.

Why stdlib (no `structlog`):
  - Zero extra dependency. `structlog` would buy nicer ergonomics around
    contextvars and processor pipelines, but at the cost of another
    library to pin and another piece of API surface to learn — for ~30
    lines of code we get the same useful artifact (one JSON line per log
    record, all extras flattened) and stay aligned with FastAPI/uvicorn's
    own use of the stdlib `logging` framework.
  - On Cloud Run, anything we write to stdout is captured by Cloud Logging.
    JSON lines are auto-parsed into structured entries — `intent`,
    `confidence`, `path_taken`, `latency_ms` become first-class indexed
    fields in the log explorer without any extra agent.

What gets emitted: one log line per /route request from the dispatcher.
**The raw user input is never logged** — only `input_length`. Same for
provider credentials and any field shaped like a secret.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

# Standard LogRecord attributes we should NOT re-emit (they're either
# duplicated under nicer names or are noise). Anything added by the caller
# via `logger.info(..., extra={...})` lands in record.__dict__ and we want to
# pick it up — minus this set.
_RESERVED_LOG_RECORD_ATTRS = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "message", "module",
    "msecs", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "taskName", "thread", "threadName",
}

# Field names we treat as sensitive and drop on the floor before serialization.
# Defensive — none of the callers in this project pass these, but a future
# regression where someone adds `extra={"input": text}` would silently leak
# user data into Cloud Logging without this guard.
_SENSITIVE_FIELDS = {
    "input", "user_input", "text", "prompt", "messages",
    "api_key", "openai_api_key", "anthropic_api_key", "authorization",
    "password", "token", "secret",
}


class JSONFormatter(logging.Formatter):
    """Render each LogRecord as a single JSON line on stdout."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key in _RESERVED_LOG_RECORD_ATTRS:
                continue
            if key.startswith("_"):
                continue
            if key in _SENSITIVE_FIELDS:
                continue
            payload[key] = value

        if record.exc_info:
            # Keep the stack on logger.exception(...) but in a single field so
            # the JSON line stays parseable.
            payload["exc"] = self.formatException(record.exc_info)

        # `default=str` is a safety net for anything that's not natively JSON
        # serializable (e.g. Path, datetime in extras). Better than crashing the
        # log emitter on an unusual value.
        return json.dumps(payload, default=str)


def configure_json_logging(level: int = logging.INFO) -> None:
    """Install the JSON formatter on the root logger's stdout handler.

    Idempotent — replaces the formatter on any existing StreamHandler. Called
    once at app startup from api.py.
    """
    root = logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        handler = logging.StreamHandler()
        root.addHandler(handler)

    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(JSONFormatter())
