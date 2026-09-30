"""Structured JSON logging with request IDs, and Prometheus metrics."""
from __future__ import annotations

import contextvars
import json
import logging
import time
import uuid

from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

HTTP_LATENCY = Histogram("abstain_http_request_seconds", "HTTP request latency", ["method", "route", "status"])
LLM_CALLS = Counter("abstain_llm_calls_total", "LLM call outcomes", ["outcome"])
LLM_LATENCY = Histogram("abstain_llm_latency_seconds", "LLM call latency (uncached)",
                        buckets=(0.25, 0.5, 1, 2, 4, 8, 16, 32))
LLM_TOKENS = Counter("abstain_llm_tokens_total", "LLM tokens consumed", ["kind"])
DECISIONS = Counter("abstain_decisions_total", "Decisions made", ["action", "extraction_source"])
DEGRADED = Counter("abstain_degraded_evaluations_total", "Evaluations served by the rules fallback", ["reason"])

_STANDARD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        entry.update({k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS})
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns/propagates X-Request-ID and records per-route latency."""

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            route = request.scope.get("route")
            HTTP_LATENCY.labels(request.method, getattr(route, "path", "unmatched"), str(status)).observe(
                time.perf_counter() - started)
            request_id_var.reset(token)
