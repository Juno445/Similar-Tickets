"""
monitoring.py
=============

Tiny Prometheus / OpenMetrics helper.

• ASGI middleware measures request‐latency & status-code counts.
• `/metrics` endpoint in `webhook_server.py` just calls `metrics_response()`
  so it stays framework-agnostic (works with Starlette, FastAPI, etc.).

Why not use `prometheus_fastapi_instrumentator`?
------------------------------------------------
That package is great but drags in dependencies and starts its own
background tasks.  For this project we only need two counters and one
histogram, so a 40-line in-house helper keeps the footprint small.
"""

from __future__ import annotations

import time
from typing import Callable

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# --------------------------------------------------------------------------- #
#  Registry & metrics
# --------------------------------------------------------------------------- #
REGISTRY = CollectorRegistry(auto_describe=True)

REQ_LATENCY = Histogram(
    "http_request_latency_seconds",
    "Request latency",
    ["method", "path"],
    registry=REGISTRY,
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10),
)
REQ_COUNT = Counter(
    "http_requests_total",
    "Request count by status",
    ["method", "path", "status"],
    registry=REGISTRY,
)


# --------------------------------------------------------------------------- #
#  ASGI middleware
# --------------------------------------------------------------------------- #
class MetricsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, group_paths: bool = True):
        """
        If *group_paths* is True, dynamic path segments like /webhook/123
        are normalised to /webhook to keep the label-cardinality small.
        """
        super().__init__(app)
        self.group_paths = group_paths

    # ----------------------------------------------------
    async def dispatch(self, request: Request, call_next: Callable):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start

        path = (
            "/" + request.url.path.lstrip("/").split("/")[0]
            if self.group_paths
            else request.url.path
        )

        REQ_LATENCY.labels(request.method, path).observe(elapsed)
        REQ_COUNT.labels(request.method, path, str(response.status_code)).inc()
        return response


# --------------------------------------------------------------------------- #
#  FastAPI util for the /metrics route
# --------------------------------------------------------------------------- #
def metrics_response() -> Response:
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)