"""Small JSON-over-HTTPS helper with retries (urllib only)."""

from __future__ import annotations

import http.client
import json
import random
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from . import config

RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529})


class HttpError(Exception):
    """A request failed for good. status is 0 for network-level failures."""

    def __init__(self, status: int, body: Any, url: str, retry_after: float | None = None):
        self.status = status
        self.body = body
        self.url = url
        self.retry_after = retry_after
        text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
        super().__init__(f"HTTP {status} for {_strip_query(url)}: {text[:500]}")


def _strip_query(url: str) -> str:
    return url.split("?", 1)[0]


def _decode(raw: bytes) -> Any:
    text = raw.decode("utf-8", "replace").strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except ValueError:
        return text


def _retry_after(headers: Any) -> float | None:
    try:
        value = headers.get("Retry-After") if headers is not None else None
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    body: Any = None,
    timeout: float = 60.0,
    retries: int = 3,
    backoff: float = 2.0,
) -> tuple[int, Any]:
    """Send a request and return (status, decoded body).

    Retries on transient statuses and network errors with exponential backoff.
    Raises HttpError when the request ultimately fails.
    """
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params, doseq=True)
    data = None
    all_headers = {"Accept": "application/json", "User-Agent": config.USER_AGENT}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        all_headers["Content-Type"] = "application/json"
    if headers:
        all_headers.update(headers)

    attempt = 0
    while True:
        wait: float | None = None
        try:
            request = urllib.request.Request(url, data=data, headers=all_headers, method=method)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, _decode(response.read())
        except urllib.error.HTTPError as error:
            payload = _decode(error.read() or b"")
            wait = _retry_after(error.headers)
            if error.code not in RETRY_STATUSES or attempt >= retries:
                raise HttpError(error.code, payload, url, retry_after=wait) from None
        except (urllib.error.URLError, http.client.HTTPException, ssl.SSLError, socket.timeout, OSError) as error:
            # http.client.HTTPException covers responses cut off mid-body (IncompleteRead)
            # and garbled status lines, which are not OSErrors.
            if attempt >= retries:
                raise HttpError(0, f"{type(error).__name__}: {error}", url) from None
        attempt += 1
        if wait is None:
            wait = backoff * (2 ** (attempt - 1))
        time.sleep(min(wait, 60.0) + random.uniform(0.0, 1.0))
