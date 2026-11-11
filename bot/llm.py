"""OpenRouter chat-completions client with cost tracking."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from . import config
from .http import HttpError, request_json

# OpenRouter holds the worst-case cost of every running request (input plus max_tokens)
# against a share of the account balance. With a small balance, too many parallel
# requests are rejected with 402 although credits remain, so calls are throttled here.
MAX_PARALLEL_CALLS = 5
IN_FLIGHT_RETRIES = 8


class LLMError(Exception):
    def __init__(self, message: str, *, status: int = 0, truncated: bool = False, out_of_credits: bool = False):
        super().__init__(message)
        self.status = status
        self.truncated = truncated
        self.out_of_credits = out_of_credits


@dataclass
class LLMResult:
    text: str
    model: str
    cost: float = 0.0
    finish_reason: str = ""
    searches: int = 0
    usage: dict[str, Any] = field(default_factory=dict)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return ""


def _error_message(payload: Any) -> str:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if error:
            return str(error)
    return str(payload)[:500]


def _limit_source(payload: Any) -> str:
    if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
        metadata = payload["error"].get("metadata")
        if isinstance(metadata, dict):
            return str(metadata.get("limit_source") or "")
    return ""


def web_tools(max_searches: int, max_fetches: int) -> list[dict[str, Any]]:
    """OpenRouter server tools: the model decides when to search and what to read."""
    tools: list[dict[str, Any]] = [
        {
            "type": "openrouter:web_search",
            "parameters": {"max_uses": max_searches, "max_total_results": max_searches * 6},
        }
    ]
    if max_fetches > 0:
        tools.append(
            {
                "type": "openrouter:web_fetch",
                "parameters": {"max_uses": max_fetches, "max_content_tokens": 6000},
            }
        )
    return tools


class OpenRouter:
    def __init__(self, api_key: str, api: str = config.OPENROUTER_API):
        self._api = api.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}", "X-Title": "metac-forecast-bot"}
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(MAX_PARALLEL_CALLS)
        self.total_cost = 0.0
        self.calls = 0

    def _record(self, cost: float) -> None:
        with self._lock:
            self.total_cost += cost
            self.calls += 1

    def _post(self, body: dict[str, Any], timeout: float, retries: int) -> Any:
        """POST a completion, waiting out transient in-flight budget rejections."""
        model = body.get("model")
        waits = 0
        while True:
            try:
                with self._slots:
                    _, data = request_json(
                        "POST",
                        f"{self._api}/chat/completions",
                        headers=self._headers,
                        body=body,
                        timeout=timeout,
                        retries=retries,
                        backoff=5.0,
                    )
                return data
            except HttpError as error:
                if error.status != 402:
                    raise LLMError(f"{model}: {_error_message(error.body)}", status=error.status) from None
                if _limit_source(error.body) == "openrouter_in_flight_budget" and waits < IN_FLIGHT_RETRIES:
                    waits += 1
                    time.sleep(min(max(error.retry_after or 10.0, 2.0), 45.0))
                    continue
                raise LLMError(
                    f"{model}: {_error_message(error.body)}", status=402, out_of_credits=True
                ) from None

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        effort: str | None = None,
        max_tokens: int = 8000,
        tools: list[dict[str, Any]] | None = None,
        max_tool_steps: int | None = None,
        max_tool_cost: float | None = None,
        allow_truncated: bool = False,
        timeout: float = 420.0,
        retries: int = 2,
    ) -> LLMResult:
        """One chat completion.

        tools are OpenRouter server tools; max_tool_steps and max_tool_cost bound the
        server-side tool loop. An answer cut off by max_tokens raises LLMError(truncated)
        unless allow_truncated is set, because a forecast's final answer is at the end.
        """
        body: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if effort:
            body["reasoning"] = {"effort": effort}
        if tools:
            body["tools"] = tools
            stop: list[dict[str, Any]] = []
            if max_tool_steps:
                stop.append({"type": "step_count_is", "step_count": max_tool_steps})
            if max_tool_cost:
                stop.append({"type": "max_cost", "max_cost_in_dollars": max_tool_cost})
            if stop:
                body["stop_server_tools_when"] = stop
        data = self._post(body, timeout, retries)

        if not isinstance(data, dict):
            raise LLMError(f"{model}: unexpected response")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        try:
            cost = float(usage.get("cost") or 0.0)
        except (TypeError, ValueError):
            cost = 0.0
        self._record(cost)

        choices = data.get("choices")
        if not choices:
            raise LLMError(f"{model}: {_error_message(data)}")
        choice = choices[0] if isinstance(choices[0], dict) else {}
        if choice.get("error"):
            raise LLMError(f"{model}: {_error_message(choice)}")
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        text = _content_text(message.get("content")).strip()
        finish = str(choice.get("finish_reason") or "")
        if not text:
            raise LLMError(f"{model}: empty answer (finish_reason={finish})", truncated=finish == "length")
        if finish == "length" and not allow_truncated:
            raise LLMError(f"{model}: answer cut off by the output limit", truncated=True)
        if finish == "error":
            raise LLMError(f"{model}: generation ended with an error")

        searches = 0
        for key in ("server_tool_use_details", "server_tool_use"):  # the docs use both names
            details = usage.get(key)
            if isinstance(details, dict):
                try:
                    count = int(details.get("web_search_requests") or details.get("tool_calls_executed") or 0)
                except (TypeError, ValueError):
                    count = 0
                searches = max(searches, count)
        return LLMResult(text=text, model=model, cost=cost, finish_reason=finish, searches=searches, usage=usage)

    def _get_data(self, path: str) -> dict[str, Any] | None:
        try:
            _, data = request_json("GET", f"{self._api}/{path}", headers=self._headers, timeout=30, retries=2)
        except HttpError:
            return None
        info = data.get("data") if isinstance(data, dict) else None
        return info if isinstance(info, dict) else None

    def key_limit_remaining(self) -> float | None:
        """Remaining spend allowed for this key in USD, or None when unknown/unlimited."""
        info = self._get_data("key")
        if not info:
            return None
        try:
            return None if info.get("limit_remaining") is None else float(info["limit_remaining"])
        except (TypeError, ValueError):
            return None

    def account_balance(self) -> float | None:
        """Account balance (credits bought minus credits used), or None when unavailable."""
        info = self._get_data("credits")
        if not info:
            return None
        try:
            return float(info["total_credits"]) - float(info["total_usage"])
        except (KeyError, TypeError, ValueError):
            return None

    def budget_remaining(self) -> float | None:
        """The tighter of the key limit and the account balance; None when neither is known."""
        known = [value for value in (self.key_limit_remaining(), self.account_balance()) if value is not None]
        return min(known) if known else None
