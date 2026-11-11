"""Test doubles for the two external services."""

from __future__ import annotations

import datetime as dt
import threading
from typing import Any, Callable

from bot.llm import LLMError, LLMResult
from bot.metaculus import Question, forecast_payload

NOW = dt.datetime(2026, 11, 11, 12, 0, tzinfo=dt.timezone.utc)
RESEARCH_TEXT = "1. RESOLUTION - ...\n2. CURRENT STATE - latest value 104 on 2026-11-10 (https://example.org/data)"
FIXER_MARK = "You copy a forecaster's final answer"
ADJUDICATOR_MARK = "# Forecasts from the team"


def binary_question(**overrides: Any) -> Question:
    values: dict[str, Any] = dict(
        post_id=101,
        question_id=201,
        title="Will X happen before December 31, 2026?",
        qtype="binary",
        description="Background.",
        resolution_criteria="Resolves Yes if X happens.",
        fine_print="",
        open_time=NOW - dt.timedelta(minutes=10),
        close_time=NOW + dt.timedelta(hours=2, minutes=50),
        resolve_time=NOW + dt.timedelta(days=50),
        status="open",
    )
    values.update(overrides)
    return Question(**values)


def mc_question(**overrides: Any) -> Question:
    values: dict[str, Any] = dict(
        post_id=102, question_id=202, title="Which party wins?", qtype="multiple_choice",
        options=["Party A", "Party B", "Other"],
    )
    values.update(overrides)
    return binary_question(**values)


def numeric_question(**overrides: Any) -> Question:
    values: dict[str, Any] = dict(
        post_id=103, question_id=203, title="What will the index be on December 31, 2026?", qtype="numeric",
        unit="points", range_min=50.0, range_max=200.0, zero_point=None, open_lower=True, open_upper=True, cdf_size=201,
    )
    values.update(overrides)
    return binary_question(**values)


def discrete_question(**overrides: Any) -> Question:
    values: dict[str, Any] = dict(
        post_id=104, question_id=204, title="How many rate cuts in 2027?", qtype="discrete", unit="cuts",
        range_min=-0.5, range_max=10.5, nominal_min=0.0, nominal_max=10.0, open_lower=False, open_upper=False,
        cdf_size=12,
    )
    values.update(overrides)
    return binary_question(**values)


def percentile_answer(centre: float, spread: float) -> str:
    from statistics import NormalDist

    lines = ["Reasoning about the number."]
    for p in (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99):
        lines.append(f"Percentile {p}: {centre + spread * NormalDist().inv_cdf(p / 100):.3f}")
    return "\n".join(lines)


class FakeLLM:
    """Answers research calls with a canned brief and forecast calls via `answer`."""

    def __init__(
        self,
        answer: Callable[[str, list[dict[str, str]]], str],
        *,
        research: Callable[[str, list[dict[str, Any]]], str] | None = None,
        fixer: Callable[[list[dict[str, str]]], str] | None = None,
        remaining: float | None = 50.0,
        searches: int = 3,
    ):
        self._answer = answer
        self._research = research
        self._fixer = fixer
        self.remaining = remaining
        self.searches = searches
        self.total_cost = 0.0
        self.calls = 0
        self.log: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def chat(self, model, messages, *, effort=None, max_tokens=8000, tools=None, max_tool_steps=None,
             max_tool_cost=None, allow_truncated=False, timeout=0, retries=0):
        with self._lock:
            self.calls += 1
            self.log.append({"model": model, "effort": effort, "tools": tools, "messages": messages,
                             "max_tool_steps": max_tool_steps, "max_tool_cost": max_tool_cost})
        system = messages[0]["content"]
        if tools:
            text = self._research(model, tools) if self._research else RESEARCH_TEXT
            kind_cost = 0.02
        elif system.startswith(FIXER_MARK):
            if self._fixer is None:
                raise LLMError("fixer unavailable")
            text = self._fixer(messages)
            kind_cost = 0.001
        else:
            text = self._answer(model, messages)
            kind_cost = 0.05
        with self._lock:
            self.total_cost += kind_cost
        return LLMResult(text=text, model=model, cost=kind_cost, searches=self.searches if tools else 0)

    def budget_remaining(self):
        return self.remaining

    def forecast_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.log if not c["tools"] and not c["messages"][0]["content"].startswith(FIXER_MARK)]


class FakeClient:
    """In-memory Metaculus: posts keyed by tournament, records what the bot submits."""

    def __init__(self, posts_by_tournament: dict[str, list[dict[str, Any]]]):
        self.posts_by_tournament = posts_by_tournament
        self.forecasts: list[dict[str, Any]] = []
        self.comments: list[tuple[int, str]] = []
        self.fail_listing: dict[str, Exception] = {}
        self.fail_forecast: Exception | None = None

    def list_open_posts(self, tournament):
        if str(tournament) in self.fail_listing:
            raise self.fail_listing[str(tournament)]
        return list(self.posts_by_tournament.get(str(tournament), []))

    def get_post(self, post_id):
        for posts in self.posts_by_tournament.values():
            for post in posts:
                if post["id"] == post_id:
                    return post
        raise ValueError(f"no post {post_id}")

    def post_forecast(self, question, forecast):
        if self.fail_forecast is not None:
            raise self.fail_forecast
        self.forecasts.append(forecast_payload(question, forecast))

    def post_comment(self, post_id, text):
        self.comments.append((post_id, text))


def post_json(post_id: int, question_id: int, qtype: str = "binary", **question: Any) -> dict[str, Any]:
    """A post as the API returns it."""
    body: dict[str, Any] = {
        "id": question_id,
        "type": qtype,
        "title": f"Question {question_id}?",
        "description": "Background.",
        "resolution_criteria": "Criteria.",
        "fine_print": "",
        "status": "open",
        "open_time": (NOW - dt.timedelta(minutes=10)).isoformat(),
        "scheduled_close_time": (NOW + dt.timedelta(hours=2, minutes=50)).isoformat(),
        "scheduled_resolve_time": (NOW + dt.timedelta(days=30)).isoformat(),
        "my_forecasts": {"latest": None, "history": []},
    }
    body.update(question)
    return {"id": post_id, "title": body["title"], "question": body}
