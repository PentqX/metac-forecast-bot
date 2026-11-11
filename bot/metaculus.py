"""Metaculus API client and question model."""

from __future__ import annotations

import datetime as dt
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from . import config
from .http import request_json

SUPPORTED_TYPES = ("binary", "multiple_choice", "numeric", "discrete")
# Metaculus asks API clients to pace themselves; its own client waits this long between requests.
MIN_SECONDS_BETWEEN_REQUESTS = 3.5


def parse_time(value: Any) -> dt.datetime | None:
    """Parse an ISO timestamp from the API into an aware UTC datetime."""
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


@dataclass
class Question:
    post_id: int
    question_id: int
    title: str
    qtype: str
    description: str = ""
    resolution_criteria: str = ""
    fine_print: str = ""
    options: list[str] = field(default_factory=list)
    unit: str = ""
    range_min: float | None = None
    range_max: float | None = None
    zero_point: float | None = None
    nominal_min: float | None = None
    nominal_max: float | None = None
    open_lower: bool = True
    open_upper: bool = True
    cdf_size: int = 201
    open_time: dt.datetime | None = None
    close_time: dt.datetime | None = None
    resolve_time: dt.datetime | None = None
    status: str = ""
    already_forecast: bool = False
    group_label: str = ""

    @property
    def url(self) -> str:
        return f"https://www.metaculus.com/questions/{self.post_id}/"

    @property
    def is_discrete(self) -> bool:
        return self.qtype == "discrete"

    @property
    def outcome_count(self) -> int:
        """Number of in-range outcomes of a discrete question."""
        return self.cdf_size - 1

    @property
    def step(self) -> float | None:
        """Distance between neighbouring outcomes of a discrete question."""
        if not self.is_discrete or self.range_min is None or self.range_max is None:
            return None
        return (self.range_max - self.range_min) / self.outcome_count

    def discrete_values(self) -> list[float]:
        """The possible in-range outcomes of a discrete question (bucket centres)."""
        step = self.step
        if step is None or self.range_min is None:
            return []
        first = self.nominal_min if self.nominal_min is not None else self.range_min + step / 2.0
        return [_clean(first + step * i) for i in range(self.outcome_count)]


def _clean(value: float) -> float:
    rounded = round(value, 9)
    return float(int(rounded)) if rounded == int(rounded) else rounded


def _as_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _has_forecast(question: dict[str, Any]) -> bool:
    mine = question.get("my_forecasts")
    if not isinstance(mine, dict):
        return False
    latest = mine.get("latest")
    if isinstance(latest, dict) and latest.get("forecast_values") is not None:
        return True
    history = mine.get("history")
    return isinstance(history, list) and len(history) > 0


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _question_from(post: dict[str, Any], raw: dict[str, Any], group: dict[str, Any] | None) -> Question | None:
    qtype = raw.get("type")
    if qtype not in SUPPORTED_TYPES or raw.get("id") is None or post.get("id") is None:
        return None
    group = group or {}
    scaling = raw.get("scaling") if isinstance(raw.get("scaling"), dict) else {}
    label = _first_text(raw.get("label"))
    post_title = _first_text(post.get("title"), raw.get("title"))
    title = _first_text(raw.get("title"), post_title)
    if group and label and label.lower() not in title.lower():
        title = f"{post_title} — {label}"

    cdf_size = 201
    if qtype in ("numeric", "discrete"):
        count = scaling.get("inbound_outcome_count")
        if count is None and qtype == "numeric":
            count = 200
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            return None
        cdf_size = count + 1

    def is_open(name: str) -> bool:
        value = raw.get(name)
        if value is None:
            value = scaling.get(name)
        # Like the official client, assume an open bound when the API does not say.
        return True if value is None else bool(value)

    options = raw.get("options")
    return Question(
        post_id=int(post["id"]),
        question_id=int(raw["id"]),
        title=title,
        qtype=qtype,
        description=_first_text(raw.get("description"), group.get("description"), post.get("description")),
        resolution_criteria=_first_text(raw.get("resolution_criteria"), group.get("resolution_criteria")),
        fine_print=_first_text(raw.get("fine_print"), group.get("fine_print")),
        options=[str(option) for option in options] if isinstance(options, list) else [],
        unit=_first_text(raw.get("unit")),
        range_min=_as_float(scaling.get("range_min")),
        range_max=_as_float(scaling.get("range_max")),
        zero_point=_as_float(scaling.get("zero_point")),
        nominal_min=_as_float(scaling.get("nominal_min")),
        nominal_max=_as_float(scaling.get("nominal_max")),
        open_lower=is_open("open_lower_bound"),
        open_upper=is_open("open_upper_bound"),
        cdf_size=cdf_size,
        open_time=parse_time(raw.get("open_time")),
        close_time=parse_time(raw.get("scheduled_close_time")),
        resolve_time=parse_time(raw.get("scheduled_resolve_time")),
        status=str(raw.get("status") or ""),
        already_forecast=_has_forecast(raw),
        group_label=label,
    )


def questions_from_post(post: dict[str, Any]) -> list[Question]:
    """All supported questions contained in a post (single question or a group)."""
    found: list[Question] = []
    single = post.get("question")
    if isinstance(single, dict):
        question = _question_from(post, single, None)
        if question:
            found.append(question)
    group = post.get("group_of_questions")
    if isinstance(group, dict):
        for raw in group.get("questions") or []:
            if isinstance(raw, dict):
                question = _question_from(post, raw, group)
                if question:
                    found.append(question)
    return found


def forecast_payload(question: Question, forecast: Any) -> dict[str, Any]:
    """One entry of the body of POST /questions/forecast/."""
    payload: dict[str, Any] = {"question": question.question_id, "source": "api"}
    if question.qtype == "binary":
        payload["probability_yes"] = float(forecast)
    elif question.qtype == "multiple_choice":
        payload["probability_yes_per_category"] = dict(forecast)
    else:
        payload["continuous_cdf"] = list(forecast)
    return payload


class MetaculusClient:
    def __init__(self, token: str, api: str = config.METACULUS_API, min_interval: float = MIN_SECONDS_BETWEEN_REQUESTS):
        self._api = api.rstrip("/")
        self._headers = {"Authorization": f"Token {token}", "Accept-Language": "en"}
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._last_request = 0.0

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        with self._lock:  # one request at a time, spaced out
            wait = self._last_request + self._min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait + random.uniform(0.0, 0.5))
            try:
                _, data = request_json(method, f"{self._api}/{path}", headers=self._headers, **kwargs)
            finally:
                self._last_request = time.monotonic()
        return data

    def me(self) -> dict[str, Any]:
        data = self._request("GET", "users/me/")
        return data if isinstance(data, dict) else {}

    def list_open_posts(self, tournament: str | int) -> list[dict[str, Any]]:
        posts: list[dict[str, Any]] = []
        offset = 0
        while offset < 1000:
            data = self._request(
                "GET",
                "posts/",
                params={
                    "limit": 100,
                    "offset": offset,
                    "order_by": "-published_time",
                    "forecast_type": list(SUPPORTED_TYPES) + ["group_of_questions"],
                    "tournaments": [tournament],
                    "statuses": ["open"],
                    "include_description": "true",
                },
            )
            if not isinstance(data, dict):
                raise ValueError(f"unexpected response listing posts of {tournament}")
            results = data.get("results") or []
            posts.extend(post for post in results if isinstance(post, dict))
            if not results or not data.get("next"):
                break
            offset += len(results)
        return posts

    def get_post(self, post_id: int) -> dict[str, Any]:
        data = self._request("GET", f"posts/{post_id}/")
        if not isinstance(data, dict):
            raise ValueError(f"unexpected response for post {post_id}")
        return data

    def post_forecast(self, question: Question, forecast: Any) -> None:
        self._request("POST", "questions/forecast/", body=[forecast_payload(question, forecast)])

    def post_comment(self, post_id: int, text: str) -> None:
        self._request(
            "POST",
            "comments/create/",
            body={"on_post": post_id, "text": text, "is_private": True, "included_forecast": True},
        )
