"""One question in, one forecast out: research -> ensemble -> aggregation -> checks."""

from __future__ import annotations

import datetime as dt
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from . import aggregate, config, numeric, parsing, prompts
from .llm import LLMError, LLMResult, OpenRouter
from .metaculus import Question
from .research import Research, run_research


class ForecastError(Exception):
    """The question could not be forecast reliably; nothing must be submitted."""


@dataclass
class MemberResult:
    model: str
    weight: float
    value: Any  # binary: float; multiple choice: {option: p}; numeric: {percentile: value}
    rationale: str
    cdf: list[float] | None = None  # numeric only


@dataclass
class ForecastResult:
    question: Question
    forecast: Any  # float | {option: p} | list[float] (cdf)
    summary: str
    comment: str
    tier: str
    cost: float
    members: list[MemberResult] = field(default_factory=list)
    adjudicator: MemberResult | None = None
    notes: list[str] = field(default_factory=list)


class _Spend:
    """Cost of the calls made for one question (calls run on several threads)."""

    def __init__(self) -> None:
        self.total = 0.0
        self._lock = threading.Lock()

    def add(self, result: LLMResult) -> LLMResult:
        with self._lock:
            self.total += result.cost
        return result


# ---------------------------------------------------------------------------
# Asking one forecaster
# ---------------------------------------------------------------------------
_LOWER_EFFORT = {"high": "medium", "medium": "low"}


def _ask(llm: OpenRouter, member: config.Member, messages: prompts.Messages, spend: _Spend) -> LLMResult:
    try:
        return spend.add(
            llm.chat(
                member.model,
                messages,
                effort=member.effort,
                max_tokens=member.max_tokens,
                timeout=config.FORECAST_TIMEOUT,
                retries=1,
            )
        )
    except LLMError as error:
        # The thinking used up the whole output budget: retry once with less thinking.
        if not error.truncated or member.effort not in _LOWER_EFFORT:
            raise
    return spend.add(
        llm.chat(
            member.model,
            messages,
            effort=_LOWER_EFFORT[member.effort],
            max_tokens=member.max_tokens,
            timeout=config.FORECAST_TIMEOUT,
            retries=1,
        )
    )


def _coerce_percentiles(raw: Any) -> dict[float, float] | None:
    if not isinstance(raw, dict):
        return None
    result: dict[float, float] = {}
    for key, value in raw.items():
        try:
            percentile, number = float(key), float(value)
        except (TypeError, ValueError):
            return None  # an entry that cannot be read makes the whole list suspect
        if isinstance(value, bool) or not 0.0 < percentile < 100.0:
            return None
        result[percentile] = number
    # Keys written as fractions (0.05 ... 0.95) would otherwise be read as tiny percentiles.
    if result and max(result) <= 1.0:
        result = {percentile * 100.0: number for percentile, number in result.items()}
    return result if len(result) >= 5 else None


def _coerce_options(raw: Any, question: Question) -> dict[str, float] | None:
    if not isinstance(raw, dict):
        return None
    lookup = {str(key).strip().lower(): value for key, value in raw.items()}
    values: dict[str, float] = {}
    for option in question.options:
        try:
            values[option] = float(lookup[option.strip().lower()])
        except (KeyError, TypeError, ValueError):
            return None
    total = sum(values.values())
    if min(values.values()) < 0.0 or not (0.97 <= total <= 1.03 or 97.0 <= total <= 103.0):
        return None
    return {option: value / total for option, value in values.items()}


def _parse_answer(question: Question, text: str) -> Any:
    if question.qtype == "binary":
        return parsing.parse_probability(text)
    if question.qtype == "multiple_choice":
        return parsing.parse_options(text, question.options)
    return parsing.parse_percentiles(text)


def _fix_answer(llm: OpenRouter, question: Question, text: str, spend: _Spend) -> Any:
    """Second chance for a rationale whose final lines were not in the requested format."""
    fixer = config.FORMAT_FIXER
    try:
        result = spend.add(
            llm.chat(
                fixer.model,
                prompts.fixer_messages(question, text),
                effort=fixer.effort,
                max_tokens=fixer.max_tokens,
                timeout=120.0,
                retries=1,
            )
        )
    except LLMError as error:
        if error.out_of_credits:
            raise
        return None
    data = parsing.extract_json(result.text)
    if not isinstance(data, dict) or data.get("missing"):
        return None
    if question.qtype == "binary":
        try:
            percent = float(data.get("probability_yes_percent"))
        except (TypeError, ValueError):
            return None
        return percent / 100.0 if 0.0 <= percent <= 100.0 else None
    if question.qtype == "multiple_choice":
        return _coerce_options(data.get("options"), question)
    return _coerce_percentiles(data.get("percentiles"))


def _scale(question: Question) -> numeric.Scale:
    if question.range_min is None or question.range_max is None:
        raise ForecastError("numeric question without a range")
    try:
        return numeric.Scale(question.range_min, question.range_max, question.zero_point)
    except ValueError as error:
        raise ForecastError(str(error)) from None


def _member_from(
    llm: OpenRouter,
    question: Question,
    model: str,
    weight: float,
    result: LLMResult,
    spend: _Spend,
) -> MemberResult:
    value = _parse_answer(question, result.text)
    if value is None:
        value = _fix_answer(llm, question, result.text, spend)
    if value is None:
        raise ForecastError(f"{model}: no final answer in the expected format")
    cdf = None
    if question.qtype in ("numeric", "discrete"):
        scale = _scale(question)
        try:
            value = numeric.clean_percentiles(value, scale)
            cdf = numeric.member_cdf(
                value,
                scale,
                question.cdf_size,
                discrete=question.is_discrete,
                open_lower=question.open_lower,
                open_upper=question.open_upper,
                widen_weight=config.NUMERIC_WIDEN_WEIGHT,
                widen_factor=config.NUMERIC_WIDEN_FACTOR,
            )
        except ValueError as error:
            raise ForecastError(f"{model}: {error}") from None
    return MemberResult(model=model, weight=weight, value=value, rationale=result.text, cdf=cdf)


def _run_member(
    llm: OpenRouter, question: Question, member: config.Member, messages: prompts.Messages, spend: _Spend
) -> MemberResult:
    result = _ask(llm, member, messages, spend)
    return _member_from(llm, question, member.model, member.weight, result, spend)


# ---------------------------------------------------------------------------
# Describing forecasts
# ---------------------------------------------------------------------------
def _pct(p: float) -> str:
    return f"{100.0 * p:.1f}%"


def describe(question: Question, value: Any, cdf: list[float] | None = None) -> str:
    if question.qtype == "binary":
        return _pct(float(value))
    if question.qtype == "multiple_choice":
        ranked = sorted(value.items(), key=lambda item: -item[1])
        return ", ".join(f"{name} {_pct(p)}" for name, p in ranked)
    if cdf is None:
        return "n/a"
    low, median, high = numeric.quantiles(cdf, _scale(question), [0.05, 0.5, 0.95])
    unit = f" {question.unit}" if question.unit else ""
    return (
        f"median {prompts.fmt_number(_round_sig(median))}{unit} "
        f"(5%: {prompts.fmt_number(_round_sig(low))}, 95%: {prompts.fmt_number(_round_sig(high))})"
    )


def _round_sig(value: float, digits: int = 4) -> float:
    if value == 0.0 or not math.isfinite(value):
        return value
    return round(value, digits - 1 - int(math.floor(math.log10(abs(value)))))


def _trim(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 20].rstrip() + "\n[... trimmed]"


def _comment(
    question: Question,
    summary: str,
    tier: config.Tier,
    planned_runs: int,
    members: list[MemberResult],
    adjudicator: MemberResult | None,
    research: Research,
) -> str:
    lines = [
        f"**Forecast: {summary}**",
        "",
        f"*Automated forecast, no human in the loop. {len(members)} of {planned_runs} model runs used "
        f"(tier: {tier.name}); {len(research.briefs)} research source(s).*",
        "",
        "**Ensemble**",
    ]
    for member in members:
        lines.append(f"- {member.model}: {describe(question, member.value, member.cdf)}")
    lead = max(members, key=lambda member: member.weight)
    if adjudicator is not None:
        lines.append(
            f"- adjudicator {adjudicator.model} (called because the members disagreed): "
            f"{describe(question, adjudicator.value, adjudicator.cdf)}"
        )
        lead = adjudicator
    lines += ["", f"**Reasoning ({lead.model})**", _trim(lead.rationale, 3800)]
    if research.briefs:
        lines += ["", "**Research notes (excerpt)**", _trim(research.briefs[0][1], 2600)]
    return _trim("\n".join(lines), config.COMMENT_MAX_CHARS)


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
def _disagree(question: Question, members: list[MemberResult]) -> bool:
    if len(members) < 2:
        return False
    if question.qtype == "binary":
        logit_range, prob_range = aggregate.binary_spread([m.value for m in members])
        return logit_range > config.DISAGREE_LOGIT_RANGE or prob_range > config.DISAGREE_PROB_RANGE
    if question.qtype == "multiple_choice":
        return aggregate.options_spread([m.value for m in members], question.options) > config.DISAGREE_PROB_RANGE
    medians = [numeric.loc_at(m.cdf, 0.5) for m in members if m.cdf]
    return bool(medians) and max(medians) - min(medians) > config.DISAGREE_MEDIAN_RANGE


def _adjudicate(
    llm: OpenRouter,
    question: Question,
    now: dt.datetime,
    research: Research,
    members: list[MemberResult],
    adjudicator: config.Member,
    spend: _Spend,
) -> MemberResult | None:
    views = [(describe(question, m.value, m.cdf), _trim(m.rationale, 2200)) for m in members]
    messages = prompts.adjudicator_messages(question, now, research.briefs, views)
    try:
        result = _ask(llm, adjudicator, messages, spend)
        member = _member_from(llm, question, adjudicator.model, adjudicator.weight, result, spend)
    except Exception:  # noqa: BLE001 - the ensemble is already paid for; never lose it to the adjudicator
        return None
    if question.qtype == "binary":
        member.value = aggregate.clamp(float(member.value), 0.005, 0.995)
    return member


def _combine(
    question: Question, tier: config.Tier, members: list[MemberResult], adjudicator: MemberResult | None
) -> tuple[Any, list[float] | None]:
    """Final forecast in API form, plus the final CDF for numeric questions."""
    weights = [m.weight for m in members]
    share = config.ADJUDICATOR_BLEND
    if question.qtype == "binary":
        p = aggregate.aggregate_binary([m.value for m in members], weights, scale=config.BINARY_LOGIT_SCALE)
        if adjudicator is not None:
            p = aggregate.blend_binary(p, adjudicator.value, share)
        return aggregate.clamp(p, tier.prob_floor, tier.prob_ceil), None
    if question.qtype == "multiple_choice":
        pooled = aggregate.aggregate_options([m.value for m in members], weights, question.options)
        if adjudicator is not None:
            pooled = aggregate.blend_options(pooled, adjudicator.value, share, question.options)
        return aggregate.floor_and_normalize(pooled, config.MC_FLOOR), None
    mixed = numeric.mixture([m.cdf for m in members], weights)
    if adjudicator is not None and adjudicator.cdf is not None:
        mixed = numeric.mixture([mixed, adjudicator.cdf], [1.0 - share, share])
    try:
        final = numeric.standardize(mixed, question.open_lower, question.open_upper)
        numeric.validate(final, question.open_lower, question.open_upper, question.cdf_size)
    except ValueError as error:
        raise ForecastError(f"invalid distribution: {error}") from None
    return final, final


def forecast_question(llm: OpenRouter, question: Question, tier: config.Tier, now: dt.datetime) -> ForecastResult:
    """Forecast one question. Raises ForecastError / ResearchError / LLMError when it cannot."""
    if question.qtype == "multiple_choice" and len(question.options) < 2:
        raise ForecastError("multiple-choice question without options")
    if question.qtype in ("numeric", "discrete"):
        _scale(question)

    spend = _Spend()
    research = run_research(llm, question, now, tier)
    spend.total += research.cost
    messages = prompts.forecast_messages(question, now, research.briefs)

    planned = [member for member in tier.members for _ in range(member.runs)]
    members: list[MemberResult] = []
    notes: list[str] = list(research.errors)
    out_of_credits: LLMError | None = None
    with ThreadPoolExecutor(max_workers=len(planned)) as pool:
        futures = [(member, pool.submit(_run_member, llm, question, member, messages, spend)) for member in planned]
        for member, future in futures:
            try:
                members.append(future.result())
            except LLMError as error:
                if error.out_of_credits:
                    out_of_credits = error
                notes.append(str(error))
            except ForecastError as error:
                notes.append(str(error))
            except Exception as error:  # noqa: BLE001 - one member failing oddly must not void the others
                notes.append(f"{member.model}: unexpected {type(error).__name__}: {error}")
    if out_of_credits is not None and not members:
        raise out_of_credits

    planned_weight = sum(member.weight for member in planned)
    if not members or sum(m.weight for m in members) < config.MIN_WEIGHT_SHARE * planned_weight:
        raise ForecastError("too few ensemble members succeeded: " + "; ".join(notes))

    adjudicator: MemberResult | None = None
    if tier.adjudicator is not None and _disagree(question, members):
        adjudicator = _adjudicate(llm, question, now, research, members, tier.adjudicator, spend)
        if adjudicator is None:
            notes.append("adjudicator failed; ensemble used as is")

    forecast, final_cdf = _combine(question, tier, members, adjudicator)
    summary = describe(question, forecast, final_cdf)
    comment = _comment(question, summary, tier, len(planned), members, adjudicator, research)
    return ForecastResult(
        question=question,
        forecast=forecast,
        summary=summary,
        comment=comment,
        tier=tier.name,
        cost=spend.total,
        members=members,
        adjudicator=adjudicator,
        notes=notes,
    )
