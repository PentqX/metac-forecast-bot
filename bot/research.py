"""Research: independent agents that search the web and report facts, not forecasts."""

from __future__ import annotations

import datetime as dt
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import config, prompts
from .llm import LLMError, OpenRouter, web_tools
from .metaculus import Question

_URL = re.compile(r"https?://[^\s)>\]]+\.[^\s)>\]]+")


class ResearchError(Exception):
    """No research agent produced a brief from live web access."""


@dataclass
class Research:
    briefs: list[tuple[str, str]] = field(default_factory=list)  # (source name, text)
    cost: float = 0.0
    searches: int = 0
    errors: list[str] = field(default_factory=list)


def looks_researched(text: str, searches: int) -> bool:
    """True when a brief shows signs of live web access: reported searches or cited URLs.

    A model whose search tool failed still writes fluent text from memory; that must not
    count as research.
    """
    return searches > 0 or bool(_URL.search(text))


def _run_source(
    llm: OpenRouter, question: Question, now: dt.datetime, source: config.ResearchSource
) -> tuple[str, float, int]:
    """One research agent. Tries search plus page reading first, then search alone."""
    messages = prompts.research_messages(question, now, source, tools=True)
    plans = [web_tools(source.max_searches, source.max_fetches)]
    if source.max_fetches > 0:
        plans.append(web_tools(source.max_searches, 0))
    last_error: LLMError | None = None
    for tools in plans:
        try:
            result = llm.chat(
                source.model,
                messages,
                effort=source.effort,
                max_tokens=source.max_tokens,
                tools=tools,
                max_tool_steps=source.max_searches + source.max_fetches + 2,
                max_tool_cost=config.RESEARCH_MAX_COST,
                allow_truncated=True,  # a brief that ran long is still useful
                timeout=config.RESEARCH_TIMEOUT,
                retries=1,
            )
        except LLMError as error:
            if error.out_of_credits:
                raise
            last_error = error
            continue
        if looks_researched(result.text, result.searches):
            return result.text, result.cost, result.searches
        last_error = LLMError(f"{source.model}: the brief shows no sign of live web access")
    assert last_error is not None
    raise last_error


def run_research(llm: OpenRouter, question: Question, now: dt.datetime, tier: config.Tier) -> Research:
    """Run every research source of the tier in parallel.

    Forecasting without current information scores worse than not forecasting, so this
    raises ResearchError when no source succeeds; the question is retried on a later run.
    """
    research = Research()
    out_of_credits: LLMError | None = None
    with ThreadPoolExecutor(max_workers=max(1, len(tier.research))) as pool:
        futures = [(source, pool.submit(_run_source, llm, question, now, source)) for source in tier.research]
        for source, future in futures:
            try:
                text, cost, searches = future.result()
            except LLMError as error:
                if error.out_of_credits:
                    out_of_credits = error
                research.errors.append(f"{source.name}: {error}")
                continue
            except Exception as error:  # noqa: BLE001 - one source failing oddly must not void the other
                research.errors.append(f"{source.name}: unexpected {type(error).__name__}: {error}")
                continue
            research.briefs.append((source.name, text[: config.RESEARCH_CHARS_PER_SOURCE]))
            research.cost += cost
            research.searches += searches
    if not research.briefs:
        if out_of_credits is not None:
            raise out_of_credits
        raise ResearchError("; ".join(research.errors) or "no research source configured")
    return research
