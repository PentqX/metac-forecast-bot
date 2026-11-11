"""Bot configuration.

Everything a maintenance run is expected to tune lives here: the model line-up,
spending tiers, probability caps and tournament ids. Nothing in this file is secret.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

BOT_VERSION = "1.0.0"
USER_AGENT = f"Mozilla/5.0 (compatible; metac-forecast-bot/{BOT_VERSION})"

METACULUS_API = "https://www.metaculus.com/api"
OPENROUTER_API = "https://openrouter.ai/api/v1"

MINIBENCH_ID = "minibench"
TEST_AREA_ID = "bot-testing-area"

# Set to a slug or numeric id to pin the seasonal tournament instead of deriving it
# from the calendar (see seasonal_tournament_ids).
SEASON_OVERRIDE: str | None = None

# ---------------------------------------------------------------------------
# Models (OpenRouter ids)
# ---------------------------------------------------------------------------
OPUS = "anthropic/claude-opus-5.5"
FABLE = "anthropic/claude-fable-5.2"
SONNET = "anthropic/claude-sonnet-5.5"
HAIKU = "anthropic/claude-haiku-5.5"
GPT = "openai/gpt-5.6-sol"
GEMINI_FLASH = "google/gemini-3.8-flash"


@dataclass(frozen=True)
class Member:
    """One forecaster in the ensemble."""

    model: str
    effort: str = "high"  # low | medium | high (supported by every model we use)
    weight: float = 1.0  # weight of each run of this member
    runs: int = 1
    max_tokens: int = 14000  # thinking and answer share this budget


@dataclass(frozen=True)
class ResearchSource:
    """One research agent. Different sources use different search indexes on purpose."""

    name: str
    model: str
    effort: str = "low"
    max_searches: int = 5
    max_fetches: int = 3
    max_tokens: int = 7000


@dataclass(frozen=True)
class Tier:
    name: str
    research: tuple[ResearchSource, ...]
    members: tuple[Member, ...]
    adjudicator: Member | None
    prob_floor: float
    prob_ceil: float
    # Rough expected cost per question in USD; used only for budget pacing.
    est_cost: float


_RESEARCH_ANTHROPIC = ResearchSource(
    name="web-a", model=HAIKU, effort="low", max_searches=6, max_fetches=3
)
_RESEARCH_GOOGLE = ResearchSource(
    name="web-g", model=GEMINI_FLASH, effort="low", max_searches=5, max_fetches=2
)

TIERS: dict[str, Tier] = {
    # Seasonal tournament questions: ~8x the prize money per question of MiniBench.
    "full": Tier(
        name="full",
        research=(_RESEARCH_ANTHROPIC, _RESEARCH_GOOGLE),
        members=(
            Member(OPUS, effort="high", weight=1.5, runs=2),
            Member(GPT, effort="high", weight=1.0, runs=1),
            Member(GEMINI_FLASH, effort="high", weight=0.5, runs=2, max_tokens=12000),
        ),
        adjudicator=Member(FABLE, effort="medium", weight=1.0, max_tokens=12000),
        prob_floor=0.02,
        prob_ceil=0.98,
        est_cost=0.65,
    ),
    # MiniBench questions and the bot testing area.
    "lite": Tier(
        name="lite",
        research=(ResearchSource("web-a", HAIKU, "low", max_searches=4, max_fetches=2),),
        members=(
            Member(OPUS, effort="medium", weight=1.5, runs=1, max_tokens=10000),
            Member(GEMINI_FLASH, effort="high", weight=0.6, runs=2, max_tokens=10000),
        ),
        adjudicator=None,
        prob_floor=0.03,
        prob_ceil=0.97,
        est_cost=0.2,
    ),
    # Used when credits are nearly exhausted: still forecasts, at minimal cost.
    "economy": Tier(
        name="economy",
        research=(ResearchSource("web-a", HAIKU, "low", max_searches=3, max_fetches=1),),
        members=(Member(GEMINI_FLASH, effort="high", weight=1.0, runs=3, max_tokens=10000),),
        adjudicator=None,
        prob_floor=0.04,
        prob_ceil=0.96,
        est_cost=0.1,
    ),
}

# Base tier per tournament kind.
SEASONAL_TIER = "full"
MINIBENCH_TIER = "lite"
TEST_TIER = "lite"

# Budget pacing on remaining OpenRouter credits (USD). Below MIN_CREDITS the run stops
# instead of failing half-way through questions.
CREDITS_FOR_FULL = 20.0
CREDITS_FOR_LITE = 6.0
MIN_CREDITS = 0.6
# Hard ceiling of spend for a single run, whatever the number of open questions.
MAX_COST_PER_RUN = 12.0

# Cheap model used to recover a final answer from a rationale whose last lines did
# not follow the requested format.
FORMAT_FIXER = Member(HAIKU, effort="low", max_tokens=1500)

# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
# Multiplier on the aggregated log-odds (1.0 = no recalibration). To be tuned from
# resolved questions once there are enough of them.
BINARY_LOGIT_SCALE = 1.0
# The adjudicator runs when members disagree by more than this.
DISAGREE_LOGIT_RANGE = 1.6
DISAGREE_PROB_RANGE = 0.30
DISAGREE_MEDIAN_RANGE = 0.20  # numeric: spread of member medians, as a share of the range
ADJUDICATOR_BLEND = 0.4
# Minimum share of the configured ensemble weight that must succeed to submit.
MIN_WEIGHT_SHARE = 0.4
MC_FLOOR = 0.01

# Numeric questions: language models tend to state intervals that are too narrow. Each
# member distribution is mixed with a wider copy of itself. To be tuned from resolved
# questions: lower the weight if stated intervals turn out to be well calibrated.
NUMERIC_WIDEN_WEIGHT = 0.15
NUMERIC_WIDEN_FACTOR = 1.8
PERCENTILES = (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99)

# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------
MAX_PARALLEL_POSTS = 3
RESEARCH_TIMEOUT = 360.0
# Spend in USD at which OpenRouter stops the tool loop of one research call. It bounds
# OpenRouter's own loop; a provider's native search is bounded by max_uses where supported.
RESEARCH_MAX_COST = 0.5
FORECAST_TIMEOUT = 480.0
RESEARCH_CHARS_PER_SOURCE = 7000
COMMENT_MAX_CHARS = 9000
# Questions are open for about three hours and one question takes roughly 3-8 minutes.
# Never start one this close to its close: the answer would arrive too late and be paid for.
MIN_MINUTES_BEFORE_CLOSE = 12.0
# A question that keeps failing must not be paid for on every 20-minute run. The bot keeps
# no state between runs, so attempts are limited by the clock instead: a question is tried
# only while the time left until it closes falls in one of these 20-minute slots. For the
# usual three-hour window that is the first 40 minutes, minutes 80-100 and the last 40
# minutes: at most five attempts instead of nine. With more than three hours left, a
# question is tried once an hour.
ATTEMPT_SLOT_MINUTES = 20.0
ATTEMPT_SLOTS_BEFORE_CLOSE = frozenset({8, 7, 4, 1, 0})


def seasonal_tournament_ids(today: _dt.date) -> list[str]:
    """Slugs of the seasonal bot tournament(s) that may have open questions today.

    Seasons start in January, May and September. Around a season change the previous
    tournament can still have a few open questions, so both are returned then.
    """
    if SEASON_OVERRIDE:
        return [SEASON_OVERRIDE]
    year, month = today.year, today.month
    if month >= 9:
        current, previous = f"fall-futureeval-{year}", f"summer-futureeval-{year}"
    elif month >= 5:
        current, previous = f"summer-futureeval-{year}", f"spring-futureeval-{year}"
    else:
        current, previous = f"spring-futureeval-{year}", f"fall-futureeval-{year - 1}"
    ids = [current]
    if month in (1, 5, 9):
        ids.append(previous)
    return ids
