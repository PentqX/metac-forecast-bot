"""Run modes: live tournaments, the bot testing area, a single post, and the connection self-test."""

from __future__ import annotations

import datetime as dt
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from . import config, selfcheck
from .http import HttpError
from .llm import LLMError, OpenRouter, web_tools
from .metaculus import MetaculusClient, Question, questions_from_post
from .pipeline import ForecastError, ForecastResult, forecast_question
from .research import ResearchError

IN_CI = os.getenv("GITHUB_ACTIONS") == "true"
_PLACEHOLDERS = {"", "REPLACE_ME", "1234567890", "your-token-here", "your-api-key-here"}
_TIER_RANK = {"full": 0, "lite": 1, "economy": 2}


def log(message: str) -> None:
    print(message, flush=True)


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def load_dotenv(path: str = ".env") -> None:
    """Read KEY=VALUE lines from a local .env file (never present in CI)."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


def secret(name: str) -> str | None:
    value = (os.getenv(name) or "").strip()
    return None if value in _PLACEHOLDERS else value


@dataclass
class Target:
    question: Question
    tier: str
    tournament: str


@dataclass
class Outcome:
    target: Target
    status: str  # submitted | dry | failed | skipped
    result: ForecastResult | None = None
    error: str = ""
    rejected: bool = False  # Metaculus refused the forecast (a bug to look at, not a hiccup)


def attempt_allowed(question: Question, now: dt.datetime) -> bool:
    """Whether a tournament question may be tried now (see config.ATTEMPT_SLOTS_BEFORE_CLOSE)."""
    if question.close_time is None:
        return True
    minutes_left = (question.close_time - now).total_seconds() / 60.0
    slot = int(minutes_left // config.ATTEMPT_SLOT_MINUTES)
    if slot > max(config.ATTEMPT_SLOTS_BEFORE_CLOSE):
        return slot % 3 == 0
    return slot in config.ATTEMPT_SLOTS_BEFORE_CLOSE


def pick_tier(base: str, remaining: float | None) -> str | None:
    """Downgrade the tier as credits run low; None means stop spending."""
    if remaining is None:
        return base
    if remaining < config.MIN_CREDITS:
        return None
    if base == "full" and remaining < config.CREDITS_FOR_FULL:
        base = "lite"
    if base == "lite" and remaining < config.CREDITS_FOR_LITE:
        base = "economy"
    return base


def gather_targets(
    client: MetaculusClient,
    plan: list[tuple[str, str]],
    now: dt.datetime,
    *,
    reforecast: bool,
    limit_attempts: bool = False,
) -> tuple[list[Target], dict[str, int]]:
    """Open questions the bot has not forecast yet.

    limit_attempts applies the age-based attempt slots (tournament runs only; test
    questions stay open for months). Returns (targets, {tournament: HTTP status} for
    the tournaments that failed to load).
    """
    targets: list[Target] = []
    seen: set[int] = set()
    failed: dict[str, int] = {}
    for tournament, tier in plan:
        try:
            posts = client.list_open_posts(tournament)
        except (HttpError, ValueError) as error:
            failed[str(tournament)] = error.status if isinstance(error, HttpError) else -1
            log(f"! could not list '{tournament}': {error}")
            continue
        found = 0
        for post in posts:
            try:
                questions = questions_from_post(post)
            except Exception as error:  # noqa: BLE001 - skip a post the parser does not understand
                log(f"! could not read post {post.get('id')}: {type(error).__name__}: {error}")
                continue
            for question in questions:
                if question.question_id in seen:
                    continue
                seen.add(question.question_id)
                if question.status and question.status != "open":
                    continue
                if question.already_forecast and not reforecast:
                    continue
                targets.append(Target(question, tier, str(tournament)))
                found += 1
        log(f"- {tournament}: {len(posts)} open post(s), {found} to consider")

    confirmed: list[Target] = []
    for target in targets:
        # The list view may omit our own forecast; the post view is authoritative.
        try:
            detail = client.get_post(target.question.post_id)
        except (HttpError, ValueError) as error:
            log(f"! could not load post {target.question.post_id}, skipping this run: {error}")
            continue
        try:
            fresh = next(
                (q for q in questions_from_post(detail) if q.question_id == target.question.question_id), None
            )
        except Exception as error:  # noqa: BLE001
            log(f"! could not read post {target.question.post_id}: {type(error).__name__}: {error}")
            continue
        if fresh is None or (fresh.status and fresh.status != "open"):
            continue
        if fresh.already_forecast and not reforecast:
            continue
        if fresh.close_time and (fresh.close_time - now).total_seconds() < config.MIN_MINUTES_BEFORE_CLOSE * 60:
            log(f"- {fresh.url} closes too soon, skipped")
            continue
        if limit_attempts and not attempt_allowed(fresh, now):
            log(f"- {fresh.url} is between attempt slots, left for a later run")
            continue
        confirmed.append(Target(fresh, target.tier, target.tournament))

    far = dt.datetime.max.replace(tzinfo=dt.timezone.utc)
    confirmed.sort(key=lambda t: (_TIER_RANK.get(t.tier, 9), t.question.close_time or far))
    return confirmed, failed


def _process(
    client: MetaculusClient, llm: OpenRouter, target: Target, clock: Callable[[], dt.datetime], dry_run: bool
) -> Outcome:
    question = target.question
    # Questions wait for a free worker; the estimates the run was planned on can be far
    # below what earlier questions really cost, so check the money actually spent.
    if llm.total_cost >= config.MAX_COST_PER_RUN:
        return Outcome(target, "skipped", error="spend ceiling for one run reached")
    now = clock()
    if question.close_time and (question.close_time - now).total_seconds() < config.MIN_MINUTES_BEFORE_CLOSE * 60:
        return Outcome(target, "skipped", error="closes too soon")
    try:
        result = forecast_question(llm, question, config.TIERS[target.tier], now)
    except (ForecastError, ResearchError, LLMError, HttpError, ValueError) as error:
        return Outcome(target, "failed", error=f"{type(error).__name__}: {error}")
    except Exception as error:  # noqa: BLE001 - one odd question must not take the whole run down
        return Outcome(target, "failed", error=f"unexpected {type(error).__name__}: {error}")
    if dry_run:
        return Outcome(target, "dry", result=result)
    try:
        client.post_forecast(question, result.forecast)
    except HttpError as error:
        rejected = 400 <= error.status < 500 and error.status != 429
        return Outcome(target, "failed", result=result, error=f"forecast not accepted: {error}", rejected=rejected)
    except Exception as error:  # noqa: BLE001
        return Outcome(target, "failed", result=result, error=f"forecast not sent: {type(error).__name__}: {error}")
    for attempt in range(2):
        try:
            client.post_comment(question.post_id, result.comment)
            break
        except Exception as error:  # noqa: BLE001 - the forecast is in; a missing comment is only noted
            if attempt == 1:
                result.notes.append(f"comment not posted: {type(error).__name__}: {error}")
            else:
                time.sleep(5)
    return Outcome(target, "submitted", result=result)


def _print_details(outcome: Outcome) -> None:
    """Full forecast details. Never printed in CI: logs of a public repository are public."""
    result = outcome.result
    if result is None or IN_CI:
        return
    log("-" * 78)
    log(f"{result.question.title}\n{result.question.url}")
    log(f"final: {result.summary}   (tier {result.tier}, cost ${result.cost:.3f})")
    log(result.comment)
    for note in result.notes:
        log(f"note: {note}")


def _step_summary(lines: list[str]) -> None:
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    except OSError:
        pass


def run_forecasts(
    plan: list[tuple[str, str]],
    *,
    reforecast: bool = False,
    limit_attempts: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    limit: int | None = None,
) -> int:
    token, key = secret("METACULUS_TOKEN"), secret("OPENROUTER_API_KEY")
    if not token or not key:
        log("Missing METACULUS_TOKEN and/or OPENROUTER_API_KEY.")
        return 2
    client, llm = MetaculusClient(token), OpenRouter(key)

    now = utcnow()
    targets, failed_lists = gather_targets(
        client, plan, now, reforecast=reforecast, limit_attempts=limit_attempts
    )
    exit_code = 0
    if len(failed_lists) == len(plan):
        exit_code = 1
    main_tournament = str(plan[0][0])
    if main_tournament in failed_lists:
        # At the very start of a season the new tournament may not exist yet; anything
        # else means the bot is missing its main tournament and must be looked at.
        season_start = now.month in (1, 5, 9) and failed_lists[main_tournament] == 404
        if not season_start:
            exit_code = 1
    if limit is not None:
        targets = targets[:limit]
    if not targets:
        log("No open questions to forecast.")
        return exit_code

    remaining = llm.budget_remaining()
    log(f"Credits remaining: {'unknown' if remaining is None else f'${remaining:.2f}'}")

    planned: list[Target] = []
    planned_cost = 0.0
    skipped_for_budget = 0
    for target in targets:
        estimate_left = None if remaining is None else remaining - planned_cost
        tier = pick_tier(target.tier, estimate_left)
        if tier is None or planned_cost + config.TIERS[tier].est_cost > config.MAX_COST_PER_RUN:
            skipped_for_budget += 1
            continue
        planned.append(Target(target.question, tier, target.tournament))
        planned_cost += config.TIERS[tier].est_cost

    outcomes: list[Outcome] = []
    with ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_POSTS) as pool:
        futures = [pool.submit(_process, client, llm, target, utcnow, dry_run) for target in planned]
        for future in futures:
            outcome = future.result()
            outcomes.append(outcome)
            question = outcome.target.question
            cost = f"${outcome.result.cost:.3f}" if outcome.result else "-"
            line = f"[{outcome.status}] {question.url} ({question.qtype}, {outcome.target.tier}, {cost})"
            if outcome.error:
                line += f" :: {outcome.error}"
            log(line)
            if outcome.result:
                for note in outcome.result.notes:
                    log(f"    note: {note[:300]}")
            if verbose:
                _print_details(outcome)

    done = sum(1 for o in outcomes if o.status in ("submitted", "dry"))
    failed = sum(1 for o in outcomes if o.status == "failed")
    rejected = sum(1 for o in outcomes if o.rejected)
    skipped_for_budget += sum(1 for o in outcomes if o.status == "skipped" and "ceiling" in o.error)
    left = llm.budget_remaining()
    summary = [
        f"Forecasts {'prepared (dry run)' if dry_run else 'submitted'}: {done}; failed: {failed}; "
        f"skipped for budget: {skipped_for_budget}",
        f"Spent this run: ${llm.total_cost:.2f} in {llm.calls} model calls; "
        f"credits remaining: {'unknown' if left is None else f'${left:.2f}'}",
    ]
    for line in summary:
        log(line)
    _step_summary(["### Forecast run"] + [f"- {line}" for line in summary])

    if failed and done == 0:
        exit_code = 1
    if rejected:
        # Deterministic: the same forecast would be bought and refused again on later runs.
        log(f"::error::Metaculus refused {rejected} forecast(s); the bot needs attention.")
        exit_code = 1
    if skipped_for_budget and not planned:
        log("Out of credits: open questions were left unanswered.")
        exit_code = 1
    return exit_code


def run_single_post(post_id: int, tier: str, *, submit: bool) -> int:
    """Forecast one post by id (local debugging on test or closed questions; dry run unless submit)."""
    token, key = secret("METACULUS_TOKEN"), secret("OPENROUTER_API_KEY")
    if not token or not key:
        log("Missing METACULUS_TOKEN and/or OPENROUTER_API_KEY.")
        return 2
    client, llm = MetaculusClient(token), OpenRouter(key)
    questions = questions_from_post(client.get_post(post_id))
    if not questions:
        log("No supported question in this post.")
        return 1
    status = 0
    for question in questions:
        if submit and question.already_forecast:
            log(f"[skipped] {question.url} already has a forecast; not submitting a second one")
            continue
        outcome = _process(client, llm, Target(question, tier, "manual"), utcnow, dry_run=not submit)
        log(f"[{outcome.status}] {question.url} {outcome.error}")
        _print_details(outcome)
        status = status or (1 if outcome.status == "failed" else 0)
    log(f"Spent: ${llm.total_cost:.3f}")
    return status


def run_offline() -> int:
    problems = selfcheck.run_offline_checks()
    for problem in problems:
        log(f"! {problem}")
    log("Offline checks: " + ("FAILED" if problems else "ok"))
    return 1 if problems else 0


def run_selftest() -> int:
    """Check every external dependency with tiny requests (costs a few cents)."""
    failures = run_offline()
    token, key = secret("METACULUS_TOKEN"), secret("OPENROUTER_API_KEY")
    if not token or not key:
        log("Missing METACULUS_TOKEN and/or OPENROUTER_API_KEY.")
        return 2

    client = MetaculusClient(token)
    try:
        account = client.me()
        log(f"Metaculus: signed in as '{account.get('username', '?')}' (bot account: {account.get('is_bot', '?')})")
    except (HttpError, ValueError) as error:
        log(f"Metaculus: could not read the account ({error}); continuing")
    try:
        posts = client.list_open_posts(config.TEST_AREA_ID)
        kinds = sorted({q.qtype for post in posts for q in questions_from_post(post)})
        log(f"Metaculus: ok ({len(posts)} open post(s) in the testing area; types: {', '.join(kinds) or 'none'})")
    except (HttpError, ValueError) as error:
        failures += 1
        log(f"Metaculus: FAILED - {error}")
    for tournament in config.seasonal_tournament_ids(utcnow().date()) + [config.MINIBENCH_ID]:
        try:
            log(f"Metaculus: '{tournament}' reachable, {len(client.list_open_posts(tournament))} open post(s)")
        except (HttpError, ValueError) as error:
            failures += 1
            log(f"Metaculus: '{tournament}' FAILED - {error}")

    llm = OpenRouter(key)
    remaining = llm.budget_remaining()
    log(f"OpenRouter credits remaining: {'unknown' if remaining is None else f'${remaining:.2f}'}")

    models: dict[tuple[str, str], int] = {}
    sources: dict[str, config.ResearchSource] = {}
    for tier in config.TIERS.values():
        for member in tier.members + ((tier.adjudicator,) if tier.adjudicator else ()):
            models.setdefault((member.model, member.effort), 0)
        for source in tier.research:
            sources.setdefault(source.model, source)
    models.setdefault((config.FORMAT_FIXER.model, config.FORMAT_FIXER.effort), 0)

    ping = [{"role": "user", "content": "Reply with the single word: OK"}]
    for model, effort in sorted(models):
        try:
            result = llm.chat(model, ping, effort=effort, max_tokens=2000, timeout=120, retries=0)
            log(f"model {model} ({effort}): ok, ${result.cost:.4f}")
        except LLMError as error:
            failures += 1
            log(f"model {model} ({effort}): FAILED - {error}")

    question = [
        {
            "role": "user",
            "content": "Use web search to find one news headline published in the last 48 hours. "
            "Reply with the headline, its date and the source URL, in one line.",
        }
    ]
    for model, source in sorted(sources.items()):
        try:
            result = llm.chat(
                model, question, effort=source.effort, max_tokens=3000, tools=web_tools(2, 1), max_tool_steps=4,
                max_tool_cost=0.2, allow_truncated=True, timeout=180, retries=0,
            )
            log(f"web research via {model}: ok, {result.searches} search(es) reported, ${result.cost:.4f}")
            if not IN_CI:
                log(f"    {result.text[:300]}")
        except LLMError as error:
            failures += 1
            log(f"web research via {model}: FAILED - {error}")

    log(f"Self-test spent ${llm.total_cost:.3f}. " + ("ALL OK" if not failures else f"{failures} problem(s)"))
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Forecasting bot for the Metaculus FutureEval bot tournaments")
    parser.add_argument(
        "--mode",
        choices=["tournament", "test_questions", "metaculus_cup", "selftest", "offline"],
        default="tournament",
    )
    parser.add_argument("--dry-run", action="store_true", help="forecast but do not submit")
    parser.add_argument("--verbose", action="store_true", help="print full forecasts (ignored in CI)")
    parser.add_argument("--limit", type=int, default=None, help="forecast at most this many questions")
    parser.add_argument("--post-id", type=int, default=None, help="forecast a single post (dry run unless --submit)")
    parser.add_argument("--submit", action="store_true", help="with --post-id: actually submit")
    parser.add_argument("--tier", choices=sorted(config.TIERS), default=None, help="override the tier")
    args = parser.parse_args(argv)

    load_dotenv()
    if args.post_id is not None:
        return run_single_post(args.post_id, args.tier or config.TEST_TIER, submit=args.submit)
    if args.mode == "offline":
        return run_offline()
    if args.mode == "selftest":
        return run_selftest()
    if args.mode == "metaculus_cup":
        log("The Metaculus Cup is not part of this bot's programme; nothing to do.")
        return 0
    if args.mode == "test_questions":
        status = run_offline()
        if status:
            return status
        if not secret("METACULUS_TOKEN") or not secret("OPENROUTER_API_KEY"):
            log("Secrets are not configured yet, so the live part of the test was skipped.")
            return 0
        plan = [(config.TEST_AREA_ID, args.tier or config.TEST_TIER)]
        return run_forecasts(plan, reforecast=True, dry_run=args.dry_run, verbose=args.verbose, limit=args.limit)

    today = utcnow().date()
    plan = [(slug, args.tier or config.SEASONAL_TIER) for slug in config.seasonal_tournament_ids(today)]
    plan.append((config.MINIBENCH_ID, args.tier or config.MINIBENCH_TIER))
    return run_forecasts(plan, limit_attempts=True, dry_run=args.dry_run, verbose=args.verbose, limit=args.limit)


if __name__ == "__main__":
    sys.exit(main())
