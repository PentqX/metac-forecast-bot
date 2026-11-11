import dataclasses
import datetime as dt
import http.client
import io
import json
import os
import unittest
import urllib.error
from contextlib import redirect_stdout
from unittest import mock

from bot import config, http as http_module, llm as llm_module, metaculus, runner
from bot.http import HttpError

from . import fakes

SEASON = "fall-futureeval-2026"


def agreeing_llm(**kwargs):
    return fakes.FakeLLM(lambda model, messages: "Reasoning.\nProbability: 20%", **kwargs)


class RunnerHarness(unittest.TestCase):
    def setUp(self):
        self.client = fakes.FakeClient({})
        self.llm = agreeing_llm()
        patches = [
            mock.patch.object(runner, "MetaculusClient", lambda token: self.client),
            mock.patch.object(runner, "OpenRouter", lambda key: self.llm),
            mock.patch.object(runner, "utcnow", lambda: fakes.NOW),
            mock.patch.dict(os.environ, {"METACULUS_TOKEN": "t", "OPENROUTER_API_KEY": "k"}),
            mock.patch.object(runner, "IN_CI", False),
            mock.patch.object(runner.time, "sleep", lambda seconds: None),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def run_plan(self, plan=None, **kwargs):
        output = io.StringIO()
        with redirect_stdout(output):
            code = runner.run_forecasts(plan or [(SEASON, "full"), ("minibench", "lite")], **kwargs)
        return code, output.getvalue()


class RunForecastsTest(RunnerHarness):
    def test_submits_forecast_and_comment(self):
        self.client.posts_by_tournament = {SEASON: [fakes.post_json(1, 11)], "minibench": [fakes.post_json(2, 22)]}
        code, output = self.run_plan()
        self.assertEqual(code, 0)
        # questions are processed in parallel, so the order of submission is not fixed
        forecasts = sorted(self.client.forecasts, key=lambda f: f["question"])
        self.assertEqual([f["question"] for f in forecasts], [11, 22])
        self.assertEqual(forecasts[0], {"question": 11, "source": "api", "probability_yes": mock.ANY})
        self.assertAlmostEqual(forecasts[0]["probability_yes"], 0.2, places=6)
        self.assertEqual(sorted(post for post, _ in self.client.comments), [1, 2])
        self.assertIn("Forecasts submitted: 2; failed: 0", output)
        # the seasonal question used the full tier (5 forecasts), MiniBench the lite tier (3)
        self.assertEqual(len(self.llm.forecast_calls()), 8)

    def test_public_log_never_contains_forecasts(self):
        self.client.posts_by_tournament = {SEASON: [fakes.post_json(1, 11)]}
        with mock.patch.object(runner, "IN_CI", True):
            code, output = self.run_plan(verbose=True)
        self.assertEqual(code, 0)
        self.assertNotIn("20.0%", output)
        self.assertNotIn("Probability", output)
        self.assertNotIn("Reasoning", output)

    def test_skips_forecast_closed_and_closing_questions(self):
        done = fakes.post_json(1, 11, my_forecasts={"latest": {"forecast_values": [0.5, 0.5]}})
        history_only = fakes.post_json(2, 12, my_forecasts={"latest": None, "history": [{"x": 1}]})
        closed = fakes.post_json(3, 13, status="closed")
        closing = fakes.post_json(4, 14, scheduled_close_time=(fakes.NOW + dt.timedelta(minutes=3)).isoformat())
        fresh = fakes.post_json(5, 15)
        self.client.posts_by_tournament = {SEASON: [done, history_only, closed, closing, fresh]}
        code, _ = self.run_plan([(SEASON, "lite")])
        self.assertEqual(code, 0)
        self.assertEqual([f["question"] for f in self.client.forecasts], [15])

    def test_post_view_is_authoritative_about_existing_forecast(self):
        listed = fakes.post_json(1, 11)
        self.client.posts_by_tournament = {SEASON: [listed]}
        detailed = fakes.post_json(1, 11, my_forecasts={"latest": {"forecast_values": [0.5, 0.5]}})
        self.client.get_post = lambda post_id: detailed
        code, output = self.run_plan([(SEASON, "lite")])
        self.assertEqual(code, 0)
        self.assertEqual(self.client.forecasts, [])
        self.assertIn("No open questions", output)

    def test_same_question_in_two_tournaments_is_forecast_once(self):
        post = fakes.post_json(1, 11)
        self.client.posts_by_tournament = {SEASON: [post], "minibench": [post]}
        self.run_plan()
        self.assertEqual(len(self.client.forecasts), 1)

    def test_dry_run_submits_nothing(self):
        self.client.posts_by_tournament = {SEASON: [fakes.post_json(1, 11)]}
        code, output = self.run_plan(dry_run=True, verbose=True)
        self.assertEqual(code, 0)
        self.assertEqual(self.client.forecasts, [])
        self.assertEqual(self.client.comments, [])
        self.assertIn("final: 20.0%", output)

    def test_reforecast_mode_for_testing_area(self):
        done = fakes.post_json(1, 11, my_forecasts={"latest": {"forecast_values": [0.5, 0.5]}})
        self.client.posts_by_tournament = {config.TEST_AREA_ID: [done]}
        self.run_plan([(config.TEST_AREA_ID, "lite")], reforecast=True)
        self.assertEqual(len(self.client.forecasts), 1)

    def test_group_questions_are_unpacked(self):
        group_post = {
            "id": 7,
            "title": "How many by each date?",
            "group_of_questions": {
                "description": "Group background.",
                "resolution_criteria": "Group criteria.",
                "questions": [
                    dict(fakes.post_json(7, 71)["question"], label="June", title="June"),
                    dict(fakes.post_json(7, 72)["question"], label="July", title="July"),
                ],
            },
        }
        self.client.posts_by_tournament = {SEASON: [group_post]}
        self.run_plan([(SEASON, "lite")])
        self.assertEqual(sorted(f["question"] for f in self.client.forecasts), [71, 72])

    def test_all_failed_is_an_error(self):
        self.client.posts_by_tournament = {SEASON: [fakes.post_json(1, 11)]}
        self.llm = fakes.FakeLLM(lambda model, messages: "no answer")
        code, output = self.run_plan([(SEASON, "lite")])
        self.assertEqual(code, 1)
        self.assertEqual(self.client.forecasts, [])
        self.assertIn("[failed]", output)

    def test_rejected_forecast_is_reported_and_not_commented(self):
        self.client.posts_by_tournament = {SEASON: [fakes.post_json(1, 11)]}
        self.client.fail_forecast = HttpError(400, {"detail": "question closed"}, "https://x")
        code, output = self.run_plan([(SEASON, "lite")])
        self.assertEqual(code, 1)
        self.assertEqual(self.client.comments, [])
        self.assertIn("forecast not accepted", output)
        self.assertIn("::error::Metaculus refused 1 forecast(s)", output)

    def test_a_refused_forecast_fails_the_run_even_if_others_succeed(self):
        self.client.posts_by_tournament = {SEASON: [fakes.post_json(1, 11), fakes.post_json(2, 12)]}
        original = self.client.post_forecast

        def post_forecast(question, forecast):
            if question.question_id == 11:
                raise HttpError(400, {"detail": "CDF Invalid"}, "https://x")
            original(question, forecast)

        self.client.post_forecast = post_forecast
        code, _ = self.run_plan([(SEASON, "lite")])
        self.assertEqual(code, 1)
        self.assertEqual([f["question"] for f in self.client.forecasts], [12])

    def test_a_server_hiccup_on_submit_is_not_treated_as_a_refusal(self):
        self.client.posts_by_tournament = {SEASON: [fakes.post_json(1, 11), fakes.post_json(2, 12)]}
        original = self.client.post_forecast

        def post_forecast(question, forecast):
            if question.question_id == 11:
                raise HttpError(503, "unavailable", "https://x")
            original(question, forecast)

        self.client.post_forecast = post_forecast
        code, output = self.run_plan([(SEASON, "lite")])
        self.assertEqual(code, 0)
        self.assertNotIn("::error::", output)

    def test_unexpected_errors_while_posting_do_not_abort_the_run(self):
        self.client.posts_by_tournament = {SEASON: [fakes.post_json(1, 11), fakes.post_json(2, 12)]}
        self.client.post_comment = mock.Mock(side_effect=http.client.IncompleteRead(b""))
        code, output = self.run_plan([(SEASON, "lite")])
        self.assertEqual(code, 0)
        self.assertEqual(len(self.client.forecasts), 2)
        self.assertIn("comment not posted", output)

    def test_one_failure_does_not_fail_the_run(self):
        self.client.posts_by_tournament = {SEASON: [fakes.post_json(1, 11), fakes.post_json(2, 12)]}
        seen = []

        def answer(model, messages):
            title = "Question 11?" if "Question 11?" in messages[1]["content"] else "Question 12?"
            seen.append(title)
            return "no answer" if title == "Question 11?" else "Probability: 20%"

        self.llm = fakes.FakeLLM(answer)
        code, _ = self.run_plan([(SEASON, "lite")])
        self.assertEqual(code, 0)
        self.assertEqual([f["question"] for f in self.client.forecasts], [12])

    def test_listing_failures(self):
        self.client.posts_by_tournament = {"minibench": [fakes.post_json(2, 22)]}
        self.client.fail_listing = {SEASON: HttpError(500, "boom", "https://x")}
        code, _ = self.run_plan()
        self.assertEqual(code, 1)  # the main tournament could not be read
        self.assertEqual(len(self.client.forecasts), 1)  # MiniBench was still forecast

    def test_missing_new_season_at_season_start_is_tolerated(self):
        january = dt.datetime(2027, 1, 3, 12, 0, tzinfo=dt.timezone.utc)
        self.client.fail_listing = {"spring-futureeval-2027": HttpError(404, "not found", "https://x")}
        with mock.patch.object(runner, "utcnow", lambda: january):
            code, _ = self.run_plan([("spring-futureeval-2027", "full"), (SEASON, "full"), ("minibench", "lite")])
        self.assertEqual(code, 0)

    def test_missing_secrets(self):
        with mock.patch.dict(os.environ, {"METACULUS_TOKEN": "REPLACE_ME"}):
            code, output = self.run_plan()
        self.assertEqual(code, 2)
        self.assertIn("Missing METACULUS_TOKEN", output)


class BudgetTest(RunnerHarness):
    def test_pick_tier(self):
        self.assertEqual(runner.pick_tier("full", None), "full")
        self.assertEqual(runner.pick_tier("full", 100.0), "full")
        self.assertEqual(runner.pick_tier("full", config.CREDITS_FOR_FULL - 0.01), "lite")
        self.assertEqual(runner.pick_tier("full", config.CREDITS_FOR_LITE - 0.01), "economy")
        self.assertEqual(runner.pick_tier("lite", config.CREDITS_FOR_LITE - 0.01), "economy")
        self.assertIsNone(runner.pick_tier("full", config.MIN_CREDITS - 0.01))

    def test_low_credits_downgrade_the_tier(self):
        self.client.posts_by_tournament = {SEASON: [fakes.post_json(1, 11)]}
        self.llm = agreeing_llm(remaining=3.0)
        code, output = self.run_plan([(SEASON, "full")])
        self.assertEqual(code, 0)
        self.assertIn("economy", output)
        self.assertEqual({c["model"] for c in self.llm.forecast_calls()}, {config.GEMINI_FLASH})

    def test_out_of_credits_stops_and_reports(self):
        self.client.posts_by_tournament = {SEASON: [fakes.post_json(1, 11)]}
        self.llm = agreeing_llm(remaining=0.1)
        code, output = self.run_plan([(SEASON, "full")])
        self.assertEqual(code, 1)
        self.assertEqual(self.llm.calls, 0)
        self.assertIn("Out of credits", output)

    def test_real_spend_stops_further_questions(self):
        # Planned estimates allow all four, but the first questions turn out expensive.
        self.client.posts_by_tournament = {SEASON: [fakes.post_json(i, 100 + i) for i in range(1, 5)]}
        cheap_on_paper = dataclasses.replace(config.TIERS["lite"], est_cost=0.01)
        with mock.patch.object(config, "MAX_COST_PER_RUN", 0.2), mock.patch.object(config, "MAX_PARALLEL_POSTS", 1), \
             mock.patch.dict(config.TIERS, {"lite": cheap_on_paper}):
            code, output = self.run_plan([(SEASON, "lite")])
        self.assertEqual(code, 0)
        self.assertLess(len(self.client.forecasts), 4)
        self.assertGreaterEqual(len(self.client.forecasts), 1)
        self.assertIn("[skipped]", output)
        self.assertIn("spend ceiling", output)

    def test_run_cost_ceiling(self):
        self.client.posts_by_tournament = {SEASON: [fakes.post_json(i, 100 + i) for i in range(1, 41)]}
        code, output = self.run_plan([(SEASON, "full")])
        self.assertEqual(code, 0)
        expected = int(config.MAX_COST_PER_RUN / config.TIERS["full"].est_cost)
        self.assertEqual(len(self.client.forecasts), expected)
        self.assertIn(f"skipped for budget: {40 - expected}", output)


class AttemptSlotTest(RunnerHarness):
    def question_aged(self, minutes, window=180):
        """A question that opened `minutes` ago and closes `window` minutes after opening."""
        opened = fakes.NOW - dt.timedelta(minutes=minutes)
        return fakes.binary_question(open_time=opened, close_time=opened + dt.timedelta(minutes=window))

    def test_slots(self):
        allowed = [m for m in range(0, 180, 5) if runner.attempt_allowed(self.question_aged(m), fakes.NOW)]
        self.assertEqual(allowed, [0, 5, 10, 15, 20, 25, 30, 35, 40, 85, 90, 95, 100, 145, 150, 155, 160, 165, 170, 175])
        self.assertTrue(runner.attempt_allowed(fakes.binary_question(close_time=None), fakes.NOW))

    def test_slots_do_not_depend_on_the_open_time(self):
        # Only the time left matters, so an odd or missing open time cannot lock a question out.
        close = fakes.NOW + dt.timedelta(minutes=170)
        for opened in (None, fakes.NOW - dt.timedelta(days=30), fakes.NOW):
            self.assertTrue(runner.attempt_allowed(fakes.binary_question(open_time=opened, close_time=close), fakes.NOW))

    def test_long_open_questions_are_tried_once_an_hour(self):
        tries = [m for m in range(200, 2000, 20)
                 if runner.attempt_allowed(fakes.binary_question(close_time=fakes.NOW + dt.timedelta(minutes=m)), fakes.NOW)]
        self.assertEqual(len(tries), 30)
        self.assertTrue(all(b - a == 60 for a, b in zip(tries, tries[1:])))

    def test_every_twenty_minute_schedule_gets_at_most_five_attempts(self):
        for offset in range(20):
            ages = range(offset, 168, 20)  # the bot stops 12 minutes before the close at 180
            attempts = sum(runner.attempt_allowed(self.question_aged(age), fakes.NOW) for age in ages)
            self.assertTrue(2 <= attempts <= 5, (offset, attempts))

    def test_a_question_first_seen_late_is_still_attempted_within_forty_minutes(self):
        for first_seen in range(0, 130):
            waits = [w for w in range(0, 60, 1) if runner.attempt_allowed(self.question_aged(first_seen + w), fakes.NOW)]
            self.assertTrue(waits and waits[0] <= 40, first_seen)

    def test_tournament_runs_respect_slots_and_test_runs_do_not(self):
        between = fakes.post_json(1, 11, scheduled_close_time=(fakes.NOW + dt.timedelta(minutes=130)).isoformat())
        self.client.posts_by_tournament = {SEASON: [between]}
        code, output = self.run_plan([(SEASON, "lite")], limit_attempts=True)
        self.assertEqual((code, self.client.forecasts), (0, []))
        self.assertIn("between attempt slots", output)
        code, _ = self.run_plan([(SEASON, "lite")], limit_attempts=False)
        self.assertEqual(len(self.client.forecasts), 1)


class ModeTest(RunnerHarness):
    def run_main(self, *argv):
        output = io.StringIO()
        with redirect_stdout(output), mock.patch.object(runner, "load_dotenv", lambda: None):
            code = runner.main(list(argv))
        return code, output.getvalue()

    def test_offline_mode(self):
        code, output = self.run_main("--mode", "offline")
        self.assertEqual(code, 0)
        self.assertIn("Offline checks: ok", output)

    def test_cup_mode_does_nothing(self):
        code, _ = self.run_main("--mode", "metaculus_cup")
        self.assertEqual(code, 0)
        self.assertEqual(self.llm.calls, 0)

    def test_test_mode_without_secrets_passes(self):
        with mock.patch.dict(os.environ, {"METACULUS_TOKEN": "", "OPENROUTER_API_KEY": ""}):
            code, output = self.run_main("--mode", "test_questions")
        self.assertEqual(code, 0)
        self.assertIn("Secrets are not configured yet", output)

    def test_test_mode_forecasts_testing_area(self):
        self.client.posts_by_tournament = {config.TEST_AREA_ID: [fakes.post_json(1, 11)]}
        code, _ = self.run_main("--mode", "test_questions")
        self.assertEqual(code, 0)
        self.assertEqual(len(self.client.forecasts), 1)

    def test_tournament_mode_plan(self):
        self.client.posts_by_tournament = {SEASON: [fakes.post_json(1, 11)], "minibench": [fakes.post_json(2, 22)]}
        code, _ = self.run_main()
        self.assertEqual(code, 0)
        self.assertEqual(len(self.client.forecasts), 2)

    def test_single_post_is_a_dry_run_by_default(self):
        self.client.posts_by_tournament = {"any": [fakes.post_json(9, 99)]}
        code, output = self.run_main("--post-id", "9")
        self.assertEqual(code, 0)
        self.assertEqual(self.client.forecasts, [])
        self.assertIn("[dry]", output)
        code, _ = self.run_main("--post-id", "9", "--submit")
        self.assertEqual(len(self.client.forecasts), 1)

    def test_single_post_never_submits_a_second_forecast(self):
        done = fakes.post_json(9, 99, my_forecasts={"latest": {"forecast_values": [0.5, 0.5]}})
        self.client.posts_by_tournament = {"any": [done]}
        code, output = self.run_main("--post-id", "9", "--submit")
        self.assertEqual(self.client.forecasts, [])
        self.assertIn("already has a forecast", output)
        self.assertEqual(self.llm.calls, 0)

    def test_tournament_mode_limits_attempts(self):
        between = fakes.post_json(1, 11, scheduled_close_time=(fakes.NOW + dt.timedelta(minutes=130)).isoformat())
        self.client.posts_by_tournament = {SEASON: [between]}
        code, _ = self.run_main()
        self.assertEqual((code, self.client.forecasts), (0, []))


class MetaculusParsingTest(unittest.TestCase):
    def test_numeric_fields(self):
        post = fakes.post_json(
            3, 33, "numeric", unit="USD", open_upper_bound=True, open_lower_bound=False,
            scaling={"range_min": 0, "range_max": 500, "zero_point": None, "inbound_outcome_count": None},
        )
        (question,) = metaculus.questions_from_post(post)
        self.assertEqual((question.range_min, question.range_max, question.zero_point), (0.0, 500.0, None))
        self.assertEqual((question.open_lower, question.open_upper, question.cdf_size), (False, True, 201))
        self.assertEqual(question.unit, "USD")
        self.assertEqual(question.close_time, fakes.NOW + dt.timedelta(hours=2, minutes=50))

    def test_bounds_default_to_open(self):
        post = fakes.post_json(3, 33, "numeric", scaling={"range_min": 0, "range_max": 500})
        (question,) = metaculus.questions_from_post(post)
        self.assertTrue(question.open_lower and question.open_upper)

    def test_discrete_fields(self):
        post = fakes.post_json(
            4, 44, "discrete", open_upper_bound=False, open_lower_bound=False,
            scaling={"range_min": -0.5, "range_max": 10.5, "nominal_min": 0, "nominal_max": 10, "inbound_outcome_count": 11},
        )
        (question,) = metaculus.questions_from_post(post)
        self.assertEqual(question.cdf_size, 12)
        self.assertEqual(question.discrete_values(), [float(i) for i in range(11)])
        self.assertEqual(metaculus.questions_from_post(fakes.post_json(4, 44, "discrete", scaling={"range_min": 0, "range_max": 5})), [])

    def test_unsupported_and_malformed_posts(self):
        self.assertEqual(metaculus.questions_from_post({"id": 1, "notebook": {}}), [])
        self.assertEqual(metaculus.questions_from_post(fakes.post_json(1, 11, "date")), [])
        self.assertEqual(metaculus.questions_from_post({"id": 1, "question": {"type": "binary"}}), [])

    def test_payloads(self):
        binary = fakes.binary_question()
        self.assertEqual(metaculus.forecast_payload(binary, 0.25), {"question": 201, "source": "api", "probability_yes": 0.25})
        mc = fakes.mc_question()
        payload = metaculus.forecast_payload(mc, {"Party A": 0.5, "Party B": 0.4, "Other": 0.1})
        self.assertEqual(payload["probability_yes_per_category"], {"Party A": 0.5, "Party B": 0.4, "Other": 0.1})
        self.assertEqual(metaculus.forecast_payload(fakes.numeric_question(), [0.1, 0.9])["continuous_cdf"], [0.1, 0.9])
        json.dumps(payload)


class _Response:
    def __init__(self, status, payload):
        self.status, self._payload = status, payload

    def read(self):
        return self._payload if isinstance(self._payload, bytes) else json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _http_error(code, payload, headers=None):
    return urllib.error.HTTPError("https://x", code, "err", headers or {}, io.BytesIO(json.dumps(payload).encode()))


class HttpTest(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.object(http_module.time, "sleep", lambda seconds: self.sleeps.append(seconds))
        self.sleeps = []
        patch.start()
        self.addCleanup(patch.stop)

    def test_success_and_request_shape(self):
        captured = {}

        def urlopen(request, timeout):
            captured["url"], captured["data"], captured["headers"] = request.full_url, request.data, dict(request.header_items())
            return _Response(200, {"ok": True})

        with mock.patch.object(http_module.urllib.request, "urlopen", urlopen):
            status, data = http_module.request_json(
                "POST", "https://x/api", params={"tournaments": ["a b"], "statuses": ["open"]}, body={"k": 1},
                headers={"Authorization": "Token t"},
            )
        self.assertEqual((status, data), (200, {"ok": True}))
        self.assertEqual(captured["url"], "https://x/api?tournaments=a+b&statuses=open")
        self.assertEqual(json.loads(captured["data"]), {"k": 1})
        self.assertEqual(captured["headers"]["Authorization"], "Token t")
        self.assertIn("metac-forecast-bot", captured["headers"]["User-agent"])

    def test_retries_transient_errors_then_succeeds(self):
        responses = [_http_error(503, {"e": 1}), urllib.error.URLError("reset"), _Response(200, b"  \n {\"ok\": 1}")]

        def urlopen(request, timeout):
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        with mock.patch.object(http_module.urllib.request, "urlopen", urlopen):
            self.assertEqual(http_module.request_json("GET", "https://x")[1], {"ok": 1})
        self.assertEqual(len(self.sleeps), 2)

    def test_cut_off_responses_are_retried_and_reported_as_http_errors(self):
        class Broken:
            status = 200

            def read(self):
                raise http.client.IncompleteRead(b"partial")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        responses = [Broken(), _Response(200, {"ok": 1})]
        with mock.patch.object(http_module.urllib.request, "urlopen", lambda request, timeout: responses.pop(0)):
            self.assertEqual(http_module.request_json("GET", "https://x")[1], {"ok": 1})
        with mock.patch.object(http_module.urllib.request, "urlopen", lambda request, timeout: Broken()):
            with self.assertRaises(HttpError) as caught:
                http_module.request_json("GET", "https://x", retries=1)
        self.assertEqual(caught.exception.status, 0)
        self.assertIn("IncompleteRead", str(caught.exception))

    def test_client_errors_are_not_retried(self):
        calls = []

        def urlopen(request, timeout):
            calls.append(1)
            raise _http_error(400, {"detail": "bad cdf"})

        with mock.patch.object(http_module.urllib.request, "urlopen", urlopen):
            with self.assertRaises(HttpError) as caught:
                http_module.request_json("POST", "https://x?secret=1", body={})
        self.assertEqual(len(calls), 1)
        self.assertEqual(caught.exception.status, 400)
        self.assertIn("bad cdf", str(caught.exception))
        self.assertNotIn("secret", str(caught.exception))

    def test_gives_up_after_retries(self):
        def urlopen(request, timeout):
            raise _http_error(429, {"e": 1}, {"Retry-After": "7"})

        with mock.patch.object(http_module.urllib.request, "urlopen", urlopen):
            with self.assertRaises(HttpError) as caught:
                http_module.request_json("GET", "https://x", retries=2)
        self.assertEqual(caught.exception.status, 429)
        self.assertEqual(caught.exception.retry_after, 7.0)
        self.assertEqual(len(self.sleeps), 2)
        self.assertTrue(all(7.0 <= s <= 8.0 for s in self.sleeps))


class OpenRouterTest(unittest.TestCase):
    def setUp(self):
        self.responses = []
        self.bodies = []

        def request_json(method, url, **kwargs):
            self.bodies.append((method, url, kwargs.get("body")))
            item = self.responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return 200, item

        for patch in (mock.patch.object(llm_module, "request_json", request_json),
                      mock.patch.object(llm_module.time, "sleep", lambda seconds: None)):
            patch.start()
            self.addCleanup(patch.stop)
        self.client = llm_module.OpenRouter("key")

    def test_chat_request_and_cost(self):
        self.responses = [{
            "choices": [{"message": {"content": "  hello  "}, "finish_reason": "stop"}],
            "usage": {"cost": 0.0123, "server_tool_use_details": {"web_search_requests": 2}},
        }]
        result = self.client.chat("m", [{"role": "user", "content": "hi"}], effort="high", max_tokens=100,
                                  tools=llm_module.web_tools(3, 1), max_tool_steps=5, max_tool_cost=0.5)
        self.assertEqual((result.text, result.cost, result.searches), ("hello", 0.0123, 2))
        method, url, body = self.bodies[0]
        self.assertEqual((method, url), ("POST", "https://openrouter.ai/api/v1/chat/completions"))
        self.assertEqual(body["reasoning"], {"effort": "high"})
        self.assertEqual(
            body["stop_server_tools_when"],
            [{"type": "step_count_is", "step_count": 5}, {"type": "max_cost", "max_cost_in_dollars": 0.5}],
        )
        self.assertEqual(
            body["tools"],
            [
                {"type": "openrouter:web_search", "parameters": {"max_uses": 3, "max_total_results": 18}},
                {"type": "openrouter:web_fetch", "parameters": {"max_uses": 1, "max_content_tokens": 6000}},
            ],
        )
        # only fields of the Chat Completions API; sampling parameters are rejected by current Claude models
        self.assertEqual(set(body), {"model", "messages", "max_tokens", "reasoning", "tools", "stop_server_tools_when"})
        self.assertAlmostEqual(self.client.total_cost, 0.0123)

    def test_plain_request_has_no_tool_fields(self):
        self.responses = [{"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}]
        self.client.chat("m", [{"role": "user", "content": "hi"}], max_tool_steps=5)
        self.assertEqual(set(self.bodies[0][2]), {"model", "messages", "max_tokens"})

    def test_cut_off_answer_is_an_error_unless_allowed(self):
        cut = {"choices": [{"message": {"content": "partial reasoning"}, "finish_reason": "length"}], "usage": {"cost": 0.1}}
        self.responses = [cut]
        with self.assertRaises(llm_module.LLMError) as caught:
            self.client.chat("m", [])
        self.assertTrue(caught.exception.truncated)
        self.responses = [cut]
        self.assertEqual(self.client.chat("m", [], allow_truncated=True).text, "partial reasoning")
        self.responses = [{"choices": [{"message": {"content": "partial"}, "finish_reason": "error"}]}]
        with self.assertRaises(llm_module.LLMError):
            self.client.chat("m", [], allow_truncated=True)

    def test_search_count_falls_back_to_tool_calls(self):
        self.responses = [{"choices": [{"message": {"content": "x"}}],
                           "usage": {"server_tool_use_details": {"tool_calls_executed": 4, "web_search_requests": None}}}]
        self.assertEqual(self.client.chat("m", []).searches, 4)

    def test_content_parts_are_joined(self):
        self.responses = [{"choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}]}]
        self.assertEqual(self.client.chat("m", []).text, "ab")

    def test_empty_answer_is_truncation_when_length_limited(self):
        self.responses = [{"choices": [{"message": {"content": None}, "finish_reason": "length"}], "usage": {"cost": 0.2}}]
        with self.assertRaises(llm_module.LLMError) as caught:
            self.client.chat("m", [])
        self.assertTrue(caught.exception.truncated)
        self.assertAlmostEqual(self.client.total_cost, 0.2)  # still paid for

    def test_error_payloads(self):
        self.responses = [{"error": {"message": "model not found", "code": 404}}]
        with self.assertRaises(llm_module.LLMError) as caught:
            self.client.chat("m", [])
        self.assertIn("model not found", str(caught.exception))
        self.responses = [HttpError(400, {"error": {"message": "bad tool"}}, "u")]
        with self.assertRaises(llm_module.LLMError) as caught:
            self.client.chat("m", [])
        self.assertEqual(caught.exception.status, 400)
        self.assertFalse(caught.exception.out_of_credits)

    def test_in_flight_budget_rejection_is_waited_out(self):
        in_flight = HttpError(402, {"error": {"metadata": {"limit_source": "openrouter_in_flight_budget"}}}, "u", retry_after=3)
        self.responses = [in_flight, in_flight, {"choices": [{"message": {"content": "ok"}}]}]
        self.assertEqual(self.client.chat("m", []).text, "ok")

    def test_exhausted_credits_are_reported_as_such(self):
        self.responses = [HttpError(402, {"error": {"message": "limit", "metadata": {"limit_source": "openrouter_key_limit"}}}, "u")]
        with self.assertRaises(llm_module.LLMError) as caught:
            self.client.chat("m", [])
        self.assertTrue(caught.exception.out_of_credits)

    def test_budget_remaining_takes_the_tighter_limit(self):
        self.responses = [{"data": {"limit_remaining": 12.5}}, {"data": {"total_credits": 50, "total_usage": 45}}]
        self.assertEqual(self.client.budget_remaining(), 5.0)
        self.responses = [{"data": {"limit_remaining": None}}, HttpError(403, "forbidden", "u")]
        self.assertIsNone(self.client.budget_remaining())
        self.responses = [{"data": {"limit_remaining": 7}}, HttpError(403, "forbidden", "u")]
        self.assertEqual(self.client.budget_remaining(), 7.0)


class MetaculusClientTest(unittest.TestCase):
    def test_requests_are_spaced_and_shaped(self):
        calls, sleeps = [], []
        clock = [1000.0]

        def request_json(method, url, **kwargs):
            calls.append((method, url, kwargs))
            return 200, {"results": [], "next": None}

        with mock.patch.object(metaculus, "request_json", request_json), \
             mock.patch.object(metaculus.time, "monotonic", lambda: clock[0]), \
             mock.patch.object(metaculus.time, "sleep", lambda s: (sleeps.append(s), clock.__setitem__(0, clock[0] + s))):
            client = metaculus.MetaculusClient("tok")
            client.list_open_posts("minibench")
            client.post_comment(5, "text")
            client.post_forecast(fakes.binary_question(), 0.3)
        self.assertEqual(len(sleeps), 2)
        self.assertTrue(all(s >= metaculus.MIN_SECONDS_BETWEEN_REQUESTS for s in sleeps))
        method, url, kwargs = calls[0]
        self.assertEqual(url, "https://www.metaculus.com/api/posts/")
        self.assertEqual(kwargs["headers"]["Authorization"], "Token tok")
        self.assertEqual(kwargs["params"]["tournaments"], ["minibench"])
        self.assertEqual(kwargs["params"]["statuses"], ["open"])
        self.assertEqual(calls[1][2]["body"], {"on_post": 5, "text": "text", "is_private": True, "included_forecast": True})
        self.assertEqual(calls[2][1], "https://www.metaculus.com/api/questions/forecast/")
        self.assertEqual(calls[2][2]["body"], [{"question": 201, "source": "api", "probability_yes": 0.3}])

    def test_pagination(self):
        pages = [{"results": [{"id": i} for i in range(100)], "next": "more"}, {"results": [{"id": 100}], "next": None}]
        offsets = []

        def request_json(method, url, **kwargs):
            offsets.append(kwargs["params"]["offset"])
            return 200, pages.pop(0)

        with mock.patch.object(metaculus, "request_json", request_json):
            client = metaculus.MetaculusClient("tok", min_interval=0)
            self.assertEqual(len(client.list_open_posts("x")), 101)
        self.assertEqual(offsets, [0, 100])


if __name__ == "__main__":
    unittest.main()
