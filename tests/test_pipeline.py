import unittest

from bot import config, numeric
from bot.llm import LLMError
from bot.pipeline import ForecastError, forecast_question
from bot.research import ResearchError

from . import fakes

FULL, LITE = config.TIERS["full"], config.TIERS["lite"]


def by_model(table):
    """Answer function: text per model, cycling through a list when a model runs several times."""
    counters = {}

    def answer(model, messages):
        choices = table[model]
        if isinstance(choices, Exception):
            raise choices
        if isinstance(choices, str):
            return choices
        index = counters.get(model, 0)
        counters[model] = index + 1
        choice = choices[index % len(choices)]
        if isinstance(choice, Exception):
            raise choice
        return choice

    return answer


class BinaryPipelineTest(unittest.TestCase):
    def test_happy_path(self):
        llm = fakes.FakeLLM(
            by_model(
                {
                    config.OPUS: ["Reasoning.\nProbability: 20%", "Reasoning.\nProbability: 24%"],
                    config.GPT: "Reasoning.\nProbability: 30%",
                    config.GEMINI_FLASH: ["Probability: 26%", "Probability: 22%"],
                }
            )
        )
        result = forecast_question(llm, fakes.binary_question(), FULL, fakes.NOW)
        self.assertTrue(0.20 < result.forecast < 0.30)
        self.assertEqual(len(result.members), 5)
        self.assertIsNone(result.adjudicator)  # members agree closely
        self.assertIn("Forecast:", result.comment)
        self.assertIn("no human in the loop", result.comment)
        self.assertLessEqual(len(result.comment), config.COMMENT_MAX_CHARS)
        # two research calls and five forecasts
        self.assertAlmostEqual(result.cost, 2 * 0.02 + 5 * 0.05)
        self.assertEqual(len([c for c in llm.log if c["tools"]]), 2)

    def test_forecasters_see_question_and_research(self):
        llm = fakes.FakeLLM(by_model({config.OPUS: "Probability: 20%", config.GEMINI_FLASH: "Probability: 20%"}))
        question = fakes.binary_question(fine_print="Only official sources count.")
        forecast_question(llm, question, LITE, fakes.NOW)
        prompt = llm.forecast_calls()[0]["messages"][1]["content"]
        for expected in (question.title, "Only official sources count.", fakes.RESEARCH_TEXT, "Probability: XX.X%",
                         "Now: 2026-11-11 12:00 UTC", "about 50 days from now"):
            self.assertIn(expected, prompt)

    def test_caps_extreme_agreement(self):
        llm = fakes.FakeLLM(by_model({config.OPUS: "Probability: 99.9%", config.GEMINI_FLASH: "Probability: 99.8%"}))
        self.assertEqual(forecast_question(llm, fakes.binary_question(), LITE, fakes.NOW).forecast, LITE.prob_ceil)
        llm = fakes.FakeLLM(by_model({config.OPUS: "Probability: 0.1%", config.GEMINI_FLASH: "Probability: 0.2%"}))
        self.assertEqual(forecast_question(llm, fakes.binary_question(), LITE, fakes.NOW).forecast, LITE.prob_floor)

    def test_format_fixer_rescues_unparseable_answer(self):
        llm = fakes.FakeLLM(
            by_model({config.OPUS: "I would say roughly one in five.", config.GEMINI_FLASH: "Probability: 20%"}),
            fixer=lambda messages: '{"probability_yes_percent": 20}',
        )
        result = forecast_question(llm, fakes.binary_question(), LITE, fakes.NOW)
        self.assertEqual(len(result.members), 3)
        self.assertAlmostEqual(result.forecast, 0.20, places=6)

    def test_answer_in_the_wrong_format_goes_to_the_fixer_not_a_guess(self):
        asked = []

        def fixer(messages):
            asked.append(messages[1]["content"])
            return '{"probability_yes_percent": 12}'

        # A regular expression would read 88% here; the fixer is told to return the Yes probability.
        llm = fakes.FakeLLM(
            by_model({config.OPUS: "Reasoning.\nProbability of No: 88%", config.GEMINI_FLASH: "Probability: 12%"}),
            fixer=fixer,
        )
        result = forecast_question(llm, fakes.binary_question(), LITE, fakes.NOW)
        self.assertAlmostEqual(result.forecast, 0.12, places=6)
        self.assertEqual(len(asked), 1)
        self.assertIn("Probability of No: 88%", asked[0])
        self.assertIn("probability_yes_percent", asked[0])

    def test_one_member_crashing_does_not_void_the_others(self):
        def answer(model, messages):
            if model == config.GEMINI_FLASH:
                raise RuntimeError("connection dropped mid-body")
            return "Probability: 30%"

        result = forecast_question(fakes.FakeLLM(answer), fakes.binary_question(), LITE, fakes.NOW)
        self.assertAlmostEqual(result.forecast, 0.30, places=6)
        self.assertTrue(any("unexpected RuntimeError" in note for note in result.notes))

    def test_adjudicator_running_out_of_credits_keeps_the_ensemble(self):
        def answer(model, messages):
            if fakes.ADJUDICATOR_MARK in messages[1]["content"]:
                raise LLMError("request too expensive", status=402, out_of_credits=True)
            return {config.OPUS: "Probability: 10%", config.GPT: "Probability: 70%", config.GEMINI_FLASH: "Probability: 40%"}[model]

        result = forecast_question(fakes.FakeLLM(answer), fakes.binary_question(), FULL, fakes.NOW)
        self.assertIsNone(result.adjudicator)
        self.assertTrue(0.10 < result.forecast < 0.70)

    def test_extreme_adjudicator_cannot_swamp_the_ensemble(self):
        def answer(model, messages):
            if fakes.ADJUDICATOR_MARK in messages[1]["content"]:
                return "Probability: 0.1%"
            return {config.OPUS: "Probability: 10%", config.GPT: "Probability: 70%", config.GEMINI_FLASH: "Probability: 40%"}[model]

        result = forecast_question(fakes.FakeLLM(answer), fakes.binary_question(), FULL, fakes.NOW)
        self.assertAlmostEqual(result.adjudicator.value, 0.005)
        self.assertGreater(result.forecast, 0.04)

    def test_member_without_answer_is_dropped(self):
        llm = fakes.FakeLLM(
            by_model({config.OPUS: "Probability: 30%", config.GEMINI_FLASH: ["no idea", "Probability: 20%"]}),
            fixer=lambda messages: '{"missing": true}',
        )
        result = forecast_question(llm, fakes.binary_question(), LITE, fakes.NOW)
        self.assertEqual(len(result.members), 2)
        self.assertTrue(any("no final answer" in note for note in result.notes))

    def test_too_few_members_means_no_forecast(self):
        llm = fakes.FakeLLM(
            by_model({config.OPUS: LLMError("provider down"), config.GEMINI_FLASH: [LLMError("down"), "Probability: 20%"]})
        )
        with self.assertRaises(ForecastError):
            forecast_question(llm, fakes.binary_question(), LITE, fakes.NOW)

    def test_lead_model_alone_is_enough(self):
        llm = fakes.FakeLLM(by_model({config.OPUS: "Probability: 30%", config.GEMINI_FLASH: LLMError("down")}))
        result = forecast_question(llm, fakes.binary_question(), LITE, fakes.NOW)
        self.assertAlmostEqual(result.forecast, 0.30, places=6)

    def test_disagreement_calls_adjudicator(self):
        def answer(model, messages):
            if fakes.ADJUDICATOR_MARK in messages[1]["content"]:
                self.assertEqual(model, config.FABLE)
                self.assertIn("Final answer: 10.0%", messages[1]["content"])
                return "The crux is the deadline.\nProbability: 60%"
            return {config.OPUS: "Probability: 10%", config.GPT: "Probability: 70%", config.GEMINI_FLASH: "Probability: 40%"}[model]

        llm = fakes.FakeLLM(answer)
        result = forecast_question(llm, fakes.binary_question(), FULL, fakes.NOW)
        self.assertIsNotNone(result.adjudicator)
        self.assertAlmostEqual(result.adjudicator.value, 0.60)
        self.assertIn("adjudicator", result.comment)
        without = fakes.FakeLLM(
            by_model({config.OPUS: "Probability: 10%", config.GPT: "Probability: 70%", config.GEMINI_FLASH: "Probability: 40%",
                      config.FABLE: LLMError("down")})
        )
        baseline = forecast_question(without, fakes.binary_question(), FULL, fakes.NOW)
        self.assertIsNone(baseline.adjudicator)
        self.assertGreater(result.forecast, baseline.forecast)  # pulled towards the adjudicator's 60%
        self.assertTrue(any("adjudicator failed" in note for note in baseline.notes))

    def test_truncated_answer_retries_with_less_thinking(self):
        efforts = []

        def answer(model, messages):
            return "Probability: 20%"

        llm = fakes.FakeLLM(answer)
        original = llm.chat

        def chat(model, messages, **kwargs):
            if model == config.OPUS and not kwargs.get("tools"):
                efforts.append(kwargs.get("effort"))
                if len(efforts) == 1:
                    raise LLMError("empty answer", truncated=True)
            return original(model, messages, **kwargs)

        llm.chat = chat
        result = forecast_question(llm, fakes.binary_question(), LITE, fakes.NOW)
        self.assertEqual(efforts, ["medium", "low"])
        self.assertEqual(len(result.members), 3)

    def test_out_of_credits_propagates(self):
        llm = fakes.FakeLLM(by_model({config.OPUS: LLMError("no credits", status=402, out_of_credits=True),
                                      config.GEMINI_FLASH: LLMError("no credits", status=402, out_of_credits=True)}))
        with self.assertRaises(LLMError) as caught:
            forecast_question(llm, fakes.binary_question(), LITE, fakes.NOW)
        self.assertTrue(caught.exception.out_of_credits)


class ResearchTest(unittest.TestCase):
    def test_no_research_means_no_forecast(self):
        def research(model, tools):
            raise LLMError("search tool unavailable", status=400)

        llm = fakes.FakeLLM(by_model({config.OPUS: "Probability: 20%", config.GEMINI_FLASH: "Probability: 20%"}), research=research)
        with self.assertRaises(ResearchError):
            forecast_question(llm, fakes.binary_question(), LITE, fakes.NOW)
        self.assertEqual(llm.forecast_calls(), [])  # nothing was spent on forecasters

    def test_text_without_any_sign_of_web_access_is_not_research(self):
        llm = fakes.FakeLLM(
            by_model({config.OPUS: "Probability: 20%", config.GEMINI_FLASH: "Probability: 20%"}),
            research=lambda model, tools: "I could not use the search tool, but from memory the situation is calm.",
            searches=0,
        )
        with self.assertRaises(ResearchError):
            forecast_question(llm, fakes.binary_question(), LITE, fakes.NOW)
        self.assertEqual(llm.forecast_calls(), [])

    def test_cited_urls_count_as_research_when_searches_are_not_reported(self):
        llm = fakes.FakeLLM(
            by_model({config.OPUS: "Probability: 20%", config.GEMINI_FLASH: "Probability: 20%"}),
            research=lambda model, tools: "Latest value 104 (https://example.org/data, 2026-11-10).",
            searches=0,
        )
        self.assertAlmostEqual(forecast_question(llm, fakes.binary_question(), LITE, fakes.NOW).forecast, 0.20, places=6)

    def test_falls_back_to_search_without_page_reading(self):
        def research(model, tools):
            if any(tool["type"] == "openrouter:web_fetch" for tool in tools):
                raise LLMError("web_fetch not supported", status=400)
            return "brief from search only (https://example.org/a)"

        llm = fakes.FakeLLM(by_model({config.OPUS: "Probability: 20%", config.GEMINI_FLASH: "Probability: 20%"}), research=research)
        forecast_question(llm, fakes.binary_question(), LITE, fakes.NOW)
        self.assertIn("brief from search only", llm.forecast_calls()[0]["messages"][1]["content"])

    def test_one_source_failing_is_tolerated(self):
        def research(model, tools):
            if model == config.GEMINI_FLASH:
                raise LLMError("down", status=503)
            return fakes.RESEARCH_TEXT

        llm = fakes.FakeLLM(
            by_model({config.OPUS: "Probability: 20%", config.GPT: "Probability: 20%", config.GEMINI_FLASH: "Probability: 20%"}),
            research=research,
        )
        result = forecast_question(llm, fakes.binary_question(), FULL, fakes.NOW)
        self.assertIn("1 research source(s)", result.comment)
        self.assertTrue(any("web-g" in note for note in result.notes))

    def test_research_tools_are_bounded(self):
        llm = fakes.FakeLLM(by_model({config.OPUS: "Probability: 20%", config.GEMINI_FLASH: "Probability: 20%"}))
        forecast_question(llm, fakes.binary_question(), LITE, fakes.NOW)
        call = next(c for c in llm.log if c["tools"])
        source = LITE.research[0]
        search = next(t for t in call["tools"] if t["type"] == "openrouter:web_search")
        self.assertEqual(search["parameters"]["max_uses"], source.max_searches)
        self.assertEqual(call["max_tool_steps"], source.max_searches + source.max_fetches + 2)
        self.assertEqual(call["max_tool_cost"], config.RESEARCH_MAX_COST)


class MultipleChoicePipelineTest(unittest.TestCase):
    def test_happy_path(self):
        llm = fakes.FakeLLM(
            by_model(
                {
                    config.OPUS: "Party A: 60%\nParty B: 40%\nOther: 0%",
                    config.GEMINI_FLASH: ["Party A: 50%\nParty B: 45%\nOther: 5%", "Party A: 55%\nParty B: 44%\nOther: 1%"],
                }
            )
        )
        result = forecast_question(llm, fakes.mc_question(), LITE, fakes.NOW)
        self.assertEqual(list(result.forecast), ["Party A", "Party B", "Other"])
        self.assertAlmostEqual(sum(result.forecast.values()), 1.0, places=9)
        self.assertGreaterEqual(result.forecast["Other"], config.MC_FLOOR - 1e-12)
        self.assertGreater(result.forecast["Party A"], result.forecast["Party B"])
        self.assertIn("Party A: XX%", llm.forecast_calls()[0]["messages"][1]["content"])

    def test_fixer_for_options(self):
        llm = fakes.FakeLLM(
            by_model({config.OPUS: "A is most likely.", config.GEMINI_FLASH: "Party A: 50%\nParty B: 45%\nOther: 5%"}),
            fixer=lambda messages: '{"options": {"party a": 70, "Party B": 25, "Other": 5}}',
        )
        result = forecast_question(llm, fakes.mc_question(), LITE, fakes.NOW)
        self.assertEqual(len(result.members), 3)
        self.assertGreater(result.forecast["Party A"], 0.55)

    def test_question_without_options(self):
        llm = fakes.FakeLLM(by_model({}))
        with self.assertRaises(ForecastError):
            forecast_question(llm, fakes.mc_question(options=[]), LITE, fakes.NOW)


class NumericPipelineTest(unittest.TestCase):
    def test_happy_path(self):
        llm = fakes.FakeLLM(
            by_model(
                {
                    config.OPUS: fakes.percentile_answer(110, 10),
                    config.GEMINI_FLASH: [fakes.percentile_answer(105, 12), fakes.percentile_answer(114, 9)],
                }
            )
        )
        question = fakes.numeric_question()
        result = forecast_question(llm, question, LITE, fakes.NOW)
        self.assertEqual(len(result.forecast), 201)
        numeric.validate(result.forecast, True, True, 201)
        median = numeric.Scale(50, 200).to_value(numeric.loc_at(result.forecast, 0.5))
        self.assertAlmostEqual(median, 110, delta=3)
        self.assertIn("median", result.summary)
        self.assertIn("points", result.summary)
        prompt = llm.forecast_calls()[0]["messages"][1]["content"]
        self.assertIn("Units: points", prompt)
        self.assertIn("Percentile 99: X", prompt)
        self.assertIn("Outcomes above 200 are possible", prompt)

    def test_single_typo_does_not_move_the_distribution(self):
        clean = fakes.percentile_answer(108, 10)
        typo = clean.replace(clean.splitlines()[6], "Percentile 40: 1080")
        self.assertIn("Percentile 40: 1080", typo)
        with_typo = forecast_question(
            fakes.FakeLLM(by_model({config.OPUS: typo, config.GEMINI_FLASH: clean})), fakes.numeric_question(), LITE, fakes.NOW
        )
        without = forecast_question(
            fakes.FakeLLM(by_model({config.OPUS: clean, config.GEMINI_FLASH: clean})), fakes.numeric_question(), LITE, fakes.NOW
        )
        self.assertEqual(len(with_typo.members), 3)
        scale = numeric.Scale(50, 200)
        for probability in (0.05, 0.5, 0.95):
            a = scale.to_value(numeric.loc_at(with_typo.forecast, probability))
            b = scale.to_value(numeric.loc_at(without.forecast, probability))
            self.assertAlmostEqual(a, b, delta=1.5)

    def test_wrong_units_member_is_dropped(self):
        llm = fakes.FakeLLM(
            by_model(
                {
                    config.OPUS: fakes.percentile_answer(110, 10),
                    config.GEMINI_FLASH: [fakes.percentile_answer(110000, 10000), fakes.percentile_answer(112, 9)],
                }
            )
        )
        result = forecast_question(llm, fakes.numeric_question(), LITE, fakes.NOW)
        self.assertEqual(len(result.members), 2)
        self.assertTrue(any("wrong units" in note for note in result.notes))

    def test_closed_bounds(self):
        llm = fakes.FakeLLM(by_model({config.OPUS: fakes.percentile_answer(60, 15), config.GEMINI_FLASH: fakes.percentile_answer(62, 15)}))
        question = fakes.numeric_question(open_lower=False, open_upper=False)
        result = forecast_question(llm, question, LITE, fakes.NOW)
        self.assertEqual(result.forecast[0], 0.0)
        self.assertEqual(result.forecast[-1], 1.0)
        numeric.validate(result.forecast, False, False, 201)
        self.assertIn("cannot be below 50", llm.forecast_calls()[0]["messages"][1]["content"])

    def test_discrete(self):
        answer = "\n".join(
            f"Percentile {p}: {v}"
            for p, v in zip((1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99), (0, 1, 1, 2, 2, 3, 3, 3, 4, 5, 6, 7, 9))
        )
        llm = fakes.FakeLLM(by_model({config.OPUS: answer, config.GEMINI_FLASH: answer}))
        question = fakes.discrete_question()
        result = forecast_question(llm, question, LITE, fakes.NOW)
        self.assertEqual(len(result.forecast), 12)
        numeric.validate(result.forecast, False, False, 12)
        pmf = [b - a for a, b in zip(result.forecast, result.forecast[1:])]
        self.assertEqual(max(range(11), key=lambda k: pmf[k]), 3)
        self.assertIn("one of 11 discrete values: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10", llm.forecast_calls()[0]["messages"][1]["content"])

    def test_numeric_disagreement_calls_adjudicator(self):
        def answer(model, messages):
            if fakes.ADJUDICATOR_MARK in messages[1]["content"]:
                return fakes.percentile_answer(150, 8)
            return {config.OPUS: fakes.percentile_answer(80, 8), config.GPT: fakes.percentile_answer(150, 8),
                    config.GEMINI_FLASH: fakes.percentile_answer(85, 8)}[model]

        result = forecast_question(fakes.FakeLLM(answer), fakes.numeric_question(), FULL, fakes.NOW)
        self.assertIsNotNone(result.adjudicator)
        numeric.validate(result.forecast, True, True, 201)

    def test_fixer_for_percentiles(self):
        fixed = '{"percentiles": {"5": 90, "25": 100, "50": 110, "75": 120, "95": 135}}'
        llm = fakes.FakeLLM(
            by_model({config.OPUS: "Somewhere around 110.", config.GEMINI_FLASH: fakes.percentile_answer(110, 10)}),
            fixer=lambda messages: fixed,
        )
        result = forecast_question(llm, fakes.numeric_question(), LITE, fakes.NOW)
        self.assertEqual(len(result.members), 3)

    def test_question_without_range(self):
        with self.assertRaises(ForecastError):
            forecast_question(fakes.FakeLLM(by_model({})), fakes.numeric_question(range_min=None), LITE, fakes.NOW)

    def test_log_scaled_question(self):
        llm = fakes.FakeLLM(by_model({config.OPUS: fakes.percentile_answer(900, 250), config.GEMINI_FLASH: fakes.percentile_answer(1000, 300)}))
        question = fakes.numeric_question(range_min=10.0, range_max=100000.0, zero_point=0.0, open_lower=False, open_upper=True)
        result = forecast_question(llm, question, LITE, fakes.NOW)
        numeric.validate(result.forecast, False, True, 201)
        median = numeric.Scale(10, 100000, 0).to_value(numeric.loc_at(result.forecast, 0.5))
        self.assertAlmostEqual(median, 950, delta=120)


if __name__ == "__main__":
    unittest.main()
