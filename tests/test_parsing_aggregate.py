import datetime as dt
import unittest

from bot import aggregate, config, parsing


class ProbabilityParsingTest(unittest.TestCase):
    def test_final_line(self):
        self.assertAlmostEqual(parsing.parse_probability("blah 40% blah\nProbability: 12.5%"), 0.125)

    def test_last_statement_wins(self):
        text = "Probability: 80%\nOn reflection that is too high.\nProbability: 35%"
        self.assertAlmostEqual(parsing.parse_probability(text), 0.35)

    def test_markdown_and_qualifiers(self):
        self.assertAlmostEqual(parsing.parse_probability("**Probability:** 7%"), 0.07)
        self.assertAlmostEqual(parsing.parse_probability("Probability: **3.5%**"), 0.035)
        self.assertAlmostEqual(parsing.parse_probability("Probability: <1%"), 0.01)
        self.assertAlmostEqual(parsing.parse_probability("probability = 62 %."), 0.62)
        self.assertAlmostEqual(parsing.parse_probability("Probability: 12,5%"), 0.125)
        self.assertAlmostEqual(parsing.parse_probability("Probability: 35%\n\n(End of forecast.)"), 0.35)

    def test_a_later_number_cancels_the_answer_line(self):
        self.assertIsNone(parsing.parse_probability("Probability: 40%\nOn reflection 20 is better."))
        self.assertIsNone(parsing.parse_probability("Probability: 40%\nOne.\nTwo.\nThree."))
        self.assertAlmostEqual(parsing.parse_probability("Probability: 80%\nToo high.\nProbability: 35%"), 0.35)

    def test_only_a_clean_final_line_counts(self):
        # Each of these once produced a confident wrong number; all must go to the fixer instead.
        cases = [
            "Reasoning.\nSo my answer is 22%.",
            "Polls show 22% support.\nHard to say.",
            "Between 20% and 30%",
            "probability = about 62 %",
            "Scenario A (probability: 15%) and B (probability: 85%).\nProbability (final): 15%",
            "Polymarket implied probability: 62%\nI disagree.\nProbability: 0.12",
            "Reasoning.\nProbability of No: 88%",
            "Reasoning.\nProbability of Yes: 12%",
            "Probability: 40%\nThen I looked again.\nMore thoughts.\nEven more.\nI now think it is lower.",
            "The base rate (probability: 30%) seems right.",
        ]
        for text in cases:
            self.assertIsNone(parsing.parse_probability(text), text)

    def test_rejects_out_of_range_and_missing(self):
        self.assertIsNone(parsing.parse_probability("Probability: 250%"))
        self.assertIsNone(parsing.parse_probability("no number here"))
        self.assertIsNone(parsing.parse_probability(""))


class NumberParsingTest(unittest.TestCase):
    def test_plain_and_decorated(self):
        cases = {
            "42": 42.0,
            "-3.5": -3.5,
            "1,234,567.8": 1234567.8,
            "$1,200": 1200.0,
            "-$5": -5.0,
            "12.5%": 12.5,
            "**7.25**": 7.25,
            "1e6": 1000000.0,
            "−3": -3.0,
            "1 234": 1234.0,
            "15 (roughly)": 15.0,
            ".5": 0.5,
        }
        for raw, expected in cases.items():
            self.assertEqual(parsing.parse_number(raw), expected, raw)

    def test_ambiguous_is_rejected(self):
        for raw in ("3.2 million", "about 5", "5-6", "1,23", "n/a", "", "12 USD", "3k"):
            self.assertIsNone(parsing.parse_number(raw), raw)


class PercentileParsingTest(unittest.TestCase):
    def test_standard_block(self):
        text = "Reasoning...\n" + "\n".join(f"Percentile {p}: {p * 2}" for p in config.PERCENTILES)
        parsed = parsing.parse_percentiles(text)
        self.assertEqual(len(parsed), len(config.PERCENTILES))
        self.assertEqual(parsed[50.0], 100.0)

    def test_variants(self):
        text = "- **Percentile 10**: 1,000\n* Percentile 20 = 1,500\nPercentile 40: $2,000\nPercentile 60: 2500 (median-ish)\npercentile 90: 4000"
        parsed = parsing.parse_percentiles(text)
        self.assertEqual(parsed, {10.0: 1000.0, 20.0: 1500.0, 40.0: 2000.0, 60.0: 2500.0, 90.0: 4000.0})

    def test_last_block_wins(self):
        first = "\n".join(f"Percentile {p}: {p}" for p in (10, 20, 40, 60, 80, 90))
        second = "\n".join(f"Percentile {p}: {p + 100}" for p in (10, 20, 40, 60, 80, 90))
        self.assertEqual(parsing.parse_percentiles(first + "\nRevised:\n" + second)[10.0], 110.0)

    def test_negative_values(self):
        text = "\n".join(f"Percentile {p}: {v}" for p, v in ((10, -5.5), (30, -2), (50, 0), (70, 1.5), (90, 4)))
        self.assertEqual(parsing.parse_percentiles(text)[10.0], -5.5)

    def test_too_few_or_ambiguous(self):
        self.assertIsNone(parsing.parse_percentiles("Percentile 10: 1\nPercentile 90: 5"))
        text = "\n".join(f"Percentile {p}: {p} million" for p in (10, 20, 40, 60, 80, 90))
        self.assertIsNone(parsing.parse_percentiles(text))

    def test_never_merges_a_draft_with_a_partial_final_list(self):
        draft = "\n".join(f"Percentile {p}: {p}" for p in (10, 20, 40, 60, 80, 90))
        partial = "Percentile 10: 500\nPercentile 50: 600\nPercentile 90: 700"
        self.assertIsNone(parsing.parse_percentiles(draft + "\nOn reflection:\n" + partial))

    def test_one_unreadable_line_rejects_the_block(self):
        lines = [f"Percentile {p}: {p}" for p in (10, 20, 40, 60, 80, 90)]
        lines[2] = "Percentile 40: roughly 40"
        self.assertIsNone(parsing.parse_percentiles("\n".join(lines)))

    def test_blank_lines_inside_the_block_are_fine(self):
        text = "Percentile 10: 1\n\nPercentile 30: 2\nPercentile 50: 3\n\nPercentile 70: 4\nPercentile 90: 5\nDone."
        self.assertEqual(len(parsing.parse_percentiles(text)), 5)

    def test_prose_mentions_are_ignored(self):
        text = "The 90th percentile of past values was 12.\n" + "\n".join(
            f"Percentile {p}: {p}" for p in (10, 20, 40, 60, 80, 90)
        )
        self.assertEqual(parsing.parse_percentiles(text)[90.0], 90.0)


class OptionParsingTest(unittest.TestCase):
    OPTIONS = ["0-5", "6-10", "More than 10"]

    def test_standard(self):
        parsed = parsing.parse_options("0-5: 20%\n6-10: 50%\nMore than 10: 30%", self.OPTIONS)
        self.assertAlmostEqual(parsed["6-10"], 0.5)
        self.assertAlmostEqual(sum(parsed.values()), 1.0)

    def test_decorated_lines(self):
        text = "Final:\n- **0-5**: 20%\n- Option 6-10: 50 %\n3. More than 10: 30%"
        parsed = parsing.parse_options(text, self.OPTIONS)
        self.assertAlmostEqual(parsed["More than 10"], 0.3)

    def test_fractions_and_renormalisation(self):
        parsed = parsing.parse_options("0-5: 0.2\n6-10: 0.5\nMore than 10: 0.3", self.OPTIONS)
        self.assertAlmostEqual(parsed["0-5"], 0.2)
        parsed = parsing.parse_options("0-5: 21%\n6-10: 50%\nMore than 10: 31%", self.OPTIONS)
        self.assertAlmostEqual(sum(parsed.values()), 1.0)

    def test_last_statement_wins(self):
        text = "0-5: 90%\n6-10: 5%\nMore than 10: 5%\nRevised:\n0-5: 20%\n6-10: 50%\nMore than 10: 30%"
        self.assertAlmostEqual(parsing.parse_options(text, self.OPTIONS)["0-5"], 0.2)

    def test_never_merges_a_draft_with_the_final_answer(self):
        parties = ["Republican", "Democrat", "Other"]
        draft = "Republican: 45%\nDemocrat: 50%\nOther: 5%\n"
        cases = [
            draft + "Final:\nRepublican: 60% (previous estimate: 45%)\nDemocrat: 35%\nOther: 5%",
            draft + "Final answer:\nRepublican - 60%\nDemocrat - 35%\nOther - 5%",
            draft + "Final:\nRepublican: 60%\nDemocrat: 35%",
            "Republican: 55c / 30c / 16c on the market\nDemocrat: 30c\nOther: 16c",
            "Republican: 52 seats\nDemocrat: 46 seats\nOther: 2 seats",
            "Republican: 60%\nDemocrat: 35%\nOther: 5%\nRepublican: 50%",
        ]
        for text in cases:
            self.assertIsNone(parsing.parse_options(text, parties), text)

    def test_only_a_short_remark_without_numbers_may_follow(self):
        block = "0-5: 20%\n6-10: 50%\nMore than 10: 30%\n"
        self.assertAlmostEqual(parsing.parse_options(block + "\nThat is my final answer.", self.OPTIONS)["6-10"], 0.5)
        self.assertIsNone(parsing.parse_options(block + "\nActually make the first one 25.", self.OPTIONS))
        self.assertIsNone(parsing.parse_options(block + "\nOne.\nTwo.\nThree.", self.OPTIONS))

    def test_missing_option_or_bad_total(self):
        self.assertIsNone(parsing.parse_options("0-5: 20%\n6-10: 50%", self.OPTIONS))
        self.assertIsNone(parsing.parse_options("0-5: 20%\n6-10: 20%\nMore than 10: 20%", self.OPTIONS))
        self.assertIsNone(parsing.parse_options("0-5: 30%\n6-10: 30%\nMore than 10: 30%", self.OPTIONS))

    def test_option_names_with_colons_and_case(self):
        options = ["Before 10:00", "After 10:00"]
        parsed = parsing.parse_options("before 10:00: 35%\nAFTER 10:00: 65%", options)
        self.assertAlmostEqual(parsed["Before 10:00"], 0.35)

    def test_yes_no_style_names(self):
        options = ["Party A", "Party B", "Other"]
        parsed = parsing.parse_options("Party A: 60%\nParty B: 39%\nOther: 1%", options)
        self.assertAlmostEqual(parsed["Other"], 0.01)


class JsonExtractionTest(unittest.TestCase):
    def test_plain_and_wrapped(self):
        self.assertEqual(parsing.extract_json('{"a": 1}'), {"a": 1})
        self.assertEqual(parsing.extract_json('Here you go:\n```json\n{"a": {"b": "}"}}\n```'), {"a": {"b": "}"}})
        self.assertEqual(parsing.extract_json("not json {oops} then {\"ok\": true}"), {"ok": True})
        self.assertIsNone(parsing.extract_json("nothing"))


class BinaryAggregationTest(unittest.TestCase):
    def test_between_members_and_symmetric(self):
        p = aggregate.aggregate_binary([0.2, 0.3, 0.25], [1, 1, 1])
        self.assertTrue(0.2 < p < 0.3)
        q = aggregate.aggregate_binary([0.8, 0.7, 0.75], [1, 1, 1])
        self.assertAlmostEqual(p + q, 1.0, places=9)

    def test_identical_members(self):
        self.assertAlmostEqual(aggregate.aggregate_binary([0.37] * 4, [1, 2, 3, 4]), 0.37, places=9)

    def test_weights_matter(self):
        low = aggregate.aggregate_binary([0.1, 0.6], [3, 1])
        high = aggregate.aggregate_binary([0.1, 0.6], [1, 3])
        self.assertLess(low, high)

    def test_outlier_has_limited_pull(self):
        base = aggregate.aggregate_binary([0.10, 0.12, 0.11, 0.10], [1, 1, 1, 1])
        with_outlier = aggregate.aggregate_binary([0.10, 0.12, 0.11, 0.999999], [1, 1, 1, 1])
        self.assertLess(with_outlier, 0.25)
        self.assertGreater(with_outlier, base)

    def test_scale_extremises(self):
        plain = aggregate.aggregate_binary([0.7, 0.8], [1, 1])
        sharp = aggregate.aggregate_binary([0.7, 0.8], [1, 1], scale=1.5)
        self.assertGreater(sharp, plain)

    def test_blend_and_spread(self):
        self.assertAlmostEqual(aggregate.blend_binary(0.2, 0.2, 0.4), 0.2, places=9)
        blended = aggregate.blend_binary(0.2, 0.6, 0.4)
        self.assertTrue(0.2 < blended < 0.6)
        logit_range, prob_range = aggregate.binary_spread([0.05, 0.19])
        self.assertAlmostEqual(prob_range, 0.14)
        self.assertAlmostEqual(logit_range, 1.494, places=2)

    def test_weighted_median(self):
        self.assertEqual(aggregate.weighted_median([1, 2, 3], [1, 1, 1]), 2)
        self.assertEqual(aggregate.weighted_median([1, 2, 3, 4], [1, 1, 1, 1]), 2.5)
        self.assertEqual(aggregate.weighted_median([1, 2, 10], [1, 1, 5]), 10)
        with self.assertRaises(ValueError):
            aggregate.weighted_median([], [])


class OptionAggregationTest(unittest.TestCase):
    def test_pool_and_floor(self):
        options = ["a", "b", "c"]
        pooled = aggregate.aggregate_options(
            [{"a": 0.9, "b": 0.1, "c": 0.0}, {"a": 0.5, "b": 0.5, "c": 0.0}], [1, 1], options
        )
        self.assertAlmostEqual(pooled["a"], 0.7)
        final = aggregate.floor_and_normalize(pooled, 0.01)
        self.assertAlmostEqual(sum(final.values()), 1.0, places=12)
        self.assertAlmostEqual(final["c"], 0.01, places=12)
        self.assertAlmostEqual(final["a"] / final["b"], 0.7 / 0.3, places=9)

    def test_floor_with_many_options(self):
        options = [f"o{i}" for i in range(80)]
        final = aggregate.floor_and_normalize({o: (1.0 if o == "o0" else 0.0) for o in options}, 0.01)
        self.assertAlmostEqual(sum(final.values()), 1.0, places=9)
        self.assertGreater(min(final.values()), 0.0)
        self.assertGreater(final["o0"], 0.4)

    def test_all_zero_becomes_uniform(self):
        final = aggregate.floor_and_normalize({"a": 0.0, "b": 0.0}, 0.01)
        self.assertAlmostEqual(final["a"], 0.5)

    def test_spread_and_blend(self):
        options = ["a", "b"]
        self.assertAlmostEqual(aggregate.options_spread([{"a": 0.9, "b": 0.1}, {"a": 0.4, "b": 0.6}], options), 0.5)
        blended = aggregate.blend_options({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}, 0.4, options)
        self.assertAlmostEqual(blended["a"], 0.6)


class ConfigTest(unittest.TestCase):
    def test_season_slugs(self):
        self.assertEqual(config.seasonal_tournament_ids(dt.date(2026, 11, 11)), ["fall-futureeval-2026"])
        self.assertEqual(
            config.seasonal_tournament_ids(dt.date(2027, 1, 3)), ["spring-futureeval-2027", "fall-futureeval-2026"]
        )
        self.assertEqual(config.seasonal_tournament_ids(dt.date(2027, 3, 1)), ["spring-futureeval-2027"])
        self.assertEqual(
            config.seasonal_tournament_ids(dt.date(2027, 5, 20)), ["summer-futureeval-2027", "spring-futureeval-2027"]
        )
        self.assertEqual(
            config.seasonal_tournament_ids(dt.date(2027, 9, 30)), ["fall-futureeval-2027", "summer-futureeval-2027"]
        )

    def test_tiers_are_consistent(self):
        for tier in config.TIERS.values():
            self.assertTrue(tier.members and tier.research)
            self.assertTrue(0.0 < tier.prob_floor < 0.1 and 0.9 < tier.prob_ceil < 1.0)
            for member in tier.members:
                self.assertIn(member.effort, ("low", "medium", "high"))
                self.assertGreater(member.weight, 0)
        self.assertIn(config.SEASONAL_TIER, config.TIERS)
        self.assertIn(config.MINIBENCH_TIER, config.TIERS)


if __name__ == "__main__":
    unittest.main()
