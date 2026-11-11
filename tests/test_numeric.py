import math
import random
import unittest

from bot import numeric

PCTS = (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99)


def normal_percentiles(mean, sd):
    return {p: mean + sd * numeric._NORMAL.inv_cdf(p / 100.0) for p in PCTS}


class ScaleTest(unittest.TestCase):
    def test_linear_roundtrip(self):
        scale = numeric.Scale(-50, 150)
        for value in (-50, 0, 37.5, 150, 400):
            self.assertAlmostEqual(scale.to_value(scale.to_loc(value)), value, places=9)
        self.assertEqual(scale.to_loc(-50), 0.0)
        self.assertEqual(scale.to_loc(150), 1.0)

    def test_log_roundtrip_and_ends(self):
        scale = numeric.Scale(10, 10000, 0)
        self.assertAlmostEqual(scale.to_loc(10), 0.0, places=12)
        self.assertAlmostEqual(scale.to_loc(10000), 1.0, places=12)
        # On a log scale with zero point 0 the geometric mean sits in the middle.
        self.assertAlmostEqual(scale.to_loc(math.sqrt(10 * 10000)), 0.5, places=9)
        for value in (10, 55, 316.2, 9999, 50000):
            self.assertAlmostEqual(scale.to_value(scale.to_loc(value)), value, places=6)

    def test_log_value_at_or_below_zero_point_is_finite(self):
        scale = numeric.Scale(10, 10000, 0)
        self.assertTrue(math.isfinite(scale.to_loc(0)))
        self.assertTrue(math.isfinite(scale.to_loc(-5)))
        self.assertLess(scale.to_loc(0), 0)

    def test_invalid_ranges(self):
        with self.assertRaises(ValueError):
            numeric.Scale(5, 5)
        with self.assertRaises(ValueError):
            numeric.Scale(10, 100, 10)


class MemberCdfTest(unittest.TestCase):
    def test_reproduces_stated_percentiles(self):
        scale = numeric.Scale(0, 200)
        stated = normal_percentiles(100, 15)
        cdf = numeric.member_cdf(stated, scale, 201)
        for p, value in stated.items():
            index = scale.to_loc(value) * 200
            lower = cdf[int(math.floor(index))]
            upper = cdf[min(200, int(math.ceil(index)))]
            self.assertTrue(lower - 1e-9 <= p / 100.0 <= upper + 1e-9, (p, lower, upper))

    def test_normal_input_gives_normal_curve(self):
        scale = numeric.Scale(0, 200)
        cdf = numeric.member_cdf(normal_percentiles(100, 15), scale, 201)
        for i in range(0, 201, 5):
            expected = numeric._NORMAL.cdf((i - 100) / 15)
            self.assertAlmostEqual(cdf[i], expected, places=3)

    def test_monotone_and_bounded(self):
        rng = random.Random(7)
        for _ in range(300):
            lo = rng.uniform(-1000, 1000)
            hi = lo + rng.uniform(0.5, 5000)
            scale = numeric.Scale(lo, hi)
            values = sorted(rng.uniform(lo - 0.3 * (hi - lo), hi + 0.3 * (hi - lo)) for _ in PCTS)
            cdf = numeric.member_cdf(dict(zip(PCTS, values)), scale, 201, widen_weight=0.2)
            self.assertTrue(all(0.0 <= v <= 1.0 for v in cdf))
            self.assertTrue(all(b >= a for a, b in zip(cdf, cdf[1:])))

    def test_tied_values_become_a_jump(self):
        scale = numeric.Scale(0, 100)
        stated = {10: 40, 20: 40, 40: 40, 50: 52, 60: 55, 80: 70, 90: 70}
        cdf = numeric.member_cdf(stated, scale, 201)
        self.assertTrue(all(b >= a for a, b in zip(cdf, cdf[1:])))
        # A run of ties at 40 means the CDF jumps there from about 10% to about 40%.
        self.assertLess(cdf[78], 0.12)
        self.assertGreater(cdf[82], 0.38)

    def test_out_of_order_values_are_refused(self):
        with self.assertRaises(ValueError):
            numeric.member_cdf({10: 40, 50: 55, 60: 50, 90: 70}, numeric.Scale(0, 100), 201)

    def test_probability_stated_at_a_closed_bound_stays_inside(self):
        # 10% of the probability is stated at the lower bound itself, 20% at or below 1.
        scale = numeric.Scale(0, 100)
        stated = {1: 0, 5: 0, 10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 6, 70: 9, 80: 14, 90: 25, 95: 40, 99: 70}
        raw = numeric.member_cdf(stated, scale, 201, open_lower=False, open_upper=False)
        final = numeric.standardize(raw, False, False)
        numeric.validate(final, False, False, 201)
        self.assertAlmostEqual(final[2], 0.20, delta=0.01)  # P(outcome <= 1)
        self.assertAlmostEqual(final[8], 0.50, delta=0.01)  # P(outcome <= 4)
        # the mass at the bound sits in the first bucket, as the server asks for
        self.assertGreater(final[1], 0.09)

    def test_values_beyond_a_closed_bound_are_pulled_onto_it(self):
        scale = numeric.Scale(0, 100)
        stated = {10: 60, 30: 80, 50: 95, 70: 97, 90: 130}  # the top value is impossible
        raw = numeric.member_cdf(stated, scale, 201, open_lower=False, open_upper=False)
        self.assertGreater(raw[-1], 0.9999)  # nothing is left beyond the bound to be rescaled away
        final = numeric.standardize(raw, False, False)
        numeric.validate(final, False, False, 201)
        # the stated percentiles inside the range are kept ...
        self.assertAlmostEqual(scale.to_value(numeric.loc_at(final, 0.3)), 80, delta=1.0)
        self.assertAlmostEqual(scale.to_value(numeric.loc_at(final, 0.5)), 95, delta=1.0)
        # ... and the probability stated at or past the bound sits in the last bucket
        self.assertGreater(final[-1] - final[-2], 0.10)

    def test_without_pinning_the_same_answer_would_be_distorted(self):
        scale = numeric.Scale(0, 100)
        stated = {10: 60, 30: 80, 50: 95, 70: 97, 90: 130}
        open_top = numeric.member_cdf(stated, scale, 201, open_lower=False, open_upper=True)
        self.assertLess(open_top[-1], 0.80)  # on an open bound that probability really is above the range

    def test_more_than_one_bucket_can_hold_is_capped_and_still_valid(self):
        scale = numeric.Scale(0, 100)
        stated = {10: 60, 30: 80, 50: 110, 70: 120, 90: 130}  # half the probability at the bound
        raw = numeric.member_cdf(stated, scale, 201, open_lower=False, open_upper=False)
        final = numeric.standardize(raw, False, False)
        numeric.validate(final, False, False, 201)
        self.assertAlmostEqual(final[-1] - final[-2], 0.19, delta=0.011)  # the most one bucket may hold

    def test_widening_keeps_median_and_fattens_tails(self):
        scale = numeric.Scale(0, 200)
        stated = normal_percentiles(100, 10)
        plain = numeric.member_cdf(stated, scale, 201)
        wide = numeric.member_cdf(stated, scale, 201, widen_weight=0.2, widen_factor=1.8)
        self.assertAlmostEqual(plain[100], 0.5, places=6)
        self.assertAlmostEqual(wide[100], 0.5, places=6)
        self.assertGreater(wide[70], plain[70] * 5)  # three sigma below: far more mass
        self.assertGreater(1 - wide[130], (1 - plain[130]) * 5)

    def test_mass_outside_range_is_kept(self):
        scale = numeric.Scale(0, 100)
        cdf = numeric.member_cdf(normal_percentiles(100, 20), scale, 201)
        self.assertAlmostEqual(cdf[-1], 0.5, places=3)  # half the distribution is above the range

    def test_needs_two_percentiles(self):
        with self.assertRaises(ValueError):
            numeric.member_cdf({50: 10}, numeric.Scale(0, 100), 201)


class CleanPercentilesTest(unittest.TestCase):
    SCALE = numeric.Scale(0, 200)

    def test_accepts_reasonable_answer(self):
        stated = normal_percentiles(100, 15)
        self.assertEqual(numeric.clean_percentiles(stated, self.SCALE), {float(p): v for p, v in stated.items()})

    def test_rejects_wrong_units(self):
        with self.assertRaises(ValueError):
            numeric.clean_percentiles(normal_percentiles(100000, 15000), self.SCALE)

    def test_rejects_too_few_or_one_sided(self):
        with self.assertRaises(ValueError):
            numeric.clean_percentiles({10: 1, 50: 2}, self.SCALE)
        with self.assertRaises(ValueError):
            numeric.clean_percentiles({50: 100, 60: 105, 70: 110, 80: 115, 90: 120}, self.SCALE)
        with self.assertRaises(ValueError):
            numeric.clean_percentiles({0.05: 80, 0.25: 90, 0.5: 100, 0.75: 110, 0.95: 120}, self.SCALE)

    def test_accepts_answer_partly_outside_open_range(self):
        numeric.clean_percentiles(normal_percentiles(210, 30), self.SCALE)

    def test_single_slip_is_dropped_not_sorted_into_the_tail(self):
        stated = dict(normal_percentiles(108, 10))
        stated[40] = 1080.0  # a typo for about 105
        cleaned = numeric.clean_percentiles(stated, self.SCALE)
        self.assertNotIn(40.0, cleaned)
        self.assertEqual(len(cleaned), len(stated) - 1)
        self.assertLess(max(cleaned.values()), 140)

    def test_several_slips_reject_the_answer(self):
        stated = dict(normal_percentiles(108, 10))
        stated[40], stated[60] = 1080.0, 10.0
        with self.assertRaises(ValueError):
            numeric.clean_percentiles(stated, self.SCALE)

    def test_ties_are_not_slips(self):
        stated = {10: 3, 30: 3, 50: 4, 70: 4, 90: 6}
        self.assertEqual(len(numeric.clean_percentiles(stated, numeric.Scale(-0.5, 10.5))), 5)


class StandardizeTest(unittest.TestCase):
    def _final(self, stated, scale, open_lower, open_upper, size=201, discrete=False):
        raw = numeric.member_cdf(stated, scale, size, discrete=discrete, widen_weight=0.2)
        final = numeric.standardize(raw, open_lower, open_upper)
        numeric.validate(final, open_lower, open_upper, size)
        return final

    def test_all_bound_combinations_validate(self):
        scale = numeric.Scale(0, 200)
        for open_lower in (False, True):
            for open_upper in (False, True):
                final = self._final(normal_percentiles(100, 15), scale, open_lower, open_upper)
                self.assertEqual(final[0] == 0.0, not open_lower)
                self.assertEqual(final[-1] == 1.0, not open_upper)
                self.assertAlmostEqual(numeric.loc_at(final, 0.5), 0.5, places=2)

    def test_random_forecasts_always_validate(self):
        rng = random.Random(11)
        for _ in range(400):
            lo = rng.uniform(-100, 100)
            hi = lo + rng.uniform(1, 1000)
            zero = lo - rng.uniform(0.1, 50) if rng.random() < 0.3 else None
            scale = numeric.Scale(lo, hi, zero)
            centre = rng.uniform(lo - 0.2 * (hi - lo), hi + 0.2 * (hi - lo))
            spread = (hi - lo) * 10 ** rng.uniform(-3.5, 0.3)
            stated = normal_percentiles(centre, spread)
            if zero is not None:
                stated = {p: max(v, zero + 1e-6) for p, v in stated.items()}
            open_lower, open_upper = rng.random() < 0.5, rng.random() < 0.5
            raw = numeric.member_cdf(stated, scale, 201, widen_weight=0.2)
            try:
                final = numeric.standardize(raw, open_lower, open_upper)
            except ValueError:
                # Only acceptable when the forecast puts essentially nothing in a closed range.
                self.assertLess(raw[-1] - raw[0], 1e-6)
                continue
            numeric.validate(final, open_lower, open_upper, 201)

    def test_spike_is_capped(self):
        scale = numeric.Scale(0, 200)
        final = self._final(normal_percentiles(100, 0.01), scale, False, False)
        steps = [b - a for a, b in zip(final, final[1:])]
        self.assertLessEqual(max(steps), 0.2 + 1e-9)
        self.assertGreaterEqual(min(steps), 5e-5 - 1e-9)

    def test_discrete_question(self):
        # Outcomes 0..10: range -0.5..10.5 with 11 outcomes -> 12 CDF points.
        scale = numeric.Scale(-0.5, 10.5)
        stated = {1: 0, 5: 1, 10: 1, 20: 2, 30: 2, 40: 3, 50: 3, 60: 3, 70: 4, 80: 5, 90: 6, 95: 7, 99: 9}
        final = self._final(stated, scale, False, False, size=12, discrete=True)
        pmf = [b - a for a, b in zip(final, final[1:])]
        self.assertEqual(len(pmf), 11)
        self.assertAlmostEqual(sum(pmf), 1.0, places=9)
        self.assertEqual(max(range(11), key=lambda k: pmf[k]), 3)  # mode at the outcome stated most often
        self.assertGreater(pmf[3], 0.2)
        self.assertGreaterEqual(min(pmf), 0.01 / 11 - 1e-9)

    def test_standardize_rejects_forecast_entirely_outside_closed_range(self):
        scale = numeric.Scale(0, 100)
        raw = numeric.member_cdf(normal_percentiles(100000, 1), scale, 201)
        with self.assertRaises(ValueError):
            numeric.standardize(raw, False, False)

    def test_validate_uses_the_server_rounding(self):
        # Three outcomes: the minimum step 0.01 / 3 is not representable in 10 decimals.
        scale = numeric.Scale(-0.5, 2.5)
        for stated in ({10: 0, 30: 0, 50: 1, 70: 1, 90: 2}, {10: 2, 30: 2, 50: 2, 70: 2, 90: 2}):
            raw = numeric.member_cdf(stated, scale, 4, discrete=True, open_lower=False, open_upper=False)
            final = numeric.standardize(raw, False, False)
            numeric.validate(final, False, False, 4)
        numeric.validate([0.0, 0.0033333333, 0.5, 1.0], False, False, 4)
        with self.assertRaises(ValueError):
            numeric.validate([0.0, 0.0033333, 0.5, 1.0], False, False, 4)
        with self.assertRaises(ValueError):
            numeric.validate([0.0, 0.2, 0.5, 0.99999], False, False, 4)  # closed top must be exactly 1

    def test_validate_catches_bad_cdfs(self):
        good = self._final(normal_percentiles(100, 15), numeric.Scale(0, 200), True, True)
        numeric.validate(good, True, True, 201)
        with self.assertRaises(ValueError):
            numeric.validate(good[:-1], True, True, 201)
        flat = list(good)
        flat[50] = flat[49]
        with self.assertRaises(ValueError):
            numeric.validate(flat, True, True, 201)
        with self.assertRaises(ValueError):
            numeric.validate(good, False, True, 201)  # closed lower bound must start at 0
        with self.assertRaises(ValueError):
            numeric.validate([0.0] + [1.0] * 200, False, False, 201)


class MixtureTest(unittest.TestCase):
    def test_weighted_average(self):
        a, b = [0.0, 0.2, 1.0], [0.0, 0.8, 1.0]
        self.assertEqual(numeric.mixture([a, b], [1, 1])[1], 0.5)
        self.assertAlmostEqual(numeric.mixture([a, b], [3, 1])[1], 0.35)

    def test_mixture_of_two_opinions_is_bimodal_and_valid(self):
        scale = numeric.Scale(0, 200)
        low = numeric.member_cdf(normal_percentiles(60, 5), scale, 201)
        high = numeric.member_cdf(normal_percentiles(140, 5), scale, 201)
        final = numeric.standardize(numeric.mixture([low, high], [1, 1]), True, True)
        numeric.validate(final, True, True, 201)
        self.assertAlmostEqual(final[100], 0.5, places=2)

    def test_errors(self):
        with self.assertRaises(ValueError):
            numeric.mixture([], [])
        with self.assertRaises(ValueError):
            numeric.mixture([[0, 1]], [0])
        with self.assertRaises(ValueError):
            numeric.mixture([[0, 1], [0, 0.5, 1]], [1, 1])


class QuantileTest(unittest.TestCase):
    def test_quantiles_recover_inputs(self):
        scale = numeric.Scale(0, 200)
        cdf = numeric.member_cdf(normal_percentiles(100, 15), scale, 201)
        low, median, high = numeric.quantiles(cdf, scale, [0.05, 0.5, 0.95])
        self.assertAlmostEqual(median, 100, delta=0.2)
        self.assertAlmostEqual(low, 100 - 1.645 * 15, delta=0.5)
        self.assertAlmostEqual(high, 100 + 1.645 * 15, delta=0.5)

    def test_loc_at_clips(self):
        self.assertEqual(numeric.loc_at([0.3, 0.6, 0.9], 0.1), 0.0)
        self.assertEqual(numeric.loc_at([0.3, 0.6, 0.9], 0.95), 1.0)
        self.assertAlmostEqual(numeric.loc_at([0.3, 0.6, 0.9], 0.45), 0.25)


if __name__ == "__main__":
    unittest.main()
