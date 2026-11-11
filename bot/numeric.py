"""Numeric and discrete forecasts: stated percentiles -> a CDF Metaculus accepts.

Each forecaster states a handful of percentiles. They are turned into a smooth
distribution by interpolating the probit of the cumulative probability against the
question's own x-axis (monotone cubic interpolation, straight-line tails, i.e. tails
that fall off like a normal distribution). Member distributions are then mixed and the
mixture is standardised to the constraints of the Metaculus API.
"""

from __future__ import annotations

import bisect
import math
from statistics import NormalDist
from typing import Mapping, Sequence

_NORMAL = NormalDist()

# Largest allowed step of a 201-point CDF; it scales with the number of points.
_MAX_STEP_201 = 0.2
# Closed bounds: cumulative probability pinned at the bound, and how much probability
# beyond a closed bound is tolerated before pinning.
_ANCHOR = 1e-6
_CLOSED_BOUND_SLACK = 0.005


class Scale:
    """Maps real-world values to the question's 0..1 x-axis (linear or log-scaled)."""

    def __init__(self, range_min: float, range_max: float, zero_point: float | None = None):
        if not (math.isfinite(range_min) and math.isfinite(range_max)) or range_max <= range_min:
            raise ValueError(f"invalid question range: {range_min}..{range_max}")
        self.lo = float(range_min)
        self.hi = float(range_max)
        self.zero = None if zero_point is None else float(zero_point)
        self._ratio = 1.0
        if self.zero is not None:
            if self.lo <= self.zero:
                raise ValueError("log-scaled question with zero point at or above the lower bound")
            self._ratio = (self.hi - self.zero) / (self.lo - self.zero)

    def to_loc(self, value: float) -> float:
        if self.zero is None:
            return (value - self.lo) / (self.hi - self.lo)
        floor = self.zero + (self.lo - self.zero) * 1e-9
        value = max(value, floor)
        argument = (value - self.lo) * (self._ratio - 1.0) + (self.hi - self.lo)
        return (math.log(argument) - math.log(self.hi - self.lo)) / math.log(self._ratio)

    def to_value(self, loc: float) -> float:
        if self.zero is None:
            return self.lo + (self.hi - self.lo) * loc
        return self.lo + (self.hi - self.lo) * (self._ratio**loc - 1.0) / (self._ratio - 1.0)


def _spread_ties(
    xs: list[float], half_width: float, lower: float | None = None, upper: float | None = None
) -> list[float]:
    """Make x strictly increasing. Runs of equal values are spread around their value.

    lower / upper are closed bounds: a run sitting on one is spread inwards, so that the
    probability the forecaster put on the bound itself stays inside the range.
    """
    n = len(xs)
    out = list(xs)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(xs[j + 1] - xs[i]) <= 1e-12:
            j += 1
        if j > i:
            width = half_width
            if i > 0:
                width = min(width, 0.4 * (xs[i] - xs[i - 1]))
            if j + 1 < n:
                width = min(width, 0.4 * (xs[j + 1] - xs[i]))
            width = max(width, 1e-9)
            start, end = xs[i] - width, xs[i] + width
            if lower is not None and start < lower:
                start, end = lower, end + (lower - start)
            if upper is not None and end > upper:
                start, end = start - (end - upper), upper
            count = j - i
            for k in range(i, j + 1):
                out[k] = start + (end - start) * (k - i) / count
        i = j + 1
    for k in range(1, n):
        if out[k] <= out[k - 1]:
            out[k] = out[k - 1] + 1e-9
    return out


def _edge_slope(h0: float, h1: float, d0: float, d1: float) -> float:
    slope = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if slope * d0 <= 0.0:
        return 0.0
    if d0 * d1 <= 0.0 and abs(slope) > 3.0 * abs(d0):
        return 3.0 * d0
    return slope


def _pchip_slopes(xs: Sequence[float], ys: Sequence[float]) -> list[float]:
    """Derivatives of the shape-preserving piecewise cubic through the points."""
    n = len(xs)
    h = [xs[i + 1] - xs[i] for i in range(n - 1)]
    d = [(ys[i + 1] - ys[i]) / h[i] for i in range(n - 1)]
    if n == 2:
        return [d[0], d[0]]
    slopes = [0.0] * n
    for i in range(1, n - 1):
        if d[i - 1] * d[i] > 0.0:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            slopes[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    slopes[0] = _edge_slope(h[0], h[1], d[0], d[1])
    slopes[-1] = _edge_slope(h[-1], h[-2], d[-1], d[-2])
    return slopes


class _Curve:
    """Monotone cubic through (x, z) with straight-line continuation beyond both ends."""

    def __init__(self, xs: Sequence[float], zs: Sequence[float]):
        self.xs = list(xs)
        self.zs = list(zs)
        self.slopes = _pchip_slopes(self.xs, self.zs)
        self.left = (self.zs[1] - self.zs[0]) / (self.xs[1] - self.xs[0])
        self.right = (self.zs[-1] - self.zs[-2]) / (self.xs[-1] - self.xs[-2])

    def __call__(self, x: float) -> float:
        xs, zs = self.xs, self.zs
        if x <= xs[0]:
            return zs[0] + (x - xs[0]) * self.left
        if x >= xs[-1]:
            return zs[-1] + (x - xs[-1]) * self.right
        i = bisect.bisect_right(xs, x) - 1
        h = xs[i + 1] - xs[i]
        t = (x - xs[i]) / h
        t2, t3 = t * t, t * t * t
        return (
            (2.0 * t3 - 3.0 * t2 + 1.0) * zs[i]
            + (t3 - 2.0 * t2 + t) * h * self.slopes[i]
            + (-2.0 * t3 + 3.0 * t2) * zs[i + 1]
            + (t3 - t2) * h * self.slopes[i + 1]
        )


def _longest_non_decreasing(values: Sequence[float]) -> list[int]:
    """Indexes of a longest non-decreasing subsequence."""
    n = len(values)
    best = [1] * n
    previous = [-1] * n
    for i in range(n):
        for j in range(i):
            if values[j] <= values[i] and best[j] + 1 > best[i]:
                best[i] = best[j] + 1
                previous[i] = j
    index = max(range(n), key=lambda k: best[k])
    keep: list[int] = []
    while index != -1:
        keep.append(index)
        index = previous[index]
    return keep[::-1]


def clean_percentiles(percentiles: Mapping[float, float], scale: Scale) -> dict[float, float]:
    """Validate one forecaster's percentiles and return the usable ones.

    Raises ValueError for answers that cannot be trusted: too few points, no spread
    around the median, values out of order, or values in the wrong units. A single value
    that breaks the ordering (a typo such as 1080 for 108) is dropped rather than
    sorted into the tail, which would silently change the whole distribution.
    """
    points = sorted((float(p), float(v)) for p, v in percentiles.items() if 0.0 < p < 100.0 and math.isfinite(v))
    if len(points) < 3:
        raise ValueError("fewer than three usable percentiles")
    if points[0][0] > 40.0 or points[-1][0] < 60.0:
        raise ValueError("percentiles do not cover both sides of the median")
    keep = _longest_non_decreasing([v for _, v in points])
    dropped = len(points) - len(keep)
    if dropped > max(1, len(points) // 10):
        raise ValueError("percentile values are not in increasing order")
    points = [points[i] for i in keep]
    if len(points) < 3 or points[0][0] > 40.0 or points[-1][0] < 60.0:
        raise ValueError("too few usable percentiles after removing out-of-order values")
    locs = [scale.to_loc(v) for _, v in points]
    if all(loc < -0.25 or loc > 1.25 for loc in locs):
        raise ValueError("every percentile is far outside the question range (wrong units?)")
    middle = locs[len(locs) // 2]
    if middle < -1.0 or middle > 2.0:
        raise ValueError("central estimate is far outside the question range (wrong units?)")
    return dict(points)


def member_cdf(
    percentiles: Mapping[float, float],
    scale: Scale,
    cdf_size: int,
    *,
    discrete: bool = False,
    open_lower: bool = True,
    open_upper: bool = True,
    widen_weight: float = 0.0,
    widen_factor: float = 1.8,
) -> list[float]:
    """CDF of one forecaster on the question grid, before standardisation.

    Values must not decrease with the percentile (see clean_percentiles). On an open
    side the result may put probability outside the range: values below the first grid
    point show up as cdf[0] > 0 and values above the last one as cdf[-1] < 1. On a
    closed side, stated values are pulled onto the bound and probability stated at the
    bound is kept just inside it.
    """
    points = sorted((p / 100.0, float(v)) for p, v in percentiles.items() if 0.0 < p < 100.0 and math.isfinite(v))
    if len(points) < 2:
        raise ValueError("need at least two percentiles")
    probabilities = [p for p, _ in points]
    values = [v for _, v in points]
    if any(b < a for a, b in zip(values, values[1:])):
        raise ValueError("percentile values are not in increasing order")
    lower = None if open_lower else 0.0
    upper = None if open_upper else 1.0
    locs = [scale.to_loc(v) for v in values]
    if lower is not None:
        locs = [max(loc, lower) for loc in locs]
    if upper is not None:
        locs = [min(loc, upper) for loc in locs]
    half_width = 0.45 / (cdf_size - 1) if discrete else 5e-4

    def build(anchor_lower: bool, anchor_upper: bool) -> _Curve:
        ps, xs = list(probabilities), list(locs)
        if anchor_lower and ps[0] > _ANCHOR:
            ps.insert(0, _ANCHOR)
            xs.insert(0, 0.0)
        if anchor_upper and ps[-1] < 1.0 - _ANCHOR:
            ps.append(1.0 - _ANCHOR)
            xs.append(1.0)
        return _Curve(_spread_ties(xs, half_width, lower, upper), [_NORMAL.inv_cdf(p) for p in ps])

    curve = build(False, False)
    # Nothing can lie beyond a closed bound. If the stated percentiles leave noticeable
    # probability there (values at or past the bound), pin the curve to the bound so that
    # this probability lands in the bucket next to it instead of being rescaled away.
    anchor_lower = lower is not None and _NORMAL.cdf(curve(0.0)) > _CLOSED_BOUND_SLACK
    anchor_upper = upper is not None and 1.0 - _NORMAL.cdf(curve(1.0)) > _CLOSED_BOUND_SLACK
    if anchor_lower or anchor_upper:
        curve = build(anchor_lower, anchor_upper)

    cdf: list[float] = []
    for i in range(cdf_size):
        z = curve(i / (cdf_size - 1))
        value = _NORMAL.cdf(z)
        if widen_weight > 0.0:
            value = (1.0 - widen_weight) * value + widen_weight * _NORMAL.cdf(z / widen_factor)
        cdf.append(value)
    for i in range(1, cdf_size):
        if cdf[i] < cdf[i - 1]:
            cdf[i] = cdf[i - 1]
    return cdf


def mixture(cdfs: Sequence[Sequence[float]], weights: Sequence[float]) -> list[float]:
    if not cdfs or len(cdfs) != len(weights):
        raise ValueError("mixture needs one weight per distribution")
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("weights must be positive")
    size = len(cdfs[0])
    if any(len(cdf) != size for cdf in cdfs):
        raise ValueError("distributions of different sizes")
    return [sum(w * cdf[i] for cdf, w in zip(cdfs, weights)) / total for i in range(size)]


def _max_step(size: int, wiggle: float = 1.0) -> float:
    return _MAX_STEP_201 * (200.0 / (size - 1)) * wiggle


def standardize(cdf: Sequence[float], open_lower: bool, open_upper: bool) -> list[float]:
    """Apply the standardisation Metaculus documents for API forecasts.

    - no probability outside closed bounds, at least 0.1% outside open ones
    - every step rises by at least 0.01 / (points - 1)
    - no step rises by more than 0.2 (scaled for discrete questions)
    """
    n = len(cdf)
    lower_to = 0.0 if open_lower else cdf[0]
    upper_to = 1.0 if open_upper else cdf[-1]
    inbound = upper_to - lower_to
    if inbound <= 1e-9:
        raise ValueError("no probability inside the question range")

    shaped: list[float] = []
    for i, height in enumerate(cdf):
        loc = i / (n - 1)
        rescaled = (height - lower_to) / inbound
        if open_lower and open_upper:
            shaped.append(0.988 * rescaled + 0.01 * loc + 0.001)
        elif open_lower:
            shaped.append(0.989 * rescaled + 0.01 * loc + 0.001)
        elif open_upper:
            shaped.append(0.989 * rescaled + 0.01 * loc)
        else:
            shaped.append(0.99 * rescaled + 0.01 * loc)

    pmf = [shaped[0]] + [shaped[i] - shaped[i - 1] for i in range(1, n)] + [1.0 - shaped[-1]]
    cap = _max_step(n, wiggle=0.95)

    def capped(scale: float) -> list[float]:
        return [pmf[0]] + [min(cap, scale * mass) for mass in pmf[1:-1]] + [pmf[-1]]

    low = high = scale = 1.0
    while sum(capped(high)) < 1.0 and high < 1e9:
        high *= 1.2
    for _ in range(100):
        scale = 0.5 * (low + high)
        total = sum(capped(scale))
        if total < 1.0:
            low = scale
        else:
            high = scale
        if total == 1.0 or (high - low) < 2e-5:
            break
    final = capped(scale)
    inner = sum(final[1:-1])
    factor = (shaped[-1] - shaped[0]) / inner
    result: list[float] = []
    running = final[0]
    result.append(running)
    for mass in final[1:-1]:
        running += mass * factor
        result.append(running)
    return [round(value, 10) for value in result]


def validate(cdf: Sequence[float], open_lower: bool, open_upper: bool, cdf_size: int) -> None:
    """Raise ValueError unless the CDF passes the same checks as the Metaculus server.

    Mirrors continuous_validation in the Metaculus backend (questions/serializers/common.py):
    values rounded to 10 decimals, steps rounded to 9, minimum step 0.01 / outcomes
    (rounded to 9), maximum step 0.2 * 200 / outcomes, exact 0 and 1 at closed bounds.
    """
    if len(cdf) != cdf_size:
        raise ValueError(f"cdf has {len(cdf)} points, expected {cdf_size}")
    if any(not math.isfinite(v) for v in cdf):
        raise ValueError("cdf contains a non-finite value")
    values = [round(v, 10) for v in cdf]
    if any(v < 0.0 or v > 1.0 for v in values):
        raise ValueError("cdf value outside 0..1")
    outcomes = cdf_size - 1
    min_step = round(0.01 / outcomes, 9)
    max_step = _MAX_STEP_201 * 200.0 / outcomes
    for i in range(1, cdf_size):
        step = round(values[i] - values[i - 1], 9)
        if step < min_step:
            raise ValueError(f"cdf step {i} too small: {step}")
        if step > max_step:
            raise ValueError(f"cdf step {i} too large: {step}")
    if open_lower:
        if values[0] < 0.001:
            raise ValueError("open lower bound needs at least 0.1% below the range")
    elif values[0] != 0.0:
        raise ValueError("closed lower bound must start at exactly 0")
    if open_upper:
        if values[-1] > 0.999:
            raise ValueError("open upper bound needs at least 0.1% above the range")
    elif values[-1] != 1.0:
        raise ValueError("closed upper bound must end at exactly 1")


def loc_at(cdf: Sequence[float], probability: float) -> float:
    """Location on the 0..1 axis where the CDF reaches the given probability."""
    n = len(cdf)
    if probability <= cdf[0]:
        return 0.0
    if probability >= cdf[-1]:
        return 1.0
    i = bisect.bisect_left(list(cdf), probability)
    lower, upper = cdf[i - 1], cdf[i]
    fraction = 0.0 if upper <= lower else (probability - lower) / (upper - lower)
    return (i - 1 + fraction) / (n - 1)


def quantiles(cdf: Sequence[float], scale: Scale, probabilities: Sequence[float]) -> list[float]:
    """Real-world values at the given cumulative probabilities (clipped to the range)."""
    return [scale.to_value(loc_at(cdf, p)) for p in probabilities]
