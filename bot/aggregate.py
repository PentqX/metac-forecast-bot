"""Combining the forecasts of several ensemble members."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def logit(p: float) -> float:
    p = clamp(p, 1e-6, 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def expit(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    pairs = sorted(zip(values, weights))
    total = sum(w for _, w in pairs)
    if total <= 0 or not pairs:
        raise ValueError("weighted_median needs positive weights")
    half = total / 2.0
    running = 0.0
    for index, (value, weight) in enumerate(pairs):
        running += weight
        if running > half + 1e-12:
            return value
        if abs(running - half) <= 1e-12 and index + 1 < len(pairs):
            return (value + pairs[index + 1][0]) / 2.0
    return pairs[-1][0]


def aggregate_binary(probabilities: Sequence[float], weights: Sequence[float], *, scale: float = 1.0) -> float:
    """Aggregate yes-probabilities in log-odds space.

    Half weighted mean of log-odds (efficient when members are calibrated), half
    weighted median (robust against one member going badly wrong). Members are
    winsorised at 0.5% / 99.5% first so that a single extreme answer cannot dominate.
    """
    if not probabilities or len(probabilities) != len(weights):
        raise ValueError("need one weight per probability")
    zs = [logit(clamp(p, 0.005, 0.995)) for p in probabilities]
    total = sum(weights)
    mean = sum(z * w for z, w in zip(zs, weights)) / total
    median = weighted_median(zs, weights)
    return expit(scale * (0.5 * mean + 0.5 * median))


def blend_binary(ensemble: float, other: float, other_share: float) -> float:
    return expit((1.0 - other_share) * logit(ensemble) + other_share * logit(other))


def binary_spread(probabilities: Sequence[float]) -> tuple[float, float]:
    """(range of log-odds, range of probabilities) across members."""
    zs = [logit(clamp(p, 0.005, 0.995)) for p in probabilities]
    return max(zs) - min(zs), max(probabilities) - min(probabilities)


def floor_and_normalize(distribution: Mapping[str, float], floor: float) -> dict[str, float]:
    """Normalise to 1 with every option at or above floor (lowered if there are too many options)."""
    names = list(distribution)
    if not names:
        raise ValueError("no options")
    floor = min(floor, 1.0 / (2.0 * len(names)))
    values = {name: max(0.0, float(distribution[name])) for name in names}
    total = sum(values.values())
    if total <= 0.0:
        values = {name: 1.0 for name in names}
        total = float(len(names))
    values = {name: value / total for name, value in values.items()}

    pinned: set[str] = set()
    while True:
        free = [name for name in names if name not in pinned]
        free_mass = sum(values[name] for name in free)
        budget = 1.0 - floor * len(pinned)
        scaled = {name: values[name] * budget / free_mass for name in free} if free_mass > 0 else {}
        newly = [name for name in free if scaled.get(name, 0.0) < floor]
        if not newly:
            result = {name: floor for name in pinned}
            result.update(scaled)
            break
        pinned.update(newly)

    ordered = {name: result[name] for name in names}
    drift = 1.0 - sum(ordered.values())
    largest = max(ordered, key=lambda name: ordered[name])
    ordered[largest] += drift
    return ordered


def aggregate_options(
    distributions: Sequence[Mapping[str, float]], weights: Sequence[float], options: Sequence[str]
) -> dict[str, float]:
    """Weighted linear pool over the options (not floored)."""
    if not distributions or len(distributions) != len(weights):
        raise ValueError("need one weight per distribution")
    total = sum(weights)
    return {
        option: sum(w * float(d.get(option, 0.0)) for d, w in zip(distributions, weights)) / total
        for option in options
    }


def blend_options(
    ensemble: Mapping[str, float], other: Mapping[str, float], other_share: float, options: Sequence[str]
) -> dict[str, float]:
    return {
        option: (1.0 - other_share) * ensemble.get(option, 0.0) + other_share * other.get(option, 0.0)
        for option in options
    }


def options_spread(distributions: Sequence[Mapping[str, float]], options: Sequence[str]) -> float:
    """Largest disagreement on any single option."""
    spread = 0.0
    for option in options:
        values = [float(d.get(option, 0.0)) for d in distributions]
        spread = max(spread, max(values) - min(values))
    return spread
