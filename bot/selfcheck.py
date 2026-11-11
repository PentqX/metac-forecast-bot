"""Offline checks of the numeric core. No network, no secrets; safe to run anywhere."""

from __future__ import annotations

from . import aggregate, numeric, parsing


def run_offline_checks() -> list[str]:
    """Return a list of problems; empty means the core behaves as expected."""
    problems: list[str] = []

    def expect(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    percentiles = {1: 80, 5: 90, 10: 95, 20: 101, 30: 105, 40: 108, 50: 111, 60: 114, 70: 118, 80: 123, 90: 131, 95: 138, 99: 152}
    cases = [
        ("linear, both open", numeric.Scale(50, 200), True, True, 201, False),
        ("linear, both closed", numeric.Scale(50, 200), False, False, 201, False),
        ("log scale, upper open", numeric.Scale(50, 5000, 0), False, True, 201, False),
        ("discrete", numeric.Scale(49.5, 200.5), False, False, 152, True),
    ]
    for name, scale, open_lower, open_upper, size, discrete in cases:
        try:
            raw = numeric.member_cdf(
                percentiles, scale, size, discrete=discrete, open_lower=open_lower, open_upper=open_upper,
                widen_weight=0.2,
            )
            final = numeric.standardize(numeric.mixture([raw, raw], [1.0, 2.0]), open_lower, open_upper)
            numeric.validate(final, open_lower, open_upper, size)
            median = scale.to_value(numeric.loc_at(final, 0.5))
            expect(100 < median < 122, f"{name}: median {median:.2f} far from the stated 111")
        except Exception as error:  # noqa: BLE001 - report every failure of the self-check
            problems.append(f"{name}: {type(error).__name__}: {error}")

    p = aggregate.aggregate_binary([0.2, 0.3, 0.25], [1.0, 1.0, 1.0])
    expect(0.2 < p < 0.3, f"binary aggregate {p} outside the member range")
    pooled = aggregate.floor_and_normalize({"a": 0.9, "b": 0.1, "c": 0.0}, 0.01)
    expect(abs(sum(pooled.values()) - 1.0) < 1e-9 and min(pooled.values()) >= 0.01 - 1e-12, "option floor failed")
    expect(parsing.parse_probability("reasoning...\nProbability: 12.5%") == 0.125, "probability parsing failed")
    parsed = parsing.parse_percentiles("\n".join(f"Percentile {k}: {v}" for k, v in percentiles.items()))
    expect(parsed is not None and parsed.get(50.0) == 111.0, "percentile parsing failed")
    return problems
