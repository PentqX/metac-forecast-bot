"""Extracting the final answer from a forecaster's written rationale.

These parsers are deliberately strict: they accept only the requested final-answer
format, at the end of the text. Anything else returns None and is handed to the format
fixer, a small model that reads the text. A regular expression that guesses can pick up
a number from the reasoning (a market price, a draft, the probability of the opposite
outcome) and submit it with full confidence, which is far worse than not answering.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

# "Probability: 12.5%" alone on a line (markdown and punctuation around it are fine,
# other words are not: "Probability of No: 88%" must not match).
_FINAL_PROBABILITY = re.compile(
    r"^[^\w\n]*probability[^\w\n%]*[:：=][^\w\n%]*([0-9]{1,3}(?:[.,][0-9]+)?)\s*%[^\w\n]*$",
    re.IGNORECASE,
)
_PERCENTILE_LINE = re.compile(
    r"^[^\w\n]*percentile\s*\**\s*([0-9]{1,2}(?:\.[0-9]+)?)(?:st|nd|rd|th)?\s*\**\s*[:：=]\s*(.+?)\s*$",
    re.IGNORECASE,
)
# "<option text>: 35%" alone on a line; the option text runs up to the last colon.
_OPTION_LINE = re.compile(r"^(.*)[:：=]\s*\**\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*%?[\s*_`.]*$")
_PLAIN_NUMBER = re.compile(r"^[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?$")
_GROUPED_NUMBER = re.compile(r"^[-+]?[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?$")
_TRAILING_NOTE = re.compile(r"\s*\([^()]*\)\s*$")
_SPACES = dict.fromkeys(map(ord, "    "), None)

# After the final answer, at most this many non-empty lines may follow, and none of them
# may contain a digit: a later line with a number may be a revised answer in another format.
_MAX_TRAILING_LINES = 2
_DIGIT = re.compile(r"[0-9]")


def _harmless_tail(lines: Sequence[str]) -> bool:
    tail = [line for line in lines if line.strip()]
    return len(tail) <= _MAX_TRAILING_LINES and not any(_DIGIT.search(line) for line in tail)


def parse_probability(text: str) -> float | None:
    """Probability of Yes as a fraction, from a final line like 'Probability: 12.5%'."""
    lines = text.strip().splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if not lines[index].strip():
            continue
        match = _FINAL_PROBABILITY.match(lines[index])
        if match:
            if not _harmless_tail(lines[index + 1 :]):
                return None
            value = float(match.group(1).replace(",", "."))
            return value / 100.0 if 0.0 <= value <= 100.0 else None
        if _DIGIT.search(lines[index]):
            return None  # a number after any answer line: do not guess
    return None


def parse_number(raw: str) -> float | None:
    """A plain number, tolerating a currency sign, thousands separators and a trailing '%'.

    Returns None for anything ambiguous (for example '3.2 million'): guessing the scale
    of a number is how unit errors happen.
    """
    token = _TRAILING_NOTE.sub("", raw.strip()).strip("*`_ ")
    token = token.replace("−", "-").replace("–", "-")
    sign = ""
    if token[:1] in ("+", "-"):
        sign, token = token[0], token[1:].lstrip()
    token = token.lstrip("$€£¥").rstrip("%").translate(_SPACES)
    if _GROUPED_NUMBER.match(token):
        token = token.replace(",", "")
    token = sign + token
    if not _PLAIN_NUMBER.match(token):
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _last_block(lines: Sequence[str], classify) -> list[Any] | None:
    """Parsed entries of the final run of consecutive answer lines (blank lines allowed inside).

    classify(line) returns a parsed entry, None for a line that is not an answer line,
    or False for an answer line that could not be read; a block containing such a line
    is rejected as a whole. The block must end the text, apart from a short closing
    remark without numbers.
    """
    blocks: list[tuple[list[Any], int]] = []  # (entries, index of the block's last line)
    current: list[Any] | None = None
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        entry = classify(line)
        if entry is None:
            current = None
            continue
        if current is None:
            current = []
            blocks.append((current, index))
        current.append(entry)
        blocks[-1] = (current, index)
    if not blocks:
        return None
    entries, end = blocks[-1]
    if any(entry is False for entry in entries) or not _harmless_tail(lines[end + 1 :]):
        return None
    return entries


def parse_percentiles(text: str, minimum: int = 5) -> dict[float, float] | None:
    """Percentile -> value from the final block of lines like 'Percentile 10: 42.5'.

    Only the last block counts; an earlier draft is never merged into it.
    """

    def classify(line: str):
        match = _PERCENTILE_LINE.match(line)
        if not match:
            return None
        percentile = float(match.group(1))
        value = parse_number(match.group(2))
        if value is None or not 0.0 < percentile < 100.0:
            return False
        return percentile, value

    block = _last_block(text.splitlines(), classify)
    if block is None:
        return None
    found = dict(block)
    return found if len(found) >= minimum else None


def _normalise(name: str) -> str:
    return re.sub(r"[\W_]+", " ", name).strip().lower()


def _option_for(label: str, wanted: dict[str, str]) -> str | None:
    candidates = [
        label,
        re.sub(r"^\s*[-*•]?\s*option\s*", "", label, flags=re.IGNORECASE),
        re.sub(r"^\s*[-*•]?\s*\(?[0-9]{1,2}[.)]\s+", "", label),
    ]
    for candidate in candidates:
        key = _normalise(candidate)
        if key in wanted:
            return wanted[key]
    return None


def parse_options(text: str, options: Sequence[str]) -> dict[str, float] | None:
    """Option -> probability (summing to 1) from the final block of lines like 'Option name: 35%'.

    The block must name every option exactly once and contain nothing but such lines.
    """
    wanted = {_normalise(option): option for option in options}
    if len(wanted) != len(options):
        return None

    def classify(line: str):
        match = _OPTION_LINE.match(line)
        if not match:
            return None
        option = _option_for(match.group(1), wanted)
        if option is None:
            return None
        return option, float(match.group(2))

    block = _last_block(text.splitlines(), classify)
    if block is None or len(block) != len(options):
        return None
    found = dict(block)
    if len(found) != len(options):
        return None
    total = sum(found.values())
    # Accept percentages (sum near 100) and fractions (sum near 1); anything else is a misread.
    if not (0.97 <= total <= 1.03 or 97.0 <= total <= 103.0):
        return None
    return {option: found[option] / total for option in options}


def extract_json(text: str) -> Any:
    """The first parseable JSON object in a model reply, or None."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : index + 1])
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    return None
