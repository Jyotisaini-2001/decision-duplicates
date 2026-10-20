"""Experimental conflict checks for decision pairs.

These rules are general patterns, not hardcoded test sentences.
They only reject a pair when a conflict is explicit.

Return status:
- conflict: a confident mismatch in action, quantity, date, or entity
- no_conflict: no such mismatch found
- unknown: a signal was incomplete, so this check does not reject the pair
"""

from __future__ import annotations

import re

# Polarity-bearing decision words. Local negation flips the sign.
_POS = {
    "approve": 1, "approved": 1, "approves": 1,
    "allow": 1, "allowed": 1, "allows": 1,
    "enable": 1, "enabled": 1, "enables": 1,
    "continue": 1, "continued": 1,
    "keep": 1, "kept": 1, "retain": 1, "retained": 1,
    "renew": 1, "renewed": 1,
    "launch": 1, "ship": 1, "release": 1,
    "hire": 1, "hiring": 1, "recruit": 1, "recruiting": 1,
    "require": 1, "required": 1,
    "increase": 1, "increased": 1, "raise": 1, "raised": 1,
}
_NEG = {
    "reject": -1, "rejected": -1, "rejects": -1,
    "deny": -1, "denied": -1, "denies": -1,
    "disallow": -1, "forbid": -1, "forbidden": -1,
    "block": -1, "blocked": -1, "blocks": -1,
    "disable": -1, "disabled": -1,
    "stop": -1, "stopped": -1, "halt": -1, "halted": -1,
    "freeze": -1, "frozen": -1,
    "remove": -1, "removed": -1, "delete": -1, "deleted": -1,
    "cancel": -1, "canceled": -1, "cancelled": -1,
    "postpone": -1, "delay": -1, "delayed": -1,
    "sunset": -1, "deprecate": -1, "deprecated": -1,
    "retire": -1, "retiring": -1,
    "decrease": -1, "decreased": -1, "cut": -1, "reduce": -1, "reduced": -1,
    "discount": -1,
}
_POLARITY = {**_POS, **_NEG}
_NEGATION = {"not", "no", "never", "without", "dont"}
_AUX = {"will", "would", "to", "was", "were", "is", "are", "been", "be", "has", "have", "had"}
_CONTENT_STOP = {
    "the", "a", "an", "to", "of", "for", "on", "in", "by", "and", "or", "we", "our",
    "was", "were", "will", "be", "is", "are", "this", "that", "with", "from", "as",
    "at", "per", "each", "next", "starting", "beginning", "do", "did", "not", "no",
    "it", "its", "their", "them", "into", "over", "than", "then", "up", "out",
}

_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
_WEEKDAYS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
_LOCATION_SKIP = set(_MONTHS) | _WEEKDAYS | {"order", "office", "general", "total", "addition", "place"}


def _tokens(text: str) -> list[str]:
    lowered = text.lower().replace("'", "")
    return re.findall(r"[a-z0-9]+", lowered)


def _sentence_polarity(text: str) -> int | None:
    """Return +1, -1, or None when no single confident action polarity exists."""
    toks = _tokens(text)
    found: list[tuple[int, int, bool]] = []
    for i, tok in enumerate(toks):
        if tok not in _POLARITY and not (tok == "phase" and i + 1 < len(toks) and toks[i + 1] == "out"):
            continue
        sign = -1 if tok == "phase" else _POLARITY[tok]
        window = toks[max(0, i - 3):i]
        if any(w in _NEGATION or w == "nt" for w in window):
            sign *= -1
        is_main = (
            (i > 0 and toks[i - 1] in _AUX)
            or (i + 1 < len(toks) and toks[i + 1] in {"the", "a", "an"})
            or tok == "phase"
        )
        found.append((i, sign, is_main))
    if not found:
        return None
    mains = [sign for _, sign, is_main in found if is_main]
    chosen = mains if mains else [sign for _, sign, _ in found]
    unique = set(chosen)
    if len(unique) != 1:
        return None
    return unique.pop()


def _content_tokens(text: str) -> set[str]:
    return {
        tok for tok in _tokens(text)
        if tok not in _CONTENT_STOP and tok not in _POLARITY and not tok.isdigit()
    }


def _polarity_conflict(text_a: str, text_b: str) -> tuple[str, str]:
    pa, pb = _sentence_polarity(text_a), _sentence_polarity(text_b)
    if pa is None or pb is None:
        return "unknown", "action polarity not confident on both sides"
    if pa == pb:
        return "no_conflict", "action polarity matches"
    shared = _content_tokens(text_a) & _content_tokens(text_b)
    if len(shared) < 2:
        return "unknown", "opposite actions but the subject overlap is too small"
    return "conflict", f"opposite actions about the same subject (shared: {', '.join(sorted(shared)[:6])})"


def _number_token(raw: str) -> float | None:
    raw = raw.lower()
    if raw.isdigit():
        return float(raw)
    return float(_WORDS[raw]) if raw in _WORDS else None


def _money_values(text: str) -> set[float]:
    values = set()
    for match in re.finditer(r"\$(\d+(?:\.\d+)?)(\s*[kKmM])?", text):
        value = float(match.group(1))
        suffix = (match.group(2) or "").strip().lower()
        if suffix == "k":
            value *= 1000
        elif suffix == "m":
            value *= 1_000_000
        values.add(value)
    return values


def _percent_values(text: str) -> set[float]:
    values = set()
    for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*(?:%|percent)\b", text.lower()):
        values.add(float(match.group(1)))
    for match in re.finditer(r"\b([a-z]+)\s+percent\b", text.lower()):
        value = _number_token(match.group(1))
        if value is not None:
            values.add(value)
    return values


def _unit_values(text: str, units: set[str]) -> set[float]:
    values = set()
    toks = _tokens(text)
    for i, tok in enumerate(toks[:-1]):
        if toks[i + 1] not in units:
            continue
        value = _number_token(tok)
        if value is not None:
            values.add(value)
    return values


def _duration_hours(text: str) -> set[float]:
    hours = set()
    toks = _tokens(text)
    scale = {"hour": 1, "hours": 1, "day": 24, "days": 24, "week": 24 * 7, "weeks": 24 * 7}
    for i, tok in enumerate(toks[:-1]):
        mult = scale.get(toks[i + 1])
        value = _number_token(tok)
        if mult and value is not None:
            hours.add(value * mult)
    return hours


def _quantity_conflict(text_a: str, text_b: str) -> tuple[str, str]:
    checks = [
        ("money", _money_values(text_a), _money_values(text_b)),
        ("percent", _percent_values(text_a), _percent_values(text_b)),
        ("headcount", _unit_values(text_a, {"engineers", "developers", "designers", "analysts"}),
         _unit_values(text_b, {"engineers", "developers", "designers", "analysts"})),
        ("duration", _duration_hours(text_a), _duration_hours(text_b)),
    ]
    for name, left, right in checks:
        if left and right and left != right:
            return "conflict", f"different {name}: {sorted(left)} vs {sorted(right)}"
    return "no_conflict", "no conflicting quantity"


def _date_conflict(text_a: str, text_b: str) -> tuple[str, str]:
    def months(text: str) -> set[int]:
        return {_MONTHS[tok] for tok in _tokens(text) if tok in _MONTHS}

    def weekdays(text: str) -> set[str]:
        return {tok for tok in _tokens(text) if tok in _WEEKDAYS}

    def quarters(text: str) -> set[str]:
        found = set(re.findall(r"\bq([1-4])\b", text.lower()))
        if "next quarter" in text.lower():
            found.add("next")
        return found

    def year_end(text: str) -> bool:
        lowered = text.lower()
        return any(phrase in lowered for phrase in ("end of year", "year-end", "year end", "eoy"))

    for name, left, right in (
        ("weekday", weekdays(text_a), weekdays(text_b)),
        ("quarter", quarters(text_a), quarters(text_b)),
    ):
        if left and right and left != right:
            return "conflict", f"different {name}: {sorted(left)} vs {sorted(right)}"

    left_m, right_m = months(text_a), months(text_b)
    # "end of year" and December are compatible; do not invent a conflict there.
    if left_m and right_m and left_m != right_m:
        if not ({12} >= left_m and year_end(text_b) or {12} >= right_m and year_end(text_a)):
            return "conflict", f"different month: {sorted(left_m)} vs {sorted(right_m)}"
    short_dates_a = set(re.findall(r"\b(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b", text_a))
    short_dates_b = set(re.findall(r"\b(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b", text_b))
    if short_dates_a and short_dates_b and short_dates_a != short_dates_b:
        return "conflict", f"different calendar date: {sorted(short_dates_a)} vs {sorted(short_dates_b)}"
    times_a = set(re.findall(r"\b\d{1,2}:\d{2}\b", text_a))
    times_b = set(re.findall(r"\b\d{1,2}:\d{2}\b", text_b))
    if times_a and times_b and times_a != times_b:
        return "conflict", f"different time: {sorted(times_a)} vs {sorted(times_b)}"
    return "no_conflict", "no conflicting date"


def _entity_conflict(text_a: str, text_b: str) -> tuple[str, str]:
    def ids_by_prefix(text: str) -> dict[str, set[str]]:
        found: dict[str, set[str]] = {}
        for prefix, number in re.findall(r"\b([A-Za-z]{2,})-(\d+)\b", text):
            found.setdefault(prefix.upper(), set()).add(number)
        return found

    left_ids, right_ids = ids_by_prefix(text_a), ids_by_prefix(text_b)
    for prefix in left_ids.keys() & right_ids.keys():
        if left_ids[prefix] != right_ids[prefix]:
            return "conflict", f"different {prefix} id: {sorted(left_ids[prefix])} vs {sorted(right_ids[prefix])}"

    def places(text: str) -> set[str]:
        found = set()
        for match in re.finditer(r"\b(?:in|at)\s+([A-Z][a-z]{2,})\b", text):
            place = match.group(1).lower()
            if place not in _LOCATION_SKIP:
                found.add(place)
        return found

    left_places, right_places = places(text_a), places(text_b)
    if left_places and right_places and left_places != right_places:
        return "conflict", f"different place: {sorted(left_places)} vs {sorted(right_places)}"

    def labeled(text: str, label: str) -> set[str]:
        return {value.lower() for value in re.findall(rf"\b{label}\s+([A-Za-z0-9]+)\b", text, flags=re.I)}

    for label in ("plan", "team", "level"):
        left, right = labeled(text_a, label), labeled(text_b, label)
        if left and right and left != right:
            return "conflict", f"different {label}: {sorted(left)} vs {sorted(right)}"

    ips_a = set(re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}(?:/\d{1,2})?\b", text_a))
    ips_b = set(re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}(?:/\d{1,2})?\b", text_b))
    if ips_a and ips_b and ips_a != ips_b:
        return "conflict", f"different network: {sorted(ips_a)} vs {sorted(ips_b)}"
    return "no_conflict", "no conflicting entity"


def critical_conflict(text_a: str, text_b: str) -> dict[str, str]:
    """Check whether two decisions confidently conflict.

    Does not use gold labels. Unknown means the pair is not rejected.
    """
    checks = (
        _polarity_conflict(text_a, text_b),
        _quantity_conflict(text_a, text_b),
        _date_conflict(text_a, text_b),
        _entity_conflict(text_a, text_b),
    )
    conflicts = [reason for status, reason in checks if status == "conflict"]
    if conflicts:
        return {"status": "conflict", "reason": "; ".join(conflicts)}
    unknowns = [reason for status, reason in checks if status == "unknown"]
    if unknowns:
        return {"status": "unknown", "reason": "; ".join(unknowns)}
    return {"status": "no_conflict", "reason": "no confident conflict"}
