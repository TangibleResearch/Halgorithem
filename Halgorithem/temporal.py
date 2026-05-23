import re
from datetime import date


YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b")
CURRENT_TERMS = {
    "current",
    "currently",
    "latest",
    "today",
    "now",
    "present",
    "as of",
}


def extract_years(text):
    return {int(y) for y in YEAR_RE.findall(text or "")}


def has_temporal_language(text):
    lowered = (text or "").lower()
    return any(term in lowered for term in CURRENT_TERMS) or bool(extract_years(text))


def temporal_conflict(claim, chunk_text):
    claim_years = extract_years(claim)
    chunk_years = extract_years(chunk_text)
    if claim_years and chunk_years and claim_years.isdisjoint(chunk_years):
        return {
            "reason": "Date mismatch",
            "claim_years": sorted(claim_years),
            "truth_years": sorted(chunk_years),
        }

    return None


def temporal_warning(claim):
    lowered = (claim or "").lower()
    if any(term in lowered for term in CURRENT_TERMS):
        return {
            "warning": "Time-sensitive claim",
            "as_of_year": date.today().year,
        }
    return None
