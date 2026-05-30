import re
from datetime import date

from .nlp import parse


YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b")
ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\b")
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
    if not (claim_years and chunk_years and claim_years.isdisjoint(chunk_years)):
        return None
    claim_anchors = temporal_anchors(claim)
    chunk_anchors = temporal_anchors(chunk_text)
    if claim_anchors and chunk_anchors and claim_anchors.isdisjoint(chunk_anchors):
        return None
    return {
        "reason": "Date mismatch",
        "claim_years": sorted(claim_years),
        "truth_years": sorted(chunk_years),
    }


def temporal_anchors(text):
    anchors = {m.group(0).lower() for m in ENTITY_RE.finditer(text or "")}
    stop = {
        "the", "a", "an", "in", "on", "at", "by", "of", "for", "with",
        "was", "is", "are", "were", "as", "current", "currently",
        "created", "invented", "developed", "designed", "launched",
        "released", "founded", "reported", "started", "ended",
    }
    anchors.update(
        token.lower()
        for token in re.findall(r"\b[a-zA-Z][a-zA-Z'-]+\b", text or "")
        if len(token) > 3 and token.lower() not in stop
    )
    doc = parse(text)
    for ent in doc.ents:
        if ent.label_ not in {"DATE", "TIME", "CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "MONEY"}:
            anchors.add(ent.text.lower())
    for token in doc:
        if token.pos_ in {"PROPN", "NOUN"} and not token.is_stop and not token.like_num:
            anchors.add(token.lemma_.lower())
    return {anchor for anchor in anchors if anchor and not YEAR_RE.fullmatch(anchor)}


def temporal_warning(claim):
    lowered = (claim or "").lower()
    if any(term in lowered for term in CURRENT_TERMS):
        return {
            "warning": "Time-sensitive claim",
            "as_of_year": date.today().year,
        }
    return None
