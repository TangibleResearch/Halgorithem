import re

from .temporal import temporal_conflict


ROLE_VERBS = {
    "created", "create", "invented", "invent", "developed", "develop",
    "discovered", "discover", "founded", "found", "wrote", "write",
}

UNIT_ALIASES = {
    "kg": "kilogram",
    "kilogram": "kilogram",
    "kilograms": "kilogram",
    "lb": "pound",
    "lbs": "pound",
    "pound": "pound",
    "pounds": "pound",
    "km": "kilometer",
    "kilometer": "kilometer",
    "kilometers": "kilometer",
    "mile": "mile",
    "miles": "mile",
    "m": "meter",
    "meter": "meter",
    "meters": "meter",
    "cm": "centimeter",
    "centimeter": "centimeter",
    "centimeters": "centimeter",
    "c": "celsius",
    "celsius": "celsius",
    "f": "fahrenheit",
    "fahrenheit": "fahrenheit",
    "usd": "usd",
    "dollars": "usd",
    "dollar": "usd",
    "eur": "eur",
    "euros": "eur",
    "euro": "eur",
    "g": "gram",
    "gram": "gram",
    "grams": "gram",
}

UNIT_TO_BASE = {
    "gram": ("mass", 0.001),
    "kilogram": ("mass", 1.0),
    "pound": ("mass", 0.45359237),
    "meter": ("length", 1.0),
    "centimeter": ("length", 0.01),
    "kilometer": ("length", 1000.0),
    "mile": ("length", 1609.344),
}


def numbers_conflict(claim, chunk, extract_numbers):
    claim_numbers = set(extract_numbers(claim))
    truth_numbers = set(chunk.get("numbers", []))
    if not claim_numbers or not truth_numbers:
        return None
    if claim_numbers.issubset(truth_numbers):
        return None

    def skip(number):
        try:
            value = float(number)
            return 1400 <= value <= 2100
        except (ValueError, TypeError):
            return True

    for claim_number in claim_numbers:
        if skip(claim_number):
            continue
        claim_value = float(claim_number)
        for truth_number in truth_numbers:
            if skip(truth_number):
                continue
            truth_value = float(truth_number)
            if claim_value == 0 or truth_value == 0:
                continue
            if min(claim_value, truth_value) / max(claim_value, truth_value) >= 0.5:
                if claim_value != truth_value:
                    return {
                        "reason": "Number mismatch",
                        "claim_numbers": sorted(claim_numbers),
                        "truth_numbers": sorted(truth_numbers),
                    }
    return None


def _units(text):
    units = {}
    for value, unit in re.findall(r"\b(\d+(?:\.\d+)?)\s*([A-Za-z$]+)\b", text or ""):
        canonical = UNIT_ALIASES.get(unit.lower().replace("$", "usd"))
        if canonical:
            units.setdefault(value, set()).add(canonical)
    return units


def _quantities(text):
    quantities = []
    for value, unit in re.findall(r"\b(\d+(?:\.\d+)?)\s*([A-Za-z$]+)\b", text or ""):
        canonical = UNIT_ALIASES.get(unit.lower().replace("$", "usd"))
        if canonical:
            quantities.append((float(value), canonical))
    return quantities


def unit_conflict(claim, chunk_text):
    claim_units = _units(claim)
    truth_units = _units(chunk_text)
    for value, units in claim_units.items():
        truth = truth_units.get(value)
        if truth and units.isdisjoint(truth):
            return {
                "reason": "Unit mismatch",
                "claim_units": sorted(units),
                "truth_units": sorted(truth),
            }
    for claim_value, claim_unit in _quantities(claim):
        claim_base = UNIT_TO_BASE.get(claim_unit)
        if not claim_base:
            continue
        claim_dimension, claim_factor = claim_base
        for truth_value, truth_unit in _quantities(chunk_text):
            truth_base = UNIT_TO_BASE.get(truth_unit)
            if not truth_base:
                continue
            truth_dimension, truth_factor = truth_base
            if claim_dimension != truth_dimension:
                continue
            claim_normalized = claim_value * claim_factor
            truth_normalized = truth_value * truth_factor
            if truth_normalized == 0:
                continue
            relative_error = abs(claim_normalized - truth_normalized) / abs(truth_normalized)
            if relative_error <= 0.03:
                return None
            if claim_unit != truth_unit or min(claim_normalized, truth_normalized) / max(claim_normalized, truth_normalized) >= 0.2:
                return {
                    "reason": "Unit mismatch",
                    "claim_units": [claim_unit],
                    "truth_units": [truth_unit],
                }
    return None


def equivalent_unit_numbers(claim, chunk_text):
    equivalent = set()
    for claim_value, claim_unit in _quantities(claim):
        claim_base = UNIT_TO_BASE.get(claim_unit)
        if not claim_base:
            continue
        claim_dimension, claim_factor = claim_base
        for truth_value, truth_unit in _quantities(chunk_text):
            truth_base = UNIT_TO_BASE.get(truth_unit)
            if not truth_base:
                continue
            truth_dimension, truth_factor = truth_base
            if claim_dimension != truth_dimension:
                continue
            claim_normalized = claim_value * claim_factor
            truth_normalized = truth_value * truth_factor
            if truth_normalized and abs(claim_normalized - truth_normalized) / abs(truth_normalized) <= 0.03:
                equivalent.add(str(int(claim_value)) if claim_value.is_integer() else str(claim_value))
    return equivalent


def unit_representation_change(claim, chunk_text):
    for claim_value, claim_unit in _quantities(claim):
        claim_base = UNIT_TO_BASE.get(claim_unit)
        if not claim_base:
            continue
        claim_dimension, claim_factor = claim_base
        for truth_value, truth_unit in _quantities(chunk_text):
            truth_base = UNIT_TO_BASE.get(truth_unit)
            if not truth_base:
                continue
            truth_dimension, truth_factor = truth_base
            if claim_dimension != truth_dimension or claim_unit == truth_unit:
                continue
            claim_normalized = claim_value * claim_factor
            truth_normalized = truth_value * truth_factor
            if truth_normalized and abs(claim_normalized - truth_normalized) / abs(truth_normalized) <= 0.03:
                return {
                    "reason": "Equivalent value with changed unit representation",
                    "claim_quantity": [claim_value, claim_unit],
                    "truth_quantity": [truth_value, truth_unit],
                }
    return None


def _relations(text):
    lowered = (text or "").lower()
    pattern = (
        r"\b(?P<subject>[a-z][a-z .-]{1,60}?)\s+"
        r"(?:was\s+|is\s+)?"
        r"(?P<verb>created|invented|developed|discovered|founded|wrote|designed|located)\s+"
        r"(?:(?:by|in|at)\s+)?"
        r"(?P<object>[a-z0-9][a-z0-9 .-]{1,60}?)(?:\.|,|$)"
    )
    relations = []
    for match in re.finditer(pattern, lowered):
        subject = " ".join(match.group("subject").split())
        obj = re.sub(r"\b(?:in|on|at)\s+\d{3,4}\b", "", match.group("object"))
        obj = " ".join(obj.split())
        relations.append((subject, match.group("verb"), obj))
    return relations


def _relation(text):
    relations = _relations(text)
    return relations[0] if relations else None


def source_qualifier_conflict(claim, chunk_text):
    claim_reports = set(re.findall(r"\breport\s+([a-z]+)\b", (claim or "").lower()))
    truth_reports = set(re.findall(r"\breport\s+([a-z]+)\b", (chunk_text or "").lower()))
    if claim_reports and truth_reports and claim_reports.isdisjoint(truth_reports):
        return {
            "reason": "Source qualifier mismatch",
            "claim_sources": sorted(claim_reports),
            "truth_sources": sorted(truth_reports),
        }
    return None


def _locations(text):
    lowered = (text or "").lower()
    locations = {}
    for subject, place in re.findall(
        r"\b([a-z][a-z .-]{1,60}?)\s+(?:is|was)\s+(?:located\s+)?(?:in|at)\s+([a-z][a-z .-]{1,60}?)(?:\.|,|$)",
        lowered,
    ):
        locations.setdefault(" ".join(subject.split()), set()).add(" ".join(place.split()))
    for subject, place in re.findall(r"\b([a-z][a-z .-]{1,60}?),\s*([a-z][a-z .-]{1,60}?)(?:\.|,|$)", lowered):
        subject_tokens = set(subject.split())
        place_tokens = set(place.split())
        if subject_tokens & {"as", "of", "today", "current", "latest"}:
            continue
        if place_tokens & {"has", "status", "price", "version"}:
            continue
        locations.setdefault(" ".join(subject.split()), set()).add(" ".join(place.split()))
    return locations


def location_conflict(claim, chunk_text):
    claim_locations = _locations(claim)
    truth_locations = _locations(chunk_text)
    for claim_subject, claim_places in claim_locations.items():
        claim_subject_tokens = set(claim_subject.split())
        for truth_subject, truth_places in truth_locations.items():
            if not (claim_subject_tokens & set(truth_subject.split())):
                continue
            if claim_places.isdisjoint(truth_places):
                return {
                    "reason": "Location mismatch",
                    "claim_locations": sorted(claim_places),
                    "truth_locations": sorted(truth_places),
                }
    return None


def missing_location_evidence(claim, chunk_text):
    claim_locations = _locations(claim)
    if not claim_locations:
        return False
    truth_locations = _locations(chunk_text)
    if not truth_locations:
        return True
    for claim_subject in claim_locations:
        claim_subject_tokens = set(claim_subject.split())
        if any(claim_subject_tokens & set(truth_subject.split()) for truth_subject in truth_locations):
            return False
    return True


def entity_role_conflict(claim, chunk_text):
    claim_rel = _relation(claim)
    truth_relations = _relations(chunk_text)
    if not claim_rel or not truth_relations:
        return None

    claim_subject, claim_verb, claim_object = claim_rel
    claim_subject_tokens = set(claim_subject.split())
    claim_object_tokens = set(claim_object.split())
    if not any(token.isalpha() for token in claim_object_tokens):
        return None

    for truth_subject, truth_verb, truth_object in truth_relations:
        if claim_verb != truth_verb or claim_verb == "designed":
            continue
        truth_object_tokens = set(truth_object.split())
        truth_subject_tokens = set(truth_subject.split())
        subject_overlaps = bool(claim_subject_tokens & truth_subject_tokens)
        object_overlaps = bool(claim_object_tokens & truth_object_tokens)
        passive_equivalent = bool(claim_subject_tokens & truth_object_tokens) and bool(claim_object_tokens & truth_subject_tokens)
        if passive_equivalent:
            continue
        if object_overlaps and not subject_overlaps:
            return {
                "reason": "Entity-role mismatch",
                "claim_subject": claim_subject,
                "truth_subject": truth_subject,
            }
        if subject_overlaps and not object_overlaps:
            return {
                "reason": "Entity-role mismatch",
                "claim_object": claim_object,
                "truth_object": truth_object,
            }
    return None


def negation_conflict(claim, chunk, has_negation_mismatch):
    if has_negation_mismatch(claim, chunk.get("text", "")):
        return {"reason": "Negation mismatch"}
    return None


def find_contradiction(claim, chunk, extract_numbers, has_negation_mismatch, score, threshold):
    number_issue = numbers_conflict(claim, chunk, extract_numbers)
    if number_issue:
        return number_issue

    unit_issue = unit_conflict(claim, chunk.get("text", ""))
    if unit_issue and score >= threshold:
        return unit_issue

    role_issue = entity_role_conflict(claim, chunk.get("text", ""))
    if role_issue and score >= threshold:
        return role_issue

    location_issue = location_conflict(claim, chunk.get("text", ""))
    if location_issue and score >= threshold:
        return location_issue

    source_issue = source_qualifier_conflict(claim, chunk.get("text", ""))
    if source_issue and score >= threshold:
        return source_issue

    temporal_issue = temporal_conflict(claim, chunk.get("text", ""))
    if temporal_issue and score >= threshold:
        return temporal_issue

    negation_issue = negation_conflict(claim, chunk, has_negation_mismatch)
    if negation_issue and score >= threshold:
        return negation_issue

    return None
