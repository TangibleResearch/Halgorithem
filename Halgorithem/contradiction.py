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
            return 1400 <= value <= 2100 or value <= 31
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
    import re

    units = {}
    for value, unit in re.findall(r"\b(\d+(?:\.\d+)?)\s*([A-Za-z$]+)\b", text or ""):
        canonical = UNIT_ALIASES.get(unit.lower().replace("$", "usd"))
        if canonical:
            units.setdefault(value, set()).add(canonical)
    return units


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
    return None


def _relations(text):
    import re

    lowered = (text or "").lower()
    pattern = (
        r"\b(?P<subject>[a-z][a-z .-]{1,60}?)\s+"
        r"(?:was\s+|is\s+)?"
        r"(?P<verb>created|invented|developed|discovered|founded|wrote|designed)\s+"
        r"(?:by\s+)?"
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
    import re

    claim_reports = set(re.findall(r"\breport\s+([a-z]+)\b", (claim or "").lower()))
    truth_reports = set(re.findall(r"\breport\s+([a-z]+)\b", (chunk_text or "").lower()))
    if claim_reports and truth_reports and claim_reports.isdisjoint(truth_reports):
        return {
            "reason": "Source qualifier mismatch",
            "claim_sources": sorted(claim_reports),
            "truth_sources": sorted(truth_reports),
        }
    return None


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
