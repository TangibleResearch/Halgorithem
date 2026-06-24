from .nlp import parse


INFERENTIAL_TERMS = {
    "helped",
    "made",
    "easier",
    "easy",
    "beginners",
    "learn",
    "important",
    "significant",
    "influential",
}
INFERENTIAL_ROOT_LEMMAS = {
    "help",
    "ease",
    "learn",
    "influence",
    "matter",
    "signify",
}

NEGATION_TERMS = {
    "no",
    "not",
    "never",
    "neither",
    "nor",
    "without",
    "didn't",
    "doesn't",
    "wasn't",
    "isn't",
    "aren't",
    "can't",
    "cannot",
}


def is_inferential_claim(claim):
    lowered = (claim or "").lower()
    if "made" in lowered and "easier" in lowered and " made by " not in lowered:
        return True
    doc = parse(claim)
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    return bool(root and root.lemma_.lower() in INFERENTIAL_ROOT_LEMMAS)


def is_negative_claim(claim):
    words = set((claim or "").lower().replace(".", "").split())
    return bool(words & NEGATION_TERMS)


def classify_support(score, threshold=0.30, contradiction=None, unsupported_terms=None, claim=None):
    unsupported_terms = unsupported_terms or []
    supported_threshold = max(threshold + 0.10, 0.40)

    if unsupported_terms and is_negative_claim(claim):
        return "UNVERIFIABLE_DENIAL"
    if is_negative_claim(claim) and contradiction and contradiction.get("reason") == "NLI contradiction" and score < threshold:
        return "UNVERIFIABLE_DENIAL"

    hard_contradiction = contradiction and contradiction.get("reason") in {
        "Date mismatch",
        "Number mismatch",
        "Unit mismatch",
        "Negation mismatch",
        "Entity-role mismatch",
        "Location mismatch",
        "Source qualifier mismatch",
        "NLI contradiction",
    }
    if contradiction and unsupported_terms and contradiction.get("reason") in {
        "Entity-role mismatch",
        "Source qualifier mismatch",
        "Number mismatch",
        "NLI contradiction",
    }:
        return "HALLUCINATION"
    if hard_contradiction:
        return "CONTRADICTION"
    if is_inferential_claim(claim) and score >= 0.08:
        return "WEAK_SUPPORT"
    if unsupported_terms:
        return "HALLUCINATION"
    if contradiction:
        return "CONTRADICTION"
    if score >= supported_threshold and not unsupported_terms:
        return "SUPPORTED"
    lowered = (claim or "").lower()
    if "located in" in lowered and score < supported_threshold:
        return "HALLUCINATION"
    if score >= threshold:
        return "WEAK_SUPPORT"
    return "HALLUCINATION"


def confidence_score(score, evidence_count=0, contradiction=None, unsupported_terms=None, status=None):
    unsupported_terms = unsupported_terms or []
    if status == "HALLUCINATION":
        confidence = 1.0 - max(0.0, min(float(score), 1.0))
    elif status == "UNVERIFIABLE_DENIAL":
        confidence = 0.45
    else:
        confidence = max(0.0, min(float(score), 1.0))
    confidence += min(evidence_count, 3) * 0.04
    confidence -= min(len(unsupported_terms), 4) * 0.06
    if contradiction and status in {"CONTRADICTION", "HALLUCINATION"}:
        confidence += 0.10
    return round(max(0.0, min(confidence, 1.0)), 3)
