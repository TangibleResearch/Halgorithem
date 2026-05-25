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
    words = set((claim or "").lower().replace(".", "").split())
    return bool(words & INFERENTIAL_TERMS)


def is_negative_claim(claim):
    words = set((claim or "").lower().replace(".", "").split())
    return bool(words & NEGATION_TERMS)


def classify_support(score, threshold=0.30, contradiction=None, unsupported_terms=None, claim=None):
    unsupported_terms = unsupported_terms or []
    supported_threshold = max(threshold + 0.10, 0.40)

    numeric_or_logical_contradiction = contradiction and contradiction.get("reason") in {
        "Date mismatch",
        "Number mismatch",
        "Unit mismatch",
        "Negation mismatch",
    }
    relation_contradiction = contradiction and contradiction.get("reason") in {
        "Location mismatch",
        "Entity-role mismatch",
        "Source qualifier mismatch",
    }
    if contradiction and contradiction.get("reason") == "Number mismatch" and unsupported_terms:
        return "HALLUCINATION"
    if numeric_or_logical_contradiction:
        return "CONTRADICTION"
    if unsupported_terms and is_negative_claim(claim):
        return "UNVERIFIABLE_DENIAL"
    if unsupported_terms:
        return "HALLUCINATION"
    if relation_contradiction:
        return "CONTRADICTION"
    if contradiction:
        return "CONTRADICTION"
    if is_inferential_claim(claim) and score >= 0.08:
        return "WEAK_SUPPORT"
    if score >= supported_threshold and not unsupported_terms:
        return "SUPPORTED"
    if score >= threshold:
        return "WEAK_SUPPORT"
    if is_negative_claim(claim):
        return "UNVERIFIABLE_DENIAL"
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
    if contradiction:
        confidence += 0.10
    return round(max(0.0, min(confidence, 1.0)), 3)
