from .temporal import temporal_conflict


def numbers_conflict(claim, chunk, extract_numbers):
    claim_numbers = set(extract_numbers(claim))
    truth_numbers = set(chunk.get("numbers", []))
    if not claim_numbers or not truth_numbers:
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


def negation_conflict(claim, chunk, has_negation_mismatch):
    if has_negation_mismatch(claim, chunk.get("text", "")):
        return {"reason": "Negation mismatch"}
    return None


def find_contradiction(claim, chunk, extract_numbers, has_negation_mismatch, score, threshold):
    number_issue = numbers_conflict(claim, chunk, extract_numbers)
    if number_issue:
        return number_issue

    temporal_issue = temporal_conflict(claim, chunk.get("text", ""))
    if temporal_issue and score >= threshold:
        return temporal_issue

    negation_issue = negation_conflict(claim, chunk, has_negation_mismatch)
    if negation_issue and score >= threshold:
        return negation_issue

    return None
