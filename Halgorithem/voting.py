import re

from .models import AtomicCheck, FinalVerdict, NLICheck, SimilarityCheck


SIMILARITY_THRESHOLD = 0.4
NLI_CONFIDENCE_THRESHOLD = 0.6
SUPPORTED_THRESHOLD = 0.68
HALLUCINATED_THRESHOLD = 0.38


def entropy_gate(sentence_text, embedder, threshold=0.85):
    """Checks deterministic paraphrase embedding consistency before expensive verification work."""
    parts = [part.strip() for part in re.split(r"\s*(?:,|;|\band\b)\s*", sentence_text or "") if part.strip()]
    if len(parts) <= 1:
        paraphrases = [sentence_text or ""] * 5
    else:
        paraphrases = [
            " ".join(parts),
            "; ".join(parts),
            ", ".join(reversed(parts)),
            f"{parts[0]} and {' '.join(parts[1:])}",
            " ".join(part for part in sorted(parts, key=str.lower)),
        ]
    embeddings = [embedder.encode(text) for text in paraphrases[:5]]
    scores = []
    for left_index, left in enumerate(embeddings):
        for right in embeddings[left_index + 1:]:
            scores.append(embedder.similarity(left, right))
    entropy_score = sum(scores) / len(scores) if scores else 1.0
    if entropy_score < threshold:
        return "UNVERIFIABLE", 0.5, entropy_score
    return None, None, entropy_score


def nli_score(check: NLICheck):
    if check.confidence < NLI_CONFIDENCE_THRESHOLD:
        return None
    if check.verdict == "ENTAIL":
        return check.confidence
    if check.verdict == "CONTRADICT":
        return 1.0 - check.confidence
    return 0.5


def similarity_score(check: SimilarityCheck):
    if check.score < SIMILARITY_THRESHOLD:
        return None
    return check.score


def similarity_weight(check: SimilarityCheck):
    quality = max(0.0, min(check.source_quality, 1.0))
    return 0.2 * (0.75 + 0.25 * quality)


def atomic_score(check: AtomicCheck):
    return check.score


def has_strong_atomic_entailment(check: AtomicCheck):
    return any(result.verdict == "ENTAIL" and result.confidence >= 0.85 for result in check.claims)


def has_strong_atomic_contradiction(check: AtomicCheck):
    return any(result.verdict == "CONTRADICT" and result.confidence >= 0.85 for result in check.claims)


def fuse_votes(similarity: SimilarityCheck, nli: NLICheck, atomic: AtomicCheck):
    weighted = []
    sim = similarity_score(similarity)
    sent = nli_score(nli)
    atom = atomic_score(atomic)
    if sim is not None:
        weighted.append((similarity_weight(similarity), sim))
    if sent is not None:
        weighted.append((0.5, sent))
    if atom is not None:
        weighted.append((0.3, atom))

    if not weighted:
        return "UNVERIFIABLE", 0.0

    weight_total = sum(weight for weight, _ in weighted)
    final_score = sum(weight * score for weight, score in weighted) / weight_total

    strong_atomic_entail = has_strong_atomic_entailment(atomic)
    nli_override_is_contested = (
        nli.verdict == "CONTRADICT"
        and nli.confidence >= 0.85
        and similarity.score >= 0.85
        and strong_atomic_entail
    )

    if nli.verdict == "CONTRADICT" and nli.confidence >= 0.85 and not nli_override_is_contested:
        return "HALLUCINATED", max(nli.confidence, 1.0 - final_score)
    if has_strong_atomic_contradiction(atomic):
        return "HALLUCINATED", max(result.confidence for result in atomic.claims if result.verdict == "CONTRADICT")
    if final_score >= SUPPORTED_THRESHOLD:
        return "SUPPORTED", final_score
    if final_score <= HALLUCINATED_THRESHOLD:
        return "HALLUCINATED", 1.0 - final_score
    return "UNVERIFIABLE", 1.0 - abs(0.5 - final_score) * 2


def choose_evidence(similarity: SimilarityCheck, nli: NLICheck, atomic: AtomicCheck, final_verdict: FinalVerdict):
    if final_verdict == "HALLUCINATED":
        if nli.verdict == "CONTRADICT" and nli.evidence:
            return nli.evidence
        for result in atomic.claims:
            if result.verdict == "CONTRADICT" and result.evidence:
                return result.evidence
    if final_verdict == "UNVERIFIABLE" and nli.evidence:
        return nli.evidence
    if nli.verdict == "ENTAIL" and nli.evidence:
        return nli.evidence
    if atomic.evidence:
        return atomic.evidence
    return similarity.evidence
