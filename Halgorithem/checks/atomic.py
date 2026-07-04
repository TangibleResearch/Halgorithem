from .nli import NLIModel
from .utils import token_set
from ..models import AtomicCheck, AtomicClaimResult, IngestedDocument, ProcessedSentence


def claim_text(claim):
    return claim.text or f"{claim.subject} {claim.relation} {claim.object}".strip()


def claim_overlap(left, right):
    left_text = f"{left.subject} {left.relation} {left.object}".strip() or claim_text(left)
    right_text = f"{right.subject} {right.relation} {right.object}".strip() or claim_text(right)

    left_tokens = token_set(left_text)
    right_tokens = token_set(right_text)

    if not left_tokens:
        return 0.0

    return len(left_tokens & right_tokens) / len(left_tokens)


def choose_best_atomic_evidence(candidate_results):
    """
    Pick the strongest evidence result from multiple NLI checks.

    Priority:
    1. High-confidence contradiction
    2. High-confidence entailment
    3. Best neutral / weak match
    """
    if not candidate_results:
        return None

    contradictions = [
        result for result in candidate_results
        if result["verdict"] == "CONTRADICT"
    ]

    entailments = [
        result for result in candidate_results
        if result["verdict"] == "ENTAIL"
    ]

    neutrals = [
        result for result in candidate_results
        if result["verdict"] not in {"ENTAIL", "CONTRADICT"}
    ]

    def evidence_score(result, confidence_weight=0.75, overlap_weight=0.25):
        return result["confidence"] * confidence_weight + result["overlap"] * overlap_weight

    if contradictions:
        return max(contradictions, key=evidence_score)

    if entailments:
        return max(entailments, key=evidence_score)

    return max(
        neutrals,
        key=lambda result: result["confidence"] * 0.5 + result["overlap"] * 0.5,
    )


def atomic_claim_nli(ai_sentence: ProcessedSentence, document: IngestedDocument, *, nli_model=None):
    nli_model = nli_model or NLIModel()
    results = []

    for ai_claim in ai_sentence.claims:
        candidates = sorted(
            document.claims,
            key=lambda doc_claim: claim_overlap(ai_claim, doc_claim),
            reverse=True,
        )

        top_candidates = [
            doc_claim for doc_claim in candidates[:5]
            if claim_overlap(ai_claim, doc_claim) > 0.15
        ]

        if not top_candidates:
            results.append(
                AtomicClaimResult(
                    claim=claim_text(ai_claim),
                    verdict="NEUTRAL",
                    confidence=0.5,
                )
            )
            continue

        candidate_results = []

        for doc_claim in top_candidates:
            verdict, confidence = nli_model.predict(
                claim_text(doc_claim),
                claim_text(ai_claim),
            )

            candidate_results.append({
                "doc_claim": doc_claim,
                "verdict": verdict,
                "confidence": confidence,
                "overlap": claim_overlap(ai_claim, doc_claim),
            })

        best = choose_best_atomic_evidence(candidate_results)

        if best is None:
            results.append(
                AtomicClaimResult(
                    claim=claim_text(ai_claim),
                    verdict="NEUTRAL",
                    confidence=0.5,
                )
            )
            continue

        results.append(
            AtomicClaimResult(
                claim=claim_text(ai_claim),
                verdict=best["verdict"],
                confidence=best["confidence"],
                evidence=claim_text(best["doc_claim"]),
            )
        )

    evidence = ""
    if results:
        evidence = max(
            results,
            key=lambda result: result.confidence or 0.0,
        ).evidence or ""

    return AtomicCheck(
        claims=results,
        score=score_atomic_results(results),
        evidence=evidence,
    )


def score_atomic_results(results):
    if not results:
        return None

    # A strong contradiction should hurt the whole atomic score.
    strong_contradictions = [
        result for result in results
        if result.verdict == "CONTRADICT" and result.confidence >= 0.75
    ]

    if strong_contradictions:
        return min(1.0 - result.confidence for result in strong_contradictions)

    scores = []

    for result in results:
        if result.verdict == "ENTAIL":
            scores.append(result.confidence)
        elif result.verdict == "CONTRADICT":
            scores.append(1.0 - result.confidence)
        else:
            scores.append(0.5)

    return sum(scores) / len(scores)