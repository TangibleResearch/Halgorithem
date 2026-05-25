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


def atomic_claim_nli(ai_sentence: ProcessedSentence, document: IngestedDocument, *, nli_model=None):
    nli_model = nli_model or NLIModel()
    results = []
    for ai_claim in ai_sentence.claims:
        candidates = sorted(document.claims, key=lambda doc_claim: claim_overlap(ai_claim, doc_claim), reverse=True)
        best_claim = candidates[0] if candidates else None
        if not best_claim:
            results.append(AtomicClaimResult(claim=claim_text(ai_claim), verdict="NEUTRAL", confidence=0.5))
            continue
        verdict, confidence = nli_model.predict(claim_text(best_claim), claim_text(ai_claim))
        results.append(
            AtomicClaimResult(
                claim=claim_text(ai_claim),
                verdict=verdict,
                confidence=confidence,
                evidence=claim_text(best_claim),
            )
        )

    evidence = ""
    if results:
        evidence = max(results, key=lambda result: result.confidence).evidence
    return AtomicCheck(claims=results, score=score_atomic_results(results), evidence=evidence)


def score_atomic_results(results):
    if not results:
        return None
    scores = []
    for result in results:
        if result.verdict == "ENTAIL":
            scores.append(result.confidence)
        elif result.verdict == "CONTRADICT":
            scores.append(1.0 - result.confidence)
        else:
            scores.append(0.5)
    return sum(scores) / len(scores)
