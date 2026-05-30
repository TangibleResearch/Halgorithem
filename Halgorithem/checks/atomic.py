import re
from functools import lru_cache

from ..claim_extraction import split_atomic_claims
from ..models import AtomicCheck, AtomicClaim


@lru_cache(maxsize=8192)
def _tokens(text):
    return frozenset(t.lower() for t in re.findall(r"\b[a-zA-Z][a-zA-Z'-]+\b", text or "") if len(t) > 2)


def _overlap(left, right):
    if not left:
        return 0.0
    return len(left & right) / len(left)


def _priority(verdict):
    return {"CONTRADICT": 3, "ENTAIL": 2, "NEUTRAL": 1}.get(verdict, 0)


def _score_claim(verdict, confidence):
    if verdict == "ENTAIL":
        return confidence
    if verdict == "CONTRADICT":
        return -confidence
    return 0.5 * confidence


def prepare_document_claims(document):
    doc_claims = []
    for sentence in document:
        for claim in split_atomic_claims(sentence.resolved_text) or [sentence.resolved_text]:
            doc_claims.append((claim, _tokens(claim)))
    return doc_claims


def atomic_claim_nli(processed_sentence, document, nli_model=None, doc_claims=None):
    ai_text = getattr(processed_sentence, "resolved_text", processed_sentence)
    ai_claims = split_atomic_claims(ai_text) or [ai_text]
    doc_claims = doc_claims if doc_claims is not None else prepare_document_claims(document)

    if not doc_claims:
        return AtomicCheck(status="no_document_claims")

    matched = []
    for claim in ai_claims:
        claim_tokens = _tokens(claim)
        best_claim, best_tokens = max(doc_claims, key=lambda pair: _overlap(claim_tokens, pair[1]))
        matched.append((claim, best_claim, best_tokens))

    if nli_model is not None:
        nli_results = nli_model.predict_batch(
            [best_claim for _, best_claim, _ in matched],
            [claim for claim, _, _ in matched],
        )
    else:
        nli_results = [None] * len(matched)

    results = []
    for (claim, best_claim, best_tokens), nli in zip(matched, nli_results):
        claim_tokens = _tokens(claim)
        if not claim_tokens or not best_tokens:
            results.append(AtomicClaim(claim=claim, verdict="NEUTRAL", confidence=0.0, evidence=""))
            continue
        if nli is None:
            overlap = _overlap(claim_tokens, best_tokens)
            verdict = "ENTAIL" if overlap >= 0.70 else "NEUTRAL"
            confidence = overlap if verdict == "ENTAIL" else 0.50
        elif nli.label == "CONTRADICTION":
            verdict = "CONTRADICT"
            confidence = nli.score
        elif nli.label == "ENTAILMENT":
            verdict = "ENTAIL"
            confidence = nli.score
        else:
            verdict = "NEUTRAL"
            confidence = nli.score
        results.append(AtomicClaim(claim=claim, verdict=verdict, confidence=confidence, evidence=best_claim))

    entail = sum(1 for c in results if c.verdict == "ENTAIL")
    contradict = sum(1 for c in results if c.verdict == "CONTRADICT")
    total = len(results)
    if total:
        weighted_sum = sum(_score_claim(claim.verdict, claim.confidence) for claim in results)
        score = weighted_sum / total
    else:
        score = None
    results.sort(key=lambda c: (_priority(c.verdict), c.confidence), reverse=True)
    return AtomicCheck(claims=results, score=score, status="ok")
