import re
from functools import lru_cache

from ..contradiction import find_contradiction
from ..models import NLICheck
from ..text_processing import extract_numbers, has_negation_mismatch, lemmatize_tokens


@lru_cache(maxsize=8192)
def _tokens(text):
    return frozenset(t.lower() for t in re.findall(r"\b[a-zA-Z][a-zA-Z'-]+\b", text or "") if len(t) > 2)


@lru_cache(maxsize=8192)
def _content_lemmas(text):
    return frozenset(lemma for lemma in lemmatize_tokens(text) if len(lemma) > 2)


class NLIModel:
    model_quality = 0.75

    def predict(self, premise, hypothesis):
        return self.predict_batch([premise], [hypothesis])[0]

    def predict_batch(self, premises, hypotheses):
        return [rule_nli(premise, hypothesis) for premise, hypothesis in zip(premises, hypotheses)]


def rule_nli(premise, hypothesis):
    chunk = {"text": premise or "", "numbers": extract_numbers(premise)}
    issue = find_contradiction(
        claim=hypothesis,
        chunk=chunk,
        extract_numbers=extract_numbers,
        has_negation_mismatch=has_negation_mismatch,
        score=1.0,
        threshold=0.0,
    )
    if issue:
        return NLICheck("CONTRADICTION", 0.84, issue.get("reason", "Contradiction"), model_quality=0.75)

    premise_tokens = _tokens(premise)
    hypothesis_tokens = _tokens(hypothesis)
    premise_numbers = set(extract_numbers(premise))
    hypothesis_numbers = set(extract_numbers(hypothesis))
    if hypothesis_numbers and not premise_numbers:
        return NLICheck("NEUTRAL", 0.35, "Missing number evidence", model_quality=0.75)
    if hypothesis_numbers and premise_numbers and not hypothesis_numbers.issubset(premise_numbers):
        return NLICheck("NEUTRAL", 0.42, "Number evidence differs", model_quality=0.75)
    if hypothesis_tokens:
        token_overlap = len(premise_tokens & hypothesis_tokens) / len(hypothesis_tokens)
        premise_lemmas = _content_lemmas(premise)
        hypothesis_lemmas = _content_lemmas(hypothesis)
        lemma_overlap = len(premise_lemmas & hypothesis_lemmas) / len(hypothesis_lemmas) if hypothesis_lemmas else 0.0
        overlap = max(token_overlap, lemma_overlap)
        if overlap >= 0.70:
            return NLICheck("ENTAILMENT", min(0.60 + overlap * 0.30, 0.92), model_quality=0.75)
    return NLICheck("NEUTRAL", 0.50, model_quality=0.75)


def sentence_nli(processed_sentence, document=None, nli_model=None, hits=None):
    model = nli_model or NLIModel()
    claim = getattr(processed_sentence, "resolved_text", processed_sentence)
    claim = claim if isinstance(claim, str) else str(claim)
    relevant_hits = hits or []
    if not relevant_hits and document is not None:
        relevant_hits = [{"sentence": s.resolved_text, "score": 1.0} for s in document[:1]]

    best_hit_score = max(
        (
            (hit.get("score", 0.0) if isinstance(hit, dict) else getattr(hit, "score", 0.0))
            for hit in relevant_hits
        ),
        default=0.0,
    )
    min_hit_score = max(0.30, best_hit_score * 0.80)
    premises = []
    hypotheses = []
    hit_scores = []
    for hit in relevant_hits[:5]:
        premise = hit.get("sentence") if isinstance(hit, dict) else getattr(hit, "sentence", str(hit))
        hit_score = hit.get("score", 0.0) if isinstance(hit, dict) else getattr(hit, "score", 0.0)
        if hit_score < min_hit_score:
            continue
        premises.append(premise)
        hypotheses.append(claim)
        hit_scores.append(hit_score)
    if not premises:
        return NLICheck("NEUTRAL", 0.50, model_quality=getattr(model, "model_quality", 1.0))

    results = model.predict_batch(premises, hypotheses)
    entailments = [r for r in results if r.label == "ENTAILMENT"]
    strong_entailment = max(entailments, key=lambda r: r.score) if entailments else None
    contradictions = [r for r in results if r.label == "CONTRADICTION"]
    if strong_entailment and strong_entailment.score >= 0.82:
        return strong_entailment
    if contradictions:
        return max(contradictions, key=lambda r: r.score)
    if entailments:
        return max(entailments, key=lambda r: r.score)
    return max(results, key=lambda r: r.score)
