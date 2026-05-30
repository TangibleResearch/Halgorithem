from .models import AtomicCheck, NLICheck, SimilarityCheck, VoteResult
from .nlp import parse


def entropy_gate(sentence_text, threshold=0.92):
    doc = parse(sentence_text)
    if not any(t.dep_ == "conj" for t in doc):
        return None, None, 1.0
    if "," not in sentence_text and ";" not in sentence_text:
        return None, None, 1.0
    if getattr(doc._, "has_coref_resolution", False):
        return None, None, 1.0
    pronouns = {
        "it", "its", "they", "them", "their", "this", "that", "these",
        "those", "he", "him", "his", "she", "her",
    }
    has_ambiguous_pronoun = any(t.text.lower() in pronouns for t in doc)
    has_named_anchor = any(doc.ents) or any(t.pos_ == "PROPN" for t in doc)
    if has_ambiguous_pronoun and not has_named_anchor:
        return "UNVERIFIABLE", 0.5, threshold
    return None, None, 1.0


def similarity_weight(check: SimilarityCheck):
    quality = max(0.0, min(check.source_quality, 1.0))
    return 0.2 + 0.3 * quality


def nli_score(check: NLICheck):
    if check.label == "ENTAILMENT":
        return check.score
    if check.label == "CONTRADICTION":
        return 1.0 - check.score
    return 0.0


def contradiction_confidence(nli: NLICheck, atomic: AtomicCheck):
    scores = []
    if nli.label == "CONTRADICTION":
        scores.append(nli.score)
    scores.extend(claim.confidence for claim in atomic.claims if claim.verdict == "CONTRADICT")
    return max(scores, default=0.0)


def atomic_score(check: AtomicCheck):
    """Return atomic support on [-1, 1].

    Negative values are deliberate: they represent atomic contradiction strength
    and are combined with separate contradiction confidence in fuse_votes().
    """
    if check.score is not None:
        return check.score
    if not check.claims:
        return None
    entail = sum(1 for c in check.claims if c.verdict == "ENTAIL")
    contradict = sum(1 for c in check.claims if c.verdict == "CONTRADICT")
    total = len(check.claims)
    return (entail - contradict) / total if total else None


def fuse_votes(similarity: SimilarityCheck, nli: NLICheck, atomic: AtomicCheck):
    """Fuse support evidence while keeping contradiction confidence separate.

    NLI model_quality scales how much the NLI support score affects the weighted
    support average. The rule-based fallback reports 0.75; a transformer-backed
    model can report 1.0 to carry the full NLI weight.
    """
    weighted = []
    weighted.append((similarity_weight(similarity), similarity.score))
    weighted.append((0.5 * max(0.0, min(nli.model_quality, 1.0)), nli_score(nli)))
    atom = atomic_score(atomic)
    if atom is not None:
        weighted.append((0.3, atom))

    total_weight = sum(weight for weight, _ in weighted) or 1.0
    support_score = sum(weight * score for weight, score in weighted) / total_weight
    atomic_contradictions = sum(1 for claim in atomic.claims if claim.verdict == "CONTRADICT")
    atomic_entails = sum(1 for claim in atomic.claims if claim.verdict == "ENTAIL")
    contra = contradiction_confidence(nli, atomic)
    if contra >= 0.80 and (atomic_contradictions or similarity.score >= 0.25):
        verdict = "CONTRADICTION"
        confidence = contra
    elif atomic_contradictions and atom is not None and atom < -0.25:
        verdict = "CONTRADICTION"
        confidence = max(contra, abs(atom))
    elif nli.label == "ENTAILMENT" and nli.score >= 0.80 and atomic_entails and similarity.score >= 0.30:
        verdict = "SUPPORTED"
        confidence = max(support_score, nli.score * 0.9)
    elif support_score >= 0.55:
        verdict = "SUPPORTED"
        confidence = support_score
    elif support_score >= 0.30:
        verdict = "WEAK_SUPPORT"
        confidence = support_score
    elif similarity.score < 0.12 and nli.label == "NEUTRAL" and (atom is None or atom <= 0.0):
        verdict = "HALLUCINATION"
        confidence = 1.0 - max(similarity.score, support_score)
    else:
        verdict = "UNVERIFIABLE"
        confidence = 1.0 - max(support_score, contra)

    diagnostics = {
        "similarity_score": similarity.score,
        "similarity_source_quality": similarity.source_quality,
        "nli_label": nli.label,
        "nli_score": nli.score,
        "atomic_score": atom,
        "atomic_check_status": atomic.status,
        "support_score": support_score,
        "contradiction_confidence": contra,
        "atomic_contradictions": atomic_contradictions,
        "atomic_entails": atomic_entails,
    }
    return VoteResult(verdict=verdict, confidence=round(confidence, 3), diagnostics=diagnostics)
