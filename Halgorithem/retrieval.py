import re


TOKEN_ALIASES = {
    "created": "create",
    "creates": "create",
    "creating": "create",
    "creator": "create",
    "invented": "invent",
    "invents": "invent",
    "inventor": "invent",
    "developed": "develop",
    "developer": "develop",
    "developers": "develop",
    "released": "release",
    "running": "run",
    "ran": "run",
}


def _tokens(text):
    return {TOKEN_ALIASES.get(t.lower(), t.lower()) for t in re.findall(r"\b[\w'-]+\b", text or "") if len(t) > 2}


def rank_chunks(
    claim,
    chunks,
    score_fn,
    extract_numbers,
    has_negation_mismatch,
    lemmatize_fn=None,
    threshold=0.30,
    top_k=5,
):
    candidates = []
    claim_numbers = set(extract_numbers(claim))
    claim_tokens = _tokens(claim)
    claim_lemmas = set()
    if lemmatize_fn is not None:
        claim_lemmas = {TOKEN_ALIASES.get(t, t) for t in lemmatize_fn(claim) if len(t) > 2}

    for chunk in chunks:
        raw_score = score_fn(claim, chunk)
        score = raw_score
        signals = []
        chunk_tokens = {TOKEN_ALIASES.get(t, t) for t in set(chunk.get("tokens", []))}
        if not chunk_tokens:
            chunk_tokens = _tokens(chunk.get("text", ""))
        content_tokens = {t for t in claim_tokens if len(t) > 2}
        overlap = 0.0
        if content_tokens:
            token_overlap = len(content_tokens & chunk_tokens) / len(content_tokens)
            chunk_lemmas = {TOKEN_ALIASES.get(t, t) for t in set(chunk.get("lemmas", []))}
            lemma_overlap = len(claim_lemmas & chunk_lemmas) / len(claim_lemmas) if claim_lemmas and chunk_lemmas else 0.0
            overlap = max(token_overlap, lemma_overlap)
            if overlap >= 0.85:
                score = min(score + 0.18, 1.0)
                signals.append("high_token_overlap")
            elif overlap >= 0.60:
                score = min(score + 0.10, 1.0)
                signals.append("token_overlap")

        if claim_numbers and claim_numbers.issubset(set(chunk.get("numbers", []))):
            score = min(score + 0.10, 1.0)
            signals.append("number_subset")
        elif claim_numbers and set(chunk.get("numbers", [])) and overlap >= 0.60:
            signals.append("number_anchor_overlap")

        if has_negation_mismatch(claim, chunk.get("text", "")) and score >= threshold:
            score = max(score - 0.30, 0.0)
            signals.append("negation_penalty")

        candidates.append({
            "chunk": chunk,
            "score": score,
            "raw_score": raw_score,
            "signals": signals,
        })

    return sorted(candidates, key=lambda c: c["score"], reverse=True)[:top_k]
