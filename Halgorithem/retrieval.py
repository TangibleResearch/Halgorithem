import re


TOKEN_ALIASES = {
    "delivered": "made",
    "deliver": "made",
    "renowned": "famous",
    "well-known": "famous",
    "speech": "speech",
    "population": "population",
    "located": "located",
}


def _tokens(text):
    return {
        TOKEN_ALIASES.get(token, token)
        for token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", (text or "").lower())
        if len(token) > 2
    }


def rank_chunks(
    claim,
    chunks,
    score_fn,
    extract_numbers,
    has_negation_mismatch,
    threshold=0.30,
    top_k=5,
    reranker=None,
):
    candidates = []
    claim_numbers = set(extract_numbers(claim))

    for chunk in chunks:
        raw_score = score_fn(claim, chunk)
        score = raw_score
        signals = []
        claim_tokens = _tokens(claim)
        chunk_tokens = {TOKEN_ALIASES.get(t, t) for t in set(chunk.get("tokens", []))} | _tokens(chunk.get("text", ""))
        content_tokens = {t for t in claim_tokens if len(t) > 2}
        if content_tokens:
            overlap = len(content_tokens & chunk_tokens) / len(content_tokens)
            if overlap >= 0.85:
                score = min(score + 0.18, 1.0)
                signals.append("high_token_overlap")
            elif overlap >= 0.60:
                score = min(score + 0.10, 1.0)
                signals.append("token_overlap")

        if claim_numbers and claim_numbers.issubset(set(chunk.get("numbers", []))):
            score = min(score + 0.10, 1.0)
            signals.append("number_subset")

        if claim_numbers and claim_numbers & set(chunk.get("numbers", [])) and len(content_tokens & chunk_tokens) >= 2:
            score = min(score + 0.12, 1.0)
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

    ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)
    if reranker is None:
        try:
            from .model_runtime import default_reranker

            reranker = default_reranker()
        except Exception:
            reranker = None
    if reranker is not None:
        shortlist = ranked[:max(20, top_k)]
        return reranker.rerank(claim, shortlist, text_fn=lambda item: item["chunk"].get("text", ""), top_k=top_k)
    return ranked[:top_k]
