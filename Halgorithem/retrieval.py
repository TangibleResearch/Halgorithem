def rank_chunks(
    claim,
    chunks,
    score_fn,
    extract_numbers,
    has_negation_mismatch,
    threshold=0.30,
    top_k=5,
):
    candidates = []
    claim_numbers = set(extract_numbers(claim))

    for chunk in chunks:
        raw_score = score_fn(claim, chunk)
        score = raw_score
        signals = []
        claim_tokens = {t.lower() for t in claim.replace(".", " ").replace(",", " ").split() if t.strip()}
        chunk_tokens = set(chunk.get("tokens", []))
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
