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
