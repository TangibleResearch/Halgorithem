def candidate_to_evidence(candidate):
    chunk = candidate["chunk"]
    return {
        "doc_id": chunk.get("doc_id"),
        "source": chunk.get("source_name"),
        "source_quality": chunk.get("source_quality"),
        "chunk_id": chunk.get("chunk_id"),
        "sentence_start": chunk.get("sentence_start"),
        "sentence_end": chunk.get("sentence_end"),
        "score": candidate.get("score", 0.0),
        "raw_score": candidate.get("raw_score", 0.0),
        "signals": candidate.get("signals", []),
        "text": chunk.get("text", ""),
    }


def build_evidence(candidates):
    return [candidate_to_evidence(candidate) for candidate in candidates]


def best_evidence(candidates):
    if not candidates:
        return None
    return candidate_to_evidence(candidates[0])
