def _get(chunk, key, default=None):
    if isinstance(chunk, dict):
        return chunk.get(key, default)
    aliases = {
        "source_name": "source",
        "chunk_id": "sentence_id",
        "sentence_start": "sentence_id",
        "sentence_end": "sentence_id",
        "text": "resolved_text",
    }
    attr = aliases.get(key, key)
    if not hasattr(chunk, attr):
        attr = key
    return getattr(chunk, attr, default)


def candidate_to_evidence(candidate):
    chunk = candidate["chunk"]
    return {
        "doc_id": _get(chunk, "doc_id"),
        "source": _get(chunk, "source_name"),
        "source_quality": _get(chunk, "source_quality"),
        "chunk_id": _get(chunk, "chunk_id"),
        "sentence_start": _get(chunk, "sentence_start"),
        "sentence_end": _get(chunk, "sentence_end"),
        "score": candidate.get("score", 0.0),
        "raw_score": candidate.get("raw_score", 0.0),
        "signals": candidate.get("signals", []),
        "text": _get(chunk, "text", ""),
    }


def build_evidence(candidates):
    return [candidate_to_evidence(candidate) for candidate in candidates]


def best_evidence(candidates):
    if not candidates:
        return None
    return candidate_to_evidence(candidates[0])
