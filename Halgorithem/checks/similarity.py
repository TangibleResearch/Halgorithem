import heapq

from ..models import SimilarityCheck, SimilarityHit
def _token_overlap(left_tokens, right_tokens):
    left_tokens = {token for token in left_tokens if len(token) > 2}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & set(right_tokens)) / len(left_tokens)


def _similarities(embedder, left, rights):
    if hasattr(embedder, "similarity_many"):
        try:
            return embedder.similarity_many(left, rights)
        except Exception:
            pass
    return [float(embedder.similarity(left, right)) for right in rights]


def similarity_search(processed_sentence, document, embedder, top_k=5):
    query_embedding = processed_sentence.embedding
    if query_embedding is None:
        query_embedding = embedder.encode(processed_sentence.resolved_text, convert_to_tensor=True)

    embeddings = []
    for doc_sentence in document:
        embedding = doc_sentence.embedding
        if embedding is None:
            embedding = embedder.encode(doc_sentence.resolved_text, convert_to_tensor=True)
        embeddings.append(embedding)

    raw_scores = _similarities(embedder, query_embedding, embeddings)
    hits = []
    for doc_sentence, raw_score in zip(document, raw_scores):
        overlap = _token_overlap(processed_sentence.tokens, doc_sentence.tokens)
        lemma_overlap = _token_overlap(processed_sentence.lemmas, doc_sentence.lemmas)
        number_bonus = 0.05 if processed_sentence.numbers and processed_sentence.numbers.issubset(doc_sentence.numbers) else 0.0
        overlap = max(overlap, lemma_overlap)
        score = min(raw_score + 0.12 * overlap, 1.0)
        score = min(score + number_bonus, 1.0)
        hits.append(
            SimilarityHit(
                sentence=doc_sentence.context_text or doc_sentence.resolved_text,
                score=score,
                source=doc_sentence.source,
                sentence_id=doc_sentence.sentence_id,
                source_quality=doc_sentence.source_quality,
            )
        )

    selected = heapq.nlargest(top_k, hits, key=lambda hit: hit.score)
    source_quality = max((hit.source_quality for hit in selected), default=0.55)
    score = selected[0].score if selected else 0.0
    return SimilarityCheck(score=score, hits=selected, source_quality=source_quality)
