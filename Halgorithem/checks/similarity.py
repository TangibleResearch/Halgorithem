from .utils import clamp
from ..model_runtime import default_embedder, default_reranker
from ..models import IngestedDocument, ProcessedSentence, SimilarityCheck, SimilarityHit


def similarity_search(ai_sentence: ProcessedSentence, document: IngestedDocument, *, embedder=None, reranker=None, top_k=5):
    embedder = embedder or default_embedder()
    reranker = reranker or default_reranker()
    query = embedder.encode(ai_sentence.resolved_text)
    hits = []
    for doc_sentence in document.sentences:
        score = clamp(embedder.similarity(query, doc_sentence.embedding))
        hits.append(
            SimilarityHit(
                sentence_index=doc_sentence.index,
                sentence=doc_sentence.text,
                score=score,
                source=doc_sentence.source,
                source_quality=doc_sentence.source_quality,
            )
        )
    hits.sort(key=lambda hit: hit.score, reverse=True)
    shortlist = hits[:max(20, top_k)]
    top_hits = reranker.rerank(ai_sentence.resolved_text, shortlist, text_fn=lambda hit: hit.sentence, top_k=top_k)
    best = top_hits[0] if top_hits else None
    return SimilarityCheck(
        score=best.score if best else 0.0,
        evidence=best.sentence if best else "",
        source=best.source if best else "",
        source_quality=best.source_quality if best else 0.65,
        hits=top_hits,
    )
