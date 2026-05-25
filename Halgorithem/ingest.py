from .core import Halgorithm, LocalEmbedder
from .model_runtime import default_claim_extractor, default_coref, default_embedder
from .models import DocumentSentence, IngestedDocument
from .source_quality import score_source


def ingest_document(document_text, *, sentences_per_chunk=1, coref=None, extractor=None, embedder=None, source_name="inline_text"):
    splitter = Halgorithm(sentences_per_chunk=max(1, sentences_per_chunk), sentence_overlap=0, embedder=LocalEmbedder())
    coref = coref or default_coref()
    extractor = extractor or default_claim_extractor()
    embedder = embedder or default_embedder()
    source_quality = score_source(source_name, document_text)

    sentences = []
    all_claims = []
    for index, sentence in enumerate(splitter.split_sentences(document_text), 1):
        resolved = coref.resolve_text(sentence)
        claims = extractor.extract(resolved)
        all_claims.extend(claims)
        sentences.append(
            DocumentSentence(
                index=index,
                text=sentence,
                resolved_text=resolved,
                source=source_name,
                source_quality=source_quality,
                claims=claims,
                embedding=embedder.encode(resolved),
            )
        )

    diagnostics = {}
    diagnostics.update(coref.diagnostics)
    diagnostics.update(extractor.diagnostics)
    diagnostics.update(embedder.diagnostics)
    return IngestedDocument(sentences=sentences, claims=all_claims, diagnostics=diagnostics)
