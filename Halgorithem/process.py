from .core import Halgorithm, LocalEmbedder
from .model_runtime import default_claim_extractor, default_coref
from .models import ProcessedResponse, ProcessedSentence


def process_response(response_text, *, coref=None, extractor=None):
    splitter = Halgorithm(sentences_per_chunk=1, sentence_overlap=0, embedder=LocalEmbedder())
    coref = coref or default_coref()
    extractor = extractor or default_claim_extractor()

    sentences = []
    for index, sentence in enumerate(splitter.split_sentences(response_text), 1):
        resolved = coref.resolve_text(sentence)
        sentences.append(
            ProcessedSentence(
                index=index,
                text=sentence,
                resolved_text=resolved,
                claims=extractor.extract(resolved),
            )
        )

    diagnostics = {}
    diagnostics.update(coref.diagnostics)
    diagnostics.update(extractor.diagnostics)
    return ProcessedResponse(sentences=sentences, diagnostics=diagnostics)
