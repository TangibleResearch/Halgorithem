from .core import Halgorithm
from .model_runtime import default_claim_extractor, default_coref, default_embedder, encode_texts, maybe_resolve
from .models import ProcessedSentence
from .text_processing import extract_numbers, lemmatize_tokens, tokenize


def process_response(text, embedder=None, coref=None, claim_extractor=None):
    embedder = embedder or default_embedder()
    coref = coref or default_coref()
    claim_extractor = claim_extractor or default_claim_extractor
    splitter = Halgorithm()
    raw_sentences = splitter.split_sentences(text)
    resolved_sentences = [maybe_resolve(sentence, coref) for sentence in raw_sentences]
    embeddings = encode_texts(embedder, resolved_sentences)
    sentences = []
    for sentence_id, (sentence, resolved, embedding) in enumerate(zip(raw_sentences, resolved_sentences, embeddings), 1):
        sentences.append(
            ProcessedSentence(
                sentence_id=sentence_id,
                text=sentence,
                resolved_text=resolved,
                embedding=embedding,
                claims=claim_extractor(resolved),
                tokens=set(tokenize(resolved)),
                lemmas=set(lemmatize_tokens(resolved)),
                numbers=set(extract_numbers(resolved)),
            )
        )
    return sentences
