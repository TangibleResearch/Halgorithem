from .core import Halgorithm
from .model_runtime import default_coref, default_embedder, encode_texts, maybe_resolve
from .models import DocumentSentence
from .source_quality import score_source
from .text_processing import extract_numbers, lemmatize_tokens, tokenize


def ingest_document(text, source_name="inline_text", doc_id=1, embedder=None, coref=None):
    embedder = embedder or default_embedder()
    coref = coref or default_coref()
    splitter = Halgorithm(sentences_per_chunk=2, sentence_overlap=1, embedder=embedder)
    quality = score_source(source_name, text)
    raw_sentences = splitter.split_sentences(text)
    resolved_sentences = [maybe_resolve(sentence, coref) for sentence in raw_sentences]
    embeddings = encode_texts(embedder, resolved_sentences)
    document = []
    for sentence_id, (sentence, resolved, embedding) in enumerate(zip(raw_sentences, resolved_sentences, embeddings), 1):
        context_parts = resolved_sentences[max(0, sentence_id - 2): min(len(resolved_sentences), sentence_id + 1)]
        context = " ".join(context_parts)
        document.append(
            DocumentSentence(
                doc_id=doc_id,
                source=source_name,
                sentence_id=sentence_id,
                text=sentence,
                resolved_text=resolved,
                context_text=context,
                embedding=embedding,
                source_quality=quality,
                tokens=set(tokenize(resolved)),
                lemmas=set(lemmatize_tokens(resolved)),
                numbers=set(extract_numbers(resolved)),
            )
        )
    return document


def ingest_documents(docs, embedder=None, coref=None):
    if isinstance(docs, str):
        docs = [{"file_id": 1, "file_path": "inline_text", "text": docs}]
    document = []
    for index, doc in enumerate(docs, 1):
        if isinstance(doc, str):
            doc = {"file_id": index, "file_path": f"inline_text_{index}", "text": doc}
        document.extend(
            ingest_document(
                doc.get("text", ""),
                source_name=doc.get("file_path", f"inline_text_{index}"),
                doc_id=doc.get("file_id", index),
                embedder=embedder,
                coref=coref,
            )
        )
    return document


def document_sentence_to_chunk(sentence):
    return {
        "doc_id": sentence.doc_id,
        "source_name": sentence.source,
        "source_quality": sentence.source_quality,
        "chunk_id": sentence.sentence_id,
        "sentence_start": sentence.sentence_id,
        "sentence_end": sentence.sentence_id,
        "text": sentence.resolved_text,
        "tokens": list(sentence.tokens),
        "lemmas": list(sentence.lemmas),
        "numbers": list(sentence.numbers),
        "embedding": sentence.embedding,
    }


def chunks_from_document(document):
    return [document_sentence_to_chunk(sentence) for sentence in document]
