from tempfile import TemporaryDirectory

from .checks.atomic import atomic_claim_nli, prepare_document_claims
from .checks.nli import sentence_nli
from .checks.similarity import similarity_search
from .ingest import ingest_documents
from .model_runtime import default_coref, default_embedder, default_nli_model
from .models import VerificationResult
from .process import process_response
from .voting import entropy_gate, fuse_votes
from .web import WebScraper


class HalgorithemVerifier:
    def __init__(self, embedder=None, coref=None, nli_model=None, max_workers=4):
        self.embedder = embedder or default_embedder()
        self.coref = coref or default_coref()
        self.nli_model = nli_model or default_nli_model()
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def ingest(self, docs):
        return ingest_documents(docs, embedder=self.embedder, coref=self.coref)

    def process(self, response_text):
        return process_response(response_text, embedder=self.embedder, coref=self.coref)

    def verify_sentence(self, sentence, document, doc_claims=None):
        gated_status, gated_confidence, _ = entropy_gate(sentence.resolved_text)
        similarity = similarity_search(sentence, document, self.embedder)
        nli = sentence_nli(sentence, document, nli_model=self.nli_model, hits=similarity.hits)
        atomic = atomic_claim_nli(sentence, document, nli_model=self.nli_model, doc_claims=doc_claims)
        vote = fuse_votes(similarity, nli, atomic)
        verdict = gated_status or vote.verdict
        confidence = gated_confidence if gated_confidence is not None else vote.confidence
        return VerificationResult(
            sentence=sentence.resolved_text,
            verdict=verdict,
            confidence=confidence,
            similarity=similarity,
            nli=nli,
            atomic=atomic,
            diagnostics=vote.diagnostics,
        )

    def verify(self, docs, response_text):
        document = self.ingest(docs)
        sentences = self.process(response_text)
        doc_claims = prepare_document_claims(document)
        return [self.verify_sentence(sentence, document, doc_claims=doc_claims) for sentence in sentences]

    def verify_urls(self, urls, response_text):
        with TemporaryDirectory() as tmp:
            paths = WebScraper(urls, output_dir=tmp).scrape()
            docs = []
            for index, path in enumerate(paths, 1):
                with open(path, encoding="utf-8") as handle:
                    docs.append({"file_id": index, "file_path": path, "text": handle.read()})
            return self.verify(docs, response_text)


def verify(docs, response_text):
    with HalgorithemVerifier() as verifier:
        return verifier.verify(docs, response_text)


def verify_urls(urls, response_text):
    with HalgorithemVerifier() as verifier:
        return verifier.verify_urls(urls, response_text)
