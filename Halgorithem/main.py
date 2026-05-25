import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from .checks.atomic import atomic_claim_nli
from .checks.nli import NLIModel, sentence_nli
from .checks.similarity import similarity_search
from .ingest import ingest_document
from .model_runtime import default_claim_extractor, default_coref, default_embedder, default_reranker
from .models import SentenceVerification
from .process import process_response
from .voting import choose_evidence, entropy_gate, fuse_votes
from .web import scrape_url_texts


class HalgorithemVerifier:
    def __init__(self, top_k=5, max_workers=3):
        self.top_k = top_k
        self.max_workers = max_workers
        self.coref = default_coref()
        self.extractor = default_claim_extractor()
        self.embedder = default_embedder()
        self.reranker = default_reranker()
        self.nli_model = NLIModel()

    @property
    def diagnostics(self):
        diagnostics = {}
        diagnostics.update(self.coref.diagnostics)
        diagnostics.update(self.extractor.diagnostics)
        diagnostics.update(self.embedder.diagnostics)
        diagnostics.update(self.reranker.diagnostics)
        diagnostics.update(self.nli_model.diagnostics)
        return diagnostics

    def verify(self, document_text, response_text, *, source_name="inline_text"):
        document = ingest_document(
            document_text,
            coref=self.coref,
            extractor=self.extractor,
            embedder=self.embedder,
            source_name=source_name,
        )
        response = process_response(response_text, coref=self.coref, extractor=self.extractor)
        results = []
        for sentence in response.sentences:
            results.append(self.verify_sentence(sentence, document))
        return results

    def verify_sentence(self, sentence, document):
        gated_verdict, gated_confidence, entropy_score = entropy_gate(sentence.resolved_text, self.embedder)
        if gated_verdict:
            return SentenceVerification(
                sentence=sentence.text,
                similarity_score=0.0,
                entropy_score=entropy_score,
                nli_verdict="NEUTRAL",
                nli_confidence=0.0,
                atomic_claims=[],
                final_verdict=gated_verdict,
                confidence=gated_confidence,
                evidence="",
                diagnostics=self.diagnostics,
            )
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            similarity_future = executor.submit(
                similarity_search,
                sentence,
                document,
                embedder=self.embedder,
                reranker=self.reranker,
                top_k=self.top_k,
            )
            nli_future = executor.submit(sentence_nli, sentence, document, nli_model=self.nli_model, top_k=self.top_k)
            atomic_future = executor.submit(atomic_claim_nli, sentence, document, nli_model=self.nli_model)
            similarity = similarity_future.result()
            nli = nli_future.result()
            atomic = atomic_future.result()

        final_verdict, confidence = fuse_votes(similarity, nli, atomic)
        evidence = choose_evidence(similarity, nli, atomic, final_verdict)
        return SentenceVerification(
            sentence=sentence.text,
            similarity_score=similarity.score,
            entropy_score=entropy_score,
            source=similarity.source,
            source_quality=similarity.source_quality,
            nli_verdict=nli.verdict,
            nli_confidence=nli.confidence,
            atomic_claims=atomic.claims,
            final_verdict=final_verdict,
            confidence=confidence,
            evidence=evidence,
            unit_mismatch=nli.unit_mismatch,
            unit_representation_change=nli.unit_representation_change,
            unit_details=nli.unit_details,
            diagnostics=self.diagnostics,
        )


def verify(document_text, response_text, *, source_name="inline_text"):
    return HalgorithemVerifier().verify(document_text, response_text, source_name=source_name)


def verify_urls(urls, response_text):
    """Scrapes URL ground truth and verifies the response against the combined web text."""
    with TemporaryDirectory(prefix="halgorithem-web-") as tmp:
        records = scrape_url_texts(urls, output_dir=tmp)
    if not records:
        raise ValueError("No URL sources could be scraped.")
    document_text = "\n\n".join(record["text"] for record in records)
    source_name = records[0]["url"] if len(records) == 1 else "web_sources"
    return HalgorithemVerifier().verify(document_text, response_text, source_name=source_name)


def parse_urls(values):
    urls = []
    for value in values or []:
        urls.extend(part.strip() for part in value.split(",") if part.strip())
    return urls


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run deterministic Halgorithem verification.")
    parser.add_argument("--document", help="Path to source/ground-truth document text file.")
    parser.add_argument("--url", "--urls", action="append", default=[], help="Source URL(s), comma-separated or repeated.")
    parser.add_argument("--response", required=True, help="Path to AI response text file.")
    parser.add_argument("--source", default=None, help="Optional source URL or provenance label for trust scoring.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation level.")
    args = parser.parse_args(argv)

    response_text = Path(args.response).read_text(encoding="utf-8")
    urls = parse_urls(args.url)
    if urls:
        results = verify_urls(urls, response_text)
    elif args.document:
        document_text = Path(args.document).read_text(encoding="utf-8")
        results = verify(document_text, response_text, source_name=args.source or str(args.document))
    else:
        parser.error("Provide --document or --url/--urls.")
    print(json.dumps([result.model_dump(mode="json") for result in results], indent=args.indent))


if __name__ == "__main__":
    main()
