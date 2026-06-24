import pytest
import asyncio

from Halgorithem import Halgorithm, HalgorithemVerifier
from Halgorithem.claim_extraction import split_atomic_claims
from Halgorithem.contradiction import equivalent_unit_numbers, find_contradiction, numbers_conflict
from Halgorithem.math_utils import safe_eval
from Halgorithem.retrieval import rank_chunks
from Halgorithem.source_quality import score_source
from Halgorithem.temporal import temporal_conflict
from Halgorithem.voting import atomic_score, similarity_weight
from Halgorithem.models import AtomicCheck, AtomicClaim, SimilarityCheck
from Halgorithem.models import DocumentSentence, NLICheck
from Halgorithem.evidence import candidate_to_evidence
from Halgorithem.model_runtime import default_claim_extractor
from Halgorithem.process import process_response
from Halgorithem.text_processing import has_negation_mismatch
from Halgorithem.voting import entropy_gate, fuse_votes, nli_score
from Halgorithem.web import WebScraper


@pytest.fixture()
def algo():
    return Halgorithm(sentences_per_chunk=2, sentence_overlap=1)


@pytest.fixture()
def docs():
    return [
        {
            "file_id": 1,
            "file_path": "facts.txt",
            "text": (
                "BASIC was created in 1964 by John Kemeny at Dartmouth College. "
                "BASIC was designed for students. "
                "The sample has a mass of 10 kilograms. "
                "As of 2024, Project Helios is active."
            ),
        }
    ]


def first_status(algo, docs, claim):
    return algo.compare_to_docs(docs, claim)[0]["status"]


def test_claim_extraction_splits_atomic_claims():
    claims = split_atomic_claims("BASIC was created in 1964 and it was designed for students.")
    assert len(claims) == 2
    assert "BASIC was created in 1964." in claims


def test_claim_extraction_splits_verb_led_conjunction():
    claims = split_atomic_claims("Python was created in 1991 and released publicly in 1994.")
    assert "Python was created in 1991." in claims
    assert "released publicly in 1994." in claims


def test_retrieval_ranks_best_chunk(algo):
    chunks = algo.chunk_text(
        "Cats sleep often. BASIC was created in 1964 at Dartmouth College.",
        source_name="inline",
    )
    ranked = rank_chunks(
        claim="BASIC was created in 1964.",
        chunks=chunks,
        score_fn=algo.support_score,
        extract_numbers=algo.extract_numbers,
        has_negation_mismatch=algo.has_negation_mismatch,
    )
    assert "basic was created" in ranked[0]["chunk"]["text"]


def test_supported_claim(algo, docs):
    assert first_status(algo, docs, "BASIC was created in 1964.") == "SUPPORTED"


def test_weak_support(algo, docs):
    assert first_status(algo, docs, "BASIC helped beginners learn programming.") == "WEAK_SUPPORT"


def test_hallucination(algo, docs):
    assert first_status(algo, docs, "BASIC was created by NASA.") == "HALLUCINATION"


def test_denial(algo, docs):
    assert first_status(algo, docs, "BASIC was not created by NASA.") == "UNVERIFIABLE_DENIAL"


def test_date_contradiction(algo, docs):
    result = algo.compare_to_docs(docs, "BASIC was created in 1972.")[0]
    assert result["status"] == "CONTRADICTION"
    assert result["reason"] == "Date mismatch"


def test_unit_contradiction(algo, docs):
    result = algo.compare_to_docs(docs, "The sample has a mass of 10 pounds.")[0]
    assert result["status"] == "CONTRADICTION"
    assert result["reason"] == "Unit mismatch"


def test_equivalent_unit_numbers_support_grams():
    assert equivalent_unit_numbers("The sample weighs 1000 grams.", "The sample weighs 1 kilogram.")


def test_percentage_rounding_tolerance():
    chunk = {"numbers": ["31"]}
    assert numbers_conflict("The rate was 30%.", chunk, lambda text: ["30"] if "30" in text else ["31"]) is None


def test_temporal_conflict_requires_shared_anchor():
    assert temporal_conflict("Apollo launched in 1969.", "Gemini launched in 1965.") is None
    assert temporal_conflict("Apollo launched in 1970.", "Apollo launched in 1969.")["reason"] == "Date mismatch"


def test_trusted_short_sources_keep_domain_quality():
    assert score_source("https://www.nasa.gov/example", "short text") == 0.92


def test_math_checks(algo):
    supported = algo.compare_to_docs("Math source.", "2 + 2 = 4.")[0]
    contradicted = algo.compare_to_docs("Math source.", "2 + 2 = 5.")[0]
    malformed = algo.verify_math_claim("2 + = 4")
    assert supported["status"] == "SUPPORTED"
    assert contradicted["status"] == "CONTRADICTION"
    assert malformed["status"] == "ERROR"


def test_safe_eval_blocks_non_math_input():
    assert safe_eval("2^3") == 8.0
    with pytest.raises(ValueError):
        safe_eval("__import__('os').system('echo nope')")
    with pytest.raises(ValueError):
        safe_eval("9**999999")


def test_temporal_warning(algo, docs):
    result = algo.compare_to_docs(docs, "The current status of Project Helios is active.")[0]
    assert result["warning"] == "Time-sensitive claim"


def test_compare_to_docs_schema(algo, docs):
    result = algo.compare_to_docs(docs, "BASIC was created in 1964.")[0]
    expected_keys = {
        "claim",
        "status",
        "confidence",
        "score",
        "matched_source",
        "matched_chunk",
        "evidence",
        "unsupported_terms",
        "reason",
        "warning",
    }
    assert expected_keys <= set(result)


def test_compare_to_files(algo, tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("BASIC was created in 1964.", encoding="utf-8")
    result = algo.compare_to_files([str(source)], "BASIC was created in 1964.")[0]
    assert result["status"] == "SUPPORTED"


def test_runtime_hardening_errors(algo, tmp_path):
    with pytest.raises(FileNotFoundError):
        algo.compare_to_files([str(tmp_path / "missing.txt")], "A claim.")
    with pytest.raises(ValueError):
        algo.compare_to_docs([], "A claim.")
    with pytest.raises(ValueError):
        algo.compare_to_docs([{"file_path": "bad"}], "A claim.")
    assert algo.compare_to_docs("A source.", "") == []


def test_verifier_stack_exists_and_returns_result(docs):
    with HalgorithemVerifier() as verifier:
        result = verifier.verify(docs, "BASIC was created in 1964.")[0]
    assert result.verdict in {"SUPPORTED", "WEAK_SUPPORT"}
    assert result.similarity.hits
    assert "atomic_check_status" in result.diagnostics


def test_voting_weights_and_atomic_fallback():
    weight = similarity_weight(SimilarityCheck(score=0.8, source_quality=0.92))
    assert weight > 0.45
    check = AtomicCheck(claims=[
        AtomicClaim("a", "ENTAIL", 0.9),
        AtomicClaim("b", "CONTRADICT", 0.8),
    ])
    assert atomic_score(check) == 0


def test_entropy_gate_returns_unverifiable_for_ambiguous_compound():
    status, confidence, entropy = entropy_gate("It rose quickly, and they said it changed.")
    assert status == "UNVERIFIABLE"
    assert confidence == 0.5
    assert entropy == 0.92


def test_fuse_votes_uses_unverifiable_fallback_not_hallucination():
    vote = fuse_votes(
        SimilarityCheck(score=0.20, source_quality=0.5),
        NLICheck("NEUTRAL", 0.50),
        AtomicCheck(claims=[], score=None, status="empty"),
    )
    assert vote.verdict == "UNVERIFIABLE"


def test_nli_contradiction_score_stays_in_support_range():
    assert nli_score(NLICheck("CONTRADICTION", 0.90)) == pytest.approx(0.10)


def test_did_is_not_negation():
    assert not has_negation_mismatch("She did create Python.", "She created Python.")


def test_default_claim_extractor_works_as_factory_and_function():
    extractor = default_claim_extractor()
    assert extractor.extract("Python was created in 1991.")
    assert default_claim_extractor("Python was created in 1991.")


def test_document_sentence_evidence_compatibility():
    sentence = DocumentSentence(
        doc_id=7,
        source="doc.txt",
        sentence_id=3,
        text="Raw",
        resolved_text="Resolved",
        source_quality=0.8,
    )
    evidence = candidate_to_evidence({"chunk": sentence, "score": 0.9})
    assert evidence["doc_id"] == 7
    assert evidence["source"] == "doc.txt"
    assert evidence["chunk_id"] == 3
    assert evidence["text"] == "Resolved"


def test_web_scraper_accepts_output_dir(tmp_path):
    scraper = WebScraper([], output_dir=tmp_path)
    assert scraper.scrape() == []


def test_web_scraper_scrape_inside_event_loop(tmp_path):
    async def run():
        scraper = WebScraper([], output_dir=tmp_path)
        return scraper.scrape()

    assert asyncio.run(run()) == []


def test_process_response_batches_embeddings():
    class BatchEmbedder:
        def __init__(self):
            self.calls = []

        def encode(self, text, convert_to_tensor=False):
            self.calls.append(text)
            if isinstance(text, list):
                return list(text)
            return text

    embedder = BatchEmbedder()
    sentences = process_response("BASIC was created in 1964. It was designed for students.", embedder=embedder)
    assert len(sentences) == 2
    assert len(embedder.calls) == 1
    assert isinstance(embedder.calls[0], list)


def test_atomic_check_batches_nli(docs):
    class BatchNLI:
        def __init__(self):
            self.batch_calls = 0

        def predict_batch(self, premises, hypotheses):
            self.batch_calls += 1
            from Halgorithem.models import NLICheck
            return [NLICheck("ENTAILMENT", 0.9) for _ in hypotheses]

    from Halgorithem.ingest import ingest_documents
    from Halgorithem.process import process_response
    from Halgorithem.checks.atomic import atomic_claim_nli, prepare_document_claims

    document = ingest_documents(docs)
    sentence = process_response("BASIC was created in 1964 and it was designed for students.")[0]
    nli = BatchNLI()
    result = atomic_claim_nli(sentence, document, nli_model=nli, doc_claims=prepare_document_claims(document))
    assert nli.batch_calls == 1
    assert result.score is not None


def test_atomic_empty_token_claim_stays_neutral():
    from Halgorithem.checks.atomic import atomic_claim_nli

    result = atomic_claim_nli("It.", [])
    assert result.status == "no_document_claims"

    from Halgorithem.models import DocumentSentence
    doc = [DocumentSentence(doc_id=1, source="x", sentence_id=1, text="A.", resolved_text="A.")]
    result = atomic_claim_nli("It.", doc)
    assert result.claims[0].verdict == "NEUTRAL"
    assert result.claims[0].evidence == ""


def test_similarity_search_does_not_mutate_embeddings():
    from Halgorithem.checks.similarity import similarity_search
    from Halgorithem.models import DocumentSentence, ProcessedSentence

    class Embedder:
        def encode(self, text, convert_to_tensor=False):
            return {text}

        def similarity(self, left, right):
            return 1.0 if left == right else 0.0

    sentence = ProcessedSentence(sentence_id=1, text="A.", resolved_text="A.")
    doc = [DocumentSentence(doc_id=1, source="x", sentence_id=1, text="A.", resolved_text="A.")]
    similarity_search(sentence, doc, Embedder())
    assert sentence.embedding is None
    assert doc[0].embedding is None
