import pytest

import tui
from Halgorithem import Halgorithm
from Halgorithem.claim_extraction import split_atomic_claims
from Halgorithem.contradiction import find_contradiction
from Halgorithem.core import LocalEmbedder
from Halgorithem.retrieval import rank_chunks


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


def test_semantic_paraphrase_support(algo, docs):
    assert first_status(algo, docs, "John Kemeny developed BASIC at Dartmouth in 1964.") == "SUPPORTED"


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


def test_unit_conversion_support(algo, docs):
    result = algo.compare_to_docs(docs, "The sample has a mass of 10000 grams.")[0]
    assert result["status"] == "SUPPORTED"


def test_math_checks(algo):
    supported = algo.compare_to_docs("Math source.", "2 + 2 = 4.")[0]
    contradicted = algo.compare_to_docs("Math source.", "2 + 2 = 5.")[0]
    malformed = algo.verify_math_claim("2 + = 4")
    assert supported["status"] == "SUPPORTED"
    assert contradicted["status"] == "CONTRADICTION"
    assert malformed["status"] == "ERROR"


def test_percent_claims_are_verified_as_source_claims(algo):
    docs = "The drug reduced mortality by 20 percent."
    result = algo.compare_to_docs(docs, "The drug reduced mortality by 20 percent.")[0]
    assert result["type"] == "SOURCE"
    assert result["status"] == "SUPPORTED"


def test_math_rejects_unsafe_symbols(algo):
    result = algo.verify_math_claim("__import__('os') = 1")
    assert result["status"] == "ERROR"
    assert "unsupported symbols" in result["reason"]


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
    with pytest.raises(ValueError):
        Halgorithm(sentences_per_chunk=0, sentence_overlap=0)
    with pytest.raises(ValueError):
        Halgorithm(sentences_per_chunk=1, sentence_overlap=1)
    with pytest.raises(FileNotFoundError):
        algo.compare_to_files([str(tmp_path / "missing.txt")], "A claim.")
    with pytest.raises(ValueError):
        algo.compare_to_docs([], "A claim.")
    with pytest.raises(ValueError):
        algo.compare_to_docs([{"file_path": "bad"}], "A claim.")
    assert algo.compare_to_docs("A source.", "") == []


def test_location_mismatch(algo):
    docs = "Lima, Peru, 9752000. Tokyo, Japan, 14000000."
    result = algo.compare_to_docs(docs, "Lima is located in Japan.")[0]
    assert result["status"] == "CONTRADICTION"
    assert result["reason"] == "Location mismatch"


def test_missing_table_entity_with_number_is_hallucination(algo):
    docs = "Product A price 19 USD. Product B price 25 USD."
    result = algo.compare_to_docs(docs, "Product C has price 31 USD.")[0]
    assert result["status"] == "HALLUCINATION"
    assert "c" in result["unsupported_terms"]


def test_britannica_nehru_paraphrase_with_lexical_fallback():
    algo = Halgorithm(sentences_per_chunk=2, sentence_overlap=1, embedder=LocalEmbedder())
    docs = (
        "Midnight on August 14-15, 1947, was a landmark moment. "
        "Shortly before the stroke of midnight on August 14, India's first prime minister, "
        "Jawaharlal Nehru, made a famous speech entitled A Tryst with Destiny."
    )
    claim = (
        "India's first prime minister, Jawaharlal Nehru, delivered his renowned "
        "Tryst with Destiny speech shortly before midnight on August 14, 1947."
    )
    result = algo.compare_to_docs(docs, claim)[0]
    assert result["status"] in {"SUPPORTED", "WEAK_SUPPORT"}
    assert result["status"] != "HALLUCINATION"


def test_semantic_paraphrase_does_not_require_nearby_year(algo):
    docs = (
        "Shortly before the stroke of midnight on August 14, India's first prime minister, "
        "Jawaharlal Nehru, made a famous speech entitled A Tryst with Destiny."
    )
    claim = (
        "India's first prime minister, Jawaharlal Nehru, delivered his renowned "
        "Tryst with Destiny speech shortly before midnight on August 14, 1947."
    )
    result = algo.compare_to_docs(docs, claim)[0]
    assert result["status"] in {"SUPPORTED", "WEAK_SUPPORT"}


def test_result_exposes_embedder_diagnostics():
    algo = Halgorithm(embedder=LocalEmbedder())
    result = algo.compare_to_docs("BASIC was created in 1964.", "BASIC was created in 1964.")[0]
    assert result["embedder"] == "lexical"
    assert result["embedding_model"] == "HashingVectorizer"


def test_tui_api_key_prompt_is_visible(monkeypatch, tmp_path):
    answers = iter([
        "test-key",
        "files",
        str(tmp_path / "source.txt"),
        "0.30",
        "2",
        "1",
        "What is true?",
    ])
    prompt_kwargs = []

    (tmp_path / "source.txt").write_text("A source.", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(tui.console, "clear", lambda: None)
    monkeypatch.setattr(tui.Prompt, "ask", lambda *args, **kwargs: prompt_kwargs.append(kwargs) or next(answers))

    config = tui.collect_inputs()

    assert config["prompt"] == "What is true?"
    assert prompt_kwargs[0].get("password") is None
