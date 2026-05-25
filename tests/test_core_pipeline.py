from Halgorithem.checks.nli import rule_nli
from Halgorithem.checks.units import normalize_units, unit_representation_mismatch
from Halgorithem.ingest import ingest_document
from Halgorithem.model_runtime import RebelClaimExtractor, dedupe_claims
from Halgorithem.models import AtomicClaim


def test_rule_claim_extractor_normalizes_short_location_relation():
    extractor = RebelClaimExtractor(model_name="rule")

    claims = extractor.extract("Lima is in Peru.")

    assert len(claims) == 1
    assert claims[0].subject.lower() == "lima"
    assert claims[0].relation == "located"
    assert claims[0].object.lower() == "peru"


def test_rule_nli_contradicts_short_and_explicit_location_mismatch():
    verdict, confidence = rule_nli("Lima is in Peru.", "Lima is located in Japan.")

    assert verdict == "CONTRADICT"
    assert confidence >= 0.8


def test_unit_normalization_matches_equivalent_rewrites():
    normalized, changes = normalize_units("The Mars sample container has a mass of 10000 grams.")

    assert "10 kilogram" in normalized
    assert changes[0]["original"] == "10000 grams"
    assert changes[0]["normalized"] == "10 kilogram"


def test_unit_representation_mismatch_flags_equivalent_rewrite():
    mismatch = unit_representation_mismatch(
        "The Mars sample container has a mass of 10 kilograms.",
        "The Mars sample container has a mass of 10000 grams.",
    )

    assert mismatch["source"] == "10 kilograms"
    assert mismatch["response"] == "10000 grams"
    assert mismatch["normalized"] == "10 kilogram"


def test_rebel_triplet_filter_rejects_malformed_artifacts():
    claims = dedupe_claims(
        [
            AtomicClaim(subject="nasa.basic", relation="owned by", object="nasa", text="bad"),
            AtomicClaim(subject="nasa", relation="owner of", object="nasa.basic", text="bad"),
            AtomicClaim(subject="Lima", relation="country", object="Peru", text="good"),
        ]
    )

    assert [claim.text for claim in claims] == ["good"]


def test_ingest_records_source_quality():
    document = ingest_document(
        "NASA launched the test mission.",
        source_name="https://www.nasa.gov/example",
        extractor=RebelClaimExtractor(model_name="rule"),
    )

    assert document.sentences[0].source == "https://www.nasa.gov/example"
    assert document.sentences[0].source_quality > 0.7
