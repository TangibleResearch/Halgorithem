import pytest

from Halgorithem.models import AtomicCheck, AtomicClaimResult, NLICheck, SimilarityCheck
from Halgorithem.voting import fuse_votes


def test_voting_supports_strong_entailment():
    verdict, confidence = fuse_votes(
        SimilarityCheck(score=0.82, evidence="source"),
        NLICheck(verdict="ENTAIL", confidence=0.91, evidence="source"),
        AtomicCheck(claims=[AtomicClaimResult(claim="A", verdict="ENTAIL", confidence=0.88)], score=0.88),
    )
    assert verdict == "SUPPORTED"
    assert confidence >= 0.85


def test_voting_discards_weak_nli():
    verdict, confidence = fuse_votes(
        SimilarityCheck(score=0.78, evidence="source"),
        NLICheck(verdict="CONTRADICT", confidence=0.40, evidence="source"),
        AtomicCheck(claims=[], score=None),
    )
    assert verdict == "SUPPORTED"
    assert confidence == pytest.approx(0.78)


def test_voting_hallucinates_confident_contradiction():
    verdict, confidence = fuse_votes(
        SimilarityCheck(score=0.76, evidence="source"),
        NLICheck(verdict="CONTRADICT", confidence=0.93, evidence="source"),
        AtomicCheck(claims=[], score=None),
    )
    assert verdict == "HALLUCINATED"
    assert confidence >= 0.93


def test_voting_does_not_let_contested_nli_override_win_alone():
    verdict, confidence = fuse_votes(
        SimilarityCheck(score=0.94, evidence="source"),
        NLICheck(verdict="CONTRADICT", confidence=0.93, evidence="source"),
        AtomicCheck(
            claims=[AtomicClaimResult(claim="A", verdict="ENTAIL", confidence=0.92)],
            score=0.92,
        ),
    )
    assert verdict == "UNVERIFIABLE"
    assert confidence > 0.9


def test_voting_source_quality_scales_similarity_weight_only():
    trusted_verdict, trusted_confidence = fuse_votes(
        SimilarityCheck(score=0.82, evidence="source", source_quality=0.95),
        NLICheck(verdict="NEUTRAL", confidence=0.7, evidence="source"),
        AtomicCheck(claims=[], score=None),
    )
    weak_verdict, weak_confidence = fuse_votes(
        SimilarityCheck(score=0.82, evidence="source", source_quality=0.25),
        NLICheck(verdict="NEUTRAL", confidence=0.7, evidence="source"),
        AtomicCheck(claims=[], score=None),
    )

    assert trusted_verdict == "UNVERIFIABLE"
    assert weak_verdict == "UNVERIFIABLE"
    assert trusted_confidence < weak_confidence


def test_voting_unverifiable_when_only_weak_signals_exist():
    verdict, confidence = fuse_votes(
        SimilarityCheck(score=0.22),
        NLICheck(verdict="NEUTRAL", confidence=0.52),
        AtomicCheck(claims=[], score=None),
    )
    assert verdict == "UNVERIFIABLE"
    assert confidence == 0.0
