from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


Verdict = Literal["ENTAIL", "NEUTRAL", "CONTRADICT"]
FinalVerdict = Literal["SUPPORTED", "HALLUCINATED", "UNVERIFIABLE"]


class AtomicClaim(BaseModel):
    subject: str = ""
    relation: str = ""
    object: str = ""
    text: str


class DocumentSentence(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: int
    text: str
    resolved_text: str
    source: str = ""
    source_quality: float = 0.65
    claims: list[AtomicClaim] = Field(default_factory=list)
    embedding: Any = Field(default=None, exclude=True)


class IngestedDocument(BaseModel):
    sentences: list[DocumentSentence]
    claims: list[AtomicClaim]
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class ProcessedSentence(BaseModel):
    index: int
    text: str
    resolved_text: str
    claims: list[AtomicClaim] = Field(default_factory=list)


class ProcessedResponse(BaseModel):
    sentences: list[ProcessedSentence]
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class SimilarityHit(BaseModel):
    sentence_index: int
    sentence: str
    score: float
    source: str = ""
    source_quality: float = 0.65


class SimilarityCheck(BaseModel):
    score: float
    evidence: str = ""
    source: str = ""
    source_quality: float = 0.65
    hits: list[SimilarityHit] = Field(default_factory=list)


class NLICheck(BaseModel):
    verdict: Verdict
    confidence: float
    evidence: str = ""
    evidence_index: int | None = None
    unit_mismatch: bool = False
    unit_representation_change: bool = False
    unit_details: list[dict[str, Any]] = Field(default_factory=list)


class AtomicClaimResult(BaseModel):
    claim: str
    verdict: Verdict
    confidence: float
    evidence: str = ""


class AtomicCheck(BaseModel):
    claims: list[AtomicClaimResult] = Field(default_factory=list)
    score: float | None = None
    evidence: str = ""


class SentenceVerification(BaseModel):
    sentence: str
    similarity_score: float
    entropy_score: float = 1.0
    source: str = ""
    source_quality: float = 0.65
    nli_verdict: Verdict
    nli_confidence: float
    atomic_claims: list[AtomicClaimResult]
    final_verdict: FinalVerdict
    confidence: float
    evidence: str
    unit_mismatch: bool = False
    unit_representation_change: bool = False
    unit_details: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
