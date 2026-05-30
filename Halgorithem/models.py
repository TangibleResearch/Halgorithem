from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentSentence:
    doc_id: int
    source: str
    sentence_id: int
    text: str
    resolved_text: str
    context_text: str = ""
    embedding: Any = None
    source_quality: float = 0.55
    tokens: set[str] = field(default_factory=set)
    lemmas: set[str] = field(default_factory=set)
    numbers: set[str] = field(default_factory=set)


@dataclass
class ProcessedSentence:
    sentence_id: int
    text: str
    resolved_text: str
    embedding: Any = None
    claims: list[str] = field(default_factory=list)
    tokens: set[str] = field(default_factory=set)
    lemmas: set[str] = field(default_factory=set)
    numbers: set[str] = field(default_factory=set)


@dataclass
class SimilarityHit:
    sentence: str
    score: float
    source: str = ""
    sentence_id: int | None = None
    source_quality: float = 0.55


@dataclass
class SimilarityCheck:
    score: float
    hits: list[SimilarityHit] = field(default_factory=list)
    source_quality: float = 0.55


@dataclass
class NLICheck:
    label: str
    score: float
    reason: str = ""
    model_quality: float = 1.0


@dataclass
class AtomicClaim:
    claim: str
    verdict: str
    confidence: float
    evidence: str = ""


@dataclass
class AtomicCheck:
    claims: list[AtomicClaim] = field(default_factory=list)
    score: float | None = None
    status: str = "ok"


@dataclass
class VoteResult:
    verdict: str
    confidence: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    sentence: str
    verdict: str
    confidence: float
    similarity: SimilarityCheck
    nli: NLICheck
    atomic: AtomicCheck
    diagnostics: dict[str, Any] = field(default_factory=dict)
