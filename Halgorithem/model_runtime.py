from functools import lru_cache
import re

from .claim_extraction import split_atomic_claims
from .core import _embedder


PRONOUN_RE = re.compile(
    r"\b(it|he|she|they|his|her|their|its|him|them|this|that|these|those)\b",
    re.IGNORECASE,
)


class NoOpCoref:
    def resolve_text(self, text):
        return text or ""


def maybe_resolve(text, coref):
    if not PRONOUN_RE.search(text or ""):
        return text or ""
    return coref.resolve_text(text or "")


@lru_cache(maxsize=1)
def default_embedder():
    return _embedder


@lru_cache(maxsize=1)
def default_coref():
    return NoOpCoref()


@lru_cache(maxsize=1)
def default_nli_model():
    from .checks.nli import NLIModel

    return NLIModel()


class RuleClaimExtractor:
    def extract(self, text):
        return split_atomic_claims(text)

    def __call__(self, text):
        return self.extract(text)


@lru_cache(maxsize=1)
def _default_claim_extractor_instance():
    return RuleClaimExtractor()


def default_claim_extractor(text=None):
    extractor = _default_claim_extractor_instance()
    if text is None:
        return extractor
    return extractor.extract(text)


def encode_texts(embedder, texts):
    texts = [text or "" for text in texts]
    if not texts:
        return []
    try:
        encoded = embedder.encode(texts, convert_to_tensor=True)
        return [encoded[index] for index in range(len(texts))]
    except Exception:
        return [embedder.encode(text, convert_to_tensor=True) for text in texts]
