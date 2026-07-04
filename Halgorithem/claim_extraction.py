import re

from .nlp import nlp


CLAIM_SPLIT_RE = re.compile(r"\s*(?:;|\n+|\s+-\s+)\s*")
CONJUNCTION_RE = re.compile(
    r"\s+(?:and|but|while|whereas)\s+"
    r"(?=(?:[A-Z][a-z]+|\d|it\b|he\b|she\b|they\b|the\b|a\b|an\b))",
    re.IGNORECASE,
)


def _has_factual_shape(text):
    doc = nlp(text)
    has_subject = any(t.dep_ in {"nsubj", "nsubjpass"} for t in doc)
    has_verb = any(t.pos_ in {"VERB", "AUX"} for t in doc)
    has_anchor = any(doc.ents) or any(t.like_num for t in doc) or any(t.pos_ == "PROPN" for t in doc)
    tokens = {t.text.lower() for t in doc if not t.is_punct and not t.is_space}
    technical_anchors = {"language", "programming"}
    factual_relation_verbs = {
        "create", "invent", "develop", "originate", "build", "design", "use",
        "interpret", "allow", "become"
    }
    has_relation_verb = any(t.lemma_.lower() in factual_relation_verbs for t in doc)
    return has_anchor or (has_subject and has_verb) or bool(tokens & technical_anchors) or has_relation_verb


def _has_event_verb(text):
    doc = nlp(text)
    return any(t.pos_ in {"VERB", "AUX"} for t in doc)

import re


_PRONOUN_STARTERS = {"it", "they", "he", "she"}
_VERB_STARTERS = {
    "is", "are", "was", "were",
    "has", "have", "had",
    "became", "becomes", "become",
    "created", "invented", "developed", "built",
    "founded", "launched", "released", "published",
    "introduced", "located", "based", "headquartered",
    "uses", "used", "supports", "supported",
}


def _guess_subject(text: str) -> str | None:
    """
    Guess the subject of a factual sentence.

    Conservative examples:
    'Python was created by Guido...' -> Python
    'John Kemeny and Thomas Kurtz created BASIC' -> John Kemeny and Thomas Kurtz
    'OpenAI released GPT-4...' -> OpenAI
    """
    text = text.strip()

    # Subject before common factual verbs.
    match = re.match(
        r"^(.+?)\s+("
        r"is|are|was|were|has|have|had|"
        r"created|invented|developed|built|founded|launched|"
        r"released|published|introduced|uses|used|supports|supported|"
        r"became|becomes|become"
        r")\b",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    subject = match.group(1).strip(" ,.;:")

    # Avoid huge fake subjects.
    if 1 <= len(subject.split()) <= 8:
        return subject

    return None


def _repair_split_parts(parts: list[str], original_section: str) -> list[str]:
    """
    Repair fragments created by conjunction splitting.

    Examples:
    'Python was created by Guido, but it was released in 1980'
    -> ['Python was created by Guido', 'Python was released in 1980']

    'BASIC was created in 1964 and became popular in schools'
    -> ['BASIC was created in 1964', 'BASIC became popular in schools']
    """
    subject = _guess_subject(original_section)
    repaired = []

    for part in parts:
        part = part.strip(" ,.;:-\t")
        if not part:
            continue

        words = part.split()
        if not words:
            continue

        first = words[0].lower()

        # Replace leading pronoun with original subject.
        if subject and first in _PRONOUN_STARTERS and len(words) > 1:
            part = f"{subject} {' '.join(words[1:])}"

        # Add missing subject to verb-starting fragments.
        # Example: 'became popular in schools' -> 'BASIC became popular in schools'
        elif subject and first in _VERB_STARTERS:
            part = f"{subject} {part}"

        repaired.append(part)

    return repaired
def _split_conjunctions(section):
    """
    Split on conjunctions only when the right side looks clause-like.
    Avoid splitting noun/name lists.
    """
    section = section.strip()
    if not section:
        return []

    pieces = []
    current = section

    # Split on strong separators first.
    for sep in [", but ", "; but ", " but ", ", while ", "; while ", " while "]:
        if sep in current.lower():
            # Simpler case-insensitive split while preserving text enough.
            return [
                p.strip(" ,.;:")
                for p in re.split(re.escape(sep), current, flags=re.IGNORECASE)
                if p.strip(" ,.;:")
            ]

    # Conservative "and" split.
    match = re.search(r"\s+and\s+", current, flags=re.IGNORECASE)
    if not match:
        return [current]

    left = current[:match.start()].strip()
    right = current[match.end():].strip()

    if _looks_like_new_clause(right):
        return [left, right]

    return [current]


def _looks_like_new_clause(text: str) -> bool:
    words = text.strip().split()
    if not words:
        return False

    first = words[0].lower()

    if first in _PRONOUN_STARTERS:
        return True

    if first in _VERB_STARTERS:
        return True

    # Has its own subject + verb shape.
    return bool(re.search(
        r"\b(is|are|was|were|has|have|had|created|invented|developed|founded|released|published|became)\b",
        text,
        flags=re.IGNORECASE,
    ))

def _dedupe(items):
    seen, unique = set(), []
    for item in items:
        key = item.lower()
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return unique


def split_atomic_claims(sentence):
    """Split one sentence into smaller factual claims without calling a model."""
    sentence = sentence.strip()
    if not sentence:
        return []

    parts = []
    for section in CLAIM_SPLIT_RE.split(sentence):
        section = section.strip(" -\t")
        if not section:
            continue
        parts.extend(_repair_split_parts(_split_conjunctions(section), section))

    claims = []
    for part in parts:
        if len(part.split()) < 3:
            continue
        if part[-1] not in ".!?":
            part += "."
        if _has_factual_shape(part):
            claims.append(part)
    return _dedupe(claims)


def extract_claims(text, sentence_splitter, meaningful_filter):
    """Return atomic, meaningful claims with stable ids and source sentences."""
    claims = []
    for sentence_id, sentence in enumerate(sentence_splitter(text), 1):
        atomics = split_atomic_claims(sentence)
        if not atomics:
            atomics = [sentence]
        for claim in atomics:
            if not meaningful_filter(claim):
                continue
            claims.append({
                "claim_id": len(claims) + 1,
                "sentence_id": sentence_id,
                "claim": claim,
                "source_sentence": sentence,
            })
    return claims
