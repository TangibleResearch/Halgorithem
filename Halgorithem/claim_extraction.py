import re

from .nlp import parse


CLAIM_SPLIT_RE = re.compile(r"\s*(?:;|\n+|\s+-\s+)\s*")
CONJUNCTION_RE = re.compile(r"\s+(?:and|but|while|whereas)\s+", re.IGNORECASE)


def _has_factual_shape(text):
    doc = parse(text)
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
    doc = parse(text)
    return any(t.pos_ in {"VERB", "AUX"} for t in doc)


def _split_conjunctions(section):
    pieces = [p.strip(" ,") for p in CONJUNCTION_RE.split(section) if p.strip(" ,")]
    if len(pieces) <= 1:
        return [section]

    claims = [pieces[0]]
    for piece in pieces[1:]:
        if _has_event_verb(piece):
            claims.append(piece)
        else:
            claims[-1] = f"{claims[-1]} and {piece}"
    return claims


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
        parts.extend(_split_conjunctions(section))

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
