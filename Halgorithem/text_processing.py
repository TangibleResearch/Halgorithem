import re
from functools import lru_cache

from cleantext import clean
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import textacy.preprocessing as tprep

from .nlp import WORDNET_AVAILABLE, nlp


TOKEN_GRAMMAR = re.compile(
    r"[A-Za-z]+(?:[-'][A-Za-z]+)*|"
    r"[0-9]+(?:\.[0-9]+)?|"
    r"[A-Z]\."
)

STOPWORDS = set(ENGLISH_STOP_WORDS)
NEGATION_WORDS = {"not", "never", "no", "none", "isn't", "wasn't", "aren't", "weren't"}


@lru_cache(maxsize=4096)
def get_synonyms(word):
    if not WORDNET_AVAILABLE:
        return set()

    from nltk.corpus import wordnet

    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower().replace("_", " "))

    return synonyms


def clean_text(text):
    if not text:
        return ""

    text = tprep.normalize.unicode(text)
    text = tprep.normalize.whitespace(text)
    text = tprep.normalize.quotation_marks(text)
    text = tprep.remove.punctuation(text, only=["(", ")", ";", ":", "\""])

    if text and text[-1] not in ".!?":
        text += "."

    return clean(
        text,
        fix_unicode=True,
        to_ascii=True,
        no_urls=True,
        no_emails=True,
        replace_with_punct="",
        lang="en",
    )


def normalize_tokens(raw_tokens):
    useful_tokens = []

    for token in raw_tokens:
        normalized = token.lower().strip(".")

        if not normalized or normalized in STOPWORDS:
            continue

        useful_tokens.append(normalized)

    return useful_tokens


def tokenize(text):
    raw_tokens = TOKEN_GRAMMAR.findall(text)
    return normalize_tokens(raw_tokens)


def lemmatize_tokens(text):
    doc = nlp(text)
    tokens = []

    for token in doc:
        if token.is_space or token.is_punct:
            continue

        lemma = token.lemma_.lower().strip(".")

        if not lemma or lemma == "-pron-" or lemma in STOPWORDS:
            continue

        if TOKEN_GRAMMAR.fullmatch(token.text) or TOKEN_GRAMMAR.fullmatch(lemma):
            tokens.append(lemma)

    return tokens


def extract_numbers(text):
    return re.findall(r"\b\d+(?:\.\d+)?\b", text)


def extract_entities(text):
    doc = nlp(text)
    entities = set()

    for ent in doc.ents:
        ent_tokens = tuple(tokenize(ent.text))
        if ent_tokens and not all(token.isdigit() for token in ent_tokens):
            entities.add(ent_tokens)

    title_tokens = [
        token.lower().strip(".")
        for token in TOKEN_GRAMMAR.findall(text)
        if token[:1].isupper() and not token.isdigit()
    ]
    entities.update((token,) for token in title_tokens if token not in STOPWORDS)

    return entities


def has_negation_mismatch(claim, chunk_text):
    claim_words = set(re.findall(r"[a-z']+", claim.lower()))
    chunk_words = set(re.findall(r"[a-z']+", chunk_text.lower()))
    return bool(claim_words & NEGATION_WORDS) != bool(chunk_words & NEGATION_WORDS)
