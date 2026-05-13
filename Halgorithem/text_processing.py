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


WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1000000,
    "billion": 1000000000,
}

def extract_numbers(text):
    # catch digit numbers as before
    digit_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    
    # catch written numbers
    words = re.findall(r"\b[a-z]+\b", text.lower())
    word_numbers = [
        str(WORD_TO_NUM[w]) for w in words 
        if w in WORD_TO_NUM
    ]
    
    return digit_numbers + word_numbers


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
