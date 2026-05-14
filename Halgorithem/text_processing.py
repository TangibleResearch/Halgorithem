import re
from functools import lru_cache

from cleantext import clean
from markdown_it import MarkdownIt
from quantulum3 import parser as qparser
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import textacy.preprocessing as tprep

from .nlp import WORDNET_AVAILABLE, nlp


STOPWORDS = set(ENGLISH_STOP_WORDS)
md = MarkdownIt()


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


def strip_markdown(text):
    # parse markdown to plain text via markdown-it-py
    tokens = md.parse(text)
    plain = []
    for token in tokens:
        if token.children:
            for child in token.children:
                if child.type == "text" or child.type == "code_inline":
                    plain.append(child.content)
        elif token.type == "fence" or token.type == "code_block":
            plain.append(token.content)
    return " ".join(plain) if plain else text


def clean_text(text):
    if not text:
        return ""
    text = strip_markdown(text)
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


def tokenize(text):
    doc = nlp(text)
    return [
        t.text.lower() for t in doc
        if not t.is_punct and not t.is_space
        and t.text.lower() not in STOPWORDS
    ]


def lemmatize_tokens(text):
    doc = nlp(text)
    return [
        t.lemma_.lower() for t in doc
        if not t.is_punct and not t.is_space
        and t.lemma_.lower() not in STOPWORDS
        and t.lemma_ != "-PRON-"
    ]


def extract_numbers(text):
    # quantulum3 handles "seven billion", "3.5 million", "$4.2B", ordinals
    quantities = qparser.parse(text)
    extracted = [str(q.value) for q in quantities if q.value is not None]
    # fallback for bare digits quantulum3 might miss
    digit_fallback = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    seen = set(extracted)
    for d in digit_fallback:
        if d not in seen:
            extracted.append(d)
            seen.add(d)
    return extracted


def extract_entities(text):
    doc = nlp(text)
    entities = set()
    for ent in doc.ents:
        tokens = tuple(
            t.lower() for t in tokenize(ent.text)
            if not t.isdigit()
        )
        if tokens:
            entities.add(tokens)
    return entities


def has_negation_mismatch(claim, chunk_text):
    # negspacy marks negated entities on the doc
    claim_doc = nlp(claim)
    chunk_doc = nlp(chunk_text)
    claim_has_negation = any(
        getattr(t._, "negex", False) for t in claim_doc
    )
    chunk_has_negation = any(
        getattr(t._, "negex", False) for t in chunk_doc
    )
    return claim_has_negation != chunk_has_negation