import nltk
import spacy


nlp = spacy.load("en_core_web_sm")

try:
    nltk.data.find("corpora/wordnet")
    WORDNET_AVAILABLE = True
except LookupError:
    WORDNET_AVAILABLE = False
