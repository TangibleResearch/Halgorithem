import nltk
import spacy
from negspacy.negation import Negex

nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("negex", last=True)

try:
    nltk.data.find("corpora/wordnet")
    WORDNET_AVAILABLE = True
except LookupError:
    WORDNET_AVAILABLE = False