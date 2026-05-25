import nltk
import spacy
from negspacy.negation import Negex

SPACY_MODEL = None
SPACY_MODEL_WARNING = None


def _load_spacy_model():
    global SPACY_MODEL, SPACY_MODEL_WARNING
    for model_name in ("en_core_web_trf", "en_core_web_lg", "en_core_web_sm"):
        try:
            SPACY_MODEL = model_name
            return spacy.load(model_name)
        except OSError:
            continue

    SPACY_MODEL = "blank_en"
    SPACY_MODEL_WARNING = (
        "spaCy model 'en_core_web_trf', 'en_core_web_lg', or 'en_core_web_sm' is not installed; "
        "falling back to spacy.blank('en'). Install one with "
        "'python -m spacy download en_core_web_trf' for better accuracy."
    )
    blank = spacy.blank("en")
    blank.add_pipe("sentencizer")
    return blank


nlp = _load_spacy_model()
try:
    if "negex" not in nlp.pipe_names:
        nlp.add_pipe("negex", last=True)
except Exception:
    # Some blank/minimal pipelines may not expose the parser hooks negspacy wants.
    # Negation then falls back to token-level checks in text_processing.
    pass

try:
    nltk.data.find("corpora/wordnet")
    WORDNET_AVAILABLE = True
except LookupError:
    WORDNET_AVAILABLE = False
