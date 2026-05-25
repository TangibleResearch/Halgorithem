import os
import re
import warnings
from functools import lru_cache

from sklearn.metrics.pairwise import cosine_similarity

from .core import LocalEmbedder
from .models import AtomicClaim
from .nlp import SPACY_MODEL, nlp


def model_flag(name, default="1"):
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


class SentenceEmbedder:
    def __init__(self, model_name=None):
        self.model_name = model_name or os.getenv("HALGORITHEM_RETRIEVAL_MODEL", "sentence-transformers/all-mpnet-base-v2")
        self.kind = "sentence-transformer"
        self.fallback_reason = None
        self._local = None
        if self.model_name.lower() in {"local", "lexical", "hashing"}:
            self.kind = "lexical"
            self.model_name = "HashingVectorizer"
            self._model = None
            self._local = LocalEmbedder()
            return
        try:
            from sentence_transformers import SentenceTransformer, util

            allow_download = model_flag("HALGORITHEM_ALLOW_MODEL_DOWNLOAD", "0")
            self._util = util
            self._model = SentenceTransformer(self.model_name, local_files_only=not allow_download)
        except Exception as exc:
            warnings.warn(
                f"Could not load retrieval embedder {self.model_name!r} ({exc}); using lexical hashing fallback.",
                RuntimeWarning,
            )
            self.kind = "lexical"
            self.fallback_reason = str(exc)
            self._model = None
            self._local = LocalEmbedder()

    def encode(self, text):
        if self._model is not None:
            return self._model.encode(text or "", convert_to_tensor=True)
        return self._local.encode(text or "", convert_to_tensor=True)

    def similarity(self, left, right):
        if self._model is not None:
            return float(self._util.cos_sim(left, right))
        return float(cosine_similarity(left, right)[0][0])

    @property
    def diagnostics(self):
        return {
            "retrieval_embedder": self.kind,
            "retrieval_model": self.model_name if self.kind == "sentence-transformer" else "HashingVectorizer",
            "retrieval_fallback_reason": self.fallback_reason,
        }


class CrossEncoderReranker:
    """Reranks a bi-encoder shortlist with a cross-encoder, falling back to original order offline."""

    def __init__(self, model_name=None):
        self.model_name = model_name or os.getenv("HALGORITHEM_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.kind = "cross-encoder"
        self.fallback_reason = None
        if self.model_name.lower() in {"none", "off", "disabled", "local", "passthrough"}:
            self.kind = "passthrough"
            self.model_name = "passthrough"
            self._model = None
            return
        try:
            from sentence_transformers import CrossEncoder

            allow_download = model_flag("HALGORITHEM_ALLOW_MODEL_DOWNLOAD", "0")
            self._model = CrossEncoder(self.model_name, local_files_only=not allow_download)
        except Exception as exc:
            warnings.warn(
                f"Could not load cross-encoder reranker {self.model_name!r} ({exc}); using bi-encoder order.",
                RuntimeWarning,
            )
            self.kind = "passthrough"
            self.fallback_reason = str(exc)
            self._model = None

    def rerank(self, query, items, *, text_fn, top_k=5):
        if not items or self._model is None:
            return list(items)[:top_k]
        pairs = [(query or "", text_fn(item) or "") for item in items]
        try:
            scores = self._model.predict(pairs)
        except Exception as exc:
            self.kind = "passthrough"
            self.fallback_reason = str(exc)
            return list(items)[:top_k]
        ranked = sorted(zip(items, scores), key=lambda pair: float(pair[1]), reverse=True)
        return [item for item, _ in ranked[:top_k]]

    @property
    def diagnostics(self):
        return {
            "cross_encoder_reranker": self.kind,
            "cross_encoder_model": self.model_name if self.kind == "cross-encoder" else "passthrough",
            "cross_encoder_fallback_reason": self.fallback_reason,
        }


class CoreferenceResolver:
    """Resolves pronouns with fastcoref when available, otherwise leaves text unchanged."""

    def __init__(self):
        self.kind = "spacy"
        self.fallback_reason = None
        self.model_name = os.getenv("HALGORITHEM_COREF_MODEL", "biu-nlp/f-coref")
        if model_flag("HALGORITHEM_USE_COREF", "1"):
            try:
                from fastcoref.modeling import FCorefModel
                from fastcoref import FCoref

                # Compatibility shim for newer transformers versions expecting this attribute.
                if not hasattr(FCorefModel, "all_tied_weights_keys"):
                    FCorefModel.all_tied_weights_keys = {}
                device = os.getenv("HALGORITHEM_COREF_DEVICE") or None
                self._model = FCoref(
                    model_name_or_path=self.model_name,
                    device=device,
                    nlp=SPACY_MODEL or "en_core_web_sm",
                    enable_progress_bar=False,
                )
                self.kind = "fastcoref"
            except Exception as exc:
                self._model = None
                self.fallback_reason = str(exc)
        else:
            self._model = None

    def resolve_text(self, text):
        if not text or self._model is None:
            return text or ""
        try:
            preds = self._model.predict(texts=[text])
            clusters = preds[0].get_clusters(as_strings=False)
            if not clusters:
                return text
            # replace each non-first mention with the antecedent text
            chars = list(text)
            replacements = []
            for cluster in clusters:
                if len(cluster) < 2:
                    continue
                antecedent_start, antecedent_end = cluster[0]
                antecedent = text[antecedent_start:antecedent_end]
                for mention_start, mention_end in cluster[1:]:
                    mention = text[mention_start:mention_end]
                    # only replace short pronouns, not full noun phrases
                    if len(mention.split()) <= 3:
                        replacements.append((mention_start, mention_end, antecedent))
            # apply replacements in reverse so indices stay valid
            for start, end, replacement in sorted(replacements, reverse=True):
                chars[start:end] = list(replacement)
            return "".join(chars)
        except Exception as exc:
            self.fallback_reason = str(exc)
            return text

    @property
    def diagnostics(self):
        return {
            "coreference": self.kind,
            "coreference_model": self.model_name if self.kind == "fastcoref" else None,
            "spacy_model": SPACY_MODEL,
            "coreference_fallback_reason": self.fallback_reason,
        }


class RebelClaimExtractor:
    def __init__(self, model_name=None):
        self.model_name = model_name or os.getenv("HALGORITHEM_REBEL_MODEL", "Babelscape/rebel-large")
        self.kind = "rebel"
        self.fallback_reason = None
        if self.model_name.lower() in {"rule", "local", "deterministic"}:
            self.kind = "rule"
            self.model_name = "rule"
            self._tokenizer = None
            self._model = None
            return
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            allow_download = model_flag("HALGORITHEM_ALLOW_MODEL_DOWNLOAD", "0")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=not allow_download)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, local_files_only=not allow_download)
        except Exception as exc:
            self.kind = "rule"
            self.fallback_reason = str(exc)
            self._tokenizer = None
            self._model = None

    def extract(self, text):
        if not text:
            return []
        if self._model is not None:
            try:
                return self._extract_rebel(text)
            except Exception as exc:
                self.kind = "rule"
                self.fallback_reason = str(exc)
        return self._extract_rules(text)

    def _extract_rebel(self, text):
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        output = self._model.generate(**inputs, max_length=256, num_beams=3)
        decoded = self._tokenizer.batch_decode(output, skip_special_tokens=False)[0]
        return parse_rebel_output(decoded) or self._extract_rules(text)

    def _extract_rules(self, text):
        claims = []
        patterns = [
            r"(?P<subject>[A-Za-z][A-Za-z0-9 .'-]{1,80}?)\s+(?:was|is)\s+(?P<relation>created|invented|developed|discovered|founded|designed|maintained|located|priced)\s+(?:by|in|at)?\s*(?P<object>[A-Za-z0-9][A-Za-z0-9 .'-]{0,80}?)(?:\.|,|$)",
            r"(?P<subject>[A-Za-z][A-Za-z0-9 .'-]{1,80}?)\s+(?:is|was)\s+(?P<relation>located\s+)?(?:in|at)\s+(?P<object>[A-Za-z][A-Za-z .'-]{1,80}?)(?:\.|,|$)",
            r"(?P<subject>[A-Za-z][A-Za-z0-9 .'-]{1,80}?)\s+(?P<relation>has|had|contains|weighs|costs)\s+(?P<object>[^.]{1,100})(?:\.|$)",
            r"(?P<subject>[A-Za-z][A-Za-z0-9 .'-]{1,80}?),\s*(?P<object>[A-Za-z][A-Za-z .'-]{1,80}),\s*(?P<relation>\d+(?:\.\d+)?)",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                subject = clean_part(match.group("subject"))
                relation = normalize_relation(clean_part(match.group("relation")))
                obj = clean_part(match.group("object"))
                if subject and relation and obj:
                    claims.append(AtomicClaim(subject=subject, relation=relation, object=obj, text=f"{subject} {relation} {obj}"))
        if not claims:
            doc = nlp(text)
            root = next((t for t in doc if t.dep_ == "ROOT"), None)
            subj = next((t for t in doc if t.dep_ in {"nsubj", "nsubjpass"}), None)
            obj = next((t for t in doc if t.dep_ in {"dobj", "attr", "pobj"}), None)
            if root and subj and obj:
                claims.append(AtomicClaim(subject=subj.text, relation=root.lemma_, object=obj.text, text=text.strip()))
        return dedupe_claims(claims)

    @property
    def diagnostics(self):
        return {
            "claim_extractor": self.kind,
            "claim_model": self.model_name if self.kind == "rebel" else "rule",
            "claim_fallback_reason": self.fallback_reason,
        }


class FactScoreDecomposer:
    """Turns a sentence into standalone atomic English facts for NLI-friendly atomic checking."""

    prompt_template = (
        "Decompose the following sentence into simple atomic facts.\n"
        "Each fact should be a complete standalone English sentence.\n"
        "Output one fact per line, nothing else.\n"
        "Sentence: {sentence}"
    )

    def __init__(self, model_name=None):
        self.model_name = model_name or os.getenv("HALGORITHEM_DECOMPOSER_MODEL", "google/flan-t5-base")
        self.kind = "factscore"
        self.fallback_reason = None
        self._fallback = RebelClaimExtractor(model_name="rule")
        if self.model_name.lower() in {"rule", "local", "deterministic"}:
            self.kind = "rule"
            self.model_name = "rule"
            self._tokenizer = None
            self._model = None
            return
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            allow_download = model_flag("HALGORITHEM_ALLOW_MODEL_DOWNLOAD", "0")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=not allow_download)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, local_files_only=not allow_download)
        except Exception as exc:
            warnings.warn(
                f"Could not load FActScore decomposer {self.model_name!r} ({exc}); using rule-based extraction.",
                RuntimeWarning,
            )
            self.kind = "rule"
            self.fallback_reason = str(exc)
            self._tokenizer = None
            self._model = None

    def extract(self, text):
        if not text:
            return []
        if self._model is None:
            return self._fallback.extract(text)
        prompt = self.prompt_template.format(sentence=text)
        try:
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            output = self._model.generate(**inputs, max_length=256, num_beams=3)
            decoded = self._tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            facts = [
                clean_part(line)
                for line in decoded.splitlines()
                if 10 <= len(clean_part(line)) <= 150
            ]
            claims = [AtomicClaim(subject="", relation="", object="", text=fact) for fact in facts]
            return dedupe_claims(claims) or self._fallback.extract(text)
        except Exception as exc:
            self.kind = "rule"
            self.fallback_reason = str(exc)
            return self._fallback.extract(text)

    @property
    def diagnostics(self):
        return {
            "claim_extractor": self.kind,
            "claim_model": self.model_name if self.kind == "factscore" else "rule",
            "claim_fallback_reason": self.fallback_reason,
            "decomposer_model": self.model_name if self.kind == "factscore" else "rule",
            "decomposer_fallback_reason": self.fallback_reason,
        }


def clean_part(value):
    return " ".join((value or "").strip(" .,;:-").split())


def normalize_relation(value):
    normalized = " ".join((value or "").lower().split())
    if normalized in {"", "in", "at", "located"}:
        return "located"
    return normalized


def is_valid_triplet(claim):
    if not claim.subject and not claim.relation and not claim.object:
        fact = clean_part(claim.text)
        return 10 <= len(fact) <= 150
    subject = clean_part(claim.subject)
    relation = clean_part(claim.relation)
    obj = clean_part(claim.object)
    if not subject or not relation or not obj:
        return False
    if "." in subject or "." in obj:
        return False
    if subject.lower() == obj.lower():
        return False
    if len(subject) < 2 or len(obj) < 2:
        return False
    return True


def dedupe_claims(claims):
    seen = set()
    unique = []
    for claim in claims:
        if not is_valid_triplet(claim):
            continue
        key = (
            claim.subject.lower(),
            claim.relation.lower(),
            claim.object.lower(),
            claim.text.lower() if not (claim.subject or claim.relation or claim.object) else "",
        )
        if key not in seen:
            unique.append(claim)
            seen.add(key)
    return unique


def parse_rebel_output(text):
    triplets = []
    current = {"subject": "", "relation": "", "object": ""}
    field = None
    tokens = text.replace("<s>", "").replace("</s>", "").split()
    for token in tokens:
        if token == "<triplet>":
            if all(current.values()):
                triplets.append(AtomicClaim(**current, text=f"{current['subject']} {current['relation']} {current['object']}"))
            current = {"subject": "", "relation": "", "object": ""}
            field = "subject"
        elif token == "<subj>":
            field = "object"
        elif token == "<obj>":
            field = "relation"
        elif field:
            current[field] = clean_part(f"{current[field]} {token}")
    if all(current.values()):
        triplets.append(AtomicClaim(**current, text=f"{current['subject']} {current['relation']} {current['object']}"))
    return dedupe_claims(triplets)


@lru_cache(maxsize=1)
def default_coref():
    return CoreferenceResolver()


@lru_cache(maxsize=1)
def default_claim_extractor():
    return FactScoreDecomposer()


@lru_cache(maxsize=1)
def default_embedder():
    return SentenceEmbedder()


@lru_cache(maxsize=1)
def default_reranker():
    return CrossEncoderReranker()
