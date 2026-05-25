import os
import warnings

import torch

from .similarity import similarity_search
from .units import normalize_units, unit_representation_mismatch
from .utils import clamp, overlap_ratio
from ..contradiction import find_contradiction
from ..models import IngestedDocument, NLICheck, ProcessedSentence
from ..text_processing import extract_numbers, has_negation_mismatch


class NLIModel:
    def __init__(self, model_name=None):
        self.model_name = model_name or os.getenv("HALGORITHEM_NLI_MODEL", "cross-encoder/nli-deberta-v3-large")
        self.kind = "deberta-nli"
        self.fallback_reason = None
        if self.model_name.lower() in {"rule", "local", "deterministic"}:
            self.kind = "rule"
            self.model_name = "rule"
            self.tokenizer = None
            self.model = None
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            allow_download = os.getenv("HALGORITHEM_ALLOW_MODEL_DOWNLOAD", "").lower() in {"1", "true", "yes"}
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=not allow_download)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, local_files_only=not allow_download)
            self.model.eval()
        except Exception as exc:
            warnings.warn(
                f"Could not load NLI model {self.model_name!r} ({exc}); using deterministic NLI fallback.",
                RuntimeWarning,
            )
            self.kind = "rule"
            self.fallback_reason = str(exc)
            self.tokenizer = None
            self.model = None

    def predict(self, premise, hypothesis):
        if self.model is None:
            return rule_nli(premise, hypothesis)
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
        labels = self.model.config.id2label
        best_idx = int(torch.argmax(probs).item())
        raw_label = labels.get(best_idx, str(best_idx)).upper()
        confidence = float(probs[best_idx].item())
        if "ENTAIL" in raw_label:
            verdict = "ENTAIL"
        elif "CONTRAD" in raw_label:
            verdict = "CONTRADICT"
        else:
            verdict = "NEUTRAL"
        return verdict, confidence

    @property
    def diagnostics(self):
        return {"nli": self.kind, "nli_model": self.model_name if self.kind != "rule" else "rule", "nli_fallback_reason": self.fallback_reason}


def rule_nli(premise, hypothesis):
    chunk = {"text": premise, "numbers": extract_numbers(premise)}
    contradiction = find_contradiction(
        claim=hypothesis,
        chunk=chunk,
        extract_numbers=extract_numbers,
        has_negation_mismatch=has_negation_mismatch,
        score=1.0,
        threshold=0.0,
    )
    if contradiction:
        return "CONTRADICT", 0.86
    overlap = overlap_ratio(hypothesis, premise)
    if overlap >= 0.72:
        return "ENTAIL", clamp(0.55 + overlap * 0.4)
    if overlap >= 0.35:
        return "NEUTRAL", clamp(0.50 + overlap * 0.25)
    return "NEUTRAL", 0.62


def sentence_nli(ai_sentence: ProcessedSentence, document: IngestedDocument, *, nli_model=None, top_k=5):
    nli_model = nli_model or NLIModel()
    hits = similarity_search(ai_sentence, document, top_k=top_k).hits
    if not hits:
        return NLICheck(verdict="NEUTRAL", confidence=0.0)
    best_score = hits[0].score
    relevant_hits = [
        hit
        for hit in hits
        if hit.score >= 0.4 and hit.score >= best_score * 0.75
    ] or [hits[0]]
    premise = " ".join(hit.sentence for hit in relevant_hits)
    normalized_premise, premise_unit_changes = normalize_units(premise)
    normalized_hypothesis, hypothesis_unit_changes = normalize_units(ai_sentence.resolved_text)
    unit_mismatch = unit_representation_mismatch(premise, ai_sentence.resolved_text)
    verdict, confidence = nli_model.predict(normalized_premise, normalized_hypothesis)
    unit_details = []
    if unit_mismatch:
        unit_details.append(unit_mismatch)
    unit_details.extend({"source_change": change} for change in premise_unit_changes)
    unit_details.extend({"response_change": change} for change in hypothesis_unit_changes)
    return NLICheck(
        verdict=verdict,
        confidence=confidence,
        evidence=hits[0].sentence,
        evidence_index=hits[0].sentence_index,
        unit_mismatch=bool(unit_mismatch),
        unit_representation_change=bool(unit_mismatch),
        unit_details=unit_details,
    )
