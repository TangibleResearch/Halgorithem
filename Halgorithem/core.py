import re
from pathlib import Path

import pysbd
from sentence_transformers import SentenceTransformer, util

from .claim_extraction import extract_claims
from .confidence import classify_support, confidence_score
from .contradiction import find_contradiction, numbers_conflict
from .evidence import best_evidence, build_evidence
from .math_utils import numbers_close, safe_eval
from .retrieval import rank_chunks
from .source_quality import score_source
from .temporal import temporal_warning
from .text_processing import (
    clean_text,
    extract_entities,
    extract_numbers,
    get_synonyms,
    has_negation_mismatch,
    lemmatize_tokens,
    tokenize,
)
from .nlp import nlp


_embedder = SentenceTransformer("all-MiniLM-L6-v2")
INITIAL_RE = re.compile(r"\b[a-z]\.$", re.IGNORECASE)


class Halgorithm:
    def __init__(self, sentences_per_chunk=2, sentence_overlap=1):
        self.sentences_per_chunk = sentences_per_chunk
        self.sentence_overlap = sentence_overlap
        self.parser = pysbd.Segmenter(language="en", clean=False)

    # ── Text prep ─────────────────────────────────────────────────────────────

    def clean_text(self, text):
        return clean_text(text)

    def split_sentences(self, text):
        text = self.clean_text(text)
        sentences = self.parser.segment(text)
        merged = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if merged and INITIAL_RE.search(merged[-1]):
                merged[-1] = f"{merged[-1]} {sentence}"
            else:
                merged.append(sentence)
        return merged

    def tokenize(self, text):
        return tokenize(text)

    def lemmatize_tokens(self, text):
        return lemmatize_tokens(text)

    def extract_numbers(self, text):
        return extract_numbers(text)

    def extract_entities(self, text):
        return extract_entities(text)

    def has_negation_mismatch(self, claim, chunk_text):
        return has_negation_mismatch(claim, chunk_text)

    def get_synonyms(self, word):
        return get_synonyms(word)

    # ── File loading ──────────────────────────────────────────────────────────

    def load_file(self, file_path):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")
        return path.read_text(encoding="utf-8")

    def load_files(self, file_paths):
        return [
            {"file_id": i, "file_path": str(fp), "text": self.load_file(fp)}
            for i, fp in enumerate(file_paths, 1)
        ]

    # ── Chunking ──────────────────────────────────────────────────────────────

    def chunk_text(self, text, doc_id=1, source_name=None):
        sentences = self.split_sentences(text)
        chunks, start, chunk_id = [], 0, 1
        quality = score_source(source_name, text)
        while start < len(sentences):
            end = start + self.sentences_per_chunk
            chunk = " ".join(sentences[start:end])
            chunks.append({
                "doc_id": doc_id,
                "source_name": source_name,
                "source_quality": quality,
                "chunk_id": chunk_id,
                "sentence_start": start + 1,
                "sentence_end": min(end, len(sentences)),
                "text": chunk,
                "tokens": self.tokenize(chunk),
                "entities": self.extract_entities(chunk),
                "numbers": self.extract_numbers(chunk),
                "embedding": _embedder.encode(chunk, convert_to_tensor=True),
            })
            chunk_id += 1
            if end >= len(sentences):
                break
            start = end - self.sentence_overlap
        return chunks

    # ── Scoring ───────────────────────────────────────────────────────────────

    def support_score(self, claim, chunk):
        # semantic similarity via sentence-transformers — topic-agnostic
        claim_emb = _embedder.encode(claim, convert_to_tensor=True)
        return float(util.cos_sim(claim_emb, chunk["embedding"]))

    # ── Math claims ───────────────────────────────────────────────────────────

    def classify_claim_type(self, claim):
        if re.search(r"\d+\s*[\+\-\*/%]\s*\d+|(?<!\w)=(?!\w)|\d+\s*(percent|%)", claim.lower()):
            return "MATH"
        return "SOURCE"

    def verify_math_claim(self, claim):
        if "=" not in claim:
            return {"status": "UNKNOWN", "claim": claim, "reason": "No expression found"}
        parts = claim.split("=", 1)
        if len(parts) != 2:
            return {"status": "UNKNOWN", "claim": claim, "reason": "Malformed expression"}
        try:
            left, right = safe_eval(parts[0].strip()), safe_eval(parts[1].strip())
            if numbers_close(left, right):
                return {"status": "SUPPORTED", "claim": claim, "type": "MATH"}
            return {"status": "CONTRADICTION", "claim": claim, "type": "MATH",
                    "expected": left, "got": right}
        except Exception as e:
            return {"status": "ERROR", "claim": claim, "reason": str(e), "type": "MATH"}

    # ── Number conflict ───────────────────────────────────────────────────────

    def has_number_conflict(self, claim, chunk):
        issue = numbers_conflict(claim, chunk, self.extract_numbers)
        claim_numbers = set(self.extract_numbers(claim))
        truth_numbers = set(chunk["numbers"])
        return bool(issue), claim_numbers, truth_numbers

    # ── Meaningful claim filter ───────────────────────────────────────────────

    def is_meaningful_claim(self, claim):
        claim_l = claim.lower().strip()

        # filter metadata/citations
        BAD_PATTERNS = [
            "adapted from",
            "source:",
            "sources:",
            "http://",
            "https://",
            "www.",
        ]
        if any(p in claim_l for p in BAD_PATTERNS):
            return False

        tokens = self.tokenize(claim)

        last_word = claim.strip().rstrip(".").split()[-1].lower()
        if last_word in {"including", "such", "namely", "follows", "following", "as"}:
            return False

        doc = nlp(claim)

        has_anchor = any(doc.ents) or any(t.like_num for t in doc) or any(t.pos_ == "PROPN" for t in doc)
        if len(tokens) < 4 and not has_anchor:
            return False

        # reject vague summary sentences
        subject = next((t for t in doc if t.dep_ == "nsubj"), None)
        if subject and subject.text.lower() in {"these", "this", "those", "such"}:
            return False

        root = next((t for t in doc if t.dep_ == "ROOT"), None)
        SUMMARY_VERBS = {
            "reflect", "demonstrate", "highlight", "illustrate", "suggest",
            "indicate", "underscore", "emphasize", "represent", "signal",
            "mark", "mean", "position", "pivot",
        }
        if root and root.lemma_.lower() in SUMMARY_VERBS:
            return False

        # NEW: accept definition/explanation claims
        TECHNICAL_ANCHORS = {
            "class", "classes", "object", "objects", "instance", "instances",
            "attribute", "attributes", "method", "methods", "function",
            "functions", "type", "data", "namespace", "inheritance",
            "module", "argument", "variable", "state"
        }

        if any(t in TECHNICAL_ANCHORS for t in tokens):
            return True

        # original anchor logic, but now only fallback
        if has_anchor:
            return True

        # accept normal factual sentences with subject + verb
        has_subject = any(t.dep_ in {"nsubj", "nsubjpass"} for t in doc)
        has_verb = any(t.pos_ in {"VERB", "AUX"} for t in doc)

        if has_subject and has_verb and len(tokens) >= 4:
            return True

        if has_verb and len(tokens) >= 4:
            return True

        return False

    # ── Unsupported terms ─────────────────────────────────────────────────────

    def get_unsupported_terms(self, claim, all_truth_tokens):
        claim_tokens = set(self.tokenize(claim))
        all_truth_tokens = set(all_truth_tokens)
        unsupported = {
            t for t in claim_tokens
            if t not in all_truth_tokens
            and not (self.get_synonyms(t) & all_truth_tokens)
        }
        doc = nlp(claim)
        # only proper nouns and numbers are real hallucination signals
        content = {t.lemma_.lower() for t in doc if t.pos_ in {"PROPN", "NUM"} and not t.is_stop}
        return sorted(t for t in unsupported if t in content or (t.isdigit() and len(t) != 4))

    # ── Core claim checker ────────────────────────────────────────────────────

    def check_claim_against_chunks(self, claim, chunks, all_truth_tokens, threshold=0.30):
        candidates = rank_chunks(
            claim=claim,
            chunks=chunks,
            score_fn=self.support_score,
            extract_numbers=self.extract_numbers,
            has_negation_mismatch=self.has_negation_mismatch,
            threshold=threshold,
            top_k=5,
        )
        unsupported_terms = self.get_unsupported_terms(claim, all_truth_tokens)
        evidence = build_evidence(candidates)
        best = best_evidence(candidates)

        if not best:
            return {
                "status": "HALLUCINATION", "claim": claim, "score": 0.0,
                "confidence": 0.0,
                "reason": "No matching chunk found",
                "matched_doc_id": None, "matched_source": None,
                "matched_chunk_id": None, "chunk_text": "",
                "unsupported_terms": unsupported_terms,
                "evidence": [],
            }

        best_chunk = candidates[0]["chunk"]
        best_score = candidates[0]["score"]
        contradiction = find_contradiction(
            claim=claim,
            chunk=best_chunk,
            extract_numbers=self.extract_numbers,
            has_negation_mismatch=self.has_negation_mismatch,
            score=best_score,
            threshold=threshold,
        )
        status = classify_support(
            score=best_score,
            threshold=threshold,
            contradiction=contradiction,
            unsupported_terms=unsupported_terms,
            claim=claim,
        )
        confidence = confidence_score(
            score=best_score,
            evidence_count=len(evidence),
            contradiction=contradiction,
            unsupported_terms=unsupported_terms,
            status=status,
        )
        warning = temporal_warning(claim)

        result = {
            "status": status, "claim": claim, "score": best_score,
            "confidence": confidence,
            "matched_doc_id": best["doc_id"],
            "matched_source": best["source"],
            "matched_chunk_id": best["chunk_id"],
            "chunk_text": best["text"],
            "unsupported_terms": unsupported_terms,
            "evidence": evidence,
        }
        if warning:
            result["warning"] = warning["warning"]
            result["as_of_year"] = warning["as_of_year"]
        if contradiction:
            result["reason"] = contradiction["reason"]
            if "claim_numbers" in contradiction:
                result["ai_numbers"] = contradiction["claim_numbers"]
                result["truth_numbers"] = contradiction["truth_numbers"]
            if "claim_years" in contradiction:
                result["ai_years"] = contradiction["claim_years"]
                result["truth_years"] = contradiction["truth_years"]
        return result

    # ── Public API ────────────────────────────────────────────────────────────

    def compare_to_docs(self, truth_docs, ai_output, threshold=0.30):
        if isinstance(truth_docs, str):
            truth_docs = [{"file_id": 1, "file_path": "inline_text", "text": truth_docs}]
        elif truth_docs and isinstance(truth_docs[0], str):
            truth_docs = [
                {"file_id": i, "file_path": f"inline_text_{i}", "text": t}
                for i, t in enumerate(truth_docs, 1)
            ]

        all_chunks, all_truth_tokens = [], set()
        for doc in truth_docs:
            chunks = self.chunk_text(doc["text"], doc_id=doc["file_id"], source_name=doc["file_path"])
            all_chunks.extend(chunks)
            for chunk in chunks:
                all_truth_tokens.update(chunk["tokens"])

        results = []
        claims = extract_claims(
            ai_output,
            sentence_splitter=self.split_sentences,
            meaningful_filter=self.is_meaningful_claim,
        )
        for claim_data in claims:
            claim = claim_data["claim"]
            claim_type = self.classify_claim_type(claim)
            if claim_type == "MATH":
                result = self.verify_math_claim(claim)
            else:
                result = self.check_claim_against_chunks(
                    claim=claim,
                    chunks=all_chunks,
                    all_truth_tokens=all_truth_tokens,
                    threshold=threshold,
                )
            result["claim_id"] = claim_data["claim_id"]
            result["sentence_id"] = claim_data["sentence_id"]
            result["source_sentence"] = claim_data["source_sentence"]
            result["type"] = claim_type
            results.append(result)
        return results

    def compare_to_files(self, truth_file_paths, ai_output, threshold=0.30):
        return self.compare_to_docs(self.load_files(truth_file_paths), ai_output, threshold)

    def compare_with_reasoning(self, truth_file_paths, ai_output, threshold=0.30):
        return self.compare_to_files(truth_file_paths, ai_output, threshold)

    def print_report(self, results):
        supported = [r for r in results if r["status"] == "SUPPORTED"]
        weak = [r for r in results if r["status"] == "WEAK_SUPPORT"]
        bad = [r for r in results if r["status"] in {"HALLUCINATION", "CONTRADICTION"}]
        uncertain = [r for r in results if r["status"] == "UNVERIFIABLE_DENIAL"]
        total = len(results)
        confidence = (len(supported) + 0.5 * len(weak)) / total if total else 0

        print("\nHalgorithm Report")
        print("=" * 80)
        print(
            f"Strongly supported: {len(supported)}  Weak: {len(weak)}  "
            f"Unverifiable denials: {len(uncertain)}  Issues: {len(bad)}"
        )
        print(f"Confidence: {round(confidence * 100, 2)}%  —  {'reliable' if not bad else 'not reliable'}")
        print("=" * 80)

        if not bad and not uncertain:
            print("No hallucinations found.\n")
            return

        for r in uncertain + bad:
            print("=" * 80)
            print(f"Claim #{r['claim_id']} | {r['status']} | score {round(r.get('score', 0), 3)}")
            print(f"\n{r['claim']}\n")
            if r.get("reason"):
                print(f"Reason: {r['reason']}")
            if r.get("ai_numbers"):
                print(f"AI numbers: {r['ai_numbers']}  Truth numbers: {r['truth_numbers']}")
            if r.get("unsupported_terms"):
                print(f"Unsupported terms: {', '.join(r['unsupported_terms'])}")
            if r.get("chunk_text"):
                print(f"\nClosest chunk ({r['matched_source']}, chunk {r['matched_chunk_id']}):")
                print(r["chunk_text"])
            print()
