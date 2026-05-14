import re
from pathlib import Path

import pysbd
from sentence_transformers import SentenceTransformer, util

from .math_utils import numbers_close, safe_eval
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
        return [s.strip() for s in sentences if s.strip()]

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
        while start < len(sentences):
            end = start + self.sentences_per_chunk
            chunk = " ".join(sentences[start:end])
            chunks.append({
                "doc_id": doc_id,
                "source_name": source_name,
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
        claim_numbers = set(self.extract_numbers(claim))
        truth_numbers = set(chunk["numbers"])
        if not claim_numbers or not truth_numbers:
            return False, claim_numbers, truth_numbers

        def skip(n):
            try:
                v = float(n)
                return 1400 <= v <= 2100 or v <= 31  # years and small ordinals
            except (ValueError, TypeError):
                return True

        for cn in claim_numbers:
            if skip(cn):
                continue
            cv = float(cn)
            for tn in truth_numbers:
                if skip(tn):
                    continue
                tv = float(tn)
                if cv == 0 or tv == 0:
                    continue
                if min(cv, tv) / max(cv, tv) >= 0.5 and cv != tv:
                    return True, claim_numbers, truth_numbers
        return False, claim_numbers, truth_numbers

    # ── Meaningful claim filter ───────────────────────────────────────────────

    def is_meaningful_claim(self, claim):
        if len(self.tokenize(claim)) < 4:
            return False
        last_word = claim.strip().rstrip(".").split()[-1].lower()
        if last_word in {"including", "such", "namely", "follows", "following", "as"}:
            return False
        doc = nlp(claim)
        # summary sentence — demonstrative subject
        subject = next((t for t in doc if t.dep_ == "nsubj"), None)
        if subject and subject.text.lower() in {"these", "this", "those", "such"}:
            return False
        # summary sentence — interpretive root verb
        root = next((t for t in doc if t.dep_ == "ROOT"), None)
        SUMMARY_VERBS = {
            "reflect", "demonstrate", "highlight", "illustrate", "suggest",
            "indicate", "underscore", "emphasize", "represent", "signal",
            "mark", "mean", "position", "pivot",
        }
        if root and root.lemma_.lower() in SUMMARY_VERBS:
            return False
        # no verifiable anchor — no named entity, number, or proper noun
        if not any(doc.ents) and not any(t.like_num for t in doc) and not any(t.pos_ == "PROPN" for t in doc):
            return False
        return True

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
        best_chunk, best_score = None, 0.0
        best_number_conflict, best_negation = None, False

        for chunk in chunks:
            score = self.support_score(claim, chunk)

            # number subset bonus
            claim_numbers = set(self.extract_numbers(claim))
            if claim_numbers and claim_numbers.issubset(set(chunk["numbers"])):
                score = min(score + 0.10, 1.0)

            # negation penalty
            negation = self.has_negation_mismatch(claim, chunk["text"])
            if negation and score >= threshold:
                score -= 0.30

            # number conflict
            conflict, cnums, tnums = self.has_number_conflict(claim, chunk)

            if score > best_score:
                best_score = score
                best_chunk = chunk
                best_negation = negation
                best_number_conflict = (
                    {"claim_numbers": sorted(cnums), "truth_numbers": sorted(tnums)}
                    if conflict else None
                )

        unsupported_terms = self.get_unsupported_terms(claim, all_truth_tokens)

        if not best_chunk:
            return {
                "status": "HALLUCINATION", "claim": claim, "score": 0.0,
                "reason": "No matching chunk found",
                "matched_doc_id": None, "matched_source": None,
                "matched_chunk_id": None, "chunk_text": "",
                "unsupported_terms": unsupported_terms,
            }

        if best_number_conflict:
            return {
                "status": "CONTRADICTION", "claim": claim, "score": best_score,
                "reason": "Number mismatch",
                "ai_numbers": best_number_conflict["claim_numbers"],
                "truth_numbers": best_number_conflict["truth_numbers"],
                "matched_doc_id": best_chunk["doc_id"],
                "matched_source": best_chunk["source_name"],
                "matched_chunk_id": best_chunk["chunk_id"],
                "chunk_text": best_chunk["text"],
                "unsupported_terms": unsupported_terms,
            }

        if best_negation and best_score >= threshold:
            return {
                "status": "CONTRADICTION", "claim": claim, "score": best_score,
                "reason": "Negation mismatch",
                "matched_doc_id": best_chunk["doc_id"],
                "matched_source": best_chunk["source_name"],
                "matched_chunk_id": best_chunk["chunk_id"],
                "chunk_text": best_chunk["text"],
                "unsupported_terms": unsupported_terms,
            }

        # score-only status — no hardcoded word lists
        if best_score >= 0.65:
            status = "SUPPORTED"
        elif best_score >= threshold:
            status = "WEAK_SUPPORT"
        else:
            status = "HALLUCINATION"

        return {
            "status": status, "claim": claim, "score": best_score,
            "matched_doc_id": best_chunk["doc_id"],
            "matched_source": best_chunk["source_name"],
            "matched_chunk_id": best_chunk["chunk_id"],
            "chunk_text": best_chunk["text"],
            "unsupported_terms": unsupported_terms,
        }

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
        for claim_id, claim in enumerate(self.split_sentences(ai_output), 1):
            if not self.is_meaningful_claim(claim):
                continue
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
            result["claim_id"] = claim_id
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
        total = len(results)
        confidence = (len(supported) + 0.5 * len(weak)) / total if total else 0

        print("\nHalgorithm Report")
        print("=" * 80)
        print(f"Strongly supported: {len(supported)}  Weak: {len(weak)}  Issues: {len(bad)}")
        print(f"Confidence: {round(confidence * 100, 2)}%  —  {'reliable' if not bad else 'not reliable'}")
        print("=" * 80)

        if not bad:
            print("No hallucinations found.\n")
            return

        for r in bad:
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