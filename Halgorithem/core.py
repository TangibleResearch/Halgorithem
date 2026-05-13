import re
from difflib import SequenceMatcher
from pathlib import Path

import pysbd

from .math_utils import numbers_close, safe_eval
from .text_processing import (
    extract_entities,
    extract_numbers,
    get_synonyms,
    has_negation_mismatch,
    lemmatize_tokens,
    clean_text,
    tokenize,
)


class Halgorithm:
    def __init__(self, sentences_per_chunk=2, sentence_overlap=1):
        self.sentences_per_chunk = sentences_per_chunk
        self.sentence_overlap = sentence_overlap
        self.parser = pysbd.Segmenter(language="en", clean=False)

    def get_synonyms(self, word):
        return get_synonyms(word)

    def classify_claim_type(self, claim):
        claim = claim.lower()
        math_patterns = [
            r"\d+\s*[\+\-\*/%]\s*\d+",
            r"=",
            r"\d+\s*(percent|%)",
        ]

        for pattern in math_patterns:
            if re.search(pattern, claim):
                return "MATH"

        return "SOURCE"

    def extract_math_expression(self, claim):
        if "=" in claim:
            parts = claim.split("=")
            if len(parts) == 2:
                return [part.strip() for part in parts]
        return None

    def verify_math_claim(self, claim):
        parts = self.extract_math_expression(claim)

        if not parts:
            return {
                "status": "UNKNOWN",
                "reason": "No valid math expression",
            }

        left, right = parts

        try:
            result = safe_eval(left)
            expected = safe_eval(right)

            if numbers_close(result, expected):
                return {
                    "status": "SUPPORTED",
                    "claim": claim,
                    "type": "MATH",
                }

            return {
                "status": "CONTRADICTION",
                "claim": claim,
                "expected": result,
                "got": expected,
                "type": "MATH",
            }

        except Exception as e:
            return {
                "status": "ERROR",
                "claim": claim,
                "reason": str(e),
                "type": "MATH",
            }

    def verify_claim(self, claim, chunks, all_truth_tokens):
        claim_type = self.classify_claim_type(claim)

        if claim_type == "MATH":
            return self.verify_math_claim(claim)

        if claim_type == "SOURCE":
            return self.check_claim_against_chunks(
                claim=claim,
                chunks=chunks,
                all_truth_tokens=all_truth_tokens,
            )

        return {
            "status": "UNKNOWN",
            "claim": claim,
        }

    def load_file(self, file_path):
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        return path.read_text(encoding="utf-8")

    def load_files(self, file_paths):
        docs = []

        for file_id, file_path in enumerate(file_paths, start=1):
            text = self.load_file(file_path)

            docs.append({
                "file_id": file_id,
                "file_path": str(file_path),
                "text": text,
            })

        return docs

    def clean_text(self, text):
        return clean_text(text)

    def split_sentences(self, text):
        text = self.clean_text(text)
        sentences = self.parser.segment(text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def tokenize(self, text):
        return tokenize(text)

    def synonym_overlap_score(self, claim_tokens, chunk_tokens):
        overlap = 0
        total = len(claim_tokens)
        chunk_tokens = set(chunk_tokens)

        for token in claim_tokens:
            synonyms = self.get_synonyms(token)

            if token in chunk_tokens or any(s in chunk_tokens for s in synonyms):
                overlap += 1

        return overlap / total if total > 0 else 0

    def lemmatize_tokens(self, text):
        return lemmatize_tokens(text)

    def extract_numbers(self, text):
        return extract_numbers(text)

    def extract_entities(self, text):
        return extract_entities(text)

    def has_negation_mismatch(self, claim, chunk_text):
        return has_negation_mismatch(claim, chunk_text)

    def relevant_truth_numbers(self, claim, chunks, threshold=0.30):
        claim_tokens = {
            token for token in self.tokenize(claim)
            if not token.replace(".", "", 1).isdigit()
        }

        def is_year(n):
            try:
                return 1400 <= float(n) <= 2100
            except (ValueError, TypeError):
                return False

        date_words = {"year", "founded", "created", "invented",
                    "launched", "born", "died", "since"}
        claim_is_about_date = any(
            word in claim.lower() for word in date_words
        )

        numbers = set()
        for chunk in chunks:
            chunk_tokens = set(chunk["tokens"])
            lexical_overlap = claim_tokens & chunk_tokens

            if lexical_overlap or self.support_score(claim, chunk) >= threshold:
                chunk_numbers = set(chunk["numbers"])
                if not claim_is_about_date:
                    chunk_numbers = {n for n in chunk_numbers if not is_year(n)}
                numbers.update(chunk_numbers)

        return numbers

    def chunk_text(self, text, doc_id=1, source_name=None):
        sentences = self.split_sentences(text)
        chunks = []

        if not sentences:
            return chunks

        start = 0
        chunk_id = 1

        while start < len(sentences):
            end = start + self.sentences_per_chunk
            chunk_sentences = sentences[start:end]
            chunk_text = " ".join(chunk_sentences)

            chunks.append({
                "doc_id": doc_id,
                "source_name": source_name,
                "chunk_id": chunk_id,
                "sentence_start": start + 1,
                "sentence_end": min(end, len(sentences)),
                "text": chunk_text,
                "tokens": self.tokenize(chunk_text),
                "lemmas": self.lemmatize_tokens(chunk_text),
                "entities": self.extract_entities(chunk_text),
                "numbers": self.extract_numbers(chunk_text),
            })

            chunk_id += 1

            if end >= len(sentences):
                break

            start = end - self.sentence_overlap

        return chunks

    def token_similarity(self, tokens1, tokens2):
        set1 = set(tokens1)
        set2 = set(tokens2)

        if not set1 or not set2:
            return 0.0

        return len(set1 & set2) / len(set1 | set2)

    def important_overlap_score(self, claim_tokens, chunk_tokens):
        claim_set = set(claim_tokens)
        chunk_set = set(chunk_tokens)

        if not claim_set:
            return 0.0

        return len(claim_set & chunk_set) / len(claim_set)

    def string_similarity(self, text1, text2):
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def support_score(self, claim, chunk):
        claim_tokens = self.tokenize(claim)
        claim_lemmas = self.lemmatize_tokens(claim)
        chunk_tokens = chunk["tokens"]
        chunk_lemmas = chunk.get("lemmas", [])

        token_score = self.token_similarity(claim_tokens, chunk_tokens)
        overlap_score = self.important_overlap_score(claim_tokens, chunk_tokens)
        lemma_score = self.important_overlap_score(claim_lemmas, chunk_lemmas)
        string_score = self.string_similarity(claim, chunk["text"])
        syn_score = self.synonym_overlap_score(claim_tokens, chunk_tokens)

        return (
            token_score * 0.20
            + overlap_score * 0.25
            + lemma_score * 0.20
            + syn_score * 0.20
            + string_score * 0.15
        )

    def has_number_conflict(self, claim, chunk):
        claim_numbers = set(self.extract_numbers(claim))
        truth_numbers = set(chunk["numbers"])

        if not claim_numbers or not truth_numbers:
            return False, claim_numbers, truth_numbers

        def is_year(n):
            try:
                return 1400 <= float(n) <= 2100
            except (ValueError, TypeError):
                return False

        def is_ordinal_or_date(n):
            # mission numbers, day numbers, small ordinals
            # should not be compared against each other
            try:
                return float(n) <= 31
            except (ValueError, TypeError):
                return False

        for claim_num in claim_numbers:
            try:
                claim_val = float(claim_num)
            except (ValueError, TypeError):
                continue

            # skip years and small ordinals entirely
            if is_year(claim_val) or is_ordinal_or_date(claim_val):
                continue

            for truth_num in truth_numbers:
                try:
                    truth_val = float(truth_num)
                except (ValueError, TypeError):
                    continue

                if is_year(truth_val) or is_ordinal_or_date(truth_val):
                    continue

                if claim_val == 0 or truth_val == 0:
                    continue

                ratio = min(claim_val, truth_val) / max(claim_val, truth_val)

                if ratio >= 0.5 and claim_val != truth_val:
                    return True, claim_numbers, truth_numbers

        return False, claim_numbers, truth_numbers
    def get_unsupported_terms(self, claim, all_truth_tokens):
        claim_tokens = set(self.tokenize(claim))
        all_truth_tokens = set(all_truth_tokens)

        unsupported = {
            token for token in claim_tokens
            if token not in all_truth_tokens
            and not (self.get_synonyms(token) & all_truth_tokens)
        }

        unsupported = {
            term for term in unsupported
            if len(term) > 2 or term.isdigit()
        }

        return sorted(unsupported)

    def compare_with_reasoning(self, truth_file_paths, ai_output, threshold=0.30):
        docs = self.load_files(truth_file_paths)

        return self.compare_to_docs(
            truth_docs=docs,
            ai_output=ai_output,
            threshold=threshold,
        )

    def check_claim_against_chunks(self, claim, chunks, all_truth_tokens, threshold=0.30):
        best_chunk = None
        best_score = 0.0
        best_number_conflict = None
        best_negation_mismatch = False
        claim_tokens = set(self.tokenize(claim))
        claim_entities = self.extract_entities(claim)

        for chunk in chunks:
            chunk_tokens = set(chunk["tokens"])
            score = min(self.support_score(claim, chunk), 1.0)

            technical_hits = [
                token for token in claim_tokens
                if len(token) > 7 and token in chunk_tokens
            ]
            if technical_hits:
                score += 0.02 * len(technical_hits)

            generic_words = {
                "easier", "easy", "simple", "students",
                "people", "helped", "helps", "made", "useful",
            }
            generic_hits = [
                token for token in claim_tokens
                if token in generic_words and token not in chunk_tokens
            ]
            if generic_hits:
                score -= 0.04 * len(generic_hits)

            negation_mismatch = self.has_negation_mismatch(claim, chunk["text"])
            if negation_mismatch and score >= threshold:
                score -= 0.35

            number_conflict, claim_numbers, truth_numbers = self.has_number_conflict(
                claim, chunk
            )

            if claim_numbers and claim_numbers.issubset(truth_numbers):
                score += 0.15

            chunk_entities = set(chunk.get("entities", set()))
            if claim_entities and claim_entities & chunk_entities:
                score += 0.05 * len(claim_entities & chunk_entities)

            if score > best_score:
                best_score = score
                best_chunk = chunk
                best_negation_mismatch = negation_mismatch
                best_number_conflict = (
                    {
                        "claim_numbers": sorted(claim_numbers),
                        "truth_numbers": sorted(truth_numbers),
                    }
                    if number_conflict
                    else None
                )

        if not best_chunk:
            return {
                "status": "HALLUCINATION",
                "claim": claim,
                "score": 0.0,
                "reason": "No matching chunk found",
                "matched_doc_id": None,
                "matched_source": None,
                "matched_chunk_id": None,
                "chunk_text": "",
                "unsupported_terms": self.get_unsupported_terms(claim, all_truth_tokens),
            }

        unsupported_terms = self.get_unsupported_terms(claim, all_truth_tokens)
        
        if best_number_conflict:
            return {
                "status": "CONTRADICTION",
                "claim": claim,
                "score": best_score,
                "reason": "Number mismatch",
                "ai_numbers": best_number_conflict["claim_numbers"],
                "truth_numbers": best_number_conflict["truth_numbers"],
                "matched_doc_id": best_chunk["doc_id"],
                "matched_source": best_chunk["source_name"],
                "matched_chunk_id": best_chunk["chunk_id"],
                "chunk_text": best_chunk["text"],
                "unsupported_terms": unsupported_terms,
            }

        if best_negation_mismatch and best_score >= threshold:
            return {
                "status": "CONTRADICTION",
                "claim": claim,
                "score": best_score,
                "reason": "Negation mismatch",
                "matched_doc_id": best_chunk["doc_id"],
                "matched_source": best_chunk["source_name"],
                "matched_chunk_id": best_chunk["chunk_id"],
                "chunk_text": best_chunk["text"],
                "unsupported_terms": unsupported_terms,
            }

        chunk_tokens = set(best_chunk["tokens"])
        chunk_entities = set(best_chunk.get("entities", set()))
        missing_entities = []

        for ent_tokens in claim_entities:
            if ent_tokens not in chunk_entities and not set(ent_tokens).issubset(chunk_tokens):
                missing_entities.append(" ".join(ent_tokens))

        if missing_entities and best_score < 0.85:
            return {
                "status": "HALLUCINATION",
                "claim": claim,
                "score": best_score,
                "reason": "Unsupported key entity",
                "matched_doc_id": best_chunk["doc_id"],
                "matched_source": best_chunk["source_name"],
                "matched_chunk_id": best_chunk["chunk_id"],
                "chunk_text": best_chunk["text"],
                "unsupported_terms": unsupported_terms,
            }

        descriptive_words = {
            "easier", "easy", "simple", "students", "people",
            "helped", "helps", "made", "useful", "better",
        }
        is_descriptive = any(token in descriptive_words for token in claim_tokens)
        strong_tokens = [
            token for token in claim_tokens
            if len(token) > 6 and token in set(best_chunk["tokens"])
        ]
        has_strong_match = len(strong_tokens) >= 1

        if best_score >= 0.65 or (best_score >= 0.60 and has_strong_match):
            status = "WEAK_SUPPORT" if is_descriptive else "SUPPORTED"
        elif best_score >= threshold:
            status = "WEAK_SUPPORT"
        else:
            status = "HALLUCINATION"

        return {
            "status": status,
            "claim": claim,
            "score": best_score,
            "matched_doc_id": best_chunk["doc_id"],
            "matched_source": best_chunk["source_name"],
            "matched_chunk_id": best_chunk["chunk_id"],
            "chunk_text": best_chunk["text"],
            "unsupported_terms": unsupported_terms,
        }
    def is_meaningful_claim(self, claim):
        tokens = self.tokenize(claim)
        # skip if too short
        if len(tokens) < 4:
            return False
        # skip if no content words (just transition phrases)
        filler = {
            "here", "are", "the", "key", "details", "following",
            "below", "above", "note", "please", "this", "these",
            "is", "was", "were", "be", "been"
        }
        content_tokens = [t for t in tokens if t not in filler]
        if len(content_tokens) < 2:
            return False
        return True
    def compare_to_docs(self, truth_docs, ai_output, threshold=0.30):
        if isinstance(truth_docs, str):
            truth_docs = [{
                "file_id": 1,
                "file_path": "inline_text",
                "text": truth_docs,
            }]
        elif truth_docs and isinstance(truth_docs[0], str):
            truth_docs = [
                {
                    "file_id": i,
                    "file_path": f"inline_text_{i}",
                    "text": text,
                }
                for i, text in enumerate(truth_docs, start=1)
            ]

        all_chunks = []
        all_truth_tokens = set()

        for doc in truth_docs:
            chunks = self.chunk_text(
                doc["text"],
                doc_id=doc["file_id"],
                source_name=doc["file_path"],
            )
            all_chunks.extend(chunks)

            for chunk in chunks:
                all_truth_tokens.update(chunk["tokens"])

        results = []

        for claim_id, claim in enumerate(self.split_sentences(ai_output), start=1):
            
            if not self.is_meaningful_claim(claim):  # add this
                continue
                
            claim_type = self.classify_claim_type(claim)

            if claim_type == "MATH":
                result = self.verify_math_claim(claim)
            else:
                parts = self.decompose_simple(claim)
                part_results = [
                    self.check_claim_against_chunks(
                        claim=part,
                        chunks=all_chunks,
                        all_truth_tokens=all_truth_tokens,
                        threshold=threshold,
                    )
                    for part in parts
                ]
                result = self.merge_part_results(claim, part_results)

            result["claim_id"] = claim_id
            result["type"] = claim_type
            results.append(result)

        return results

    def compare_to_files(self, truth_file_paths, ai_output, threshold=0.30):
        docs = self.load_files(truth_file_paths)

        return self.compare_to_docs(
            truth_docs=docs,
            ai_output=ai_output,
            threshold=threshold,
        )

    def merge_part_results(self, claim, part_results):
        if len(part_results) == 1:
            return part_results[0]

        final_status, final_score = self.score_parts(part_results)
        if final_status == "PARTIAL":
            final_status = "WEAK_SUPPORT"

        weakest_part = min(part_results, key=lambda part: part.get("score", 0))
        return {
            "status": final_status,
            "claim": claim,
            "score": final_score,
            "parts": part_results,
            "reason": weakest_part.get("reason", "See decomposed claim parts"),
            "matched_doc_id": weakest_part.get("matched_doc_id"),
            "matched_source": weakest_part.get("matched_source"),
            "matched_chunk_id": weakest_part.get("matched_chunk_id"),
            "chunk_text": weakest_part.get("chunk_text", ""),
            "unsupported_terms": weakest_part.get("unsupported_terms", []),
        }

    def score_parts(self, part_results):
        score = 0

        for result in part_results:
            if result["status"] == "SUPPORTED":
                score += 1.0
            elif result["status"] == "WEAK_SUPPORT":
                score += 0.5
            elif result["status"] == "HALLUCINATION":
                score -= 0.75
            elif result["status"] == "CONTRADICTION":
                score -= 2.0

        if any(result["status"] == "CONTRADICTION" for result in part_results):
            return "CONTRADICTION", score
        if any(result["status"] == "HALLUCINATION" for result in part_results) and score < 0:
            return "HALLUCINATION", score
        if any(result["status"] == "HALLUCINATION" for result in part_results):
            return "PARTIAL", score
        if score >= 1:
            return "SUPPORTED", score
        if score <= -2:
            return "CONTRADICTION", score
        return "PARTIAL", score

    def decompose_simple(self, claim):
        parts = []
        claim = claim.lower()
        parts.append(claim)  # full claim always first

        for separator in (" and ", " but ", " while "):
            if separator in claim:
                new_parts = [
                    part.strip()
                    for part in claim.split(separator)
                    if len(part.strip().split()) >= 3
                ]
                parts.extend(new_parts)

        if " by " in claim:
            before, after = claim.split(" by ", 1)
            parts.append(before.strip())
            parts.append(f"creator is {after.strip().split()[0]}")

        for word in claim.split():
            if word.isdigit() and len(word) == 4:
                parts.append(f"year is {word}")

        # dedupe but preserve order
        seen = set()
        unique_parts = []
        for part in parts:
            if part not in seen:
                seen.add(part)
                unique_parts.append(part)

        return unique_parts

    def print_report(self, results):
        strong_supported = [r for r in results if r["status"] == "SUPPORTED"]
        weak_supported = [r for r in results if r["status"] == "WEAK_SUPPORT"]
        bad_results = [
            r for r in results
            if r["status"] in {"HALLUCINATION", "CONTRADICTION"}
        ]
        total = len(results)
        confidence = (
            len(strong_supported) + 0.5 * len(weak_supported)
        ) / total if total > 0 else 0

        print()
        print("Halgorithm Report")
        print("=" * 80)
        print(f"Strongly supported claims: {len(strong_supported)}")
        print(f"Weakly supported claims: {len(weak_supported)}")
        print(f"Possible hallucinations/contradictions: {len(bad_results)}")

        reliability = "not reliable" if bad_results else "reliable"

        print(
            f"Summary: {round(confidence * 100, 2)}% confidence in AI output based on truth documents. "
            f"Meaning {round(len(bad_results) / total * 100, 2)}% may be hallucinated or contradictory. "
            f"Which concludes that the AI output is {reliability}."
        )
        print("=" * 80)
        print()

        if not bad_results:
            print("No hallucinations found.")
            return

        for result in bad_results:
            print("=" * 80)
            print(f"Claim #{result['claim_id']}")
            print(f"Status: {result['status']}")
            print(f"Score: {round(result['score'], 3)}")
            print()
            print("AI Claim:")
            print(result["claim"])
            print()

            if "parts" in result:
                print("\nDecomposed Parts:")
                for part in result["parts"]:
                    print(f"- {part['claim']} -> {part['status']}")

            if result["status"] == "CONTRADICTION":
                print(f"Reason: {result['reason']}")
                if "ai_numbers" in result and "truth_numbers" in result:
                    print(f"AI says numbers: {result['ai_numbers']}")
                    print(f"Truth has numbers: {result['truth_numbers']}")
            else:
                print(f"Reason: {result['reason']}")

            if result.get("unsupported_terms"):
                print()
                print("Unsupported AI terms:")
                print(", ".join(result["unsupported_terms"]))

            print()
            print(
                f"Closest truth chunk "
                f"(File: {result['matched_source']}, Chunk {result['matched_chunk_id']}):"
            )
            print(result["chunk_text"])
            print("=" * 80)
            print()
