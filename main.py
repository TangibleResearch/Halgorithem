from lark import Lark, Tree, Token
import re
from difflib import SequenceMatcher
from pathlib import Path
from cleantext import clean
import spacy
import textacy.preprocessing as tprep
nlp = spacy.load("en_core_web_sm")
import pysbd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


import ast
import operator

ops = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod
}

def eval_expr(node):
    if isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.BinOp):
        return ops[type(node.op)](
            eval_expr(node.left),
            eval_expr(node.right)
        )
    else:
        raise ValueError("Unsafe expression")

def safe_eval(expr):
    tree = ast.parse(expr, mode='eval')
    return eval_expr(tree.body)

def classify_claim_type(self, claim):
    claim = claim.lower()

    # math patterns
    math_patterns = [
        r"\d+\s*[\+\-\*/%]\s*\d+",   # 1 + 1
        r"=",                        # equation
        r"\d+\s*(percent|%)",        # 20%
    ]

    for pattern in math_patterns:
        if re.search(pattern, claim):
            return "MATH"

    return "SOURCE" 
def extract_math_expression(self, claim):
    # simple version (can improve later)
    if "=" in claim:
        return claim.split("=")
    return None
def verify_math_claim(self, claim):
    parts = self.extract_math_expression(claim)

    if not parts:
        return {
            "status": "UNKNOWN",
            "reason": "No valid math expression"
        }

    left, right = parts

    try:
        result = safe_eval(left)
        expected = safe_eval(right)

        if result == expected:
            return {
                "status": "SUPPORTED",
                "claim": claim,
                "type": "MATH"
            }
        else:
            return {
                "status": "CONTRADICTION",
                "claim": claim,
                "expected": result,
                "got": expected,
                "type": "MATH"
            }

    except Exception as e:
        return {
            "status": "ERROR",
            "claim": claim,
            "reason": str(e),
            "type": "MATH"
        }
def verify_claim(self, claim, chunks, all_truth_tokens):
    claim_type = self.classify_claim_type(claim)

    if claim_type == "MATH":
        return self.verify_math_claim(claim)

    elif claim_type == "SOURCE":
        return self.check_claim_against_chunks(
            claim=claim,
            chunks=chunks,
            all_truth_tokens=all_truth_tokens
        )

    return {
        "status": "UNKNOWN",
        "claim": claim
    }
class Halgorithm:
    def __init__(self, sentences_per_chunk=2, sentence_overlap=1):
        self.sentences_per_chunk = sentences_per_chunk
        self.sentence_overlap = sentence_overlap

        # 1. The 'Grammar': We keep your logic but move it to a 
        # compiled Regex for 10x speed and better precision.
        # This matches words, decimal numbers, and A-Z dots.
        self.grammar = re.compile(
            r"[A-Za-z]+(?:[-'][A-Za-z]+)*|"  # Words/Hyphens
            r"[0-9]+(?:\.[0-9]+)?|"          # Numbers/Decimals
            r"[A-Z]\."                       # Single letter initials
        )

        # 2. The 'Parser': PySBD is a rule-based segmenter.
        # It's 'perfect' because it replaces Lark's rigidity with 
        # actual English language rules for sentence boundaries.
        self.parser = pysbd.Segmenter(language="en", clean=False)

        # 3. The Stopwords: Static and reliable
        self.stopwords = set(ENGLISH_STOP_WORDS)

    # ------------------------------------------------- LOADING
    # -------------------------------------------------

    def load_file(self, file_path):
        """
        Loads one text file and returns its content.
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        return path.read_text(encoding="utf-8")

    def load_files(self, file_paths):
        """
        Loads multiple text files.

        Returns:
        [
            {
                "file_id": 1,
                "file_path": "...",
                "text": "..."
            }
        ]
        """

        docs = []

        for file_id, file_path in enumerate(file_paths, start=1):
            text = self.load_file(file_path)

            docs.append({
                "file_id": file_id,
                "file_path": str(file_path),
                "text": text
            })

        return docs

    # -------------------------------------------------
    # TEXT CLEANING
    # -------------------------------------------------

    def clean_text(self, text):
        
        if not text: return ""

        # Handles your quotes, whitespace, and strip() automatically
        text = tprep.normalize.unicode(text)
        text = tprep.normalize.whitespace(text)
        text = tprep.normalize.quotation_marks(text)
        
        # Removes the extra punctuation you listed ((), ;, :)
        text = tprep.remove.punctuation(text, only=[ "(", ")", ";", ":", "\"" ])

        # Final logic for the period
        if text and text[-1] not in ".!?":
            text += "."
            
        return clean(text,
            fix_unicode=True,               # Fixes the ’ and “ characters
            to_ascii=True,                  # Simplifies everything to standard text
            no_urls=True,                   # Replaces URLs with <URL>
            no_emails=True,                 # Replaces emails with <EMAIL>
            replace_with_punct="",          # Can help strip specific punct
            lang="en"
        )

    # -------------------------------------------------
    # AST PARSING
    # -------------------------------------------------

    def get_ast(self, text):
        clean = self.clean_text(text)

        try:
            return self.parser.parse(clean)
        except Exception as e:
            return f"Parse Error: {e}"

    def extract_tokens_from_ast(self, ast):
        tokens = []

        def walk(node):
            if isinstance(node, Tree):
                for child in node.children:
                    walk(child)
            elif isinstance(node, Token):
                if node.type == "DATA":
                    tokens.append(str(node))

        walk(ast)
        return tokens

    # -------------------------------------------------
    # SENTENCES + TOKENS
    # -------------------------------------------------

    def split_sentences(self, text):
        text = self.clean_text(text)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def tokenize(self, text):
        ast = self.get_ast(text)

        if isinstance(ast, str):
            # fallback tokenizer if Lark cannot parse a weird sentence
            return self.fallback_tokenize(text)

        raw_tokens = self.extract_tokens_from_ast(ast)
        return self.normalize_tokens(raw_tokens)

    def fallback_tokenize(self, text):
        """
        Backup tokenizer so one weird character does not destroy the whole run.
        """
        raw_tokens = re.findall(
            r"[A-Za-z]+(?:[-'][A-Za-z]+)*|[0-9]+(?:\.[0-9]+)?|[A-Z]\.",
            text
        )
        return self.normalize_tokens(raw_tokens)

    def normalize_tokens(self, raw_tokens):
        useful_tokens = []

        for token in raw_tokens:
            normalized = token.lower().strip(".")

            if not normalized:
                continue

            if normalized in self.stopwords:
                continue

            useful_tokens.append(normalized)

        return useful_tokens

    def extract_numbers(self, text):
        return re.findall(r"\b\d+(?:\.\d+)?\b", text)

    # -------------------------------------------------
    # CHUNKING
    # -------------------------------------------------

    def chunk_text(self, text, doc_id=1, source_name=None):
        """
        Sentence-based chunks.

        Chunk 1 = sentence 1 + sentence 2
        Chunk 2 = sentence 2 + sentence 3
        """

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
                "numbers": self.extract_numbers(chunk_text)
            })

            chunk_id += 1

            if end >= len(sentences):
                break

            start = end - self.sentence_overlap

        return chunks

    # -------------------------------------------------
    # SCORING
    # -------------------------------------------------

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
        chunk_tokens = chunk["tokens"]

        token_score = self.token_similarity(claim_tokens, chunk_tokens)
        overlap_score = self.important_overlap_score(claim_tokens, chunk_tokens)
        string_score = self.string_similarity(claim, chunk["text"])

        return (
            token_score * 0.40 +
            overlap_score * 0.45 +
            string_score * 0.15
        )

    # -------------------------------------------------
    # CONTRADICTION CHECKS
    # -------------------------------------------------

    def has_number_conflict(self, claim, chunk):
        claim_numbers = set(self.extract_numbers(claim))
        truth_numbers = set(chunk["numbers"])

        if claim_numbers and truth_numbers and claim_numbers != truth_numbers:
            return True, claim_numbers, truth_numbers

        return False, claim_numbers, truth_numbers

    def get_unsupported_terms(self, claim, all_truth_tokens):
        claim_tokens = set(self.tokenize(claim))

        unsupported = claim_tokens - all_truth_tokens

        unsupported = {
            term for term in unsupported
            if len(term) > 2 or term.isdigit()
        }

        return sorted(unsupported)

    # -------------------------------------------------
    # CLAIM CHECKING
    # -------------------------------------------------
    def compare_with_reasoning(self, truth_file_paths, ai_output, threshold=0.30):
        docs = self.load_files(truth_file_paths)

        all_chunks = []
        all_truth_tokens = set()

        for doc in docs:
            chunks = self.chunk_text(
                doc["text"],
                doc_id=doc["file_id"],
                source_name=doc["file_path"]
            )

            all_chunks.extend(chunks)

            for chunk in chunks:
                all_truth_tokens.update(chunk["tokens"])

        ai_claims = self.split_sentences(ai_output)

        results = []

        for claim_id, claim in enumerate(ai_claims, start=1):
            claim_type = self.classify_claim_type(claim)

            if claim_type == "MATH":
                result = self.verify_math_claim(claim)

            else:
                result = self.check_claim_against_chunks(
                    claim=claim,
                    chunks=all_chunks,
                    all_truth_tokens=all_truth_tokens,
                    threshold=threshold
                )

            result["claim_id"] = claim_id
            result["type"] = claim_type

            results.append(result)

        return results
    def check_claim_against_chunks(self, claim, chunks, all_truth_tokens, threshold=0.30):
        best_chunk = None
        best_score = 0.0
        best_number_conflict = None

        for chunk in chunks:
            score = self.support_score(claim, chunk)

            number_conflict, claim_numbers, truth_numbers = self.has_number_conflict(
                claim,
                chunk
            )

            if score > best_score:
                best_score = score
                best_chunk = chunk

                if number_conflict:
                    best_number_conflict = {
                        "claim_numbers": sorted(claim_numbers),
                        "truth_numbers": sorted(truth_numbers)
                    }
                else:
                    best_number_conflict = None

            if score >= threshold and not number_conflict:
                return {
                    "status": "SUPPORTED",
                    "claim": claim,
                    "score": score,
                    "matched_doc_id": chunk["doc_id"],
                    "matched_source": chunk["source_name"],
                    "matched_chunk_id": chunk["chunk_id"],
                    "chunk_text": chunk["text"],
                    "unsupported_terms": []
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
                "matched_doc_id": best_chunk["doc_id"] if best_chunk else None,
                "matched_source": best_chunk["source_name"] if best_chunk else None,
                "matched_chunk_id": best_chunk["chunk_id"] if best_chunk else None,
                "chunk_text": best_chunk["text"] if best_chunk else "",
                "unsupported_terms": unsupported_terms
            }

        return {
            "status": "HALLUCINATION",
            "claim": claim,
            "score": best_score,
            "reason": "No truth chunk supports this claim",
            "matched_doc_id": best_chunk["doc_id"] if best_chunk else None,
            "matched_source": best_chunk["source_name"] if best_chunk else None,
            "matched_chunk_id": best_chunk["chunk_id"] if best_chunk else None,
            "chunk_text": best_chunk["text"] if best_chunk else "",
            "unsupported_terms": unsupported_terms
        }

    # -------------------------------------------------
    # COMPARE AGAINST TEXT DOCS
    # -------------------------------------------------

    def compare_to_docs(self, truth_docs, ai_output, threshold=0.30):
        """
        truth_docs can be:
        - string
        - list of strings
        - list of dicts from load_files()
        """

        if isinstance(truth_docs, str):
            truth_docs = [{
                "file_id": 1,
                "file_path": "inline_text",
                "text": truth_docs
            }]

        elif truth_docs and isinstance(truth_docs[0], str):
            truth_docs = [
                {
                    "file_id": i,
                    "file_path": f"inline_text_{i}",
                    "text": text
                }
                for i, text in enumerate(truth_docs, start=1)
            ]

        all_chunks = []
        all_truth_tokens = set()

        for doc in truth_docs:
            doc_id = doc["file_id"]
            source_name = doc["file_path"]
            doc_text = doc["text"]

            chunks = self.chunk_text(
                doc_text,
                doc_id=doc_id,
                source_name=source_name
            )

            all_chunks.extend(chunks)

            for chunk in chunks:
                all_truth_tokens.update(chunk["tokens"])

        ai_claims = self.split_sentences(ai_output)

        results = []

        for claim_id, claim in enumerate(ai_claims, start=1):
            result = self.verify_claim(
                claim=claim,
                chunks=all_chunks,
                all_truth_tokens=all_truth_tokens
            )

            result["claim_id"] = claim_id
            results.append(result)

        return results

    # -------------------------------------------------
    # COMPARE AGAINST FILES
    # -------------------------------------------------

    def compare_to_files(self, truth_file_paths, ai_output, threshold=0.30):
        """
        Loads files, chunks them, and compares AI output against them.
        """

        docs = self.load_files(truth_file_paths)

        return self.compare_to_docs(
            truth_docs=docs,
            ai_output=ai_output,
            threshold=threshold
        )

    # -------------------------------------------------
    # REPORT
    # -------------------------------------------------

    def print_report(self, results):
        bad_results = [
            r for r in results
            if r["status"] in {"HALLUCINATION", "CONTRADICTION"}
        ]

        supported_results = [
            r for r in results
            if r["status"] == "SUPPORTED"
        ]

        print()
        print("Halgorithm Report")
        print("=" * 80)
        print(f"Supported claims: {len(supported_results)}")
        print(f"Possible hallucinations/contradictions: {len(bad_results)}")
        reliability = "Hallucinated" if len(bad_results) > len(supported_results) else "Real"

        print(
            f"Summary: {round(len(supported_results) / len(results) * 100, 2)}% of claims are supported by the truth documents. "
            f"Meaning {round(len(bad_results) / len(results) * 100, 2)}% may be hallucinated or contradictory. "
            f"Which concludes that the AI output is {reliability} reliable based on the provided truth documents."
        )
        print("=" * 80)
        print()

        if not bad_results:
            print("✅ No hallucinations found.")
            return

        for r in bad_results:
            print("=" * 80)
            print(f"Claim #{r['claim_id']}")
            print(f"Status: {r['status']}")
            print(f"Score: {round(r['score'], 3)}")
            print()
            print("AI Claim:")
            print(r["claim"])
            print()

            if r["status"] == "CONTRADICTION":
                print(f"Reason: {r['reason']}")
                print(f"AI says numbers: {r['ai_numbers']}")
                print(f"Truth has numbers: {r['truth_numbers']}")
            else:
                print(f"Reason: {r['reason']}")

            if r["unsupported_terms"]:
                print()
                print("Unsupported AI terms:")
                print(", ".join(r["unsupported_terms"]))

            print()
            print(
                f"Closest truth chunk "
                f"(File: {r['matched_source']}, Chunk {r['matched_chunk_id']}):"
            )
            print(r["chunk_text"])
            print("=" * 80)
            print()


# EXAMPLE USAGE
# NOTE This is just a demo. In real usage, you would load actual text files with accurate information to compare against the AI output.
if __name__ == "__main__":
    ai_output = """
    The programming language BASIC was originally developed in 1972 by a secret NASA research group to control early satellite communication systems.
    It was later adapted for educational use after engineers realized its simplicity could help train astronauts in computational thinking.
    BASIC original compiler was written in assembly and deployed on spacecraft navigation computers, allowing real-time trajectory adjustments.
    This early success led to its adoption in universities, where it became the foundation for modern AI systems.
    """

    algo = Halgorithm(
        sentences_per_chunk=2,
        sentence_overlap=1
    )

    results = algo.compare_with_reasoning(
        truth_file_paths=[
            "sources/BASIC.txt",
            "sources/BASIC2.txt"
        ],
        ai_output=ai_output,
        threshold=0.30
    )

    algo.print_report(results)