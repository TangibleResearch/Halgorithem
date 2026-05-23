<p align="center">
  <img src="assets/Tangible.png" style="width: 60%; height: auto;">
</p>

# Halgorithem

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

> Detecting AI hallucinations before they spread through a workflow.

Halgorithem is a deterministic hallucination detection engine for checking AI output against trusted source material. The verifier itself does not call an LLM. It uses parsing, sentence chunking, embeddings, entity extraction, number checks, negation checks, source scoring, evidence retrieval, and confidence scoring to decide whether each factual claim is supported by the supplied sources.

The optional `engine.py` wrapper can call OpenAI to generate an answer, but the verification package in `Halgorithem/` is rule-based and model-free in the generative sense.

## What It Does

Halgorithem answers one question:

> Given source documents and an AI response, which claims are supported, weakly supported, contradicted, hallucinated, or unverifiable?

It returns claim-level results with:

- `status`: `SUPPORTED`, `WEAK_SUPPORT`, `CONTRADICTION`, `HALLUCINATION`, or `UNVERIFIABLE_DENIAL`
- `confidence`: normalized confidence score from `0.0` to `1.0`
- `score`: semantic evidence score
- `claim`: the atomic factual claim being checked
- `matched_source`: source file or URL for the closest evidence
- `chunk_text`: closest evidence chunk
- `evidence`: top ranked evidence chunks, not just the best match
- `unsupported_terms`: proper nouns or numbers that appear in the claim but not the source material
- `reason`: contradiction reason when one is detected
- `warning`: optional risk hint, such as a time-sensitive "current/latest" claim

## Core Pipeline

```text
AI output
  -> clean text
  -> split into sentences
  -> extract atomic claims
  -> filter meaningful factual claims
  -> retrieve top evidence chunks
  -> check contradictions
  -> score confidence
  -> return claim-level report
```

Source documents follow a matching path:

```text
source text
  -> clean text
  -> split into sentences
  -> chunk with overlap
  -> extract tokens, entities, numbers
  -> score source quality
  -> embed chunks
  -> use for retrieval and verification
```

## New Smart Modules

The verifier has been split into smaller modules so each part can get smarter without making `core.py` messy.

| Module | Purpose | AI-Free? |
|---|---|---|
| `claim_extraction.py` | Splits AI output into smaller atomic factual claims. | Yes |
| `retrieval.py` | Ranks the top evidence chunks for each claim. | Yes |
| `evidence.py` | Converts ranked chunks into structured evidence records. | Yes |
| `contradiction.py` | Detects number, date, and negation conflicts. | Yes |
| `confidence.py` | Converts evidence strength and risk signals into final labels. | Yes |
| `temporal.py` | Extracts years and checks time-sensitive/date claims. | Yes |
| `source_quality.py` | Scores source reliability and scraped text quality. | Yes |
| `text_processing.py` | Cleans text, tokenizes, extracts numbers/entities, and handles negation helpers. | Yes |
| `math_utils.py` | Safely verifies simple math expressions. | Yes |
| `web.py` | Scrapes URLs into source text. | Yes |
| `engine.py` | Optional wrapper that can generate text with OpenAI before verification. | No, generation is optional |

## Why Atomic Claims Matter

A sentence can contain multiple facts:

```text
Apollo 11 launched in 1969 and landed at Tranquility Base.
```

Halgorithem now tries to split that into smaller claims:

```text
Apollo 11 launched in 1969.
landed at Tranquility Base.
```

This improves detection because one part of a sentence can be correct while another part is unsupported or contradicted.

## Verdicts

### `SUPPORTED`

The claim has strong evidence in the supplied documents and no major unsupported terms.

### `WEAK_SUPPORT`

The claim is close to the source material but not strong enough for full support. This usually means the source is semantically similar but missing a precise entity, number, or phrase.

### `CONTRADICTION`

The claim matches relevant source material but conflicts with it. Current checks include:

- number mismatch
- date/year mismatch
- negation mismatch

Time-sensitive claims such as "current", "latest", or "today" are marked with a warning so downstream apps can require fresher sources before trusting the claim.

### `HALLUCINATION`

The claim does not have enough support in the supplied documents.

### `UNVERIFIABLE_DENIAL`

The claim denies an unsupported entity or fact, such as "NASA did not invent BASIC" when the sources never mention NASA. This is not treated as the same thing as a positive hallucination, because the verifier cannot prove the denial from absence alone.

## Quick Start

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

Run the local benchmark:

```bash
python bench.py
```

Run the TUI:

```bash
python tui.py
```

## Python Usage

Verify AI output against local files:

```python
from Halgorithem import Halgorithm

algo = Halgorithm(sentences_per_chunk=2, sentence_overlap=1)

results = algo.compare_to_files(
    truth_file_paths=["sources/basic.txt", "sources/basic2.txt"],
    ai_output="BASIC was developed in 1964. BASIC was created by NASA.",
    threshold=0.30,
)

for claim in results:
    print(claim["status"], claim["confidence"], claim["claim"])
```

Verify against in-memory source text:

```python
from Halgorithem import Halgorithm

algo = Halgorithm()

results = algo.compare_to_docs(
    truth_docs=[
        {
            "file_id": 1,
            "file_path": "internal_note",
            "text": "BASIC was developed at Dartmouth College in 1964.",
        }
    ],
    ai_output="BASIC was developed at Dartmouth College in 1964.",
)
```

Use the optional generation wrapper:

```python
from engine import run

result = run(
    prompt="What was Apollo 11?",
    urls=["https://en.wikipedia.org/wiki/Apollo_11"],
    threshold=0.30,
)

print(result["summary"])
```

## Example Result

```python
{
    "status": "CONTRADICTION",
    "claim": "BASIC was developed in 1972.",
    "score": 0.71,
    "confidence": 0.58,
    "reason": "Date mismatch",
    "matched_source": "sources/basic.txt",
    "matched_chunk_id": 1,
    "chunk_text": "BASIC was developed at Dartmouth College in 1964.",
    "evidence": [
        {
            "source": "sources/basic.txt",
            "chunk_id": 1,
            "score": 0.71,
            "text": "BASIC was developed at Dartmouth College in 1964."
        }
    ]
}
```

## Project Layout

```text
Halgorithem/
  Halgorithem/
    __init__.py
    core.py
    claim_extraction.py
    confidence.py
    contradiction.py
    evidence.py
    math_utils.py
    nlp.py
    retrieval.py
    source_quality.py
    temporal.py
    text_processing.py
    web.py
  assets/
  sources/
  bench.py
  engine.py
  requirements.txt
  test.py
  tui.py
```

## Design Principles

- The verifier should not depend on generated explanations from another AI system.
- Every claim should be traceable to source text.
- Evidence should be inspectable by humans.
- Contradictions should explain the exact conflict where possible.
- Thresholds should be adjustable.
- The core should stay modular enough for new checkers to be added safely.

## Current Limits

Halgorithem is not a full theorem prover and not a replacement for human review. It can miss:

- claims that require deep multi-hop reasoning
- claims that need current real-world knowledge not present in the sources
- paraphrases that are too far from the source chunk
- table-heavy facts if scraping loses structure
- claims where the source itself is wrong

The best results come from high-quality source documents with clear factual wording.

## Benchmark

The included benchmark checks a small BASIC-language dataset:

```bash
python bench.py
```

The benchmark covers:

- supported claims
- weakly supported claims
- unsupported claims
- contradiction claims

Future benchmark work should add:

- date-heavy claims
- entity-role swaps
- unit conversion errors
- current/latest claims
- multi-source disagreement
- table extraction cases

## License

See `LICENCE`.
