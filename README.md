# Halgorithem

Deterministic hallucination detection for checking AI output against trusted source material.

Halgorithem takes source documents and AI output, extracts factual claims, retrieves the closest evidence, and labels each claim as supported, weakly supported, contradicted, hallucinated, or an unverifiable denial.

## What It Does

- Verifies AI-generated factual claims against supplied sources.
- Splits multi-fact text into atomic claims.
- Retrieves evidence chunks from one or more sources.
- Detects date, number, unit, negation, source-qualifier, and simple entity-role conflicts.
- Flags time-sensitive claims such as "current", "latest", "today", and "now".
- Returns a structured result for every extracted claim.

## What It Does Not Do

- It does not prove truth in the real world.
- It does not browse the web unless you use the optional URL wrapper.
- It does not replace source quality review.
- It does not guarantee perfect paraphrase understanding, especially with the local fallback embedder.
- It does not use an LLM for verification.

## Install

```bash
python -m pip install -e .
```

Recommended NLP model:

```bash
python -m spacy download en_core_web_lg
```

Lightweight fallback:

```bash
python -m spacy download en_core_web_sm
```

If neither spaCy model is installed, Halgorithem falls back to `spacy.blank("en")` with reduced linguistic accuracy instead of crashing.

## Quick Start

Recommended high-accuracy verifier:

```python
from Halgorithem import verify

results = verify(
    docs="BASIC was created in 1964 by John Kemeny at Dartmouth College.",
    response_text="BASIC was created in 1964.",
)

for result in results:
    print(result.verdict, result.confidence, result.diagnostics)
```

The `verify()` helper and `HalgorithemVerifier` return dataclass results and use the newer similarity, NLI, atomic-claim, and vote-fusion pipeline. This is the preferred API for new integrations.

Vote fusion keeps support and contradiction evidence separate. Atomic scores intentionally use `[-1, 1]`, where negative values represent contradiction strength. NLI checks also expose `model_quality`: the built-in rule-based fallback is weighted lower than a transformer-backed NLI model that reports full quality.

Legacy chunk API:

```python
from Halgorithem import Halgorithm

algo = Halgorithm()

results = algo.compare_to_docs(
    truth_docs=[
        {
            "file_id": 1,
            "file_path": "source.txt",
            "text": "BASIC was created in 1964 by John Kemeny at Dartmouth College.",
        }
    ],
    ai_output="BASIC was created in 1972 by NASA.",
)

for result in results:
    print(result["status"], result["claim"], result["reason"])
```

`Halgorithm.compare_to_docs()` remains supported for compatibility and returns dictionaries. It uses the lower-level chunk scoring path, so its output shape differs from `verify()` / `HalgorithemVerifier`.

## Python API

Preferred verifier:

```python
from Halgorithem import HalgorithemVerifier

with HalgorithemVerifier() as verifier:
    results = verifier.verify(
        docs="BASIC was created in 1964.",
        response_text="BASIC was created in 1964.",
    )
```

Legacy verifier:

```python
from Halgorithem import Halgorithm

algo = Halgorithm(sentences_per_chunk=2, sentence_overlap=1)
```

Verify in-memory documents:

```python
algo.compare_to_docs(
    truth_docs="BASIC was created in 1964.",
    ai_output="BASIC was created in 1964.",
)
```

Verify files:

```python
algo.compare_to_files(
    truth_file_paths=["sources/basic.txt"],
    ai_output="BASIC was created by NASA.",
)
```

Optional generation wrapper:

```python
from engine import run

result = run(
    prompt="Summarize this source.",
    truth_file_paths=["sources/basic.txt"],
)
```

The wrapper may call OpenAI for generation. The verifier in `Halgorithem/` remains deterministic.

## CLI Usage

Interactive terminal UI:

```bash
python tui.py
```

Benchmark:

```bash
python bench.py
```

## Tests

```bash
python -m pytest
```

The pytest suite is designed to be network-free and uses local documents.

## Benchmark

`bench.py` runs a release benchmark across:

- supported claims
- paraphrases
- weak support
- hallucinations
- date mismatches
- entity-role swaps
- unit errors
- current/latest claims
- table-like facts
- multi-source disagreement
- denial claims
- missing-source cases

It reports accuracy, accuracy by category, a confusion matrix, failures, temporal warning checks, and a pass/fail threshold.

## Output Schema

The preferred verifier returns `VerificationResult` dataclasses:

```python
{
    "sentence": str,
    "verdict": "SUPPORTED | WEAK_SUPPORT | CONTRADICTION | HALLUCINATION | UNVERIFIABLE",
    "confidence": float,
    "similarity": SimilarityCheck,
    "nli": NLICheck,
    "atomic": AtomicCheck,
    "diagnostics": dict,
}
```

The legacy `Halgorithm` API returns dictionaries:

```python
{
    "claim": str,
    "status": "SUPPORTED | WEAK_SUPPORT | CONTRADICTION | HALLUCINATION | UNVERIFIABLE_DENIAL | ERROR",
    "confidence": float,
    "score": float,
    "matched_source": str | None,
    "matched_chunk_id": int | None,
    "matched_chunk": str,
    "chunk_text": str,
    "evidence": list,
    "unsupported_terms": list[str],
    "reason": str,
    "warning": str | None,
}
```

## Verdict Meanings

- `SUPPORTED`: strong evidence is present in the supplied sources.
- `WEAK_SUPPORT`: related evidence exists, but the claim is inferential or not fully direct.
- `CONTRADICTION`: relevant source evidence conflicts with the claim.
- `HALLUCINATION`: the claim lacks adequate source support.
- `UNVERIFIABLE_DENIAL`: the claim denies a fact or entity absent from the sources, so absence alone cannot prove it.
- `ERROR`: the verifier could not parse or evaluate the claim, mostly for malformed math.

## Runtime Hardening

Halgorithem handles:

- missing files
- empty sources
- empty AI output
- malformed `truth_docs`
- missing `text` fields
- bad UTF-8 file encodings
- no extracted claims
- math parse errors
- missing spaCy or embedding models

## Limitations

- Rule-based entity-role detection handles common simple patterns, not arbitrary grammar.
- Local hashing embeddings are deterministic and CI-safe but less semantic than sentence-transformers.
- Multi-source disagreement is surfaced when the source qualifier is explicit.
- Table-like facts work best when row values are near each other in text.
- Current/latest claims are warned, not externally refreshed.

## Roadmap

- Optional structured table parser.
- Better contradiction handling for passive and nested clauses.
- Calibrated benchmark sets by domain.
- Machine-readable benchmark artifacts.
- More CLI commands beyond the interactive TUI.

## Release Readiness

v1.0 readiness means tests pass, the benchmark meets its threshold, CI passes, packaging installs, README limitations are documented, and the release checklist is complete.
