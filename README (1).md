<img alt="Halgorithem logo" src="assets/Halgorithem.png?raw=true" width="400">

Halgorithem (codename **CORE** — Claim-Oriented Recognition Engine) is a deterministic hallucination detection library for checking AI output against trusted source material. It extracts factual claims from AI-generated text, retrieves the closest evidence from your documents, and labels each claim as supported, weakly supported, contradicted, hallucinated, or an unverifiable denial. Verification is meaning-based by default, using sentence-transformer embeddings to match paraphrases, with deterministic guardrails for names, numbers, dates, units, negation, and source qualifiers.

## Documentation

Full API reference, design notes, and usage examples can be found in the [docs](docs/). The output schema, verdict definitions, and benchmark details are covered below.

## Forums & Community

Have a question, idea, or bug report? Open a [GitHub Issue](https://github.com/TangibleResearch/Halgorithem/issues) or start a [Discussion](https://github.com/TangibleResearch/Halgorithem/discussions). Please review the [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

## Contributing

Contributions to the codebase, benchmark datasets, and documentation are all welcome. See the [contributing guide](CONTRIBUTING.md) for details on how to get started.

## Getting Started

### Install

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

If neither spaCy model is installed, Halgorithem falls back to `spacy.blank("en")` with reduced linguistic accuracy rather than crashing.

### Quick Start

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

## Python API

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

Verify files on disk:

```python
algo.compare_to_files(
    truth_file_paths=["sources/basic.txt"],
    ai_output="BASIC was created by NASA.",
)
```

Optional generation wrapper (may call OpenAI; the verifier itself remains fully deterministic):

```python
from engine import run

result = run(
    prompt="Summarize this source.",
    truth_file_paths=["sources/basic.txt"],
)
```

## Verification Mode

By default, Halgorithem loads `sentence-transformers/all-MiniLM-L6-v2` from the local model cache and verifies claims by semantic similarity rather than strict word matching.

```bash
HALGORITHEM_EMBEDDER=semantic python tui.py
```

To allow the model to download when missing from the local cache:

```bash
HALGORITHEM_ALLOW_MODEL_DOWNLOAD=1 python tui.py
```

For fully lexical, deterministic fallback behavior:

```bash
HALGORITHEM_EMBEDDER=local python tui.py
```

## CORE Pipeline

The newer deterministic pipeline is available through `HalgorithemVerifier` and the JSON CLI. It runs document ingestion once, then processes each AI sentence through three independent checks:

- Similarity retrieval with `sentence-transformers/all-mpnet-base-v2`
- Sentence-level NLI with `cross-encoder/nli-deberta-v3-large`
- Atomic claim NLI over REBEL-style triplets from `Babelscape/rebel-large`

All model loads are local/offline by default. If REBEL, DeBERTa, mpnet, or Coreferee are not installed or cached, Halgorithem falls back to deterministic local checks and surfaces that in `diagnostics`.

```bash
python main.py --document doc.txt --response response.txt
```

```python
from Halgorithem import HalgorithemVerifier

results = HalgorithemVerifier().verify(document_text, response_text)
print([result.model_dump(mode="json") for result in results])
```

## CLI Usage

Interactive terminal UI:

```bash
python tui.py
```

Benchmark runner:

```bash
python bench.py
```

## Tests

```bash
python -m pytest
```

The test suite is designed to be fully network-free and uses local documents only.

## Output Schema

Every claim result includes:

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

### Verdict Meanings

| Verdict | Meaning |
|:-------:|---------|
| `SUPPORTED` | Strong evidence is present in the supplied sources. |
| `WEAK_SUPPORT` | Related evidence exists, but the claim is inferential or not fully direct. |
| `CONTRADICTION` | Relevant source evidence conflicts with the claim. |
| `HALLUCINATION` | The claim lacks adequate source support. |
| `UNVERIFIABLE_DENIAL` | The claim denies a fact or entity absent from the sources — absence alone cannot prove it. |
| `ERROR` | The verifier could not parse or evaluate the claim (mostly malformed math). |

## Benchmark

`bench.py` runs a release benchmark across the following categories:

- Supported claims
- Paraphrases
- Weak support
- Hallucinations
- Date mismatches
- Entity-role swaps
- Unit errors
- Current/latest claims
- Table-like facts
- Multi-source disagreement
- Denial claims
- Missing-source cases

It reports accuracy, per-category accuracy, a confusion matrix, failures, temporal warning checks, and a pass/fail threshold.

## Runtime Hardening

Halgorithem handles the following gracefully:

- Missing or empty files
- Empty AI output
- Malformed `truth_docs` or missing `text` fields
- Bad UTF-8 file encodings
- No extracted claims
- Math parse errors
- Missing spaCy or embedding models

## Limitations

- Rule-based entity-role detection handles common simple patterns, not arbitrary grammar.
- Semantic verification depends on a local sentence-transformer model; the lexical hashing fallback is deterministic and CI-safe but less meaning-aware.
- Multi-source disagreement is only surfaced when the source qualifier is explicit.
- Table-like facts work best when row values are near each other in the source text.
- Current/latest claims are warned about, not externally refreshed.

## Roadmap

- Optional structured table parser
- Better contradiction handling for passive and nested clauses
- Calibrated benchmark sets by domain
- Machine-readable benchmark artifacts
- Additional CLI commands beyond the interactive TUI

## Release Readiness

A v1.0 release requires: tests passing, benchmark meeting its threshold, CI green, packaging installing cleanly, README limitations documented, and the release checklist complete.
