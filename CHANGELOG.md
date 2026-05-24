# Changelog

## v1.0.0

### Added

- Pytest suite covering claim extraction, retrieval, supported claims, weak support, hallucinations, contradictions, denials, math checks, temporal warnings, `compare_to_docs`, and `compare_to_files`.
- Release benchmark covering BASIC and non-BASIC scenarios, including date mismatches, entity-role swaps, unit errors, current/latest claims, table-like facts, paraphrases, multi-source disagreement, denial claims, and missing-source cases.
- GitHub Actions CI for pytest and benchmark runs on push and pull request.
- `pyproject.toml` packaging metadata.
- Release checklist.

### Changed

- Default embedding runtime is now local and network-free for reliable installs and CI.
- spaCy loading now falls back from `en_core_web_lg` to `en_core_web_sm` to `spacy.blank("en")`.
- Result objects now consistently include evidence, source, reason, warning, unsupported terms, score, and confidence fields.
- Requirements were cleaned up and documented.

### Fixed

- Short factual claims are no longer filtered out solely because they have fewer than four tokens.
- Unsupported relation entities are treated as hallucination signals.
- Date, unit, source-qualifier, and simple entity-role contradictions are handled more consistently.
- Runtime errors for malformed sources, empty inputs, bad encodings, and math parse failures are clearer.

### Known Limitations

- Rule-based relation extraction handles simple factual sentences, not all English constructions.
- Local embeddings trade semantic depth for deterministic, network-free operation.
- Current/latest claims are flagged as time-sensitive but not refreshed from live sources.
