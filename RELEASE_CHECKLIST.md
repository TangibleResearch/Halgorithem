# Release Checklist

Definition of done for v1.0.0:

- [ ] `python -m pytest` passes.
- [ ] `python bench.py` passes the configured threshold.
- [ ] GitHub Actions CI passes on `push` and `pull_request`.
- [ ] Package installs with `python -m pip install -e .`.
- [ ] `from Halgorithem import Halgorithm` works.
- [ ] `python bench.py` remains backward compatible.
- [ ] `python tui.py` remains backward compatible.
- [ ] README is updated for v1.0 usage and limitations.
- [ ] Known limitations are documented.
- [ ] Sample verification works.
- [ ] Version is bumped in `pyproject.toml`.
- [ ] CHANGELOG includes v1.0.0 notes.
