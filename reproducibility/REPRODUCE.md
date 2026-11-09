# Reproduction levels

## Level 1 — verified offline reconstruction

From the repository root, install `requirements-offline.txt`, then run:

```sh
python scripts/verify_release.py
python scripts/reproduce_readout.py
python scripts/recompute_tables.py
python -m unittest discover -s tests -v
```

The readout script loads the frozen 14B checkpoint, recomputes Cloud11's 16 scores from saved raw states and Cloud12's eight diagnostic scores/factorial effects. It checks all saved intermediate vectors, probabilities and decisions against archived values. It does not refit, tune or run a model. Counts are recomputed from saved scoring records, not by changing or re-executing model code. Outputs are written to ignored `reproduction-output/`.

The minimal failure is boolean-complement N: changing the saved wording-conditioned hidden state changes the frozen module decision in both formats. This demonstrates the archived computation, not a new model replication of the input-to-hidden-state mapping.

The default dependency set runs five offline fixture tests; four backend tests are skipped unless PyTorch and Transformers are installed. To run those additional tests, install `requirements-runtime-reference.txt`. They instantiate a random 32-dimensional synthetic architecture on CPU, never a research model checkpoint. All nine passed in the local publication audit. This is an interface test, not new behavioral evidence.

## Level 2 — source and data available, full model rerun not validated here

Per-experiment `data/`, `original/PROTOCOL.md.txt`, source, generation logs, scores and configuration are exported. `src/statemod/backend.py` is an import-portable copy of the Cloud03 attention interface. It requires the recorded Transformers API and a separately acquired frozen 8B model. Its 36-slot shape is intentionally specific; it is not a 14B backend. The compiler source is in Cloud03's source snapshot.

Historical scripts assume frozen package layouts, pre-existing reference files and output directories. Private locations are redacted; they are **not** a one-command portable deployment. Do not execute them by substituting guessed paths or treat historical manifest checks as public-export hashes. A clean GPU rerun with a resolved external artifact set is not verified by this release.

The supplied offline scripts are the supported reproduction entrypoints. No new training or GPU inference was performed to prepare publication.

## Level 3 — evidence-only large tensors

Full-vocabulary per-step logits, attention arrays and full native KV can be large. They are retained locally and represented by historical source manifests and selected numerical audit records; they are not all distributed here. No public download URL or automatic artifact-fetch command is promised. Hashes prove byte identity only when the corresponding object is available; they do not prove completeness of absent evidence.

Model weights are excluded. The original transport tar files and local redownloadable 8B weights were removed in an authorized cleanup after extracted results were checked. Extracted records survive; archive-container hashes are historical transport metadata.

## Execution warning

The experimental code scorer uses restricted builtins and resource limits, not a hardened sandbox. Publication replay reads existing scores and does not execute archived generated programs. Unit tests execute only tiny authored fixtures. Run unknown generated code only within an appropriate separate security boundary.
