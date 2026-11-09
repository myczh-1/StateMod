# Research evidence audit

Review date: 2026-11-09. Scope: archived Cloud02–12, Cloud05 Rescore01, related failure interpretation and the earlier-source provenance required by these experiments. No new training, model inference, threshold selection or rescoring was performed.

## Authority and coverage

Primary per-task outputs, scores, frozen configs and DECISION files take precedence over narrative summaries. The [experiment index](experiments/README.md) links purpose, units, preserved outcome and detailed reports. This is a focused cloud-series release; it is not an assertion that every prior StateMod branch has been independently audited or bundled.

## Resolved interpretation conflicts

| Potentially misleading shorthand | Audited interpretation | Evidence |
|---|---|---|
| Cloud03 success establishes generalization | 16 tasks from 6 families; development selection history; independent attempts later fail | Cloud03 data/config, scored results; Cloud04/06 decisions |
| Cloud05 N all correct after score fix | Corrected aggregate N baseline/auto 7/8; one real/unsupported class failure remains | Rescore01 SUMMARY and SCORE_DIFF |
| Every Cloud06 error is a matcher error | Two N misactivations; manual correct routing also causes an E computation regression | Cloud06 PAIRED, scores; Cloud07 controlled continuation |
| 14B text teacher failed | Cloud10 text choice 8/8 and code E/N 4/4 each; formal stop is its original BASE N coverage requirement | Cloud10 SUMMARY and DECISION |
| Cloud11 overall 7/8 passes | All 4 teacher rejections must be preserved; only 3/4 are | Cloud11 Attempt02 config, FIDELITY_METRICS, DECISION |
| Fences caused the boolean error | Held-out wording misactivates even with bare prefix; fences change but do not explain it alone | Cloud12 8 cells and factorial effects |
| Cached/Fresh no decision flips means numerical equality | BF16 arrays differ substantially; later protocols use Cached primary, Fresh diagnostic | Cloud09/10/12 numerical records |
| Later execution completion upgrades an earlier stop | Each experiment/attempt keeps its original state | Cloud09 numerical stop and Cloud11 initial disk stop |
| N `joint` means exact frozenset success | Historical N `joint` means functional mutable-contract success | Historical scoring source |

## Confirmed observations versus hypotheses

Confirmed within their samples: preserved 8B development behavior; failed new-family checks; a legal-set-subclass scoring defect and separate historical correction; false routing; constructor-conditioned continuation failure; exact saved-readout arithmetic; wording/format score differences.

Not established: universal semantic direction, keyword-only learning, total absence of conditional information in the backbone, a causal interpretation of each bottleneck neuron, pure model-size effects, broad numerical robustness, lower production cost, general deployment safety, or a 14B module→V loop.

## Publication transformations

An allowlist copies protocols, task/config JSON, output/score/step JSON, selected code and selected small NPZ artifacts. Private absolute machine paths and host aliases are redacted; no scientific metric is rewritten. Markdown source reports are stored as `.md.txt` transcripts to distinguish their original path references from current repository navigation. Machine-operation connection/shutdown records are excluded from public distribution; their scientific stop/audit summaries remain.

Original and exported hashes are paired in [EXPORT_PROVENANCE.json](reproducibility/EXPORT_PROVENANCE.json). The old source files selected for this export are rehashed after preparation. This verifies that selected sources were not changed; it is not a fresh all-model-weight audit. Historical model audits and revision manifests are included as recorded evidence.

## Reproduction claim

Offline module outputs and saved-score counts are recomputed. Synthetic code-contract checks and source syntax checks are run. No full GPU experiment was rerun, no new weights were downloaded and no cloud instance was contacted. Unbundled tensor hashes and historical audit summaries are not represented as independently replayed public results.
