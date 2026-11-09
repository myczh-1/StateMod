# Publication checklist

Snapshot prepared 2026-11-09. Author **myczh** and **MIT** license for project-owned code/documentation were explicitly confirmed. Third-party/model terms remain separate.

| Check | Result | Scope / limitation |
|---|---|---|
| Original outcomes and evidence levels | PASS | Development, attempted confirmation, diagnostics and historical rescoring separated |
| Score-to-table consistency | PASS | Cloud03/04/05/05-rescore01/06 saved per-task counts; Cloud08/10 teacher summaries |
| Frozen readout replay | PASS | Cloud11 16 states; Cloud12 8 cells and factorial effects; no fitting |
| Low-cost tests | PASS, 9 tests | Five offline fixtures, four random tiny CPU architecture checks; no research model loaded |
| Original/export SHA audit | PASS | 10,934 selected sources and exports checked; original files unchanged |
| Private paths / credentials | PASS within scan scope | Private-location redaction, credential/email patterns and structured key review; no guarantee against arbitrary undiscovered secrets |
| Python syntax | PASS | Public source parses; does not establish every historical entrypoint is portable |
| Local document links | PASS | Authored Markdown checked; original `.md.txt` transcripts preserve historical references |
| Large files | PASS | Largest bundled object ~9.88 MB; no base-model shards |
| Licensing / attribution | PASS for this scope | MIT choice confirmed; Apache attribution retained; no unverified external code vendored |
| Full fresh GPU reproduction | NOT RUN | No new research experiment authorized or needed for this preparation |
| Large raw tensor download availability | PARTIAL | Historical manifests and summaries available; not all raw arrays bundled; no public download service |
| Public repository creation / push | AUTHORIZED 2026-11-09 | Author approved this reviewed scope; no tagged GitHub Release authorized |

Detailed machine-readable checks: [PUBLICATION_AUDIT.json](reproducibility/PUBLICATION_AUDIT.json). File integrity: [manifest.json](manifest.json).

## Approved public scope

- English/Chinese report and README, evidence audit, methods, limitations, related-work comparison.
- Synthetic task contracts, frozen prompts/configurations, archived generated code/Token IDs/scores and step summaries, including all exported failures.
- Core and historical source snapshots; source-to-export provenance and hashes.
- Small 8B/14B judgment modules, original 8B side Bank, selected Cloud11/12 hidden-state arrays for offline replay.
- No main-model weights, SSH details/credentials, private conversations, transport archives or cloud deployment scripts.

## Repository

`myczh-1/StateMod`

Description: **External conditional control of frozen language models: code, local successes, generalization failures, and failure analysis.**

Visibility: public, explicitly approved by the author on 2026-11-09. No GitHub Release is created at this step.

The approved snapshot is a research report with **partial reproducibility**, not a validated product or fully rerun benchmark release. The machine-readable publication audit records the earlier preparation checkpoint; its pending-authorization note is historical.
