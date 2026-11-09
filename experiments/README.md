# Evidence index

This release focuses on the Cloud02–12 research arc, not the entire earlier StateMod archive. Chronological labels are experiment identifiers, not ranks of evidential strength. Earlier static-V results motivate the work but cannot fill missing 14B loop evidence.

| Experiment | Role | Statistical units / coverage | Outcome |
|---|---|---|---|
| [Cloud02](cloud02/README.md) | Development teacher-to-module transfer | 24 previously observed tasks; 8 fit / 16 held-out; 6 families | `DEVELOPMENT_8B_TEACHER_FIDELITY_SUPPORTED` |
| [Cloud03](cloud03/README.md) | Development module→V loop | 16 tasks in 6 families; 48 new generations and 16 archived text-code references | `DEVELOPMENT_8B_MODULE_V_LOOP_SUPPORTED` |
| [Cloud04](cloud04/README.md) | Attempted independent new-family confirmation | 8 new computation families × 2 wordings × E/N; 128 conditions | `CAPABILITY_OR_COVERAGE_LIMITED` |
| [Cloud05](cloud05/README.md) | Known-simple-family development control | 4 known families × 2 wordings × E/N; 64 conditions; original scorer | `SIMPLE_BASELINE_NOT_STABLE` |
| [Cloud05-rescore01](cloud05-rescore01/README.md) | Post-hoc scorer correction; no inference | 64 unchanged outputs; 6 score changes; original results retained | `DEVELOPMENT_SIMPLE_WORDING_LOOP_STABLE` |
| [Cloud06](cloud06/README.md) | Attempted independent new-family confirmation | 4 new families × 2 wordings × E/N; 64 conditions | `CAPABILITY_LIMITED` |
| [Cloud07](cloud07/README.md) | Targeted development failure diagnosis | 24 natural generations; 4 controlled continuations; 8 fixed-prefix readouts | `See stage decision` |
| [Cloud08](cloud08/README.md) | 8B text-reference development precheck | 4 families, 8 tasks; type choice separate from code | `TEXT_REFERENCE_NOT_READY` |
| [Cloud09](cloud09/README.md) | Initial 14B numerical attempt | Stopped numerical admission; missing scheduled behavior is not zero accuracy | `STOP_NUMERICAL` |
| [Cloud10](cloud10/README.md) | New Cached-primary text-reference development protocol | Same 8 tasks; old Cloud09 STOP_NUMERICAL unchanged | `TEXT_REFERENCE_NOT_READY` |
| [Cloud11](cloud11/README.md) | Initial collection attempt: disk preflight stop | No forward, no fitting; not a failed learned checkpoint | `STOP_ENGINEERING` |
| [Cloud11-attempt02](cloud11-attempt02/README.md) | 14B teacher-to-module development transfer | 8 fit + 8 held-out wording; same 4 families; no V | `TEACHER_FIDELITY_NOT_ESTABLISHED` |
| [Cloud12](cloud12/README.md) | Controlled wording × format diagnostic | 2 previously seen N tasks × 2 wordings × 2 formats; no natural-generation evaluation | `CONTROLLED_DIAGNOSTIC_COMPLETE` |

Original numerical values and scores are retained. Private machine paths are replaced and JSON may be reformatted; exported files are not byte-identical originals unless provenance says `verbatim`. Model tensors and very large per-step arrays are not all included. `.md.txt` documents preserve original reports, including obsolete instructions and statuses. New reports explain differences without overwriting them.

Cloud05 Rescore01 is a completed historical correction, not a new publication rescore. Cloud09 and the first Cloud11 disk stop are deliberately retained. A successful rerun under a different protocol does not erase these failures.
