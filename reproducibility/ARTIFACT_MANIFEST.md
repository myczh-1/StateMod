# Artifact availability

[Public manifest](../manifest.json) inventories current release files. [Export provenance](EXPORT_PROVENANCE.json) maps each exported object to its original experiment-relative path, original SHA256, public SHA256, byte size and transformation. [Source adaptations](SOURCE_ADAPTATIONS.json) identifies portable helper extraction.

| Object | Availability | Use |
|---|---|---|
| 8B judgment checkpoint | `artifacts/judgment_8b.npz`, bundled | Frozen module; 4096-dimensional input |
| 14B judgment checkpoint | `artifacts/judgment_14b.npz`, bundled | Frozen module; 5120-dimensional input |
| Original 8B side Bank | `artifacts/bank_8b.npz`, bundled | Original full stored array, not a 14B Bank |
| Cloud11 hidden states | `artifacts/cloud11_states.npz`, bundled | 16 saved natural no-rule decision states |
| Cloud12 intermediate arrays | `artifacts/cloud12_layer_values.npz`, bundled | Eight controlled diagnostic states and computed intermediates |
| Prompts/tasks/generations/scores | `experiments/`, bundled JSON exports | Inspect contracts, tokens, failures and unchanged scoring |
| Per-step large logits / native KV / attention arrays | Mostly not bundled; historical manifests retained | Locally retained evidence, no public download promised |
| Base Qwen3 weights | Not bundled | Obtain separately at pinned official revisions |
| Old transport archives | Containers removed in authorized cleanup | SHA metadata is historical; extracted evidence retained |

Historical manifests under `source_manifests/` describe original layouts and can mention absent files. They do not certify that this public package includes those files. The root manifest is the authoritative public inventory. Privacy transformations can change JSON formatting and private-location strings; source and export hashes must not be confused.

The Bank stores all 36 layers, although only 14 are active. Stored array slots are not independent training examples. Its file size and actual active slot cost should not be treated as a compressed rule-semantic capacity result.
