# Related work and reuse boundaries

Sources below are primary papers or author/official repositories, checked during this publication review (2026-11-09). StateMod does not claim to originate activation steering, conditional activation, hidden-state probing, teacher distillation, or KV intervention.

| Prior work | Established idea relevant here | StateMod relationship |
|---|---|---|
| Rimsky et al., *Steering Llama 2 via Contrastive Activation Addition*, ACL 2024 ([paper](https://aclanthology.org/2024.acl-long.828/), [code](https://github.com/nrimsky/CAA)) | Contrastive activation directions can alter behavior | Historical residual-direction experiments are related; the focal V uses side KV rather than the CAA intervention |
| *Improving Instruction-Following in Language Models through Activation Steering*, ICLR 2025 ([paper](https://openreview.net/pdf?id=wozhdnRCtw), [Microsoft code](https://github.com/microsoft/llm-steer-instruct)) | Instruction-conditioned activation differences and steering | Earlier StateMod direction studies drew on this approach; the published cloud loop is not its reproduction |
| Lee et al., *Programming Refusal with Conditional Activation Steering*, ICLR 2025 ([IBM paper page](https://research.ibm.com/publications/programming-refusal-with-conditional-activation-steering--1), [paper](https://arxiv.org/abs/2409.05907), [official code](https://github.com/IBM/activation-steering)) | Condition representations gate behavior steering | Conditional gating already exists. StateMod's cloud module is supervised by a text teacher and gates a static KV payload; it is not the CAST condition-vector algorithm |
| Zou et al., *Representation Engineering: A Top-Down Approach to AI Transparency*, 2023 ([paper](https://arxiv.org/abs/2310.01405)) | Reading and controlling high-level representations | General conceptual precedent, not evidence that StateMod has isolated a pure rule representation |
| Hewitt & Liang, *Designing and Interpreting Probes with Control Tasks*, EMNLP-IJCNLP 2019 ([paper](https://aclanthology.org/D19-1275/)) | Probe accuracy must be interpreted against controls and probe capacity | Motivates separating train fit from readable/transferable information; this release does not claim a comprehensive control-task probing study |
| Hinton, Vinyals & Dean, *Distilling the Knowledge in a Neural Network*, 2015 ([paper](https://arxiv.org/abs/1503.02531)) | Soft teacher supervision for a smaller student | StateMod uses a small binary student from teacher candidate probabilities; no new distillation principle is claimed |
| Belitsky et al., *KV Cache Steering for Controlling Frozen LLMs*, 2025 ([paper](https://arxiv.org/abs/2507.08799), [author repository](https://github.com/MaxBelitsky/cache-steering)) | One-shot intervention on existing KV caches | StateMod's focal side-bank operation adds attention slots rather than simply translating an existing cache |
| Liu et al., *Memory Inception: Latent-Space KV Cache Manipulation for Steering LLMs*, 2026 ([paper](https://arxiv.org/abs/2605.06225), [reviewed full-text version](https://arxiv.org/html/2605.06225v1)) | Text-derived KV banks at selected layers | Direct methodological reference. The local adapter uses fixed layers and pre-RoPE side attention with joint softmax; it does not reproduce all calibration, normalization or runtime paths in the paper |

## What is reused

The runnable attention dependency is Transformers/Qwen3; StateMod imports its rotary and grouped-query helpers and builds a local side-attention path. The release includes project source, not vendored CAST, Microsoft, cache-steering or Memory Inception repositories. IBM CAST was explored in earlier experiments, but the cloud judgment here is a separate teacher-supervised MLP. Attribution and licenses are tracked in [third-party notices](../THIRD_PARTY_NOTICES.md).

## Sensitivity and capability preservation

The Memory Inception paper evaluates control versus drift and reports backbone/task-dependent behavior; the probing literature cautions against treating successful probe fitting as a uniquely identified semantic mechanism. These are related methodological concerns, not prior demonstrations of StateMod's particular boolean-wording or Python-constructor failures. This review did not establish an external paper reproducing those exact cases. StateMod supplies its own archived counterexamples and paired diagnostics; it should not generalize them to all steering methods.

## Differences worth reporting, without a priority claim

The narrow combination is teacher-supervised hidden-state applicability judgment, checkpoint replay, one Python return decision, a separately frozen side-KV payload, and executable mutability/member contracts. Its value in this release is an auditable experimental arrangement, including failed new-family evaluations and a wording/format diagnostic. A literature review of these sources does not establish that this combination is novel, nor does it prove a general text-rule compiler or universal conditional state representation.

## License check boundaries

IBM activation-steering is Apache-2.0; Microsoft llm-steer-instruct is MIT. Qwen model cards/licenses and Transformers use Apache-2.0. Memory Inception is used as a paper-method reference; no author implementation is claimed or bundled. The cache-steering repository's code license was not reliably retrieved during this review; none of its code is redistributed. Paper access does not imply code redistribution permission.
