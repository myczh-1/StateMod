# External Conditional Control of Frozen Language Models: Local Successes, Generalization Failures, and Failure Analysis

**myczh · Research snapshot: 2026-11-09 · Publication preparation**

## Abstract

We study an external control pipeline with a frozen Qwen3 backbone, a small teacher-supervised judgment module, and a text-compiled side-KV Bank. The module predicts whether an external immutable-container preference should be applied; the Bank influences a single return-expression prediction, after which generation continues normally. In an 8B development evaluation, automatic routing achieved functional `frozenset` returns on 8/8 applicable tasks and preserved the mutation contract on 8/8 inapplicable tasks. New-family evaluations did not confirm stable generalization. A later 14B text reference correctly selected types and generated contract-compliant code on eight development tasks, but a separately trained 14B judgment module reproduced only 7/8 held-out wording decisions, failing a rejection requirement. Controlled diagnostics show that a wording change suffices to produce the observed boolean-task misactivation without a Markdown fence; format modulates the score but is not necessary for this failure. These results establish local feasibility and concrete failure cases, not general rule understanding or a validated 14B B→V loop.

## 1. Introduction

A frozen model can use a visible instruction, but external runtime control asks a different engineering question: can a separately stored rule representation affect selected computation without appending its body to the current user prompt? We separate applicability judgment from behavioral intervention to measure each failure mode. The motivation is controllable external state, not a demonstrated latency or context-cost advantage over text prompting.

The task domain is intentionally narrow: Python functions that collect members. E permits unique unordered hashable members and does not require mutability; the external preference is `frozenset`. N requires the returned object to support caller-side additions and removals. A correct E `set` is functionally valid even when it misses the external preference. We do not equate preference noncompliance with universal task error.

Text prompting is a substantive baseline. Its performance depends on model, prompt wording, answer order, output format, and generation task. Direct type choice and complete code generation are measured separately. The study does not show that external state generally outperforms visible rules.

## 2. Method

### 2.1 Frozen model and judgment data

The module reads the zero-indexed Block13 output at the last generated `return` token, before the next token chooses the expression. A deterministic tokenizer/AST detector examines only the generated code prefix. No gold code, current-task teacher states, future tokens, or execution labels are available to the online module. This is a Python-specific decision window, not a general semantic event detector.

Training captures no-rule natural-generation hidden states. Separately, a visible-rule teacher chooses between complete `frozenset` and `set` answers; candidate scores include the end-of-sequence token. The selected teacher format lists `frozenset` first. Teacher answer-order checks preceded Cloud02, so its 24 tasks are development evidence, not a fresh teacher confirmation.

For a hidden vector h, use sample RMS normalization with epsilon 1e-6, then training-only centering and one global RMS scale. A linear map to eight dimensions, GELU, and a final linear map produce logit z and sigmoid score p. The frozen acceptance rule is strictly p > 0.5. These scores are not calibrated correctness probabilities. Soft teacher targets supervise binary cross entropy (equivalently KL up to fixed target entropy). Only the small external module is trained; backbone parameters remain unchanged. The eight-dimensional representation is a bottleneck, not eight independently identified semantic conditions.

The 14B module uses 5120→8→1, CPU float64 AdamW, 256 steps, learning rate .003, weight decay .01, seed 42916, and the last checkpoint. Its eight training states and eight reference states come from the same four computation families with changed wording. The 8B module uses 4096→8→1 and its own frozen checkpoint; they are not interchangeable.

### 2.2 Static behavior payload

The 8B payload is compiled offline from static rule text without evaluation task code or same-task PF Oracle states. It is a static preference, not a conditional policy. The stored arrays have 36 model layers; only Blocks14–27 are activated, with 36 side slots per layer and weight 1. Qwen3-8B has 36 layers, so these are not the final fourteen layers.

Native attention uses post-RoPE queries and cached native keys. The side path uses normalized pre-RoPE queries and pre-RoPE Bank keys, a local virtual-relative-phase-zero convention. KV heads are repeated for grouped-query attention. A joint softmax covers native logits and side logits plus log(weight), and combines their respective Values. Side slots are not native sequence positions, but still consume memory and attention compute. Value-zeroing alone would not remove key competition; an actual zero/off condition bypasses the entire side path.

This is a local KV-method adaptation, not an exact reproduction of Memory Inception or existing-cache steering. In particular, it does not reproduce automated paper-level layer/head calibration or all bank-size normalization choices. See the [method comparison](RELATED_WORK.md).

### 2.3 Runtime intervention

At the first legal return boundary, read the module from an unmodified trial forward. If enabled, replay that prediction from an independent pre-decision cache checkpoint with the Bank. On rejection continue natively. FIRST_DECISION_ONLY allows exactly one prediction to access the Bank; later predictions use their own resulting history without further Bank access. Turning access off does not erase the modified state or generated tokens. The effect is within model computation, not code rewriting.

MANUAL_V uses offline known applicability as a V-side upper reference. MODULE_V uses only module decisions and the same V interface. Manual results are not counted as automatic routing success. Session state and writable caches are separate. [Source and equations](METHODOLOGY.md) document these boundaries.

## 3. Experimental setup

Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218` runs FP32/CUDA/eager. Qwen3-14B revision `40c069824f4251a91eefaf281ebe4c544efd3e18` runs BF16/CUDA/eager. Both use frozen tokenizers, non-thinking chat formatting, greedy decoding, and a 96-token code budget in the core cloud series. The original environment records are included; model size and precision change together and cannot be disentangled by this comparison.

FUNCTIONAL checks member values/types, uniqueness, input preservation, and applicable mutation operations. EXACT_TYPE requires the actual returned type, not constructor text. E JOINT requires functional correctness and exact `frozenset`. N success means the mutable contract is satisfied. Historical JSON uses `joint` for N contract success; it must not be interpreted as N `frozenset` success. Regression is paired loss on a baseline-correct task, not simply all post-intervention failures. Parsing, resource limits, unsupported code, truncated output, and missed windows remain recorded.

Counts are task instances unless stated otherwise. Wordings of the same computation family are dependent. Family counts are retained; no binomial significance or population reliability claims are made from these small engineered sets. Frozen independent attempts remain failed when gates fail. Post-hoc diagnostics and rescoring have their own evidence levels.

## 4. Results

### 4.1 Evidence ledger

| Experiment | Units and role | Main observation | Preserved status |
|---|---|---|---|
| Cloud02 | 24 prior development tasks; 8 fit, 16 hold-out instances, 6 families | 16/16 held-out teacher decisions reproduced, including 8/8 N | DEVELOPMENT_8B_TEACHER_FIDELITY_SUPPORTED |
| Cloud03 | 16 tasks, 6 families; 48 new generations + 16 archived text references | MANUAL and MODULE E joint 8/8, N contract 8/8; identical outputs across the two modes | DEVELOPMENT_8B_MODULE_V_LOOP_SUPPORTED |
| Cloud04 | 8 new families × 2 wordings × E/N = 32 tasks, 128 conditions | Both manual and automatic V decline | CAPABILITY_OR_COVERAGE_LIMITED |
| Cloud05 | 4 known simple families × 2 wordings × E/N | Routing 16/16, E joint 8/8; original scorer falsely rejected legal set subclasses | SIMPLE_BASELINE_NOT_STABLE |
| Cloud05 Rescore01 | Same 64 outputs; no new inference | 6 changed score records, 58 unchanged; N baseline/auto 7/8, not 8/8 | DEVELOPMENT_SIMPLE_WORDING_LOOP_STABLE, separate post-hoc result |
| Cloud06 | 4 new families × 2 wordings × E/N, 64 conditions | E accept 8/8, N reject 6/8; automatic N contract 0/8 | CAPABILITY_LIMITED |
| Cloud07 | 24 natural runs, 4 controlled continuations, 8 fixed-prefix readouts | Wording clarification helps; constructor-conditioned calculation failure also exists | Development diagnosis, no confirmation |
| Cloud08 | 8 development tasks, 4 families, 8B | Type choice 6/8; visible-rule code E 4/4, N 0/4 | TEXT_REFERENCE_NOT_READY |
| Cloud09 | Initial 14B numerical attempt | Cross-path audit stopped the planned evaluation | STOP_NUMERICAL |
| Cloud10 | Same 8 tasks, newly frozen Cached-primary protocol, 14B | Type choice 8/8; text code E 4/4, N 4/4; baseline N 2/4 | TEXT_REFERENCE_NOT_READY under original baseline gate |
| Cloud11 initial | Disk preflight | No model forward or training | STOP_ENGINEERING |
| Cloud11 Attempt02 | 8 fit + 8 held-out wording, same 4 families | Fit 8/8; hold-out 7/8; N reject 3/4 | TEACHER_FIDELITY_NOT_ESTABLISHED |
| Cloud12 | 2 seen N tasks × 2 wordings × 2 formats | Boolean misactivation remains without fence | CONTROLLED_DIAGNOSTIC_COMPLETE (evaluation); not confirmation |

Evidence and original stage-specific status files are indexed [per experiment](../experiments/README.md). Stage labels sometimes differ from run-completion labels; the latter do not supersede scientific admission decisions.

### 4.2 8B end-to-end results

| Evaluation | Condition | E functional | E joint | N contract |
|---|---|---:|---:|---:|
| Cloud03 | BASE | 8/8 | 0/8 | 8/8 |
| | TEXT_CODE (archived) | 8/8 | 8/8 | 2/8 |
| | MANUAL_V | 8/8 | 8/8 | 8/8 |
| | MODULE_V | 8/8 | 8/8 | 8/8 |
| Cloud04 | BASE | 12/16 | 0/16 | 1/16 |
| | TEXT_CODE | 11/16 | 9/16 | 0/16 |
| | MANUAL_V | 10/16 | 3/16 | 1/16 |
| | MODULE_V | 10/16 | 3/16 | 3/16 |
| Cloud06 | BASE | 5/8 | 0/8 | 2/8 |
| | TEXT_CODE | 6/8 | 6/8 | 2/8 |
| | MANUAL_V | 4/8 | 4/8 | 2/8 |
| | MODULE_V | 4/8 | 4/8 | 0/8 |

Cloud03 demonstrates actual automatic local behavior, not just logits. However, no random/wrong-payload control establishes that this specific compiled payload is uniquely responsible relative to all alternative perturbations. It is also not independent task confirmation.

Cloud04 changes both task and wording, and its baseline coverage is poor. N rejection differs strongly across wording (6/8 versus 0/8). Some N prompts leave caller mutation operations insufficiently explicit relative to the scorer. This limits attribution; it does not authorize repairing old tasks or upgrading the original result.

Cloud06 has two N false accepts and one E functional regression even under manual correct routing. Its N automatic regressions are the two baseline-correct N tasks. An E joint increase over zero is insufficient to pass the frozen safety and capability requirements.

### 4.3 Text reference and 14B readout

Cloud08's text code has valid executable formats and correct member calculations, but all four N outputs fail mutability. This is distinct from syntax failure. In Cloud10, the same visible rule succeeds on all eight text tasks. Its formal stop remains because the no-rule N baseline achieves 2/4 against a required 3/4. The report separates this protocol outcome from the observed usable teacher reference.

Cloud11 Attempt02 trains a fresh 14B module, leaving older modules and Bank unchanged. Held-out boolean-complement N is misaccepted with score 0.857910. Four E decisions are accepted and only three of four N decisions rejected. The overall 7/8 threshold alone is insufficient: the frozen protocol separately requires all four teacher rejections to be preserved. No 14B V Bank or end-to-end behavior result follows from this experiment.

## 5. Failure analysis

### 5.1 Scoring versus functional errors

Cloud05's original member-shape filter excluded valid set subclasses. Rescore01 first reproduced the original scores on all 64 outputs, then applied a separately versioned correction and synthetic contract tests. Six records changed (two tasks under three non-text conditions); one defective custom class remained a failure. Exact `frozenset` checks also use Python type identity rather than a class-name string. Original scores and status remain alongside the correction. This publication performs no new favorable rescoring.

### 5.2 Contract ambiguity and routing

Cloud07 makes the caller's later mutation obligation explicit. N baseline success in the four targeted cases changes from 0/4 to 3/4 and unwanted embedded validation disappears. Two false accepts disappear under clearer wording while the old-wording errors persist under a standardized prefix. This identifies a local input-expression sensitivity; it does not validate general robustness or prove that the module ignores semantic constraints.

### 5.3 Type change can alter subsequent computation

The Cloud06 character-count example generates invalid assignment-expression code when the constructor changes. Cloud07 controlled continuations find correct computation after a fixed `set` prefix and incorrect computation after fixed `frozenset`, both with and without Bank. This supports constructor-conditioned continuation as sufficient for the observed error in this case; it does not establish a general mechanism or absolve V of the real end-to-end regression.

### 5.4 Wording and Markdown fences

Cloud12 crosses training/held-out wording with bare/fenced code prefixes. All four historical states are exactly reproduced. Both contracts require mutation; neither should enable the static immutable preference.

| Task | Train bare | Train fenced | Held-out bare | Held-out fenced |
|---|---:|---:|---:|---:|
| Boolean complement N | 0.000002131 | 0.000012998 | **0.959215** | **0.857910** |
| Lowercase N | 1.491e-10 | 5.985e-11 | 0.076782 | 0.001768 |

Scores above .5 enable. In module-logit units, changing wording shifts boolean scores by +16.2165 (bare) and +13.0488 (fenced). Adding a fence shifts them by +1.8080 under training wording and −1.3598 under held-out wording. Lowercase moves even more with wording but retains sufficient rejection margin. A fence is neither necessary nor sufficient for the boolean failure in this small matrix.

An exact final-layer contribution decomposition attributes roughly 55–62% of the boolean wording-induced logit shift to the loss of a negative contribution from coordinate 7 (zero indexed). This is arithmetic accounting, not a causal intervention or identification of a mutability neuron. Wording also changes prompt length by seven tokens, and a fence adds three. Lexical content, length, absolute position and their representation effects remain coupled. The evidence does not establish a keyword-only classifier or absence of semantic information in the backbone.

### 5.5 Numerical paths

8B FP32 uses the frozen Cached/Fresh RMS and selected-log-probability gates and independent attention checks. The initial 14B BF16 attempt stopped numerically. Cloud10 separately preregisters Cached as the behavioral path, with Fresh comparison diagnostic only; it does not relax the old experiment retroactively. Its maximum logit RMS is about .22629 and selected-token log-probability discrepancy about .20421, with no observed argmax or candidate-ranking flips in those comparisons. That is not elementwise equivalence or a proof of identical full Fresh generations.

Cloud12 has maximum cross-path logit RMS .11212 and maximum element discrepancy 1.125. All eight module decisions agree across paths, including the mistaken ones. Repeated Cached probes and historical replay pass exact comparisons. The observed wording failure is therefore not explained by a measured decision flip between these paths. General numerical robustness remains unproven.

## 6. Discussion

Readable hidden-state features do not by themselves establish reliable routing. Likewise, correct routing does not ensure that a type-biased continuation preserves computation. Our evidence separates teacher limitations, student transfer failures, scorer/contract defects and V-induced generation changes. None is a universal explanation for all negative results.

The work combines known ideas: teacher distillation, low-dimensional hidden-state readout, conditional steering and KV-mediated intervention. Its contribution is an auditable narrow implementation and a sequence of positive and negative measurements. No broad architectural novelty, priority, universal rule compiler, or lossless text-to-state extraction is established.

The observed costs include side slots and a replayed decision step, as well as offline teacher/compilation work. Cloud03 records 277.17 seconds for a job including audits, 844 newly generated tokens, and 48 new conditions; it is not a production throughput benchmark. Its Bank file is 9,877,016 bytes, including inactive stored layers. This does not establish lower total compute or memory than prompting.

## 7. Limitations

The datasets are small, hand-designed, partly reused and dependent across wordings. Independent attempts did not meet their gates. Prompt clarity, baseline ability and scorer versions differ across studies; post-hoc findings are labeled. The 14B model is evaluated at a different precision, and its judgment has no V-loop test. Random/wrong payload baselines are missing in the focal loop studies. Fixed-prefix diagnosis is not natural-generation accuracy. The detector is Python-specific, and the scorer's restricted subprocess is not a secure sandbox. Large raw tensors are manifest-indexed rather than fully bundled; the public snapshot has not rerun the GPU experiments. See [limitations](LIMITATIONS.md) and [reproduction scope](../reproducibility/REPRODUCE.md).

## 8. Conclusion

External judgment plus a static side-KV intervention can produce correct local control on a frozen 8B model. Stable generalization is not established. A stronger text teacher improves the available supervision in the examined 14B tasks, but the trained reader still fails a rejection requirement under changed wording. The actionable scientific result is the separation of these failure modes and their preserved evidence, not a claim of reliable automatic rule execution.
