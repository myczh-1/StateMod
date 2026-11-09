# Method and implementation contract

## Judgment

For hidden row h, r = h / sqrt(mean(h²)+1e-6), x = (r−training_center)/training_scale, g = GELU(W1 x+b1), z = W2 g+b2. Enable iff sigmoid(z) > 0.5. Center and scale come from training rows only. Teacher probabilities use complete type-name-plus-EOS totals, not a single token.

[Readout arithmetic](../src/statemod/readout.py) reproduces saved weights without fitting. [Training reference](../src/statemod/student_reference.py) preserves the 14B fitting algorithm but is not invoked by release checks. [Frozen module inputs](../experiments/cloud11-attempt02/data/config.json) record hyperparameters.

## One-prediction side attention

Let qR/kR denote native RoPE-transformed states, q0/kB denote the pre-RoPE side convention, and vN/vB denote native/side Values. With grouped KV heads expanded to query heads:

```
aN = qR @ kR.T / sqrt(d) + causal_mask
aB = q0 @ kB.T / sqrt(d) + log(weight)
p  = softmax(concat(aN, aB))
y  = p @ concat(vN, vB)
```

The output projection remains native. A zero weight bypasses this computation entirely. This formula is not a universally position-invariant KV transplant. The frozen local convention and its audit matter. The payload stores K/V for all 36 layers; only layers 14..27 participate.

[Portable import adaptation](../src/statemod/backend.py) changes the helper import, not the attention equations. [Original source](../experiments/cloud03/source/backend.py) is separately preserved. The backend is tested on a random 32-dimensional synthetic CPU architecture in the publication workflow; real Qwen3 execution is inherited historical evidence.

## Gate and checkpoint

The [detector](../src/statemod/gate.py) sees generated text only. A prefix ending in a syntactically valid bare `return` can form the event, including a recognized Python fence. It is conservative about token boundaries but is not a proof that every event is a semantically undecided constructor choice. Contract/task design and per-run coverage remain essential. A completed `return result` or fused constructor token cannot be silently converted into a new decision window.

Module capture occurs during an unmodified trial prediction. Accepted runs replay from the same unmodified cache checkpoint; branch tensors must be independent. Exactly one prediction accesses the Bank. Closing access does not undo past activations or tokens.

## Scoring versions

[Current archived scorer](../src/statemod/scoring.py) accepts legal set subclasses for member checks, performs real `.add()` and `.remove()`, and checks exact `frozenset` identity. Each experiment's source scorer is also preserved. Cloud05 original and Rescore01 are distinct; neither is retroactively replaced. Publication scripts sum saved scores without executing generated code or modifying scores.

The restricted subprocess, builtin list, timeout and memory checks are experimental resource controls, **not a security sandbox**. Do not run untrusted internet-submitted code with this evaluator.

## Frozen comparisons

Use the original per-experiment config, protocol, scoring version and task identifiers. No pooled accuracy across independent/development/diagnostic sets. Failed admission, missing data and not-run cases remain failures or unavailable observations as recorded. A post-hoc corrected scorer is not an independently preregistered result.
