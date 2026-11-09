# Third-party attribution and license scope

Project-owned source and documentation are MIT, copyright 2026 myczh. This does not relicense third-party components or model-derived artifacts.

- **Transformers / Qwen3 attention**: Hugging Face / Qwen implementation, Apache-2.0. StateMod imports rotary-position and grouped-query helpers and follows the Qwen3 attention structure in its local patched forward. The runtime adapter changes the side-attention computation; it is not an upstream Qwen3 file. The [Apache license text](third_party/transformers/LICENSE) is retained. Source reference: [Transformers 4.57.6 Qwen3](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3/modeling_qwen3.py).
- **Qwen3-8B and Qwen3-14B**: Qwen Team, Apache-2.0 model licenses. No base-model weights are bundled. [8B license](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/LICENSE), [14B license](https://huggingface.co/Qwen/Qwen3-14B/blob/40c069824f4251a91eefaf281ebe4c544efd3e18/LICENSE). Included Bank activations and hidden-state arrays were computed using these models; upstream attribution and terms are retained for these artifacts. They are not assigned a new claim of exclusive ownership by the project license.
- **NumPy**: BSD-3-Clause dependency, not vendored. **PyTorch**: BSD-style dependency, not vendored. **psutil**: BSD-3-Clause dependency for the historical scoring worker, not vendored.
- **IBM activation-steering (CAST)**: Apache-2.0; cited prior research and historical dependency, no IBM source included in this cloud-series release.
- **Microsoft llm-steer-instruct**: MIT; cited methodological reference, no Microsoft source included.
- **Memory Inception**: method referenced in the local side-KV adaptation. No author code repository is claimed to have been copied. Do not describe the local backend as a full official reproduction.
- **MaxBelitsky/cache-steering**: cited paper/repository only. Repository license retrieval was inconclusive in this review; no code is redistributed.

The generated Python examples and synthetic contracts are archived research outputs, including errors. A model card's license is not evidence that every generated snippet is novel. No external private dataset or user conversation transcript is distributed.
