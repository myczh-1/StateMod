# Recorded environments

| Property | 8B cloud series | 14B cloud series |
|---|---|---|
| Model | Qwen/Qwen3-8B | Qwen/Qwen3-14B |
| Revision | b968826d9c46dd6066d109eabc6255188de91218 | 40c069824f4251a91eefaf281ebe4c544efd3e18 |
| Precision/device | FP32/CUDA | BF16/CUDA |
| Attention | eager | eager |
| Hidden / blocks | 4096 / 36 | 5120 / 40 |
| Q / KV heads, head dimension | 32 / 8, 128 | 40 / 8, 128 |
| Runtime | torch 2.8.0+cu128, Transformers 4.57.6 | same recorded cloud versions |
| Other recorded packages | tokenizers 0.22.2, safetensors 0.8.0, numpy 2.4.6, psutil 7.2.2 | same |
| Generation | greedy, non-thinking, code budget 96 | greedy, non-thinking, code budget 96 |
| Model-run seed | 42801 | 42801 |
| 14B CPU fitting | not applicable | torch 2.14.1, numpy 2.4.6, float64, one CPU thread, seed 42916 |

These are recorded versions, not a claim that every version is available from every package index or that arbitrary upgrades preserve behavior. The release's offline path needs NumPy only; it does not require the historical cloud runtime. Public model locations: [8B](https://huggingface.co/Qwen/Qwen3-8B), [14B](https://huggingface.co/Qwen/Qwen3-14B). Download the specified revision separately under its license. No automatic download is performed by this repository.

Private hostnames and absolute workstation/cloud paths have been redacted from exported records. Data filenames, numerical results, model revisions and hashes are retained. Original source hashes and public-export hashes are different when redaction or JSON formatting occurred.
