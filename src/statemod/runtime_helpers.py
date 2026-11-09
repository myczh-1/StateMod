"""Portable helpers copied from Cloud03; no model loading or remote operations."""

import hashlib, json

from pathlib import Path

def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)

def save(path, **arrays):
    import numpy as np
    with Path(path).open('xb') as f:
        np.savez_compressed(f, **arrays)

def metrics(a, b):
    import numpy as np
    d = a.astype(np.float64) - b.astype(np.float64)
    return dict(max_abs=float(np.abs(d).max()), rms=float(np.sqrt(np.mean(d*d))),
                relative_l2=float(np.linalg.norm(d)/max(np.linalg.norm(b), 1e-30)),
                exact=bool(np.array_equal(a, b)))

def logp(z, token):
    import numpy as np
    z = z.astype(np.float64)
    m = z.max()
    return float(z[token] - m - np.log(np.exp(z-m).sum()))

def forward(model, ids, cache=None):
    import torch
    from transformers import DynamicCache
    cache = DynamicCache(config=model.config) if cache is None else cache
    n = cache.get_seq_length()
    device = next(model.parameters()).device
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([ids], device=device),
                    attention_mask=torch.ones((1, n+len(ids)), dtype=torch.long, device=device),
                    position_ids=torch.arange(n, n+len(ids), device=device)[None],
                    cache_position=torch.arange(n, n+len(ids), device=device),
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
    assert cache.get_seq_length() == n+len(ids)
    for layer in cache.layers:
        assert layer.keys.shape == layer.values.shape == (
            1, model.config.num_key_value_heads, n+len(ids), model.config.head_dim)
    return out.logits[0, -1].float().cpu().numpy().copy(), cache

def cache_hash(cache):
    return {f'{i}/{part}': hashlib.sha256(getattr(layer, part).detach().cpu().numpy().tobytes()).hexdigest()
            for i, layer in enumerate(cache.layers) for part in ('keys', 'values')}
