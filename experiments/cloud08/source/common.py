"""Offline-only, immutable-input CUDA teacher precheck utilities."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8')

def read(path):
    return json.loads(Path(path).read_text())

def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def verify_package():
    for relative, expected in read(ROOT / 'freeze.json')['files'].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError('PACKAGE_SHA_MISMATCH: ' + relative)

def seal_output(path):
    path=Path(path)
    write(path/'manifest.json',dict(files={str(p.relative_to(path)):sha(p)
        for p in sorted(path.rglob('*')) if p.is_file() and p.name!='manifest.json'}))

def verify_model(path):
    expected = read(ROOT / 'data/model_sha.json')
    actual = {}
    for name, record in expected.items():
        p = Path(path) / name
        actual[name] = {'bytes': p.stat().st_size, 'sha256': sha(p)}
        if actual[name] != record:
            raise RuntimeError('MODEL_SHA_MISMATCH: ' + name)
    return actual

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

def load_model(path):
    import torch
    from transformers import AutoModelForCausalLM
    cfg = read(ROOT / 'data/config.json')
    assert torch.cuda.is_available(), 'CUDA_REQUIRED'
    assert torch.cuda.device_count() == 1, 'EXPOSE_EXACTLY_ONE_GPU'
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    free, total = torch.cuda.mem_get_info()
    assert free >= cfg['minimum_free_gpu_gib'] * 2**30, 'INSUFFICIENT_FREE_GPU_MEMORY'
    import psutil
    available=psutil.virtual_memory().available
    cgroup_limit=Path('/sys/fs/cgroup/memory.max')
    cgroup_usage=Path('/sys/fs/cgroup/memory.current')
    if cgroup_limit.exists() and cgroup_limit.read_text().strip().isdigit():
        available=min(available,int(cgroup_limit.read_text())-int(cgroup_usage.read_text()))
    assert available >= 40*2**30, 'INSUFFICIENT_HOST_MEMORY'
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
        dtype=torch.float32, attn_implementation='eager').to('cuda:0').eval()
    model.requires_grad_(False)
    assert all(p.device.type == 'cuda' and p.dtype == torch.float32 for p in model.parameters())
    return model

def parameter_versions(model):
    return {n: p._version for n, p in model.named_parameters()}

def check_parameters(model, versions):
    assert parameter_versions(model) == versions, 'PARAMETER_VERSION_CHANGED'
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
