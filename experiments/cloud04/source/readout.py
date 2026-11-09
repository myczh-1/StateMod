"""Frozen Cloud02 judgment; numeric state only, no task text or labels."""
import math
import numpy as np

class Judgment:
    def __init__(self, path):
        with np.load(path) as a:
            self.w = {k: a[k].copy() for k in a.files}
        assert self.w['0.weight'].shape == (8, 4096)
        for a in self.w.values():
            assert np.isfinite(a).all()
            a.flags.writeable = False

    def predict(self, hidden):
        h = np.asarray(hidden, dtype=np.float64)
        assert h.shape == (4096,) and np.isfinite(h).all()
        x = h / np.sqrt(np.mean(h*h) + 1e-6)
        x = (x-self.w['center']) / self.w['scale']
        z = self.w['0.weight'] @ x + self.w['0.bias']
        g = .5*z*(1+np.array([math.erf(float(v)/math.sqrt(2)) for v in z]))
        logit = float((self.w['2.weight'] @ g + self.w['2.bias']).item())
        p = math.exp(-np.logaddexp(0., -logit))
        return dict(logit=logit, p=p, enabled=p > .5, threshold=.5)
