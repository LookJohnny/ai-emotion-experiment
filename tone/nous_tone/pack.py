"""A tone pack: everything needed to steer and read one attribute in one model. JSON (metadata) + NPZ (vectors)."""
import json, os
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class TonePack:
    attribute: str                      # e.g. "anger"
    model: str                          # model folder name it was built on
    layers: int
    heads_per_layer: int
    head_dim: int
    heads: list                         # [[layer, head], ...] selected steering/readout heads, best first
    vectors: np.ndarray                 # [K, head_dim] per-head direction (attribute minus reference, mean at aligned positions)
    default_lambda: float = 4.0         # strength 1.0 = this lambda
    max_lambda: float = 8.0             # guard: the engine never steers harder
    readout: dict = field(default_factory=dict)    # {"a": slope, "b": intercept} logistic map of the projection score to 0..1
    expression: str = ""                # avatar preset it drives (happy / sad / angry / ...), "" = none
    meta: dict = field(default_factory=dict)        # build info: selection scores, dose curve, timings, data size

    def save(self, d):
        os.makedirs(d, exist_ok=True)
        m = asdict(self); m.pop("vectors")
        json.dump(m, open(os.path.join(d, "pack.json"), "w"), ensure_ascii=False, indent=1)
        np.savez(os.path.join(d, "vectors.npz"), vectors=self.vectors.astype(np.float32))

    @classmethod
    def load(cls, d):
        m = json.load(open(os.path.join(d, "pack.json"), encoding="utf-8"))
        return cls(vectors=np.load(os.path.join(d, "vectors.npz"))["vectors"], **m)

    def unit_vectors(self):
        n = np.linalg.norm(self.vectors, axis=1, keepdims=True); return self.vectors / np.maximum(n, 1e-9)
