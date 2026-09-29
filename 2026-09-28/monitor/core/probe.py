"""Binary concept probe = standardise -> logistic regression at one layer, applied to a pooled state.
pool: 'last_prompt' (pre-reply: state at the last prompt position), 'token' (per generated token),
      'mean_span' (mean over a token span, e.g. the last user message)."""
import json, numpy as np, torch
from dataclasses import dataclass, asdict, field


@dataclass
class ProbeSpec:
    concept: str; name: str; layer: int; pool: str
    mean: list; scale: list; coef: list; intercept: float
    threshold: float = 0.5            # alarm threshold on probability (chosen on val)
    meta: dict = field(default_factory=dict)

    def prob(self, h):
        """h: torch [d] or [n, d] -> probability tensor."""
        mu = torch.tensor(self.mean, device=h.device); sd = torch.tensor(self.scale, device=h.device)
        w = torch.tensor(self.coef, device=h.device)
        return torch.sigmoid(((h - mu) / sd) @ w + self.intercept)

    def save(self, path): json.dump(asdict(self), open(path, "w"))

    @staticmethod
    def load(path): return ProbeSpec(**json.load(open(path)))

    @staticmethod
    def from_sklearn(concept, name, layer, pool, scaler, clf, **kw):
        return ProbeSpec(concept, name, layer, pool, scaler.mean_.tolist(), scaler.scale_.tolist(),
                         clf.coef_[0].tolist(), float(clf.intercept_[0]), **kw)
