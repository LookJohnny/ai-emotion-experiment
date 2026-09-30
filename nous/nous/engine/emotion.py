"""Expression readout (emotion pre-sync): 4-class token probe + pre-reply ridge probe + frozen smoothing.
Pipeline per token: EMA of class probabilities -> calibrated intensities -> sustain rule -> pre-gate.
  alpha  : EMA weight of the new token (smaller = smoother)
  sustain: an emotion may rise to/above THR only after N consecutive tokens at/above THR (decreases are immediate)
  gamma  : if the PRE readout says 'neutral' (max emotional intensity < THR), token-phase emotional intensities are scaled by gamma"""
import numpy as np, torch

THR = 0.3
EMO = {"joy": "happy", "sadness": "sad", "anger": "angry"}


def calibrate(p, cal):
    w = {v: max(0.0, min(1.0, (p[k] - cal[k]["floor"]) / (cal[k]["ceil"] - cal[k]["floor"]))) for k, v in EMO.items()}
    w["neutral"] = 1 - max(w.values()); return w


class Smoother:
    def __init__(self, cal, alpha=0.3, sustain=1, gamma=1.0, pre=None):
        self.cal, self.alpha, self.N, self.gamma = cal, alpha, sustain, gamma
        self.ema = dict(pre) if pre else {"neutral": 1.0, "joy": 0.0, "sadness": 0.0, "anger": 0.0}
        pre_expr = calibrate(self.ema, cal) if pre else None
        self.pre_neutral = (pre_expr is not None) and max(pre_expr[v] for v in EMO.values()) < THR
        self.count = {v: 0 for v in EMO.values()}

    def pre_expr(self):
        return {k: round(v, 3) for k, v in calibrate(self.ema, self.cal).items()}

    def step(self, p):
        self.ema = {c: self.alpha * p[c] + (1 - self.alpha) * self.ema[c] for c in self.ema}
        e = calibrate(self.ema, self.cal)
        for v in EMO.values():
            if self.pre_neutral: e[v] *= self.gamma
            self.count[v] = self.count[v] + 1 if e[v] >= THR else 0
            if e[v] >= THR and self.count[v] < self.N: e[v] = THR - 0.01       # not yet sustained -> stay just below visible
        e["neutral"] = 1 - max(e[v] for v in EMO.values())
        return {k: round(v, 3) for k, v in e.items()}


class EmotionReadout:
    """token probe: softmax LR over (neutral, joy, sadness, anger) at layer L; pre probe: ridge at layer LP -> clipped, renormalised."""

    def __init__(self, token_npz, pre_npz, calib, smoothing, device):
        z = np.load(token_npz, allow_pickle=True); q = np.load(pre_npz, allow_pickle=True)
        T = lambda x: torch.tensor(x, device=device, dtype=torch.float32)
        self.L, self.cls = int(z["layer"]), [str(c) for c in z["classes"]]
        self.mu, self.sd, self.W, self.B = T(z["mean"]), T(z["scale"]), T(z["coef"]), T(z["intercept"])
        self.LP, self.pcls = int(q["layer"]), [str(c) for c in q["classes"]]
        self.pmu, self.psd, self.pW, self.pB = T(q["mean"]), T(q["scale"]), T(q["coef"]), T(q["intercept"])
        self.cal, self.smo = calib, smoothing

    def token(self, h):
        p = torch.softmax(((h - self.mu) / self.sd) @ self.W.T + self.B, -1).cpu().numpy()
        return {c: float(p[i]) for i, c in enumerate(self.cls)}

    def pre(self, h):
        y = torch.clamp(((h - self.pmu) / self.psd) @ self.pW.T + self.pB, min=0); y = (y / (y.sum() + 1e-6)).cpu().numpy()
        return {c: round(float(y[self.pcls.index(c)]), 4) for c in self.cls}

    def smoother(self, pre):
        s = self.smo; return Smoother(self.cal, s["alpha"], s["sustain"], s["gamma"], pre=pre)
