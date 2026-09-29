"""Expression post-processing shared by readout_server.py (live) and tune_smoothing.py (offline tuning).
Pipeline per token: EMA of class probabilities -> calibrated intensities -> sustain rule -> pre-gate.
  alpha  : EMA weight of the new token (smaller = smoother)
  sustain: an emotion may rise to/above THR only after N consecutive tokens at/above THR (decreases are immediate)
  gamma  : if the PRE readout says 'neutral' (max emotional intensity < THR), token-phase emotional intensities are scaled by gamma
"""
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
