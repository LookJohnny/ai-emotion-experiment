"""Two-tier alert policy (frozen in the V1.1 monitor).
tier 1 'watch'    : running score over threshold (after min_tokens) -- informational, never stops generation
tier 2 'confirmed': over threshold for `sustain` consecutive tokens  -- the only tier allowed to stop / rewrite"""


class TwoTier:
    def __init__(self, thr, min_tokens=8, sustain=3):
        self.thr, self.min_tokens, self.sustain, self.streak, self.fired_at = thr, min_tokens, sustain, 0, None

    def update(self, n, score):
        """n: tokens so far; score: running score. Returns (watch, confirmed)."""
        over = n >= self.min_tokens and score >= self.thr
        self.streak = self.streak + 1 if over else 0
        confirmed = self.streak >= self.sustain
        if confirmed and self.fired_at is None: self.fired_at = n
        return over and not confirmed, confirmed


def percentile(value, quantiles):
    """position of value among validation-negative quantiles (0..100) -> 'how unusual vs normal traffic'."""
    if not quantiles: return None
    k = sum(1 for q in quantiles if q <= value)
    return round(100 * max(0, k - 1) / (len(quantiles) - 1), 1)
