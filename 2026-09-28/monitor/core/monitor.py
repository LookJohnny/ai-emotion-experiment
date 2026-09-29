"""Conflict-escalation monitor (V1).
  pre_check(messages) : before generation -> C2 (user distress, last user message) and C1-pre (predicted hostility of the coming reply)
  on_token()          : after each generated token -> C1 per-token score, EMA-smoothed and gated by C1-pre
Alarm policy (frozen on val, see train/calibrate.py): C2 alarm if p >= thr; C1 alarm if pre p >= thr_pre (fires before the first
token) or smoothed per-token p >= thr_tok; when C1-pre is below its threshold, per-token increments are scaled by gamma."""
import json
from .probe import ProbeSpec
from .hooks import ResidualTap


class Monitor:
    def __init__(self, model, tok, specs, policy):
        self.model, self.tok, self.P, self.pol = model, tok, specs, policy       # specs: dict name -> ProbeSpec
        self.tap = ResidualTap(model, [s.layer for s in specs.values()])

    def user_span(self, messages):
        """token span of the last user message inside the chat-templated prompt."""
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        e = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        u = messages[-1]["content"]; c0 = text.rindex(u); c1 = c0 + len(u)
        idx = [i for i, (s, t) in enumerate(e["offset_mapping"]) if s >= c0 and t <= c1 and t > s]
        return e["input_ids"], (idx[0], idx[-1] + 1)

    def pre_check(self, messages):
        """run AFTER the prefill forward of `messages` (the caller does the forward so generation can reuse the KV cache)."""
        out = {}
        if "c2_user" in self.P:
            _, (s, e) = self.user_span(messages); sp = self.P["c2_user"]
            out["C2"] = float(sp.prob(self.tap.full[sp.layer][s:e].mean(0)))
        if "c1_pre" in self.P:
            sp = self.P["c1_pre"]; out["C1_pre"] = float(sp.prob(self.tap.states[sp.layer]))
        self.ema = out.get("C1_pre", 0.0)
        self.gated = "C1_pre" in out and out["C1_pre"] < self.pol["c1_pre_thr"]
        out["alarm"] = {"C2": out.get("C2", 0) >= self.pol["c2_thr"], "C1_pre": out.get("C1_pre", 0) >= self.pol["c1_pre_thr"]}
        return out

    def on_token(self):
        sp = self.P["c1_token"]; p = float(sp.prob(self.tap.states[sp.layer]))
        a = self.pol["alpha"]; inc = a * (p - self.ema)
        if self.gated and inc > 0: inc *= self.pol["gamma"]
        self.ema += inc
        return {"C1_token": round(p, 4), "C1_smooth": round(self.ema, 4), "alarm": self.ema >= self.pol["c1_tok_thr"]}
