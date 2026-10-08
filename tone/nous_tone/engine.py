"""Runtime: generate with tone control and per-token readouts.
  eng = ToneEngine(model_dir, ["packs/anger", "packs/joy"])
  for ev in eng.stream(messages, tone={"anger": -1.0}, target={"joy": 0.7}): ...
tone   = open-loop strength per attribute (-2..2); 1.0 = the pack's default lambda; negative pushes away from the attribute
target = closed loop: desired intensity 0..1; lambda is adjusted every token from the readout (never above max_lambda)
Readouts come from the same heads and directions that steer: the head output (before our injection) projected on the
pack's unit vectors, mapped to 0..1 by the pack's calibration. Expression weights for avatars follow the readouts."""
import math, threading
import numpy as np, torch
from .hooks import HeadHooks
from .pack import TonePack


def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


class ToneEngine:
    def __init__(self, model_dir=None, pack_dirs=(), device=None, model=None, tok=None, gain=1.0, ema=0.3, readout_dir=None):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.device = device or pick_device()
        if model is None:
            tok = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32 if self.device == "cpu" else torch.bfloat16).to(self.device).eval()
        self.model, self.tok, self.gain, self.ema_a = model, tok, gain, ema
        self.H = HeadHooks(model); self.lock = threading.Lock(); self.packs = {}
        g = getattr(model.generation_config, "eos_token_id", None)
        self.eos = {tok.eos_token_id} | (set(g) if isinstance(g, (list, tuple)) else {g} if g is not None else set())
        for d in pack_dirs: self.add_pack(TonePack.load(d))
        self.probe, self._res = None, None
        if readout_dir: self.set_readout(readout_dir)

    def set_readout(self, d):
        """A multinomial residual probe (nous_tone.readout) for readouts that tell tones apart; replaces the head projection."""
        from .readout import ReadoutProbe
        self.probe = ReadoutProbe(d)
        def tap(mod, a, out): self._res = (out[0] if isinstance(out, tuple) else out)[0, -1].detach().float().cpu().numpy()
        self.model.model.layers[self.probe.layer].register_forward_hook(tap)

    def add_pack(self, p: TonePack):
        assert (p.layers, p.heads_per_layer, p.head_dim) == (self.H.n_layers, self.H.n_heads, self.H.head_dim), f"pack {p.attribute} was built for another model shape"
        p._vec = [torch.tensor(v, device=self.device) for v in p.vectors]
        p._unit = [torch.tensor(u, device=self.device) for u in p.unit_vectors()]
        self.packs[p.attribute] = p; self.H.read_layers |= {L for L, _ in p.heads}

    def _V(self, lams):
        V = {}
        for a, lam in lams.items():
            if not lam: continue
            p = self.packs[a]
            for (L, h), v in zip(p.heads, p._vec):
                V.setdefault(L, torch.zeros(self.H.n_heads * self.H.head_dim, device=self.device))[self.H.head_slice(h)] += lam * v
        return V

    def _readouts(self, run=None):
        if self.probe is not None and run is not None:                    # probe on the running-mean residual
            p = self.probe.probs(run); return {a: p[a] for a in self.packs if a in p} | {"neutral": p.get("reference", 0.0)}
        out = {}
        for a, p in self.packs.items():
            s = float(np.mean([float(self.H.tap[L][self.H.head_slice(h)] @ u) for (L, h), u in zip(p.heads, p._unit)]))
            out[a] = 1 / (1 + math.exp(-(p.readout["a"] * s + p.readout["b"])))
        return out

    def _render(self, messages):
        try: text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError: text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True); ids = enc["input_ids"]
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        c = text.rfind(last_user) if last_user else -1
        start = next((i for i, (s, _) in enumerate(enc["offset_mapping"]) if s >= c), len(ids) - 1) if c >= 0 else 0
        return ids, start                                                # steer from the last user turn on, like the packs were built

    @staticmethod
    def strength_for(p, t):
        """Invert the pack's calibration curve (monotone envelope); without calibration: target 0.5 -> strength 1."""
        c = p.meta.get("calibration")
        if not c or len(c["strength"]) < 2: return 2 * t
        s, r = np.array(c["strength"], float), np.maximum.accumulate(np.array(c["readout"], float))
        if t <= r[0]: return float(s[0]) if t < r[0] - 0.05 else 0.0
        if t >= r[-1]: return float(s[-1])
        return float(np.interp(t, r, s))

    def expressions(self, r):
        w = {}
        for a, v in r.items():
            if a not in self.packs: continue
            e = self.packs[a].expression
            lo, hi = (0.25, 0.75) if self.probe is not None else (0.5, 1.0)          # probe: share among 5 tones; projection: 0.5 = undecided
            if e: w[e] = max(w.get(e, 0.0), round(max(0.0, min(1.0, (v - lo) / (hi - lo))), 3))
        w["neutral"] = round(1 - max(w.values(), default=0.0), 3); return w

    @torch.no_grad()
    def stream(self, messages, tone=None, target=None, max_new_tokens=256, temperature=0.0, top_p=1.0, seed=None):
        tone, target = dict(tone or {}), dict(target or {})
        unknown = (set(tone) | set(target)) - set(self.packs)
        if unknown: raise ValueError(f"no tone pack for {sorted(unknown)}; loaded: {sorted(self.packs)}")
        lams = {a: max(-p.max_lambda, min(p.max_lambda, tone.get(a, 0.0) * p.default_lambda)) for a, p in self.packs.items()}
        for a, t in target.items():
            # the tone of a reply is set in its first tokens (later lambda changes barely move it), so target mode picks the
            # strength up front from the pack's held-out calibration (strength -> mean readout); within the reply the readout
            # only pulls lambda DOWN when it overshoots the target
            p = self.packs[a]; lams[a] = max(-p.max_lambda, min(p.max_lambda, self.strength_for(p, t) * p.default_lambda))
        ids, start = self._render(messages); g = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
        ema, run_sum, run_n = None, None, 0
        with self.lock:
            H = self.H; H.mode, H.V, H.start = "steer", self._V(lams), start
            try:
                out = self.model(torch.tensor([ids], device=self.device), use_cache=True); past = out.past_key_values
                logits = out.logits[0, -1].float(); text_so_far, toks = "", []
                for step in range(max_new_tokens):
                    if self.probe is not None and self._res is not None:
                        # the probe was trained on reply tokens: the running mean starts at the first generated token
                        # (step 0's state is the last prompt token, a different distribution; a pre-reply probe is separate work)
                        if step > 0:
                            run_sum = self._res.copy() if run_sum is None else run_sum + self._res; run_n += 1
                        r = self._readouts(run_sum / run_n) if run_n else {a: 0.0 for a in self.packs if a in self.probe.classes} | {"neutral": 1.0}
                        ema = r
                    else:
                        r = self._readouts()                                              # state that produced this token
                        ema = r if ema is None else {a: (1 - self.ema_a) * ema[a] + self.ema_a * r[a] for a in r}
                    if temperature and temperature > 0:
                        probs = torch.softmax(logits / temperature, -1)
                        if top_p < 1:
                            sp, si = probs.sort(descending=True); keep = sp.cumsum(0) - sp < top_p
                            probs = torch.zeros_like(probs).scatter(0, si[keep], sp[keep]); probs /= probs.sum()
                        nxt = int(torch.multinomial(probs.cpu(), 1, generator=g))
                    else: nxt = int(logits.argmax())
                    if nxt in self.eos:
                        yield {"done": True, "finish_reason": "stop", "readout": ema, "lambda": dict(lams)}; return
                    toks.append(nxt); full = self.tok.decode(toks, skip_special_tokens=True)
                    if full.endswith("\ufffd"): delta = ""                    # a multi-byte character split across tokens: wait for the rest
                    else: delta = full[len(text_so_far):]; text_so_far = full
                    for a, t in target.items():                                          # overshoot guard only
                        over = ema.get(a, 0.0) - (t + 0.15)
                        if over > 0 and step > 8: lams[a] = max(0.0, lams[a] - self.gain * over)
                    yield {"delta": delta, "readout": {a: round(v, 4) for a, v in ema.items()}, "expression": self.expressions(ema),
                           "lambda": {a: round(v, 3) for a, v in lams.items() if v}}
                    H.V = self._V(lams)
                    out = self.model(torch.tensor([[nxt]], device=self.device), past_key_values=past, use_cache=True)
                    past = out.past_key_values; logits = out.logits[0, -1].float()
                yield {"done": True, "finish_reason": "length", "readout": ema, "lambda": dict(lams)}
            finally:
                H.mode, H.V = None, None

    def generate(self, messages, **kw):
        text, last = "", {}
        for ev in self.stream(messages, **kw):
            if ev.get("done"): last = ev; break
            text += ev["delta"]
        return {"text": text, "finish_reason": last.get("finish_reason"), "readout": last.get("readout"), "lambda": last.get("lambda")}
