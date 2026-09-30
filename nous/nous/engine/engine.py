"""Nous engine: one forward pass -> reply tokens + internal-state readouts, shared by the gateway and the companion SDK.

Events (dicts), in order:
  {"phase":"pre",   "C2", "C1_pre", "C3H_pre", "emo":{"probs","expr"}, "inner", "alarm":{...}}    before the first token
  {"phase":"token", "i", "delta", "C1_token", "C1_mean", "C3H_mean", "emo":{"expr"}, "inner", "alarm":{...}}
  {"phase":"done",  "tokens", "C1_mean", "C3H_mean", "finish", "text"}
alarm keys: C2, C1_pre, C3H_pre (tier-1, pre-reply) ; C1_watch / C1 and C3H_watch / C3H (tier-1 / tier-2 while generating).

Two ways to drive it:
  stream()  : local generation (self-hosted model; full white-box incl. pre-reply warning)
  reader()  : teacher-force text produced elsewhere (closed API) through the local model -- 'reader proxy'.
              Pre-reply readouts about the *other* model's reply are not meaningful there; C2 (user message) still is."""
import asyncio, contextlib, os, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .hooks import ResidualTap
from .pack import Pack
from .emotion import EmotionReadout
from .inner import InnerView
from .policy import TwoTier


class Engine:
    def __init__(self, model_path, pack_dir, device="cuda", visualize=True, min_tokens=8, sustain=3):
        self.pack = Pack(pack_dir); self.device = device; self.model_name = (self.pack.m.get("base_model") or os.path.basename(model_path.rstrip("/\\"))).split("/")[-1]
        self.min_tokens, self.sustain = min_tokens, sustain
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        m = self.pack.m; self.emotion = None
        if m.get("emotion"):
            e = m["emotion"]; import json
            self.emotion = EmotionReadout(self.pack.path(e["token"]), self.pack.path(e["pre"]),
                                          json.load(open(self.pack.path(e["calib"])))["emotions"], e["smoothing"], device)
        layers = [p.layer for p in self.pack.probes.values()] + ([self.emotion.L, self.emotion.LP] if self.emotion else [])
        vis = visualize and m.get("layer_probes")
        if vis: layers += list(range(self.model.config.num_hidden_layers))
        self.tap = ResidualTap(self.model, layers)
        self.inner = InnerView(self.model, self.tap, self.pack.path(m["layer_probes"]), m.get("heads", [])) if vis else None
        self.eos = {i for i in (self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")) if i is not None}
        self.lock = asyncio.Lock()

    # ---------- helpers ----------
    def ids(self, messages):
        return self.tok(self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]

    def c2(self, messages):
        """C2 = mean state over the last user message, from a prefill WITHOUT the system prompt (as trained)."""
        m = [x for x in messages if x["role"] != "system"]
        text = self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        e = self.tok(text, return_offsets_mapping=True, add_special_tokens=False); u = m[-1]["content"]; c0 = text.rindex(u); c1 = c0 + len(u)
        idx = [i for i, (s, t) in enumerate(e["offset_mapping"]) if s >= c0 and t <= c1 and t > s]
        self.model(torch.tensor([e["input_ids"]], device=self.device), use_cache=False)
        sp = self.pack.probes["c2"]; return float(sp.prob(self.tap.full[sp.layer][idx[0]:idx[-1] + 1].mean(0)))

    def info(self):
        return {"model": self.model_name, "pack": self.pack.card(), "thresholds": {k: v for k, v in self.pack.thr.items()},
                "policy": {"min_tokens": self.min_tokens, "sustain": self.sustain}, "neg_quantiles": self.pack.neg_q,
                "emotion": bool(self.emotion), "inner": self.inner.describe() if self.inner else None}

    # ---------- local generation ----------
    async def stream(self, messages, *, max_tokens=200, temperature=0.7, concepts=None, emotion=False, inner=False,
                     force=None, force_max=None, stop_on=None):
        """async generator of events. force: teacher-force this text instead of sampling (replay / audit).
        stop_on: tier-2 alarm names that end generation (finish_reason 'content_filter'), e.g. ("C1",)."""
        forced = self.tok(force, add_special_tokens=False)["input_ids"][:force_max] if force else None
        async with self.lock:
            run = await asyncio.to_thread(Run, self, messages, concepts, emotion, inner)
            yield run.pre
            finish = "length"
            for step in range(len(forced) if forced else max_tokens):
                ev = await asyncio.to_thread(run.advance, forced[step] if forced else None, temperature)
                if ev is None: finish = "stop"; break
                yield ev
                if stop_on and any(ev["alarm"].get(c) for c in stop_on): finish = "content_filter"; break
            else:
                if forced: finish = "stop"
            yield run.done(finish)

    # ---------- reader proxy ----------
    @contextlib.asynccontextmanager
    async def reader(self, messages, concepts=None, max_read=512):
        async with self.lock:
            run = await asyncio.to_thread(Run, self, messages, concepts, False, False)
            yield ReaderFeed(self, run, max_read)


class Run:
    """one reply: prefill + pre readouts, then one token at a time."""

    @torch.no_grad()
    def __init__(self, E, messages, concepts, emotion, inner):
        self.E = E; self.T0 = time.time(); P = E.pack.probes; st = E.tap.states
        self.c = set(concepts or E.pack.concepts) & set(E.pack.concepts)
        self.emo = E.emotion if emotion else None; self.vis = E.inner if inner else None
        pre = {"phase": "pre"}; al = {}
        if "C2" in self.c:
            pre["C2"] = round(E.c2(messages), 4); al["C2"] = pre["C2"] >= E.pack.thr["C2"]
        out = E.model(torch.tensor([E.ids(messages)], device=E.device), use_cache=True); self.past, self.logits = out.past_key_values, out.logits[0, -1]
        if "C1" in self.c:
            pre["C1_pre"] = round(float(P["c1_pre"].prob(st[P["c1_pre"].layer])), 4); al["C1_pre"] = pre["C1_pre"] >= E.pack.thr["C1_pre"]
            self.a1 = TwoTier(E.pack.thr["C1_mean"], E.min_tokens, E.sustain); self.ps = []
        if "C3H" in self.c:
            pre["C3H_pre"] = round(float(P["c3h_pre"].prob(st[P["c3h_pre"].layer])), 4); al["C3H_pre"] = pre["C3H_pre"] >= E.pack.thr["C3H_pre"]
            self.a3 = TwoTier(E.pack.thr["C3H_mean"], E.min_tokens, E.sustain); self.hsum = torch.zeros_like(st[P["c3h_token"].layer]); self.hm = None
        if self.emo:
            pp = self.emo.pre(st[self.emo.LP]); self.sm = self.emo.smoother(pp); pre["emo"] = {"probs": pp, "expr": self.sm.pre_expr()}
        if self.vis: pre["inner"] = self.vis.view()
        pre["alarm"] = al; pre["t"] = self.ms(); self.pre = pre; self.gen, self.text = [], ""

    def ms(self): return round((time.time() - self.T0) * 1000, 1)

    @torch.no_grad()
    def advance(self, tid=None, temperature=0.7):
        """feed one token (forced id, or sampled from the current logits). Returns a token event, or None at EOS."""
        E = self.E
        if tid is None:
            if temperature > 0:
                pr = torch.softmax(self.logits.float() / temperature, -1); tv, ti = pr.topk(20); tid = int(ti[torch.multinomial(tv / tv.sum(), 1)])
            else: tid = int(self.logits.argmax())
            if tid in E.eos: return None
        self.gen.append(tid); new = E.tok.decode(self.gen, skip_special_tokens=True)
        delta = "" if new.endswith("�") else new[len(self.text):]
        if delta: self.text = new
        out = E.model(torch.tensor([[tid]], device=E.device), past_key_values=self.past, use_cache=True); self.past, self.logits = out.past_key_values, out.logits[0, -1]
        n = len(self.gen); st = E.tap.states; P = E.pack.probes; ev = {"phase": "token", "i": n, "delta": delta}; al = {}
        if "C1" in self.c:
            p = float(P["c1_token"].prob(st[P["c1_token"].layer])); self.ps.append(p); mean = sum(self.ps) / n
            al["C1_watch"], al["C1"] = self.a1.update(n, mean); ev["C1_token"], ev["C1_mean"] = round(p, 4), round(mean, 4)
        if "C3H" in self.c:
            sp = P["c3h_token"]; self.hsum += st[sp.layer]; self.hm = float(sp.prob(self.hsum / n))
            al["C3H_watch"], al["C3H"] = self.a3.update(n, self.hm); ev["C3H_mean"] = round(self.hm, 4)
        if self.emo: ev["emo"] = {"expr": self.sm.step(self.emo.token(st[self.emo.L]))}
        if self.vis: ev["inner"] = self.vis.view()
        ev["alarm"] = al; ev["t"] = self.ms(); return ev

    def done(self, finish):
        d = {"phase": "done", "tokens": len(self.gen), "finish": finish, "text": self.text, "t": self.ms()}
        if "C1" in self.c: d["C1_mean"] = round(sum(self.ps) / len(self.ps), 4) if self.ps else None; d["C1_confirmed_at"] = self.a1.fired_at
        if "C3H" in self.c: d["C3H_mean"] = round(self.hm, 4) if self.hm is not None else None; d["C3H_confirmed_at"] = self.a3.fired_at
        return d


class ReaderFeed:
    """incremental teacher forcing of streamed text. Text arrives in arbitrary pieces; we re-tokenize the accumulated text and
    feed only tokens that can no longer change (all but the last), the rest at finish()."""

    def __init__(self, E, run, max_read):
        self.E, self.run, self.max_read, self.buf, self.fed, self.drift = E, run, max_read, "", [], 0
        self.pre = {k: v for k, v in run.pre.items() if k not in ("C1_pre", "C3H_pre")}           # pre-reply readouts refer to OUR reply, not theirs
        self.pre["alarm"] = {k: v for k, v in run.pre["alarm"].items() if k == "C2"}

    async def _feed(self, ids):
        evs = []
        for t in ids:
            if len(self.fed) >= self.max_read: break
            evs.append(await asyncio.to_thread(self.run.advance, t)); self.fed.append(t)
        return evs

    async def push(self, text):
        self.buf += text; ids = self.E.tok(self.buf, add_special_tokens=False)["input_ids"]
        if ids[:len(self.fed)] != self.fed: self.drift += 1; return []       # re-tokenization changed an early token: wait
        return await self._feed(ids[len(self.fed):-1])

    async def finish(self, finish="stop"):
        ids = self.E.tok(self.buf, add_special_tokens=False)["input_ids"]
        if ids[:len(self.fed)] != self.fed: self.drift += 1                  # cannot un-feed; continue from the same offset (approximation)
        evs = await self._feed(ids[len(self.fed):])
        d = self.run.done(finish); d["reader"] = {"fed_tokens": len(self.fed), "retokenize_drift": self.drift, "truncated": len(self.fed) >= self.max_read}
        return evs, d
