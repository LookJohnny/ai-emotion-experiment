"""Runtime: generate with tone control and per-token readouts.
  eng = ToneEngine(model_dir, ["packs/anger", "packs/joy"])
  for ev in eng.stream(messages, tone={"anger": -1.0}, target={"joy": 0.7}): ...
tone   = open-loop strength per attribute (-2..2); 1.0 = the pack's default lambda; negative pushes away from the attribute
target = closed loop: desired intensity 0..1; lambda is adjusted every token from the readout (never above max_lambda)
Readouts come from the same heads and directions that steer: the head output (before our injection) projected on the
pack's unit vectors, mapped to 0..1 by the pack's calibration. Expression weights for avatars follow the readouts."""
import collections, concurrent.futures, logging, math, queue, threading, time
import numpy as np, torch
from .hooks import HeadHooks
from .pack import TonePack


def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


log = logging.getLogger("nous_tone")


class JsonObjectEnd:
    """Streaming scanner: True once the first top-level JSON object in the text has closed (strings and escapes aware)."""
    def __init__(self): self.depth, self.started, self.in_str, self.esc, self.done = 0, False, False, False, False

    def feed(self, text):
        for c in text:
            if self.done: break
            if self.in_str:
                if self.esc: self.esc = False
                elif c == "\\": self.esc = True
                elif c == '"': self.in_str = False
            elif c == '"' and self.started: self.in_str = True
            elif c == "{": self.depth += 1; self.started = True
            elif c == "}" and self.started:
                self.depth -= 1
                if self.depth == 0: self.done = True
        return self.done


def json_object_slice(text):
    """The first complete top-level JSON object in text (fences / prose around it dropped), or text unchanged."""
    start = text.find("{")
    if start < 0: return text
    sc = JsonObjectEnd()
    for i, c in enumerate(text[start:]):
        if sc.feed(c): return text[start:start + i + 1]
    return text


class PriorityLock:
    """One generation at a time; when it frees, the highest-priority waiter goes next (FIFO within a priority).
    A local model serves a user's turn and a character's autonomous musings from the same weights: the user must not
    queue behind background work."""
    def __init__(self):
        self._cv, self._busy, self._queue, self._seq = threading.Condition(), False, [], 0

    def acquire(self, priority=0):
        with self._cv:
            self._seq += 1; me = (-int(priority), self._seq); self._queue.append(me)
            while self._busy or min(self._queue) != me: self._cv.wait()
            self._queue.remove(me); self._busy = True

    def release(self):
        with self._cv: self._busy = False; self._cv.notify_all()

    def outranked(self, priority) -> bool:
        """A waiter with a higher priority than `priority` is queued."""
        with self._cv:
            return bool(self._queue) and -min(self._queue)[0] > int(priority)

    def hold(self, priority=0):
        lock = self
        class _Ctx:
            def __enter__(s): lock.acquire(priority)
            def __exit__(s, *a): lock.release()
        return _Ctx()

    # threading.Lock compatibility (`with engine.lock:` = priority 0)
    def __enter__(self): self.acquire(0)
    def __exit__(self, *a): self.release()


class ToneEngine:
    def __init__(self, model_dir=None, pack_dirs=(), device=None, model=None, tok=None, gain=1.0, ema=0.3, readout_dir=None,
                 prefix_cache=3):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.device = device or pick_device()
        if model is None:
            tok = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32 if self.device == "cpu" else torch.bfloat16).to(self.device).eval()
        self.model, self.tok, self.gain, self.ema_a = model, tok, gain, ema
        self.H = HeadHooks(model); self.lock = PriorityLock(); self.packs = {}
        # Reads that arrive while a generation holds the model run between its tokens
        # (a read is one ~0.1 s forward): a streamed decision's lines get their tone
        # without waiting for the whole decision to finish.
        self._jobs: queue.SimpleQueue = queue.SimpleQueue(); self._job_mutex = threading.Lock(); self._generating = False
        # Prompt KV prefixes of recent requests (LRU). A companion's prompts share a long head
        # (character, contract, history), so the next turn prefills only what changed.
        self.prefix_cache = prefix_cache; self._prefixes: collections.OrderedDict = collections.OrderedDict()
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

    def attributes(self):
        """Attributes that get readouts: the loaded packs, or (readout-only mode, no packs) the probe's classes."""
        if self.packs or self.probe is None: return list(self.packs)
        return [c for c in self.probe.classes if c != "reference"]

    def expression_of(self, a):
        from .prompts import ATTRIBUTES
        return self.packs[a].expression if a in self.packs else ATTRIBUTES.get(a, {}).get("expression", "")

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
            p = self.probe.probs(run); return {a: p[a] for a in self.attributes() if a in p} | {"neutral": p.get("reference", 0.0)}
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
        attrs = set(self.attributes())
        for a, v in r.items():
            if a not in attrs: continue
            e = self.expression_of(a)
            lo, hi = (0.25, 0.75) if self.probe is not None else (0.5, 1.0)          # probe: share among 5 tones; projection: 0.5 = undecided
            if e: w[e] = max(w.get(e, 0.0), round(max(0.0, min(1.0, (v - lo) / (hi - lo))), 3))
        w["neutral"] = round(1 - max(w.values(), default=0.0), 3); return w

    @torch.no_grad()
    def stream(self, messages, tone=None, target=None, max_new_tokens=256, temperature=0.0, top_p=1.0, seed=None, priority=0,
               json_object=False, prefill="", preemptible=False):
        """prefill: text the reply is forced to begin with (returned as part of it), e.g. '{"selected_intent": "' so a small
        model cannot answer in prose or echo its input when a JSON object is required.
        preemptible: optional background work. It stops (finish_reason "preempted") as soon as a
        higher-priority request is waiting, so a user never waits for a background generation."""
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
        pre_ids = self.tok(prefill, add_special_tokens=False)["input_ids"] if prefill else []
        ids = ids + pre_ids
        ema, run_sum, run_n = None, None, 0
        t_queued = time.monotonic(); n_out, finish = 0, "aborted"
        scan = JsonObjectEnd() if json_object else None                   # JSON mode: stop once the object closes
        if scan is not None and prefill: scan.feed(prefill)
        with self.lock.hold(priority):
            V = self._V(lams); steered = bool(V)
            reused, cached = self._prefix_lookup(ids, start, steered)
            # the hook steers from `start` within the chunk it sees; the chunk now begins at `reused`
            H = self.H; H.mode, H.V, H.start = "steer", V, max(0, start - reused)
            t_start = time.monotonic(); past = None
            with self._job_mutex: self._generating = True
            try:
                out = self.model(torch.tensor([ids[reused:]], device=self.device), past_key_values=cached, use_cache=True)
                past = out.past_key_values
                logits = out.logits[0, -1].float(); toks = list(pre_ids)
                text_so_far = self.tok.decode(toks, skip_special_tokens=True) if toks else ""; lead = text_so_far
                for step in range(max_new_tokens):
                    self._drain_jobs()                                       # reads waiting on the model
                    if preemptible and self.lock.outranked(priority):
                        finish = "preempted"
                        yield {"done": True, "finish_reason": "preempted", "readout": ema, "lambda": dict(lams)}; return
                    if self.probe is not None and self._res is not None:
                        # the probe was trained on reply tokens: the running mean starts at the first generated token
                        # (step 0's state is the last prompt token, a different distribution; a pre-reply probe is separate work)
                        if step > 0:
                            run_sum = self._res.copy() if run_sum is None else run_sum + self._res; run_n += 1
                        r = self._readouts(run_sum / run_n) if run_n else {a: 0.0 for a in self.attributes() if a in self.probe.classes} | {"neutral": 1.0}
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
                        finish = "stop"
                        if lead: yield {"delta": lead, "readout": {a: round(v, 4) for a, v in ema.items()}, "expression": self.expressions(ema), "lambda": {}}
                        yield {"done": True, "finish_reason": "stop", "readout": ema, "lambda": dict(lams)}; return
                    toks.append(nxt); n_out += 1; full = self.tok.decode(toks, skip_special_tokens=True)
                    if full.endswith("\ufffd"): delta = ""                    # a multi-byte character split across tokens: wait for the rest
                    else: delta = full[len(text_so_far):]; text_so_far = full
                    if lead: delta, lead = lead + delta, ""                     # the prefill opens the reply
                    for a, t in target.items():                                          # overshoot guard only
                        over = ema.get(a, 0.0) - (t + 0.15)
                        if over > 0 and step > 8: lams[a] = max(0.0, lams[a] - self.gain * over)
                    yield {"delta": delta, "readout": {a: round(v, 4) for a, v in ema.items()}, "expression": self.expressions(ema),
                           "lambda": {a: round(v, 3) for a, v in lams.items() if v}}
                    if scan is not None and delta and scan.feed(delta):
                        finish = "stop"
                        yield {"done": True, "finish_reason": "stop", "readout": ema, "lambda": dict(lams)}; return
                    H.V = self._V(lams)
                    out = self.model(torch.tensor([[nxt]], device=self.device), past_key_values=past, use_cache=True)
                    past = out.past_key_values; logits = out.logits[0, -1].float()
                finish = "length"
                yield {"done": True, "finish_reason": "length", "readout": ema, "lambda": dict(lams)}
            finally:
                self._prefix_store(ids, start, steered, past)                # the prompt part only (cropped to len(ids))
                with self._job_mutex:                                        # no read is left waiting on a finished generation
                    self._generating = False
                    self._drain_jobs()
                H.mode, H.V = None, None
                dt = time.monotonic() - t_start
                log.info("generate priority=%d prompt=%d reused=%d tokens=%d wait=%.2fs gen=%.2fs (%.1f tok/s) finish=%s", priority,
                         len(ids), reused, n_out, t_start - t_queued, dt, n_out / dt if dt > 0 else 0.0, finish)

    @torch.no_grad()
    def read(self, text, segments=None, user="", shrink=8.0, priority=1):
        """Tone of a finished text, per segment, the way the readout probe was trained: the text is forced in as the
        assistant reply to `user` under the REFERENCE prompt, and the probe reads the mean residual over the reply tokens,
        so the result reflects the words, not the prompt or the JSON that carried them.
        segments: list of [start, end) character spans of `text` (default: the whole text). A segment's mean is shrunk
        toward the whole-text mean with `shrink` pseudo-tokens: short sentences keep the reply's tone instead of the noise
        of a few tokens. Returns {"readout", "expression", "tokens", "segments": [{"start","end","tokens","readout","expression"}]}."""
        if self.probe is None: raise ValueError("read needs a readout probe (start the server with --readout)")
        from .prompts import REFERENCE
        segments = [[0, len(text)]] if segments is None else segments
        for s, e in segments:
            if not (isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(text)): raise ValueError(f"bad segment {[s, e]} for text of length {len(text)}")
        msgs = [{"role": "system", "content": REFERENCE}, {"role": "user", "content": user or ""}]
        try: prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError: prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True); r_ids, offs = enc["input_ids"], enc["offset_mapping"]
        if not r_ids: raise ValueError("text is empty")
        with self._job_mutex:
            interleave = self._generating
            if interleave:
                done = concurrent.futures.Future()
                def job():
                    try: done.set_result(self._forward_reply(p_ids, r_ids))
                    except Exception as e: done.set_exception(e)
                self._jobs.put(job)
        if interleave: h = done.result(timeout=60)                           # served between the generation's tokens
        else:
            with self.lock.hold(priority):                                   # a read is on the speaking path: default high
                h = self._forward_reply(p_ids, r_ids)
        whole = h.mean(0)
        def out(vec):
            p = self.probe.probs(vec); r = {a: round(p[a], 4) for a in self.attributes() if a in p} | {"neutral": round(p.get("reference", 0.0), 4)}
            return r, self.expressions(r)
        r_all, e_all = out(whole); segs = []
        for s, e in segments:
            idx = [i for i, (a, b) in enumerate(offs) if b > s and a < e]          # tokens overlapping the span
            n = len(idx); vec = (h[idx].sum(0) + shrink * whole) / (n + shrink) if n else whole
            r, ex = out(vec); segs.append({"start": s, "end": e, "tokens": n, "readout": r, "expression": ex})
        return {"readout": r_all, "expression": e_all, "tokens": len(r_ids), "segments": segs}

    @torch.no_grad()
    def _forward_reply(self, p_ids, r_ids):
        """Residuals of the reply tokens at the probe layer, unsteered (caller holds the model)."""
        cap = {}
        def grab(mod, a, out): cap["h"] = (out[0] if isinstance(out, tuple) else out)[0, len(p_ids):].float().cpu().numpy()
        hd = self.model.model.layers[self.probe.layer].register_forward_hook(grab)
        mode, res = self.H.mode, self._res                                    # an interleaved read must leave no trace
        self.H.mode = None
        try: self.model(torch.tensor([p_ids + r_ids], device=self.device), use_cache=False)
        finally: self.H.mode, self._res = mode, res; hd.remove()
        return cap["h"]

    def _drain_jobs(self):
        while True:
            try: job = self._jobs.get_nowait()
            except queue.Empty: return
            job()

    MIN_REUSE = 32

    def _prefix_lookup(self, ids, start, steered):
        """(tokens reused, DynamicCache) from the cached prompt sharing the longest head with `ids`.
        Steered positions (from a steering start on) depend on that request's strengths, so when either
        side steers, only positions before both steering starts are reused."""
        from transformers import DynamicCache
        best, best_key = 0, None
        for key, (c_ids, c_start, c_steered, _) in self._prefixes.items():
            n, limit = 0, min(len(c_ids), len(ids) - 1)                      # at least one token left to run
            while n < limit and c_ids[n] == ids[n]: n += 1
            if steered or c_steered: n = min(n, start, c_start)
            if n > best: best, best_key = n, key
        if best < self.MIN_REUSE: return 0, None
        self._prefixes.move_to_end(best_key)
        legacy = self._prefixes[best_key][3]
        # slices are views: DynamicCache.update concatenates into new tensors, so the stored prefix stays intact
        return best, DynamicCache.from_legacy_cache(tuple((k[..., :best, :], v[..., :best, :]) for k, v in legacy))

    def _prefix_store(self, ids, start, steered, past):
        if not self.prefix_cache or past is None: return
        try: legacy = past.to_legacy_cache() if hasattr(past, "to_legacy_cache") else past
        except Exception: return
        n = len(ids)
        self._prefixes[id(legacy)] = (list(ids), start, steered, tuple((k[..., :n, :], v[..., :n, :]) for k, v in legacy))
        while len(self._prefixes) > self.prefix_cache: self._prefixes.popitem(last=False)

    def generate(self, messages, **kw):
        text, last = "", {}
        for ev in self.stream(messages, **kw):
            if ev.get("done"): last = ev; break
            text += ev["delta"]
        return {"text": text, "finish_reason": last.get("finish_reason"), "readout": last.get("readout"), "lambda": last.get("lambda")}
