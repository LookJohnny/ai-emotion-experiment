"""Nous Lens engine: read an AI reply with a local model (reader proxy) and report what the reply does to the user.

One forward pass over [prompt + reply] gives every reply token's state, so the reader never generates anything.
Readouts (frozen probes from the Qwen3-4B pack; numbers are the pre-registered sealed-test results):
  hostility   C1 token probe, mean over the first 120 reply tokens   -- reader on Kimi K3: AUROC 0.973 (validated)
  distress    C2 probe on the user's last message (no system prompt) -- keyword-free transfer AUROC 0.876 (validated)
  fabrication C3-H answer probe on short factual answers             -- reader on Kimi K3: AUROC 0.737 (EXPERIMENTAL)
Per-sentence hostility = mean token probability inside each sentence (which sentence carries it)."""
import json, os, re, time
import numpy as np, torch
from .loader import load_layers

READER_SYSTEM = "你是陪伴助手，请礼貌、专业地回答用户。"          # framing the C1 probe was validated under
QA_SYSTEM = "你是知识问答助手，请直接、简洁地回答问题。"         # framing the C3-H probe was trained under
SYCO_SYSTEM = "你是一个乐于助人的AI助手。"                    # framing the sycophancy probe was trained under
CLAIM = re.compile(r"我记得|我觉得|我认为|我查到|我看到|不对吧|对吗|对吧|是不是|没错吧|你确定|你再确认|你搞错|应该是")
FACT_Q = re.compile(r"谁|哪[个位里年一些国]|什么|多少|第几|几[年月日号个位代次类种人本条]|何时|何年|是否|^(what|who|when|where|which|how many|how much)\b", re.I)


def looks_factual(u):
    """fabrication is only read for short questions (the probe was validated on short factual QA)."""
    u = u.strip()
    return 0 < len(u) <= 120 and (u.endswith(("？", "?")) or (bool(FACT_Q.search(u)) and not u.endswith(("。", "！", "!", "~", "…"))))


SENT = re.compile(r"[^。！？!?；;\n]+[。！？!?；;\n]*[”’\"」』）)】]*")


def prompt_ids(tok, msgs):
    try: text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"]


def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


class Probe:
    def __init__(self, path, device):
        d = json.load(open(path)); self.layer = d["layer"]
        T = lambda v: torch.tensor(v, dtype=torch.float32, device=device)
        self.mu, self.sd, self.w, self.b = T(d["mean"]), T(d["scale"]), T(d["coef"]), float(d["intercept"])

    def __call__(self, h):                                   # h [n, d] or [d] float32
        return torch.sigmoid(((h - self.mu) / self.sd) @ self.w + self.b)


class Lens:
    def __init__(self, model_dir, pack_dir, device=None):
        self.device = device or pick_device(); dt = torch.float32 if self.device == "cpu" else torch.bfloat16
        t0 = time.time()
        m = json.load(open(os.path.join(pack_dir, "manifest.json"), encoding="utf-8"))
        P = lambda k: Probe(os.path.join(pack_dir, m["probes"][k]), self.device) if k in m["probes"] else None
        self.p_host, self.p_dist, self.p_fab, self.p_syco = P("c1_token"), P("c2"), P("c3h_token"), P("syco_token")   # auto-adapted packs carry a subset
        self.auto_adapted = bool(m.get("auto_adapted")); self.pack_name = m.get("name")
        probes = [p for p in (self.p_host, self.p_dist, self.p_fab, self.p_syco) if p]
        # the reader never generates: load only the layers the probes read (no upper layers, no LM head)
        keep = max(p.layer for p in probes) + 1
        self.tok, self.model = load_layers(model_dir, keep, self.device, dt)   # HF folder or GGUF file (e.g. an Ollama blob)
        self.thr, self.negq = m["thresholds"], m.get("neg_quantiles", {})
        self.states = {}
        for L in {p.layer for p in probes}:
            self.model.layers[L].register_forward_hook(lambda mod, a, out, L=L: self.states.__setitem__(L, (out[0] if isinstance(out, tuple) else out).float()))
        self.load_s = round(time.time() - t0, 1)
        self.model_name = (m.get("base_model") or model_dir).split("/")[-1]

    def pct(self, key, v):
        q = self.negq.get(key)
        if not q: return None
        return round(100 * max(0, sum(1 for x in q if x <= v) - 1) / (len(q) - 1))

    def _prompt_ids(self, msgs):
        return prompt_ids(self.tok, msgs)

    BUCKETS = (128, 256, 384, 512, 768, 1024, 1536, 2048)

    @torch.no_grad()
    def _run(self, ids, batch=None):
        """right-pad to a fixed bucket length: the model is causal, so real positions are unaffected, and a GPU/MPS
        backend reuses its compiled kernels instead of compiling for every new length. states[L] -> [B, T, d]."""
        rows = batch or [ids]; n = max(len(r) for r in rows); n = next((b for b in self.BUCKETS if b >= n), n)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self.model(torch.tensor([r + [pad] * (n - len(r)) for r in rows], device=self.device), use_cache=False)

    def warmup(self, upto=512):
        for b in self.BUCKETS:
            if b > upto: break
            self._run([0] * b)

    @torch.no_grad()
    def read(self, messages, reply, fabrication=True, max_reply=512):
        """messages: prior turns ending with the user's message (roles user/assistant); reply: the AI's answer text."""
        t0 = time.time()
        turns = [{"role": m["role"], "content": str(m["content"])[:4000]} for m in messages if m.get("role") in ("user", "assistant") and m.get("content")]
        turns = turns[-12:]
        if not turns or turns[-1]["role"] != "user": turns.append({"role": "user", "content": "（用户消息未提供）"})
        out = {"model": self.model_name, "device": self.device, "auto_adapted": self.auto_adapted}

        # hostility: [reader system + turns] + reply, one forward
        enc = self.tok(reply, add_special_tokens=False, return_offsets_mapping=True)
        rids, offs = enc["input_ids"][:max_reply], enc["offset_mapping"][:max_reply]
        pids = self._prompt_ids([{"role": "system", "content": READER_SYSTEM}] + turns)
        self._run(pids + rids)
        H = self.states[self.p_host.layer][0, len(pids):len(pids) + len(rids)]
        p = self.p_host(H).cpu().numpy() if len(rids) else np.zeros(0)
        score = float(p[:120].mean()) if len(p) else 0.0
        # which sentence carries it: each sentence read ON ITS OWN after the same prompt (one batched pass), so a calm
        # sentence after an insult is not coloured by the running context. Explanatory only; the validated number is `score`.
        spans = [(m.start(), m.end()) for m in SENT.finditer(reply) if m.group().strip()][:16]
        sents = []
        if len(spans) > 1:
            sids = [self.tok(reply[a:b].strip(), add_special_tokens=False)["input_ids"][:96] for a, b in spans]
            self._run(None, [pids + x for x in sids]); Hs = self.states[self.p_host.layer]
            for i, ((a, b), x) in enumerate(zip(spans, sids)):
                sents.append({"start": a, "end": b, "score": round(float(self.p_host(Hs[i, len(pids):len(pids) + len(x)]).mean()), 3)})
        elif spans: sents = [{"start": spans[0][0], "end": spans[0][1], "score": round(score, 3)}]
        top = max((x["score"] for x in sents), default=0)
        for x in sents: x["hot"] = bool(score >= self.thr["C1_mean"] and x["score"] >= 0.5 and x["score"] >= top - 0.15)
        out["hostility"] = {"score": round(score, 4), "threshold": self.thr["C1_mean"], "flag": score >= self.thr["C1_mean"],
                            "percentile": self.pct("C1_mean", score), "sentences": sents, "tokens_read": len(rids), "validated": True}

        # distress: last user message, no system prompt, mean over its tokens
        user_turns = [t for t in turns if t["role"] == "user"]
        u = user_turns[-1]["content"]
        if self.p_dist and u != "（用户消息未提供）":
            text = self.tok.apply_chat_template(turns, tokenize=False, add_generation_prompt=True)
            e = self.tok(text, return_offsets_mapping=True, add_special_tokens=False); c0 = text.rindex(u); c1 = c0 + len(u)
            idx = [i for i, (s, t) in enumerate(e["offset_mapping"]) if s >= c0 and t <= c1 and t > s]
            self._run(e["input_ids"]); d = float(self.p_dist(self.states[self.p_dist.layer][0, idx[0]:idx[-1] + 1].mean(0)))
            out["distress"] = {"score": round(d, 4), "threshold": self.thr["C2"], "flag": d >= self.thr["C2"], "percentile": self.pct("C2", d), "validated": True}

        # fabrication (experimental): only for a short question with a short answer
        if self.p_fab and (fabrication == "always" or (fabrication and looks_factual(u))):                 # reads the first 64 answer tokens, as validated
            qids = self._prompt_ids([{"role": "system", "content": QA_SYSTEM}, {"role": "user", "content": u}])
            aids = rids[:64]; self._run(qids + aids)
            f = float(self.p_fab(self.states[self.p_fab.layer][0, len(qids):len(qids) + len(aids)].mean(0)))
            out["fabrication"] = {"score": round(f, 4), "threshold": self.thr["C3H_mean"], "flag": f >= self.thr["C3H_mean"],
                                  "percentile": self.pct("C3H_mean", f), "validated": False}
        # sycophancy (experimental): only when the user puts forward a claim or pushes back
        if self.p_syco and u != "（用户消息未提供）" and CLAIM.search(u):
            sp = self._prompt_ids([{"role": "system", "content": SYCO_SYSTEM}] + turns); sr = rids[:120]; self._run(sp + sr)
            y = float(self.p_syco(self.states[self.p_syco.layer][0, len(sp):len(sp) + len(sr)]).mean())
            out["sycophancy"] = {"score": round(y, 4), "threshold": self.thr["SYCO_mean"], "flag": y >= self.thr["SYCO_mean"],
                                 "percentile": self.pct("SYCO_mean", y), "validated": False}
        out["ms"] = round((time.time() - t0) * 1000)
        return out
