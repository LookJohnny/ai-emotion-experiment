"""Conflict-escalation monitor: OpenAI-compatible proxy + live dashboard (plan step 6; V1.1 probes).
POST /v1/chat/completions  {"messages":[...], "stream":true, "max_tokens":200, "temperature":0.7, "x_action":"log"|"stop"}
  stream: OpenAI chat.completion.chunk SSE; every chunk carries an extra "x_risk" field (standard clients ignore it).
    first chunk  x_risk = {"phase":"pre", "C2":p, "C1_pre":p, "alarm":{...}}      <- before any token is generated
    token chunks x_risk = {"phase":"token", "C1_token":p, "C1_mean":m, "alarm":{...}}
  x_action "stop": generation is cut when a C1 alarm fires (finish_reason "content_filter").
GET /  dashboard   GET /info  probe layers / thresholds / hashes   audit log: logs/audit.jsonl
python app.py MODEL --probes ../v1_1 --port 6006
C2 is read from a system-prompt-free prefill (matches how the V1.1 C2 probe was trained)."""
import argparse, asyncio, hashlib, json, os, sys, time, uuid
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from core.probe import ProbeSpec
from core.hooks import ResidualTap

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--probes", default=os.path.join(HERE, "..", "v1_1"))
ap.add_argument("--port", type=int, default=6006); ap.add_argument("--min_tokens", type=int, default=8)
ap.add_argument("--sustain", type=int, default=3, help="confirmed (tier-2) alarm: running mean above threshold for this many consecutive tokens")
ap.add_argument("--c3h", default=None, help="dir with c3h_pre.json / c3h_token.json (hallucination probes)")
ap.add_argument("--layers", default=None, help="layer_probes.npz from train_layer_probes.py -> internal view (layer x token, head attribution)")
ap.add_argument("--heads", default="L16H12,L17H16,L14H25,L19H28,L17H4,L19H29,L18H8,L22H19,L16H3,L14H26,L20H11,L16H17", help="E01 B3 anger heads (FREEZE_5b.json)")
ap.add_argument("--emotion", default=None, help="product dir: results/readout_v2/probe_L19.npz, results/pre_v2/preprobe_L22.npz, avatar/calib.json, smoothing_freeze.json")
a = ap.parse_args()
PF = {"c1_pre": f"{a.probes}/c1_out/c1_pre.json", "c1_token": f"{a.probes}/c1_out/c1_token.json", "c2": f"{a.probes}/c2_out/c2_transfer.json"}
P = {k: ProbeSpec.load(v) for k, v in PF.items()}
TH = json.load(open(f"{a.probes}/demo_thresholds.json"))
THR = {"C1_pre": TH["p_pre"]["thr"], "C1_mean": TH["item_score"]["thr"], "C2": TH["c2_transfer"]["thr"]}
H3 = None
if a.c3h:
    H3 = {k: ProbeSpec.load(f"{a.c3h}/{k}.json") for k in ["c3h_pre", "c3h_token"]}
    THR.update(C3H_pre=TH["c3h_pre"]["thr"], C3H_mean=TH["c3h_token"]["thr"])
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
EMO = None
if a.emotion:   # E01 emotion readout (expression driver): 4-class token probe + pre-reply ridge probe + frozen smoothing
    import numpy as np
    sys.path.insert(0, a.emotion); from smoothing import Smoother
    z = np.load(f"{a.emotion}/results/readout_v2/probe_L19.npz", allow_pickle=True); q = np.load(f"{a.emotion}/results/pre_v2/preprobe_L22.npz", allow_pickle=True)
    T = lambda x: torch.tensor(x, device="cuda", dtype=torch.float32)
    EMO = dict(L=int(z["layer"]), cls=[str(c) for c in z["classes"]], mu=T(z["mean"]), sd=T(z["scale"]), W=T(z["coef"]), B=T(z["intercept"]),
               LP=int(q["layer"]), pcls=[str(c) for c in q["classes"]], pmu=T(q["mean"]), psd=T(q["scale"]), pW=T(q["coef"]), pB=T(q["intercept"]),
               cal=json.load(open(f"{a.emotion}/avatar/calib.json"))["emotions"], smo=json.load(open(f"{a.emotion}/smoothing_freeze.json"))["frozen"], Smoother=Smoother)
    P_LAYERS = [EMO["L"], EMO["LP"]]
else: P_LAYERS = []
INNER = None
if a.layers:
    import numpy as np
    cfg = model.config; z = np.load(a.layers); NL = cfg.num_hidden_layers; NH = cfg.num_attention_heads
    HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH; NKV = getattr(cfg, "num_key_value_heads", NH)
    Tt = lambda x: torch.tensor(x, device="cuda", dtype=torch.float32)
    mu, sd, co, b0 = Tt(z["mean"]), Tt(z["scale"]), Tt(z["coef"]), Tt(z["intercept"])       # [NL, d], [NL]
    HEADS = [(int(h[1:].split("H")[0]), int(h.split("H")[1])) for h in a.heads.split(",")]
    with torch.no_grad():
        dirs = co / sd                                                                          # residual direction the layer-L probe reads
        # component writes into the residual: head h -> W_O[:, h] @ x_h ; neuron n -> W_down[:, n] * act_n
        # push on the layer-L probe logit = x_h . UH[L, h]  and  act . VN[L]
        UH = torch.stack([(model.model.layers[L].self_attn.o_proj.weight.float().T @ dirs[L]).view(NH, HD) for L in range(NL)])
        VN = torch.stack([model.model.layers[L].mlp.down_proj.weight.float().T @ dirs[L] for L in range(NL)])
    OIN, MIN = {}, {}
    for L in range(NL):
        model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(lambda m, args, L=L: OIN.__setitem__(L, args[0][0, -1].detach().float()))
        model.model.layers[L].mlp.down_proj.register_forward_pre_hook(lambda m, args, L=L: MIN.__setitem__(L, args[0][0, -1].detach().float()))
    WO = [model.model.layers[L].self_attn.o_proj.weight for L in range(NL)]                     # real parameters, [d, NH*HD] each
    INNER = dict(mu=mu, sd=sd, co=co, b0=b0, UH=UH, VN=VN, OIN=OIN, MIN=MIN, NL=NL, NH=NH, HD=HD, NN=VN.shape[1], HEADS=HEADS, WO=WO, NKV=NKV,
                 cfg=dict(model_type=cfg.model_type, n_layers=NL, n_heads=NH, n_kv_heads=NKV, head_dim=HD, hidden=cfg.hidden_size,
                          intermediate=VN.shape[1], params_b=round(sum(p.numel() for p in model.parameters()) / 1e9, 2)),
                 heads=[f"L{L}H{h}" for L, h in HEADS], val_auroc=[round(float(v), 3) for v in z["val_auroc"]])
EDGES = []
if INNER:   # V-composition between heads, from weights only: ||W_V(L2, kv g) @ W_O(L1, head h)||_F / (||W_V g|| ||W_O h||), next 3 layers, top 3 per source head
    with torch.no_grad():
        NL_, NH_, HD_, NKV_ = INNER["NL"], INNER["NH"], INNER["HD"], INNER["NKV"]
        WOn = [w.float().view(-1, NH_, HD_) for w in INNER["WO"]]                                    # [d, NH, HD]
        VV = [model.model.layers[L].self_attn.v_proj.weight.float().view(NKV_, HD_, -1) for L in range(NL_)]   # [NKV, HD, d]
        on = [w.norm(dim=(0, 2)) for w in WOn]; vn = [v.norm(dim=(1, 2)) for v in VV]
        gen = torch.Generator(device="cuda").manual_seed(0)
        Rb = torch.randn(WOn[0].shape, device="cuda", generator=gen)                                    # random W_O with the same shape
        rbn = Rb.norm(dim=(0, 2))
        base = [(torch.einsum("gkd,dhe->ghke", VV[L2], Rb).pow(2).sum(dim=(2, 3)).sqrt() / (vn[L2][:, None] * rbn[None, :])).mean() for L2 in range(NL_)]
        for L1 in range(NL_ - 1):
            cand = []
            for L2 in range(L1 + 1, min(NL_, L1 + 4)):
                M = torch.einsum("gkd,dhe->ghke", VV[L2], WOn[L1]).pow(2).sum(dim=(2, 3)).sqrt() / (vn[L2][:, None] * on[L1][None, :]) - base[L2]   # baseline-corrected
                cand.append((L2, M.clamp(min=0)))
            for h in range(NH_):
                sc = [(float(M[g, h]), L2, g) for L2, M in cand for g in range(NKV_)]; sc.sort(reverse=True)
                EDGES += [[L1, h, L2, g, round(w, 4)] for w, L2, g in sc[:3] if w > 0]
        mx = max(e[4] for e in EDGES); EDGES = [e[:4] + [round(e[4] / mx, 4)] for e in EDGES]
        del WOn, Rb; torch.cuda.empty_cache()
        INNER["VV"] = [model.model.layers[L].self_attn.v_proj.weight.view(NKV_, HD_, -1) for L in range(NL_)]   # bf16 views for per-token flow
        INNER["G"] = [model.model.layers[L].input_layernorm.weight.float() for L in range(NL_)]
        INNER["EPS"] = getattr(model.config, "rms_norm_eps", 1e-6); del VV
    print("composition edges", len(EDGES), flush=True)
tap = ResidualTap(model, [s.layer for s in P.values()] + P_LAYERS + (list(range(INNER["NL"])) if INNER else []) + ([s.layer for s in H3.values()] if H3 else []))


def inner_view(topn=6):
    """Real internals at the current position (visualization; not pre-registered).
    activity : residual-stream norm per layer, each head's output norm ||W_O[:,h] x_h||, top neurons by |activation|
    safety   : per-layer C1 probe prob; each head's / top neurons' push on the layer's own hostility probe logit."""
    I = INNER; H = torch.stack([tap.states[L] for L in range(I["NL"])])
    lp = torch.sigmoid((((H - I["mu"]) / I["sd"]) * I["co"]).sum(-1) + I["b0"])
    X = torch.stack([I["OIN"][L].view(I["NH"], I["HD"]) for L in range(I["NL"])])              # [NL, NH, HD] per-head attention outputs (pre W_O)
    hp = (X * I["UH"]).sum(-1)                                                                   # [NL, NH]
    HO = [(I["WO"][L].view(-1, I["NH"], I["HD"]) * X[L].to(I["WO"][L].dtype)).sum(-1) for L in range(I["NL"])]   # [d, NH] actual per-head writes (bf16)
    hn = torch.stack([o.float().norm(dim=0) for o in HO])
    # activation flow (approximation: V path at the current position only; no attention weights, no MLP path):
    # read(L1,h -> L2,g) = || W_V(L2,g) ( gamma_L2 * o_(L1,h) / rms(resid into L2) ) ||
    lpc = lp.clamp(0, 1); rel = (hp.clamp(min=0) / hp.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)) * (.25 + .75 * lpc[:, None])   # source push, per-layer normalised
    rows = []
    for L2 in range(1, I["NL"]):
        r = tap.states[L2 - 1]; sc = (I["G"][L2] / (r.pow(2).mean() + I["EPS"]).sqrt()).to(HO[0].dtype)
        for L1 in range(max(0, L2 - 3), L2):
            R = torch.einsum("gkd,dh->gkh", I["VV"][L2], HO[L1] * sc[:, None]).float().norm(dim=1)   # [NKV, NH]
            rows.append((L1, L2, R))
    allR = torch.stack([R for _, _, R in rows]); keyL = torch.tensor([[L1, L2] for L1, L2, _ in rows], device=allR.device)
    safe = allR * rel[keyL[:, 0]][:, None, :]                                                           # [P, NKV, NH]
    flat = safe.flatten(); tot = float(flat.sum()) + 1e-9; v, ix = flat.topk(150)
    P_, G_, H_ = ix // (I["NKV"] * I["NH"]), (ix // I["NH"]) % I["NKV"], ix % I["NH"]
    rflat = allR.flatten(); rtot = float(rflat.sum()) + 1e-9; rv, rix = rflat.topk(150)
    rP, rG, rH = rix // (I["NKV"] * I["NH"]), (rix // I["NH"]) % I["NKV"], rix % I["NH"]
    kl = keyL.cpu().tolist()
    flow = {"safety": [[kl[p][0], h, kl[p][1], g, round(float(x), 4)] for p, g, h, x in zip(P_.tolist(), G_.tolist(), H_.tolist(), v.tolist())],
            "activity": [[kl[p][0], h, kl[p][1], g, round(float(x), 3)] for p, g, h, x in zip(rP.tolist(), rG.tolist(), rH.tolist(), rv.tolist())],
            "coverage": {"safety": round(float(v.sum()) / tot, 3), "activity": round(float(rv.sum()) / rtot, 3)}}
    A = torch.stack([I["MIN"][L] for L in range(I["NL"])])                                       # [NL, NN] real neuron activations
    npush = A * I["VN"]
    _, ix = npush.abs().topk(topn, dim=1); nv = npush.gather(1, ix)
    _, ax = A.abs().topk(topn, dim=1); av = A.gather(1, ax)
    out = [t.cpu().numpy() for t in (lp, hp, ix, nv, hn, ax, av, H.norm(dim=-1))]
    lp, hp, ix, nv, hn, ax, av, rn = out
    return {"layers": np.round(lp, 3).tolist(), "heads_all": np.round(hp, 2).tolist(), "heads_norm": np.round(hn, 2).tolist(),
            "resid_norm": np.round(rn, 1).tolist(), "heads": [round(float(hp[L, h]), 3) for L, h in I["HEADS"]],
            "neurons": [[[int(i), round(float(x), 3)] for i, x in zip(ix[L], nv[L])] for L in range(I["NL"])],
            "neurons_act": [[[int(i), round(float(x), 3)] for i, x in zip(ax[L], av[L])] for L in range(I["NL"])], "flow": flow}


def emo_token(h):
    p = torch.softmax(((h - EMO["mu"]) / EMO["sd"]) @ EMO["W"].T + EMO["B"], -1); return {c: float(p[i]) for i, c in enumerate(EMO["cls"])}


def emo_pre(h):
    y = torch.clamp(((h - EMO["pmu"]) / EMO["psd"]) @ EMO["pW"].T + EMO["pB"], min=0); y = y / (y.sum() + 1e-6)
    return {c: round(float(y[EMO["pcls"].index(c)]), 4) for c in EMO["cls"]}
EOS = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
os.makedirs(os.path.join(HERE, "logs"), exist_ok=True); AUDIT = os.path.join(HERE, "logs", "audit.jsonl")
app = FastAPI(); LOCK = asyncio.Lock()


def prefill(msgs):
    ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
    return ids, model(torch.tensor([ids], device="cuda"), use_cache=True)


def c2_score(msgs):
    """mean layer state over the last user message, from a prefill WITHOUT the system prompt."""
    m = [x for x in msgs if x["role"] != "system"]
    text = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    e = tok(text, return_offsets_mapping=True, add_special_tokens=False); u = m[-1]["content"]; c0 = text.rindex(u); c1 = c0 + len(u)
    idx = [i for i, (s, t) in enumerate(e["offset_mapping"]) if s >= c0 and t <= c1 and t > s]
    model(torch.tensor([e["input_ids"]], device="cuda"), use_cache=False)
    return float(P["c2"].prob(tap.full[P["c2"].layer][idx[0]:idx[-1] + 1].mean(0))), text, e["input_ids"]


def chunk(cid, delta, risk, finish=None):
    return "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": "qwen3-4b-monitored",
                                  "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], "x_risk": risk}, ensure_ascii=False) + "\n\n"


async def run(req):
    msgs = req["messages"]; n_max = int(req.get("max_tokens", 200)); temp = float(req.get("temperature", 0.7)); action = req.get("x_action", "log")
    cid = "chatcmpl-" + uuid.uuid4().hex[:12]; T0 = time.time(); ms = lambda: round((time.time() - T0) * 1000, 1)
    rec = {"id": cid, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "action": action, "user": msgs[-1]["content"][:200], "events": []}
    async with LOCK:
        with torch.no_grad():
            c2, _, _ = c2_score(msgs)
            ids, out = prefill(msgs); past = out.past_key_values; logits = out.logits[0, -1]
            c1pre = float(P["c1_pre"].prob(tap.states[P["c1_pre"].layer]))
            pre = {"phase": "pre", "t": ms(), "C2": round(c2, 4), "C1_pre": round(c1pre, 4),
                   "alarm": {"C2": c2 >= THR["C2"], "C1_pre": c1pre >= THR["C1_pre"]}}
            if EMO:
                pp = emo_pre(tap.states[EMO["LP"]]); sm = EMO["Smoother"](EMO["cal"], EMO["smo"]["alpha"], EMO["smo"]["sustain"], EMO["smo"]["gamma"], pre=pp)
                pre["emo"] = {"probs": pp, "expr": sm.pre_expr()}
            if INNER: pre["inner"] = inner_view()
            if H3:
                pre["C3H_pre"] = round(float(H3["c3h_pre"].prob(tap.states[H3["c3h_pre"].layer])), 4); pre["alarm"]["C3H_pre"] = pre["C3H_pre"] >= THR["C3H_pre"]
                hsum = torch.zeros_like(tap.states[H3["c3h_token"].layer])
            rec["pre"] = pre; yield chunk(cid, {"role": "assistant", "content": ""}, pre)
            gen, text, ps, finish = [], "", [], "stop"; streak = 0; hstreak = 0
            for step in range(n_max):
                if temp > 0:
                    pr = torch.softmax(logits.float() / temp, -1); tv, ti = pr.topk(20); nxt = int(ti[torch.multinomial(tv / tv.sum(), 1)])
                else: nxt = int(logits.argmax())
                if nxt in EOS: break
                gen.append(nxt); new = tok.decode(gen, skip_special_tokens=True)
                delta = "" if new.endswith("�") else new[len(text):]; text = new if delta else text
                out = model(torch.tensor([[nxt]], device="cuda"), past_key_values=past, use_cache=True); past = out.past_key_values; logits = out.logits[0, -1]
                p = float(P["c1_token"].prob(tap.states[P["c1_token"].layer])); ps.append(p); mean = sum(ps) / len(ps)
                over = len(ps) >= a.min_tokens and mean >= THR["C1_mean"]; streak = streak + 1 if over else 0
                al = streak >= a.sustain                                             # tier 2: confirmed (sustained) -- the only level that may stop generation
                risk = {"phase": "token", "t": ms(), "i": len(ps), "C1_token": round(p, 4), "C1_mean": round(mean, 4), "alarm": {"C1": al, "C1_watch": over and not al}}
                if EMO: risk["emo"] = {"expr": sm.step(emo_token(tap.states[EMO["L"]]))}
                if INNER: risk["inner"] = inner_view()
                if H3:
                    hsum += tap.states[H3["c3h_token"].layer]; hm = float(H3["c3h_token"].prob(hsum / len(ps)))
                    hov = len(ps) >= a.min_tokens and hm >= THR["C3H_mean"]; hstreak = hstreak + 1 if hov else 0
                    risk["C3H_mean"] = round(hm, 4); risk["alarm"]["C3H"] = hstreak >= a.sustain; risk["alarm"]["C3H_watch"] = hov and hstreak < a.sustain
                if al and not any(e["type"] == "C1_alarm" for e in rec["events"]): rec["events"].append({"type": "C1_alarm", "tier": "confirmed", "token": len(ps), "t": ms()})
                yield chunk(cid, {"content": delta}, risk); await asyncio.sleep(0)
                if al and action == "stop": finish = "content_filter"; break
            else:
                if len(gen) >= n_max: finish = "length"
    rec.update(reply=text, tokens=len(ps), C1_mean=round(sum(ps) / len(ps), 4) if ps else None, finish=finish, ms=ms())
    with open(AUDIT, "a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    yield chunk(cid, {}, {"phase": "done", "t": ms(), "tokens": len(ps), "C1_mean": rec["C1_mean"], "finish": finish}, finish)
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def completions(req: dict):
    if req.get("stream"): return StreamingResponse(run(req), media_type="text/event-stream")
    text, risk, fin = "", {}, "stop"
    async for s in run(req):
        if s.startswith("data: {"):
            d = json.loads(s[6:]); text += d["choices"][0]["delta"].get("content", ""); fin = d["choices"][0]["finish_reason"] or fin
            if d["x_risk"]["phase"] in ("pre", "done"): risk[d["x_risk"]["phase"]] = d["x_risk"]
    return JSONResponse({"id": "chatcmpl", "object": "chat.completion", "model": "qwen3-4b-monitored",
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": fin}], "x_risk": risk})


@app.get("/info")
def info():
    return {"model": os.path.basename(a.model.rstrip("/\\")), "thresholds": THR, "sustain": a.sustain, "min_tokens": a.min_tokens, "neg_quantiles": TH.get("neg_quantiles", {}), "emotion": bool(EMO), "inner": ({"heads": INNER["heads"], "layer_val_auroc": INNER["val_auroc"], "n_layers": INNER["NL"], "n_heads": INNER["NH"], "n_neurons": INNER["NN"], "config": INNER["cfg"]} if INNER else None), "threshold_note": TH["note"],
            "probes": {k: {"layer": v.layer, "sha256": hashlib.sha256(open(PF[k], "rb").read()).hexdigest()[:12]} for k, v in P.items()}}


@app.get("/audit")
def audit(n: int = 20):
    if not os.path.exists(AUDIT): return []
    return [json.loads(l) for l in open(AUDIT, encoding="utf-8").readlines()[-n:]][::-1]


@app.get("/circuit")
def circuit():
    return {"edges": EDGES, "rule": "potential wiring from weights: ||W_V(L2,g) W_O(L1,h)||_F / (||W_V|| ||W_O||) minus a random-W_O baseline, clipped at 0; next 3 layers; top-3 per source head; scaled to max 1"}


@app.get("/presets")
def presets():
    return json.load(open(os.path.join(HERE, "presets.json"), encoding="utf-8"))


@app.get("/monitor")
def monitor_page():
    return HTMLResponse(open(os.path.join(HERE, "monitor.html"), encoding="utf-8").read())


@app.get("/brain")
def brain_page():
    return HTMLResponse(open(os.path.join(HERE, "brain.html"), encoding="utf-8").read())


@app.get("/avatar")
def avatar_page():
    return HTMLResponse(open(os.path.join(HERE, "avatar.html"), encoding="utf-8").read())


if a.emotion:
    from fastapi.staticfiles import StaticFiles
    app.mount("/vrm", StaticFiles(directory=os.path.join(a.emotion, "avatar")), name="vrm")


@app.get("/")
def index():
    return HTMLResponse(open(os.path.join(HERE, "dashboard.html"), encoding="utf-8").read())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=a.port)
