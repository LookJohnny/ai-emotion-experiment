"""Real-time emotion readout server (prototype).

Per generated token: probe(residual state at layer L) -> P(neutral, joy, sadness, anger).
Also a 'pre' readout at the last prompt token, before any text exists (lead-time signal).
WebSocket /ws  in : {"messages":[{"role":"user","content":"..."}], "system": "...", "max_new_tokens":200, "temperature":0.7}
               out: {"type":"pre", ...} then {"type":"token","text":"...","probs":{...},"ema":{...},"vrm":{...}} ... {"type":"done","stats":{...}}
POST /readout  {"text": "..."} -> per-token probs for arbitrary assistant text (no generation)
GET  /          demo page (demo.html)
python readout_server.py MODEL probe_L*.npz --port 6006
"""
import json, time, asyncio, argparse, os
import numpy as np, torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("probe"); ap.add_argument("--port", type=int, default=6006)
ap.add_argument("--ema", type=float, default=0.3)
ap.add_argument("--preprobe", default=None, help="preprobe_L*.npz: predicts the upcoming reply emotion from the last prompt position")
ap.add_argument("--calib", default=None, help="avatar/calib.json: per-emotion floor/ceil for expression intensity")
ap.add_argument("--smoothing", default=None, help="smoothing_freeze.json: frozen alpha/sustain/gamma (tuned on validation prompts)")
a = ap.parse_args()
pz = np.load(a.probe, allow_pickle=True); L = int(pz["layer"]); CLS = [str(c) for c in pz["classes"]]
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
MU = torch.tensor(pz["mean"], device="cuda", dtype=torch.float32); SD = torch.tensor(pz["scale"], device="cuda", dtype=torch.float32)
W = torch.tensor(pz["coef"], device="cuda", dtype=torch.float32); B = torch.tensor(pz["intercept"], device="cuda", dtype=torch.float32)
grab = {}
model.model.layers[L].register_forward_hook(lambda m, i, o: grab.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0, -1].float()))
PRE = None
if a.preprobe:
    q = np.load(a.preprobe, allow_pickle=True); LP = int(q["layer"])
    PRE = {k: torch.tensor(q[k], device="cuda", dtype=torch.float32) for k in ["mean", "scale", "coef", "intercept"]}
    PRE_CLS = [str(c) for c in q["classes"]]
    model.model.layers[LP].register_forward_hook(lambda m, i, o: grab.__setitem__("hp", (o[0] if isinstance(o, tuple) else o)[0, -1].float()))


def pre_probs():
    # ridge regression -> predicted mean reply readout; clip + renormalise into a distribution
    y = ((grab["hp"] - PRE["mean"]) / PRE["scale"]) @ PRE["coef"].T + PRE["intercept"]
    y = torch.clamp(y, min=0); y = y / (y.sum() + 1e-6)
    return {c: round(float(y[PRE_CLS.index(c)]), 4) for c in CLS}


VRM = {"joy": "happy", "sadness": "sad", "anger": "angry", "neutral": "neutral"}
CAL = json.load(open(a.calib))["emotions"] if a.calib else None
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smoothing import Smoother
SMO = json.load(open(a.smoothing))["frozen"] if a.smoothing else {"alpha": 0.3, "sustain": 1, "gamma": 1.0}


def expr_of(p):
    """calibrated expression intensities from class probabilities: (p - floor) / (ceil - floor), clipped.
    floor = 95th pct on neutral-task replies (no false alarms), ceil = 75th pct on the emotion's own category."""
    if CAL is None: return vrm_weights(p)
    w = {VRM[e]: max(0.0, min(1.0, (p[e] - CAL[e]["floor"]) / (CAL[e]["ceil"] - CAL[e]["floor"]))) for e in ["joy", "sadness", "anger"]}
    w["neutral"] = 1 - max(w.values())
    return {k: round(v, 3) for k, v in w.items()}
app = FastAPI(); LOCK = asyncio.Lock()
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def probs(h):
    return torch.softmax(((h - MU) / SD) @ W.T + B, -1)


def pack(p):
    return {c: round(float(p[i]), 4) for i, c in enumerate(CLS)}


def vrm_weights(ema):
    # emotional classes drive their VRM preset relative to the neutral-dominant resting state; neutral = 1 - max emotion
    w = {VRM[c]: max(0.0, min(1.0, (ema[c] - 0.15) / 0.6)) for c in ["joy", "sadness", "anger"]}
    w["neutral"] = round(1 - max(w.values()), 3)
    return {k: round(v, 3) for k, v in w.items()}


@app.get("/")
def index():
    return HTMLResponse(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo.html"), encoding="utf-8").read())


@app.get("/info")
def info():
    return {"layer": L, "classes": CLS, "model": os.path.basename(a.model.rstrip("/")), "preprobe_layer": (LP if PRE is not None else None), "calibrated": CAL is not None, "smoothing": SMO}


@app.post("/readout")
async def readout(body: dict):
    msgs = body.get("messages") or [{"role": "user", "content": body.get("prompt", "你好")}]
    prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids_p = tok(prefix, add_special_tokens=False)["input_ids"]; ids = tok(prefix + body["text"], add_special_tokens=False)["input_ids"]
    out = []
    async with LOCK:
        with torch.no_grad():
            hs = model(torch.tensor([ids], device="cuda"), output_hidden_states=True, use_cache=False).hidden_states[L + 1][0].float()
        P = probs(hs[len(ids_p) - 1:])
    for j, t in enumerate(ids[len(ids_p) - 1:]):
        out.append({"token": tok.decode([ids[len(ids_p) - 1 + j]]), "probs": pack(P[j])})
    return JSONResponse({"layer": L, "tokens": out})


@app.websocket("/ws")
async def ws(sock: WebSocket):
    """modes: 'pre' = pre-reply readout + per-token correction (the invention);
              'token' = per-token readout only (no pre-readout);
              'sentence' = baseline: expression updated only when a sentence is complete (post-hoc text-level analysis).
    Every frame carries t (ms since the request arrived) so the client can measure expression-onset latency."""
    await sock.accept()
    try:
        while True:
            req = json.loads(await sock.receive_text()); T0 = time.time(); ms = lambda: round((time.time() - T0) * 1000, 1)
            mode = req.get("mode", "pre")
            msgs = ([{"role": "system", "content": req["system"]}] if req.get("system") else []) + req["messages"]
            ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
            temp = float(req.get("temperature", 0.7)); n_max = int(req.get("max_new_tokens", 200)); alpha = float(req.get("ema", a.ema))
            async with LOCK:
                gen_ids = []; text = ""; t_first = None; sent_probs = []; neutral = {c: (1.0 if c == "neutral" else 0.0) for c in CLS}
                with torch.no_grad():
                    out = model(torch.tensor([ids], device="cuda"), use_cache=True); past = out.past_key_values; logits = out.logits[0, -1]
                    use_sm = CAL is not None and mode in ("pre", "token")
                    if mode == "pre" and PRE is not None:
                        pp = pre_probs(); ema = dict(pp)
                        sm = Smoother(CAL, SMO["alpha"], SMO["sustain"], SMO["gamma"], pre=pp) if use_sm else None
                        await sock.send_text(json.dumps({"type": "frame", "phase": "pre", "t": ms(), "probs": pp,
                                                         "expr": sm.pre_expr() if sm else expr_of(ema)}, ensure_ascii=False))
                    else:
                        sm = Smoother(CAL, SMO["alpha"], SMO["sustain"], SMO["gamma"], pre=None) if use_sm else None
                        ema = dict(neutral)
                        await sock.send_text(json.dumps({"type": "frame", "phase": "start", "t": ms(), "probs": neutral, "expr": expr_of(neutral)}, ensure_ascii=False))
                    for step in range(n_max):
                        if temp > 0:
                            pr = torch.softmax(logits.float() / temp, -1); topv, topi = pr.topk(20); nxt = int(topi[torch.multinomial(topv / topv.sum(), 1)])
                        else:
                            nxt = int(logits.argmax())
                        if nxt in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")): break
                        gen_ids.append(nxt); new_text = tok.decode(gen_ids, skip_special_tokens=True)
                        if new_text.endswith("\ufffd"): delta = ""
                        else: delta = new_text[len(text):]; text = new_text
                        out = model(torch.tensor([[nxt]], device="cuda"), past_key_values=past, use_cache=True); past = out.past_key_values; logits = out.logits[0, -1]
                        p = pack(probs(grab["h"]))
                        if t_first is None: t_first = ms()
                        frame = {"type": "frame", "t": ms(), "text": delta, "probs": p}
                        if mode == "sentence":
                            sent_probs.append(p)
                            if delta and delta[-1] in "。！？!?；\n":   # sentence complete -> post-hoc analysis of that sentence
                                sp = {c: sum(x[c] for x in sent_probs) / len(sent_probs) for c in CLS}; ema = sp; sent_probs = []
                                frame.update(phase="sentence", expr=expr_of(ema))
                            else:
                                frame.update(phase="sentence_wait", expr=expr_of(ema))
                        elif sm is not None:
                            frame.update(phase="token", expr=sm.step(p))
                        else:
                            ema = {c: alpha * p[c] + (1 - alpha) * ema[c] for c in CLS}
                            frame.update(phase="token", expr=expr_of(ema))
                        await sock.send_text(json.dumps(frame, ensure_ascii=False)); await asyncio.sleep(0)
                dt = time.time() - T0
                await sock.send_text(json.dumps({"type": "done", "mode": mode, "stats": {"tokens": len(gen_ids), "seconds": round(dt, 2),
                    "tok_per_s": round(len(gen_ids) / dt, 1) if dt else None, "first_token_ms": t_first}}))
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=a.port)
