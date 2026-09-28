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
ap.add_argument("--ema", type=float, default=0.3); a = ap.parse_args()
pz = np.load(a.probe, allow_pickle=True); L = int(pz["layer"]); CLS = [str(c) for c in pz["classes"]]
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
MU = torch.tensor(pz["mean"], device="cuda", dtype=torch.float32); SD = torch.tensor(pz["scale"], device="cuda", dtype=torch.float32)
W = torch.tensor(pz["coef"], device="cuda", dtype=torch.float32); B = torch.tensor(pz["intercept"], device="cuda", dtype=torch.float32)
grab = {}
model.model.layers[L].register_forward_hook(lambda m, i, o: grab.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0, -1].float()))
VRM = {"joy": "happy", "sadness": "sad", "anger": "angry", "neutral": "neutral"}
app = FastAPI(); LOCK = asyncio.Lock()


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
    return {"layer": L, "classes": CLS, "model": os.path.basename(a.model.rstrip("/"))}


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
    await sock.accept()
    try:
        while True:
            req = json.loads(await sock.receive_text())
            msgs = ([{"role": "system", "content": req["system"]}] if req.get("system") else []) + req["messages"]
            ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
            temp = float(req.get("temperature", 0.7)); n_max = int(req.get("max_new_tokens", 200))
            async with LOCK:
                t0 = time.time(); gen_ids = []; text = ""; ema = None; t_first = None
                with torch.no_grad():
                    out = model(torch.tensor([ids], device="cuda"), use_cache=True); past = out.past_key_values; logits = out.logits[0, -1]
                    p = probs(grab["h"]); ema = {c: float(v) for c, v in pack(p).items()}
                    # 'pre' is read at the chat-template header token; the probe was trained on content tokens only -> UNCALIBRATED.
                    # Do not drive the face from it; ema starts neutral.
                    await sock.send_text(json.dumps({"type": "pre", "calibrated": False, "probs": pack(p)}, ensure_ascii=False))
                    ema = {c: (1.0 if c == "neutral" else 0.0) for c in CLS}
                    for step in range(n_max):
                        if temp > 0:
                            pr = torch.softmax(logits.float() / temp, -1); topv, topi = pr.topk(20); nxt = int(topi[torch.multinomial(topv / topv.sum(), 1)])
                        else:
                            nxt = int(logits.argmax())
                        if nxt in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")): break
                        gen_ids.append(nxt); new_text = tok.decode(gen_ids, skip_special_tokens=True)
                        if new_text.endswith("\ufffd"): delta = ""  # incomplete multi-byte char (e.g. emoji split across tokens): hold back
                        else: delta = new_text[len(text):]; text = new_text
                        out = model(torch.tensor([[nxt]], device="cuda"), past_key_values=past, use_cache=True); past = out.past_key_values; logits = out.logits[0, -1]
                        p = pack(probs(grab["h"]))  # state after reading this token = what the model 'feels' going into the next one
                        ema = {c: a.ema * p[c] + (1 - a.ema) * ema[c] for c in CLS}
                        if t_first is None: t_first = time.time() - t0
                        await sock.send_text(json.dumps({"type": "token", "text": delta, "probs": p, "ema": {c: round(v, 4) for c, v in ema.items()},
                                                         "vrm": vrm_weights(ema)}, ensure_ascii=False))
                        await asyncio.sleep(0)
                dt = time.time() - t0
                await sock.send_text(json.dumps({"type": "done", "stats": {"tokens": len(gen_ids), "seconds": round(dt, 2),
                                                                              "tok_per_s": round(len(gen_ids) / dt, 1) if dt else None,
                                                                              "first_token_s": round(t_first or 0, 3)}}))
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=a.port)
