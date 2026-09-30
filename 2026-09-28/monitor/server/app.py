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
a = ap.parse_args()
PF = {"c1_pre": f"{a.probes}/c1_out/c1_pre.json", "c1_token": f"{a.probes}/c1_out/c1_token.json", "c2": f"{a.probes}/c2_out/c2_transfer.json"}
P = {k: ProbeSpec.load(v) for k, v in PF.items()}
TH = json.load(open(f"{a.probes}/demo_thresholds.json"))
THR = {"C1_pre": TH["p_pre"]["thr"], "C1_mean": TH["item_score"]["thr"], "C2": TH["c2_transfer"]["thr"]}
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
tap = ResidualTap(model, [s.layer for s in P.values()])
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
            rec["pre"] = pre; yield chunk(cid, {"role": "assistant", "content": ""}, pre)
            gen, text, ps, finish = [], "", [], "stop"
            for step in range(n_max):
                if temp > 0:
                    pr = torch.softmax(logits.float() / temp, -1); tv, ti = pr.topk(20); nxt = int(ti[torch.multinomial(tv / tv.sum(), 1)])
                else: nxt = int(logits.argmax())
                if nxt in EOS: break
                gen.append(nxt); new = tok.decode(gen, skip_special_tokens=True)
                delta = "" if new.endswith("�") else new[len(text):]; text = new if delta else text
                out = model(torch.tensor([[nxt]], device="cuda"), past_key_values=past, use_cache=True); past = out.past_key_values; logits = out.logits[0, -1]
                p = float(P["c1_token"].prob(tap.states[P["c1_token"].layer])); ps.append(p); mean = sum(ps) / len(ps)
                al = len(ps) >= a.min_tokens and mean >= THR["C1_mean"]
                risk = {"phase": "token", "t": ms(), "i": len(ps), "C1_token": round(p, 4), "C1_mean": round(mean, 4), "alarm": {"C1": al}}
                if al and not any(e["type"] == "C1_alarm" for e in rec["events"]): rec["events"].append({"type": "C1_alarm", "token": len(ps), "t": ms()})
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
    return {"model": os.path.basename(a.model.rstrip("/\\")), "thresholds": THR, "threshold_note": TH["note"],
            "probes": {k: {"layer": v.layer, "sha256": hashlib.sha256(open(PF[k], "rb").read()).hexdigest()[:12]} for k, v in P.items()}}


@app.get("/audit")
def audit(n: int = 20):
    if not os.path.exists(AUDIT): return []
    return [json.loads(l) for l in open(AUDIT, encoding="utf-8").readlines()[-n:]][::-1]


@app.get("/presets")
def presets():
    return json.load(open(os.path.join(HERE, "presets.json"), encoding="utf-8"))


@app.get("/")
def index():
    return HTMLResponse(open(os.path.join(HERE, "dashboard.html"), encoding="utf-8").read())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=a.port)
