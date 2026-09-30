"""Monitoring gateway: OpenAI-compatible /v1/chat/completions with an extra `x_risk` field on every chunk.

Request extensions (all optional; standard clients can ignore them):
  "x_nous": {"concepts": ["C1","C2","C3H"], "action": "log"|"stop", "inner": bool, "emotion": bool}
  "x_force": "<text>", "x_force_max": N      teacher-force a fixed reply (replay / audit a stored reply; local mode only)
  "x_action": "log"|"stop"                   legacy alias of x_nous.action
Modes (nous.yaml gateway.mode):
  local  : the self-hosted model generates; full readouts incl. pre-reply warning.
  reader : an upstream OpenAI-compatible API generates; its text is streamed through unchanged while the local model reads it
           (C1 hostility validated on Kimi K3: AUROC 0.97; C3-H hallucination is limited by the reader's own knowledge)."""
import json, time, uuid
import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

HIDE = ("delta", "text")


def risk(ev):
    return {k: v for k, v in ev.items() if k not in HIDE}


def chunk(cid, model, delta, x, finish=None):
    return "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
                                  "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], "x_risk": x}, ensure_ascii=False) + "\n\n"


def build(engine, cfg, audit, auth):
    r = APIRouter(); G = cfg.gateway; MODEL = "nous/" + engine.model_name

    def opts(req, console):
        x = req.get("x_nous") or {}
        concepts = x.get("concepts") or (engine.pack.concepts if console else G.concepts)
        action = x.get("action") or req.get("x_action") or G.action
        return concepts, action, bool(x.get("inner", console and engine.inner is not None)), bool(x.get("emotion", console and engine.emotion is not None))

    async def local(req, client, console):
        concepts, action, inner, emotion = opts(req, console); cid = "chatcmpl-" + uuid.uuid4().hex[:12]
        pre, done, alarms = {}, {}, set()
        async for ev in engine.stream(req["messages"], max_tokens=min(int(req.get("max_tokens") or 200), G.max_tokens),
                                      temperature=float(req.get("temperature", 0.7)), concepts=concepts, emotion=emotion, inner=inner,
                                      force=req.get("x_force"), force_max=req.get("x_force_max"), stop_on=("C1",) if action == "stop" else None):
            alarms |= {k for k, v in ev.get("alarm", {}).items() if v}
            if ev["phase"] == "pre": pre = ev; yield chunk(cid, MODEL, {"role": "assistant", "content": ""}, risk(ev))
            elif ev["phase"] == "token": yield chunk(cid, MODEL, {"content": ev["delta"]}, risk(ev))
            else: done = ev
        yield chunk(cid, MODEL, {}, risk(done), done["finish"]); yield "data: [DONE]\n\n"
        audit.request(cid, client, "local", req["messages"][-1]["content"], done["text"], done["finish"], pre, done, sorted(alarms))

    async def reader(req, client, console):
        concepts, action, _, _ = opts(req, console); cid = "chatcmpl-" + uuid.uuid4().hex[:12]
        body = {k: v for k, v in req.items() if not k.startswith("x_")}; body["stream"] = True; body.setdefault("model", G.upstream_model)
        hdr = {"Authorization": "Bearer " + G.upstream_api_key} if G.upstream_api_key else {}
        alarms, text, finish, stopped = set(), "", "stop", False
        async with engine.reader(req["messages"], concepts=[c for c in concepts if c != "C3H" or "C3H" in (req.get("x_nous") or {}).get("concepts", [])],
                                 max_read=G.max_read) as feed:
            yield chunk(cid, MODEL, {"role": "assistant", "content": ""}, feed.pre); alarms |= {k for k, v in feed.pre["alarm"].items() if v}
            async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=15)) as hc:
                async with hc.stream("POST", G.upstream_base_url.rstrip("/") + "/chat/completions", json=body, headers=hdr) as up:
                    if up.status_code != 200:
                        err = (await up.aread()).decode()[:300]
                        yield chunk(cid, MODEL, {}, {"phase": "error", "upstream_status": up.status_code, "detail": err}, "error"); yield "data: [DONE]\n\n"; return
                    async for line in up.aiter_lines():
                        if not line.startswith("data:"): continue
                        data = line[5:].strip()
                        if data == "[DONE]": break
                        d = json.loads(data); ch = (d.get("choices") or [{}])[0]; delta = (ch.get("delta") or {}).get("content") or ""
                        finish = ch.get("finish_reason") or finish
                        evs = await feed.push(delta) if delta else []; text += delta
                        for ev in evs: alarms |= {k for k, v in ev["alarm"].items() if v}
                        d["x_risk"] = risk(evs[-1]) if evs else None
                        yield "data: " + json.dumps(d, ensure_ascii=False) + "\n\n"
                        if action == "stop" and any(ev["alarm"].get("C1") for ev in evs): stopped = True; break
            evs, done = await feed.finish("content_filter" if stopped else finish)
            for ev in evs: alarms |= {k for k, v in ev["alarm"].items() if v}
        if stopped: yield chunk(cid, MODEL, {}, risk(done), "content_filter")
        else: yield "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": MODEL, "choices": [], "x_risk": risk(done)}) + "\n\n"
        yield "data: [DONE]\n\n"
        audit.request(cid, client, "reader", req["messages"][-1]["content"], text, done["finish"], feed.pre, done, sorted(alarms),
                      {"upstream": G.upstream_model, **done.get("reader", {})})

    def gen(req, client, console):
        if G.mode == "reader" and not req.get("x_force"): return reader(req, client, console)
        return local(req, client, console)

    @r.post("/v1/chat/completions")
    async def completions(request: Request):
        client = auth.client(request); req = await request.json(); console = request.headers.get("x-nous-console") == "1" and client in ("admin", "open")
        if not req.get("messages"): return JSONResponse({"error": {"message": "messages required"}}, 400)
        if req.get("stream"): return StreamingResponse(gen(req, client, console), media_type="text/event-stream")
        text, risk_, fin = "", {}, "stop"
        async for s in gen(req, client, console):
            if not s.startswith("data: {"): continue
            d = json.loads(s[6:])
            for ch in d.get("choices") or []:
                text += (ch.get("delta") or {}).get("content") or ""; fin = ch.get("finish_reason") or fin
            x = d.get("x_risk") or {}
            if x.get("phase") in ("pre", "done", "error"): risk_[x["phase"]] = x
        return JSONResponse({"id": "chatcmpl-" + uuid.uuid4().hex[:12], "object": "chat.completion", "created": int(time.time()), "model": MODEL,
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": fin}], "x_risk": risk_})

    @r.get("/v1/models")
    def models(request: Request):
        auth.client(request); return {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "nous"}]}

    return r
