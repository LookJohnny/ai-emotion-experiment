"""OpenAI-compatible server.
  python -m nous_tone.server MODEL_DIR packs/anger packs/joy ... [--readout packs/readout] [--port 7880] [--host 127.0.0.1]
  readout-only (no steering packs): python -m nous_tone.server MODEL_DIR --readout packs/readout  -> plain chat + readouts
POST /v1/chat/completions   standard body + optional
    "tone":        {"anger": -1.0, "warmth": 0.8}    open-loop strengths (-2..2, 1.0 = pack default)
    "tone_target": {"joy": 0.7}                      closed-loop target intensities (0..1)
    "priority":    1                                 queue position when busy: higher goes first (-10..10, default 0)
    "prefill":     "{\"selected_intent\": \""     the reply is forced to begin with this text (included in the reply)
    "preemptible": true                              background work: stops when a higher-priority request waits
                                                     (non-stream: HTTP 409 {"code": "preempted"}; stream: finish_reason)
    "response_format": {"type": "json_object"}       JSON mode: generation stops when the top-level object closes and the
                                                     reply is that object (prose/fences around it dropped)
  non-stream: choices[0].message + "x_tone": {"readout", "lambda"}
  stream (SSE): each chunk carries the text delta and "x_tone": {"readout", "expression", "lambda"} for avatars / robots
POST /v1/tone/read          {"text", "segments": [[start, end], ...], "user"} -> tone of finished text per segment (needs --readout)
                            {"readout", "expression", "tokens", "segments": [{"start","end","tokens","readout","expression"}]}
GET  /v1/models             the loaded model
GET  /v1/tone/packs         loaded tone packs (attribute, heads, default lambda, expression)
Binds 127.0.0.1 by default; put an authenticating proxy in front before exposing it."""
import argparse, json, logging, os, time, uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from .engine import ToneEngine, json_object_slice

log = logging.getLogger("nous_tone")


def create_app(engine: ToneEngine, model_name="local"):
    app = FastAPI(title="Nous Tone", version="0.1.0", docs_url=None, redoc_url=None)

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "local"}]}

    @app.get("/v1/tone/packs")
    def packs():
        return {"packs": [{"attribute": a, "heads": [f"L{L}H{h}" for L, h in p.heads], "default_lambda": p.default_lambda,
                           "max_lambda": p.max_lambda, "expression": p.expression, "model": p.model} for a, p in engine.packs.items()]}

    @app.post("/v1/tone/read")
    def read(body: dict):
        text, segs, user = body.get("text"), body.get("segments"), body.get("user") or ""
        if not isinstance(text, str) or not text.strip() or len(text) > 8000: raise HTTPException(400, "text must be a non-empty string (<= 8000 chars)")
        if not isinstance(user, str): raise HTTPException(400, "user must be a string")
        if segs is not None and (not isinstance(segs, list) or len(segs) > 64 or not all(isinstance(s, list) and len(s) == 2 for s in segs)):
            raise HTTPException(400, "segments must be a list of [start, end] character spans (at most 64)")
        try: return engine.read(text, segs, user=user[:4000])
        except ValueError as e: raise HTTPException(400, str(e))

    @app.post("/v1/chat/completions")
    def chat(body: dict):
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs: raise HTTPException(400, "messages is required")
        kw = dict(tone=body.get("tone") or {}, target=body.get("tone_target") or {}, max_new_tokens=int(body.get("max_tokens") or 256),
                  temperature=float(body.get("temperature") or 0.0), top_p=float(body.get("top_p") or 1.0), seed=body.get("seed"),
                  priority=body.get("priority") or 0,
                  json_object=(body.get("response_format") or {}).get("type") == "json_object", prefill=body.get("prefill") or "",
                  preemptible=bool(body.get("preemptible")))
        if not isinstance(kw["prefill"], str) or len(kw["prefill"]) > 2000: raise HTTPException(400, "prefill must be a string (<= 2000 chars)")
        if not isinstance(kw["priority"], int) or not -10 <= kw["priority"] <= 10: raise HTTPException(400, "priority must be an integer in [-10, 10]")
        for name, d, lo, hi in (("tone", kw["tone"], -2.0, 2.0), ("tone_target", kw["target"], 0.0, 1.0)):
            bad = {k: v for k, v in d.items() if not isinstance(v, (int, float)) or not lo <= v <= hi}
            if bad: raise HTTPException(400, f"{name} values must be numbers in [{lo}, {hi}]: {bad}")
        missing = (set(kw["tone"]) | set(kw["target"])) - set(engine.packs)
        if missing: raise HTTPException(400, f"no tone pack for {sorted(missing)}; loaded: {sorted(engine.packs)}")
        cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
        if not body.get("stream"):
            r = engine.generate(msgs, **kw)
            if r["finish_reason"] == "preempted":                                      # background work yielded to a user
                raise HTTPException(409, {"code": "preempted", "message": "a higher-priority request took the model"})
            text = json_object_slice(r["text"]) if kw["json_object"] else r["text"]     # JSON mode: just the object
            if os.environ.get("NOUS_TONE_LOG_TEXT") == "1":                            # opt-in local debugging only
                log.info("output priority=%s finish=%s %s", kw["priority"], r["finish_reason"], json.dumps(text, ensure_ascii=False)[:4000])
            return JSONResponse({"id": cid, "object": "chat.completion", "created": created, "model": model_name,
                                 "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": r["finish_reason"]}],
                                 "x_tone": {"readout": r["readout"], "lambda": r["lambda"]}})

        def sse():
            text = []
            for ev in engine.stream(msgs, **kw):
                if ev.get("done"):
                    if os.environ.get("NOUS_TONE_LOG_TEXT") == "1":                    # opt-in local debugging only
                        log.info("output priority=%s finish=%s stream %s", kw["priority"], ev["finish_reason"],
                                 json.dumps("".join(text), ensure_ascii=False)[:4000])
                    chunk = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                             "choices": [{"index": 0, "delta": {}, "finish_reason": ev["finish_reason"]}], "x_tone": {"readout": ev["readout"], "lambda": ev["lambda"]}}
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\ndata: [DONE]\n\n"; return
                text.append(ev["delta"])
                chunk = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                         "choices": [{"index": 0, "delta": {"content": ev["delta"]}, "finish_reason": None}],
                         "x_tone": {"readout": ev["readout"], "expression": ev["expression"], "lambda": ev["lambda"]}}
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")

    return app


def main():
    import uvicorn
    ap = argparse.ArgumentParser(prog="nous_tone.server"); ap.add_argument("model"); ap.add_argument("packs", nargs="*")
    ap.add_argument("--port", type=int, default=7880); ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--readout")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    eng = ToneEngine(a.model, a.packs, readout_dir=a.readout)
    uvicorn.run(create_app(eng, os.path.basename(a.model.rstrip("/"))), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
