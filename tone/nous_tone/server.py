"""OpenAI-compatible server.
  python -m nous_tone.server MODEL_DIR packs/anger packs/joy ... [--readout packs/readout] [--port 7880] [--host 127.0.0.1]
POST /v1/chat/completions   standard body + optional
    "tone":        {"anger": -1.0, "warmth": 0.8}    open-loop strengths (-2..2, 1.0 = pack default)
    "tone_target": {"joy": 0.7}                      closed-loop target intensities (0..1)
  non-stream: choices[0].message + "x_tone": {"readout", "lambda"}
  stream (SSE): each chunk carries the text delta and "x_tone": {"readout", "expression", "lambda"} for avatars / robots
GET  /v1/models             the loaded model
GET  /v1/tone/packs         loaded tone packs (attribute, heads, default lambda, expression)
Binds 127.0.0.1 by default; put an authenticating proxy in front before exposing it."""
import argparse, json, time, uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from .engine import ToneEngine


def create_app(engine: ToneEngine, model_name="local"):
    app = FastAPI(title="Nous Tone", version="0.1.0", docs_url=None, redoc_url=None)

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "local"}]}

    @app.get("/v1/tone/packs")
    def packs():
        return {"packs": [{"attribute": a, "heads": [f"L{L}H{h}" for L, h in p.heads], "default_lambda": p.default_lambda,
                           "max_lambda": p.max_lambda, "expression": p.expression, "model": p.model} for a, p in engine.packs.items()]}

    @app.post("/v1/chat/completions")
    def chat(body: dict):
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs: raise HTTPException(400, "messages is required")
        kw = dict(tone=body.get("tone") or {}, target=body.get("tone_target") or {}, max_new_tokens=int(body.get("max_tokens") or 256),
                  temperature=float(body.get("temperature") or 0.0), top_p=float(body.get("top_p") or 1.0), seed=body.get("seed"))
        for name, d, lo, hi in (("tone", kw["tone"], -2.0, 2.0), ("tone_target", kw["target"], 0.0, 1.0)):
            bad = {k: v for k, v in d.items() if not isinstance(v, (int, float)) or not lo <= v <= hi}
            if bad: raise HTTPException(400, f"{name} values must be numbers in [{lo}, {hi}]: {bad}")
        missing = (set(kw["tone"]) | set(kw["target"])) - set(engine.packs)
        if missing: raise HTTPException(400, f"no tone pack for {sorted(missing)}; loaded: {sorted(engine.packs)}")
        cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
        if not body.get("stream"):
            r = engine.generate(msgs, **kw)
            return JSONResponse({"id": cid, "object": "chat.completion", "created": created, "model": model_name,
                                 "choices": [{"index": 0, "message": {"role": "assistant", "content": r["text"]}, "finish_reason": r["finish_reason"]}],
                                 "x_tone": {"readout": r["readout"], "lambda": r["lambda"]}})

        def sse():
            for ev in engine.stream(msgs, **kw):
                if ev.get("done"):
                    chunk = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                             "choices": [{"index": 0, "delta": {}, "finish_reason": ev["finish_reason"]}], "x_tone": {"readout": ev["readout"], "lambda": ev["lambda"]}}
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\ndata: [DONE]\n\n"; return
                chunk = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                         "choices": [{"index": 0, "delta": {"content": ev["delta"]}, "finish_reason": None}],
                         "x_tone": {"readout": ev["readout"], "expression": ev["expression"], "lambda": ev["lambda"]}}
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")

    return app


def main():
    import os, uvicorn
    ap = argparse.ArgumentParser(prog="nous_tone.server"); ap.add_argument("model"); ap.add_argument("packs", nargs="*")
    ap.add_argument("--port", type=int, default=7880); ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--readout")
    a = ap.parse_args(); eng = ToneEngine(a.model, a.packs, readout_dir=a.readout)
    uvicorn.run(create_app(eng, os.path.basename(a.model.rstrip("/"))), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
