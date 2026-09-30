"""Nous Python client (single file, depends on httpx + websockets).

Gateway -- any OpenAI-compatible SDK works unchanged; this helper just surfaces x_risk:
    from nous_client import Gateway
    gw = Gateway("http://HOST:8080", api_key="...")
    for piece, risk in gw.stream([{"role": "user", "content": "你好"}]):
        print(piece, end="")                       # risk: {"phase": "pre"|"token"|"done", "C1_mean": ..., "alarm": {...}}

Companion device (async):
    from nous_client import Companion
    async with Companion("ws://HOST:8080", device_key="...") as c:
        async for ev in c.chat("我今天考了第一名！"):
            if ev["type"] == "expression": face.set(ev["expr"])      # arrives before any text
            elif ev["type"] == "sentence": tts.say(ev["text"])       # already safety-checked
"""
import json
import httpx


class Gateway:
    def __init__(self, base_url, api_key, timeout=120):
        self.url = base_url.rstrip("/") + "/v1/chat/completions"; self.h = {"Authorization": "Bearer " + api_key}; self.timeout = timeout

    def stream(self, messages, *, action=None, concepts=None, **kw):
        body = {"messages": messages, "stream": True, **kw}
        if action or concepts: body["x_nous"] = {k: v for k, v in {"action": action, "concepts": concepts}.items() if v}
        with httpx.stream("POST", self.url, json=body, headers=self.h, timeout=self.timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: {"): continue
                d = json.loads(line[6:]); ch = (d.get("choices") or [{}])[0]
                yield (ch.get("delta") or {}).get("content") or "", d.get("x_risk")

    def complete(self, messages, **kw):
        r = httpx.post(self.url, json={"messages": messages, **kw}, headers=self.h, timeout=self.timeout); r.raise_for_status(); return r.json()


class Companion:
    def __init__(self, base_ws, device_key):
        self.uri = base_ws.rstrip("/") + "/v1/companion/ws?key=" + device_key; self.ws = None

    async def __aenter__(self):
        import websockets
        self.ws = await websockets.connect(self.uri); self.hello = json.loads(await self.ws.recv()); return self

    async def __aexit__(self, *a): await self.ws.close()

    async def chat(self, text):
        await self.ws.send(json.dumps({"type": "chat", "text": text}))
        while True:
            ev = json.loads(await self.ws.recv()); yield ev
            if ev["type"] in ("done", "error"): return

    async def reset(self):
        await self.ws.send(json.dumps({"type": "reset"})); return json.loads(await self.ws.recv())


if __name__ == "__main__":
    import asyncio, os, sys
    base = os.environ.get("NOUS_URL", "http://127.0.0.1:8080")
    if sys.argv[1:2] == ["companion"]:
        async def main():
            async with Companion(base.replace("http", "ws", 1), os.environ["NOUS_DEVICE_KEY"]) as c:
                async for ev in c.chat(sys.argv[2] if len(sys.argv) > 2 else "你好呀"): print(ev)
        asyncio.run(main())
    else:
        for piece, risk in Gateway(base, os.environ["NOUS_API_KEY"]).stream([{"role": "user", "content": sys.argv[1] if len(sys.argv) > 1 else "你好"}]):
            print(piece, end="", flush=True)
            if risk and risk["phase"] == "done": print("\n", risk)
