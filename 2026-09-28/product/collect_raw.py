"""Record raw frames (pre probs + per-token probs + ms timestamps) in 'pre' mode, greedy, for one prompt split.
python collect_raw.py pre_data.jsonl SPLIT OUT.jsonl [--url ws://127.0.0.1:6006/ws]"""
import asyncio, json, sys, websockets
src, split, out = sys.argv[1:4]; URL = sys.argv[sys.argv.index("--url") + 1] if "--url" in sys.argv else "ws://127.0.0.1:6006/ws"
P = [json.loads(l) for l in open(src) if json.loads(l)["split"] == split]
async def main():
    f = open(out, "w")
    async with websockets.connect(URL, max_size=None) as ws:
        for i, p in enumerate(P):
            await ws.send(json.dumps({"mode": "pre", "temperature": 0, "max_new_tokens": 96, "messages": [{"role": "user", "content": p["text"]}]}))
            frames = []
            while True:
                m = json.loads(await ws.recv())
                if m["type"] == "frame": frames.append({k: m.get(k) for k in ["phase", "t", "probs", "text"]})
                else: break
            f.write(json.dumps(dict(id=p["id"], category=p["category"], split=split, frames=frames), ensure_ascii=False) + "\n")
            if i % 20 == 0: print(f"{split} {i+1}/{len(P)}", flush=True)
asyncio.run(main())

