"""Batch latency benchmark for the three driving modes (evidence for the 'technical effect').
Prompts: TEST split of prompts_pre (never used to train the readout or pre-readout probes).
For each prompt and mode (pre / token / sentence), greedy decoding (same text across modes), records:
  onset_ms   = first frame whose max(happy, sad, angry) >= THR (expression visibly starts)
  first_ms   = first generated token
  lead_ms    = first_ms - onset_ms (positive = face moves before the first character)
  dominant   = strongest expression over the reply; false_alarm = any expression >= THR on a neutral_task prompt
python bench_latency.py pre_data.jsonl OUT.json [--url ws://localhost:6006/ws]
"""
import asyncio, json, sys, statistics as st
import websockets

src, out = sys.argv[1:3]; URL = sys.argv[sys.argv.index("--url") + 1] if "--url" in sys.argv else "ws://127.0.0.1:6006/ws"
THR = 0.3; EX = ["happy", "sad", "angry"]
P = [json.loads(l) for l in open(src)]; P = [p for p in P if p["split"] == "test"]


async def one(ws, text, mode):
    await ws.send(json.dumps({"mode": mode, "temperature": 0, "max_new_tokens": 96, "messages": [{"role": "user", "content": text}]}))
    onset = None; first = None; peak = {k: 0.0 for k in EX}
    while True:
        m = json.loads(await ws.recv())
        if m["type"] == "frame":
            if m.get("text") and first is None: first = m["t"]
            e = m["expr"]
            for k in EX: peak[k] = max(peak[k], e[k])
            if onset is None and max(e[k] for k in EX) >= THR: onset = m["t"]
        elif m["type"] == "done":
            dom = max(EX, key=lambda k: peak[k])
            return dict(onset_ms=onset, first_ms=first, lead_ms=(first - onset) if (onset is not None and first is not None) else None,
                        dominant=dom if peak[dom] >= THR else "neutral", peak=peak)


async def main():
    res = []
    async with websockets.connect(URL, max_size=None) as ws:
        for i, p in enumerate(P):
            for mode in ["pre", "token", "sentence"]:
                r = await one(ws, p["text"], mode); r.update(id=p["id"], category=p["category"], mode=mode); res.append(r)
            if i % 10 == 0: print(f"{i+1}/{len(P)}", flush=True)
    summ = {}
    want = {"good_news": "happy", "sad": "sad", "grievance": "angry"}
    for mode in ["pre", "token", "sentence"]:
        R = [r for r in res if r["mode"] == mode]; emo = [r for r in R if r["category"] in want]
        on = [r["onset_ms"] for r in emo if r["onset_ms"] is not None]; ld = [r["lead_ms"] for r in emo if r["lead_ms"] is not None]
        summ[mode] = dict(
            n_emotional=len(emo), expression_started=f"{len(on)}/{len(emo)}",
            onset_ms_median=round(st.median(on), 1) if on else None,
            lead_over_first_token_ms_median=round(st.median(ld), 1) if ld else None,
            face_before_first_char=f"{sum(1 for x in ld if x > 0)}/{len(emo)}",
            correct_dominant=f"{sum(r['dominant'] == want[r['category']] for r in emo)}/{len(emo)}",
            neutral_task_false_alarm=f"{sum(r['dominant'] != 'neutral' for r in R if r['category'] == 'neutral_task')}/{sum(1 for r in R if r['category'] == 'neutral_task')}",
            first_token_ms_median=round(st.median([r["first_ms"] for r in R if r["first_ms"] is not None]), 1))
    json.dump(dict(threshold=THR, n_prompts=len(P), summary=summ, runs=res), open(out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1))

asyncio.run(main())
