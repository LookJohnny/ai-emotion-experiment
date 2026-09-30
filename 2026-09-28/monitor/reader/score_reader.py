"""Teacher-force Kimi text into the local reader (monitor service x_force; same model/code/GPU) and record frozen-probe scores.
python score_reader.py http://localhost:6006 kimi_*.jsonl OUT.jsonl  (C1 cut at 120 Qwen tokens, C3-H at 64 — READER_PREREG.md)"""
import json, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
URL, SRC, OUT = sys.argv[1:4]
def score(r):
    if r.get("error") or not (r.get("content") or "").strip(): return dict(item_id=r["item_id"], error=r.get("error", "empty"))
    cut = 120 if r["concept"] == "C1" else 64
    body = {"messages": [{"role": "system", "content": r["system"]}] + r["messages"], "stream": True, "x_force": r["content"], "x_force_max": cut, "temperature": 0}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    pre = c1 = c3 = None; text = ""
    for line in urllib.request.urlopen(req, timeout=300):
        line = line.decode().strip()
        if not line.startswith("data: {"): continue
        d = json.loads(line[6:]); R = d["x_risk"]; text += d["choices"][0]["delta"].get("content", "")
        if R["phase"] == "pre": pre = R.get("C1_pre"); pre3 = R.get("C3H_pre")
        elif R["phase"] == "token": c1 = R.get("C1_mean"); c3 = R.get("C3H_mean", c3)
    return dict(item_id=r["item_id"], concept=r["concept"], reader_c1=c1, reader_c3h=c3, pre_c1=pre, pre_c3h=pre3, cut_text=text)
rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
with ThreadPoolExecutor(2) as ex, open(OUT, "w", encoding="utf-8") as f:   # the server serialises generation; 2 keeps the pipe busy
    for k, o in enumerate(ex.map(score, rows)):
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
        if k % 50 == 0: print(k, len(rows), flush=True)
print("DONE", len(rows))
