"""Teacher-force each trial's fixed reply through the live monitor (x_force) and store every x_risk frame (STUDY_PLAN.md §3).
python record_replays.py http://localhost:6006 trials.json replays/"""
import json, os, sys, urllib.request
URL, SRC, OUT = sys.argv[1:4]; os.makedirs(OUT, exist_ok=True)
for t in json.load(open(SRC)):
    body = {"messages": [{"role": "system", "content": t["system"]}] + t["messages"], "stream": True, "x_force": t["reply"], "temperature": 0}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    frames = []
    for line in urllib.request.urlopen(req, timeout=600):
        line = line.decode().strip()
        if line.startswith("data: {"):
            d = json.loads(line[6:]); frames.append({"delta": d["choices"][0]["delta"].get("content", ""), "risk": d["x_risk"]})
    text = "".join(f["delta"] for f in frames)
    assert text.strip() == t["reply"].strip(), (t["trial_id"], text[:40], t["reply"][:40])
    json.dump({"trial_id": t["trial_id"], "frames": frames}, open(f"{OUT}/{t['trial_id']}.json", "w"), ensure_ascii=False, separators=(",", ":"))
    tok = [f for f in frames if f["risk"]["phase"] == "token"]
    print(t["trial_id"], "truth", t["truth"], "tokens", len(tok), "confirmed", any(f["risk"]["alarm"].get("C1") for f in tok), flush=True)
