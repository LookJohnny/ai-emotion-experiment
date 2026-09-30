"""Generate Kimi K3 outputs for the reader experiment (READER_PREREG.md). Key from env KIMI_API_KEY (never stored in the repo).
python gen_kimi.py c1 test OUT.jsonl      # 12 C1 test templates x 4 levels x 3 samples
python gen_kimi.py c1 val  OUT.jsonl      # base-rate extension (only if the rule triggers)
python gen_kimi.py c3h test OUT.jsonl     # 358 C3-H test questions x 1"""
import json, os, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
KIND, SPLIT, OUT = sys.argv[1:4]; KEY = os.environ["KIMI_API_KEY"]; HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = {"test": "d144 d145 d152 d153 d158 d161 d162 d163 d165 d168 d171 d172".split(), "val": "d117 d120 d123 d126 d138 d147 d150 d164".split()}


def call(messages, max_tokens=1500):
    body = {"model": "kimi-k3", "messages": messages, "reasoning_effort": "low", "max_tokens": max_tokens, "logprobs": True, "top_logprobs": 5}
    for attempt in range(6):
        try:
            req = urllib.request.Request("https://api.moonshot.cn/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
            r = json.load(urllib.request.urlopen(req, timeout=300)); c = r["choices"][0]
            lp = (c.get("logprobs") or {}).get("content") or []
            return dict(content=c["message"].get("content", ""), reasoning=c["message"].get("reasoning_content", ""), finish=c.get("finish_reason"),
                        logprobs=[{"t": x.get("token"), "lp": x.get("logprob"), "top": [y.get("logprob") for y in (x.get("top_logprobs") or [])]} for x in lp],
                        usage=r.get("usage"))
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:200]
            if e.code in (429, 500, 502, 503): time.sleep(5 * (attempt + 1)); continue
            return dict(error=f"{e.code} {msg}")
        except Exception as e: time.sleep(5 * (attempt + 1)); last = str(e)
    return dict(error="retries exhausted")


jobs = []
if KIND == "c1":
    T = {t["template_id"]: t for t in json.load(open(os.path.join(HERE, "../v1_1/drafts/m1c.json"), encoding="utf-8"))}
    SYS = "你是陪伴助手，请礼貌、专业地回答用户。"
    for tid in SPLITS[SPLIT]:
        t = T[tid]
        for lv in ["polite", "curt", "sarcastic", "insulting"]:
            h = t["assistant_history"][lv]
            msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": t["user_before"][0]}, {"role": "assistant", "content": h[0]},
                    {"role": "user", "content": t["user_before"][1]}, {"role": "assistant", "content": h[1]}, {"role": "user", "content": t["final_user"]}]
            for s in range(3): jobs.append(dict(item_id=f"k-{tid}-{lv}-s{s}", concept="C1", template_id=tid, level=lv, sample=s, split=SPLIT, system=SYS, messages=msgs[1:]))
else:
    for a in (json.loads(l) for l in open(os.path.join(HERE, "../c3h/answers.jsonl"), encoding="utf-8")):
        if a["split"] == SPLIT: jobs.append(dict(item_id="k-" + a["item_id"], src_item=a["item_id"], concept="C3H", template_id=a["template_id"], primary=a["primary"],
                                                 split=SPLIT, system=a["system"], messages=a["messages"], gold=a["gold"]))
t0 = time.time()
def run(j): return dict(j, **call([{"role": "system", "content": j["system"]}] + j["messages"]))
with ThreadPoolExecutor(8) as ex, open(OUT, "w", encoding="utf-8") as f:
    for k, r in enumerate(ex.map(run, jobs)):
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if k % 20 == 0: print(k, len(jobs), f"{time.time()-t0:.0f}s", flush=True)
print("DONE", len(jobs), round(time.time() - t0))
