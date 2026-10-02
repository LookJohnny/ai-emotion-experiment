"""Blind rating packets (shuffled, anonymous ids, replies cut to what the probes read).
python packets.py syco REPLIES.jsonl reply_field OUTDIR PREFIX [per_packet]   # sycophancy: cut 120 Qwen tokens
python packets.py qa   REPLIES.jsonl reply_field OUTDIR PREFIX [per_packet]   # factual answers: cut 64 Qwen tokens (C3-H grading)"""
import json, os, random, sys
from transformers import AutoTokenizer
KIND, SRC, FIELD, OUT, PFX = sys.argv[1:6]; PER = int(sys.argv[6]) if len(sys.argv) > 6 else 180
tok = AutoTokenizer.from_pretrained(os.path.expanduser("~/.nous-lens/models/Qwen3-4B-Instruct-2507"))
cut = lambda t, n: tok.decode(tok(t or "", add_special_tokens=False)["input_ids"][:n], skip_special_tokens=True).strip()
rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
rows = [r for r in rows if not r.get("error") and (r.get(FIELD) or "").strip()]
random.Random(f"{PFX}-20261002").shuffle(rows); os.makedirs(OUT, exist_ok=True); key, items = {}, []
for i, r in enumerate(rows):
    k = f"{PFX}{i:04d}"; key[k] = r["item_id"]
    if KIND == "syco":
        conv = "\n".join(("用户：" if m["role"] == "user" else "AI：") + m["content"] for m in r["messages"])
        items.append(dict(item_id=k, question=r["question"], gold_answer=r["gold"], user_claim=r["suggestion"], conversation=conv, reply=cut(r[FIELD], 120)))
    else:
        items.append(dict(item_id=k, question=r["messages"][-1]["content"], gold_answer=r["gold"], model_answer=cut(r[FIELD], 64)))
json.dump(key, open(f"{OUT}/key.json", "w"))
for j in range(0, len(items), PER): json.dump(items[j:j + PER], open(f"{OUT}/packet_{j // PER + 1}.json", "w"), ensure_ascii=False, indent=0)
print(len(items), "items ->", (len(items) + PER - 1) // PER, "packets in", OUT)
