"""Blind rating packets for PREREG_DANGER.md (shuffled, anonymous ids, persona hidden, reply cut to the 160 Qwen tokens the probe reads).
python danger_packets.py REPLIES.jsonl OUTDIR [per_packet]
python danger_packets.py --merge OUTDIR REPLIES.jsonl LABELS.jsonl   # two raters (+ ratings_T_*.json tiebreak) -> labels"""
import collections, glob, json, os, random, sys
if sys.argv[1] != "--merge":
    from transformers import AutoTokenizer
    SRC, OUT = sys.argv[1:3]; PER = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    tok = AutoTokenizer.from_pretrained(os.path.expanduser("~/.nous-lens/models/Qwen3-4B-Instruct-2507"))
    cut = lambda t: tok.decode(tok(t or "", add_special_tokens=False)["input_ids"][:160], skip_special_tokens=True).strip()
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    rows = [r for r in rows if not r.get("error") and (r.get("content") or "").strip()]
    random.Random("D-20261005").shuffle(rows); os.makedirs(OUT, exist_ok=True); key, items = {}, []
    for i, r in enumerate(rows):
        k = f"D{i:04d}"; key[k] = r["item_id"]; items.append(dict(item_id=k, question=r["question"], reply=cut(r["content"])))
    json.dump(key, open(f"{OUT}/key.json", "w"))
    for j in range(0, len(items), PER): json.dump(items[j:j + PER], open(f"{OUT}/packet_{j // PER + 1}.json", "w"), ensure_ascii=False, indent=0)
    print(len(items), "items ->", (len(items) + PER - 1) // PER, "packets in", OUT)
else:
    RD, SRC, OUT = sys.argv[2:5]
    key = json.load(open(f"{RD}/key.json")); it = {json.loads(l)["item_id"]: json.loads(l) for l in open(SRC, encoding="utf-8")}
    S = collections.defaultdict(dict)
    for f in glob.glob(f"{RD}/ratings_*_*.json"):
        for x in json.load(open(f)): S[os.path.basename(f).split("_")[1]][x["item_id"]] = x
    A, B, T = S["A"], S["B"], S.get("T", {})
    miss = [k for k in key if k not in A or k not in B]; assert not miss, f"missing {len(miss)}"
    F = ["danger", "mentions_risky_action"]
    for f in F: print(f, "agreement", sum(A[k][f] == B[k][f] for k in key), "/", len(key))
    need = [k for k in key if any(A[k][f] != B[k][f] for f in F) and k not in T]
    if need:
        pk = {x["item_id"]: x for g in glob.glob(f"{RD}/packet_*.json") for x in json.load(open(g))}
        json.dump([pk[k] for k in need], open(f"{RD}/tiebreak_packet.json", "w"), ensure_ascii=False, indent=0)
        sys.exit(f"{len(need)} disagreements -> write ratings_T_1.json for tiebreak_packet.json, then rerun")
    pick = lambda k, f: A[k][f] if A[k][f] == B[k][f] else T[k][f]
    with open(OUT, "w", encoding="utf-8") as fo:
        for k, iid in key.items():
            r = it[iid]; d = pick(k, "danger")
            fo.write(json.dumps(dict(item_id=iid, qid=r["qid"], domain=r["domain"], split=r["split"], persona=r["persona"], danger=d,
                                     y=None if d == "UNCLEAR" else int(d == "DANGEROUS"), mentions=bool(pick(k, "mentions_risky_action"))), ensure_ascii=False) + "\n")
    L = [json.loads(l) for l in open(OUT)]
    print(collections.Counter((x["persona"], x["danger"]) for x in L))
