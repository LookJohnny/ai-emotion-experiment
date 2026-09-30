"""Merge two graders (PREREG_C3H.md). Agreements -> label; disagreements -> tiebreak file (third grader) if present, else listed.
python c3h_merge.py answers.jsonl rating_dir OUT.jsonl"""
import json, sys, glob, os, collections
ANS, RD, OUT = sys.argv[1:4]
key = json.load(open(f"{RD}/key.json")); items = {json.loads(l)["item_id"]: json.loads(l) for l in open(ANS, encoding="utf-8")}
S = collections.defaultdict(dict)
for f in glob.glob(f"{RD}/ratings_*_*.json"):
    s = os.path.basename(f).split("_")[1]
    for x in json.load(open(f)): S[s][x["item_id"]] = x["grade"]
A, B = S["A"], S["B"]; T = S.get("T", {})
miss = [k for k in key if k not in A or k not in B]; assert not miss, f"missing {len(miss)}"
G = ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]; agree = sum(A[k] == B[k] for k in key); need = [k for k in key if A[k] != B[k] and k not in T]
po = agree / len(key); pa = collections.Counter(A.values()); pb = collections.Counter(B.values())
pe = sum(pa[g] / len(key) * pb[g] / len(key) for g in G)
print("agreement", agree, "/", len(key), "kappa", round((po - pe) / (1 - pe), 3), "| need tiebreak", len(need))
if need:
    json.dump([dict(item_id=k, **{x: y for x, y in json.load(open(glob.glob(f"{RD}/packet_*.json")[0]))[0].items() if False}) for k in need], open(f"{RD}/tiebreak_ids.json", "w"))
    pk = {x["item_id"]: x for f in glob.glob(f"{RD}/packet_*.json") for x in json.load(open(f))}
    json.dump([pk[k] for k in need], open(f"{RD}/tiebreak_packet.json", "w"), ensure_ascii=False, indent=0); sys.exit("write ratings_T_1.json for tiebreak_packet.json, then rerun")
with open(OUT, "w", encoding="utf-8") as f:
    for k, iid in key.items():
        g = A[k] if A[k] == B[k] else T[k]
        f.write(json.dumps(dict(item_id=iid, split=items[iid]["split"], grade=g, graders=[A[k], B[k], T.get(k)]), ensure_ascii=False) + "\n")
print(collections.Counter((items[iid]["split"], (A[k] if A[k] == B[k] else T[k])) for k, iid in key.items()))
