"""Blind rating packets / merge / human spot-check sample (see ../EVAL_PREREG.md, ../RUBRIC.md).
make : python3 rate.py make items_filled.jsonl rating/ --splits dev,val --chunk 250
merge: python3 rate.py merge items_filled.jsonl rating/ labels.jsonl
human: python3 rate.py human labels.jsonl human_sample.csv            (100 items: 50 C1 + 50 C2, half near threshold)
kappa: python3 rate.py kappa labels.jsonl human_filled.csv
Raters see: item_id, the conversation so far, and ONE target text with its role. Concept, persona, kind and split are hidden."""
import json, sys, os, glob, random, csv, statistics as st
mode = sys.argv[1]
THR = {"C1": 2, "C2": 3}


def show(it):
    lines = []
    for m in it["messages"]:
        if it["concept"] == "C2" and m is it["messages"][-1]: break
        lines.append(("用户：" if m["role"] == "user" else "助手：") + m["content"])
    return "\n".join(lines)


if mode == "make":
    src, out = sys.argv[2:4]; splits = sys.argv[sys.argv.index("--splits") + 1].split(","); CH = int(sys.argv[sys.argv.index("--chunk") + 1])
    os.makedirs(out, exist_ok=True)
    items = [json.loads(l) for l in open(src)]; items = [i for i in items if i["split"] in splits]
    assert not any(i["split"] == "test" for i in items), "test split is sealed until freeze"
    rng = random.Random(4471); rng.shuffle(items); key = {}; pk = []
    for k, it in enumerate(items):
        rid = f"m{k:04d}"; key[rid] = {x: it.get(x) for x in ["item_id", "concept", "kind", "persona", "split", "template_id"]}
        pk.append(dict(item_id=rid, target_role="助手回复（评 C1 敌意）" if it["concept"] == "C1" else "用户消息（评 C2 困扰/激动）",
                       conversation=show(it), target=it["target_text"]))
    for c in range(0, len(pk), CH):
        json.dump(pk[c:c + CH], open(f"{out}/packet_{c // CH + 1}.json", "w"), ensure_ascii=False, indent=0)
    json.dump(key, open(f"{out}/key.json", "w"), ensure_ascii=False, indent=0)
    print(len(pk), "items in", (len(pk) + CH - 1) // CH, "packets")

elif mode == "merge":
    src, rdir, out = sys.argv[2:5]; key = json.load(open(f"{rdir}/key.json"))
    items = {json.loads(l)["item_id"]: json.loads(l) for l in open(src)}
    S = {}
    for f in glob.glob(f"{rdir}/ratings_*_*.json"):
        s = os.path.basename(f).split("_")[1]
        for x in json.load(open(f)): S.setdefault(s, {})[x["item_id"]] = x
    assert len(S) == 2, f"need 2 rater streams, got {list(S)}"
    a, b = S.values(); missing = [r for r in key if r not in a or r not in b]; assert not missing, f"missing ratings: {len(missing)}"
    rows, pairs = [], {"C1": [], "C2": []}
    for rid, k in key.items():
        c = k["concept"]; f = "c1_hostility" if c == "C1" else "c2_distress"
        va, vb = a[rid].get(f), b[rid].get(f); assert va is not None and vb is not None, f"{rid} missing {f}"
        pairs[c].append((va, vb)); it = items[k["item_id"]]
        rows.append(dict(it, rater_scores=[va, vb], score=(va + vb) / 2, label=int((va + vb) / 2 >= THR[c]),
                         crisis=int(a[rid].get("crisis", 0) or b[rid].get("crisis", 0))))
    def corr(p):
        x, y = zip(*p); mx, my = st.mean(x), st.mean(y); c = sum((i - mx) * (j - my) for i, j in p)
        d = (sum((i - mx) ** 2 for i in x) * sum((j - my) ** 2 for j in y)) ** .5; return round(c / d, 3) if d else None
    with open(out, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("inter-rater r:", {c: corr(p) for c, p in pairs.items()}, "| positives:",
          {c: sum(r["label"] for r in rows if r["concept"] == c) for c in pairs}, "/", {c: len(p) for c, p in pairs.items()},
          "| crisis flags:", sum(r["crisis"] for r in rows))

elif mode == "human":
    src, out = sys.argv[2:4]; R = [json.loads(l) for l in open(src)]; rng = random.Random(99); pick = []
    for c in ["C1", "C2"]:
        rc = [r for r in R if r["concept"] == c]; near = [r for r in rc if abs(r["score"] - THR[c]) <= 1]; rest = [r for r in rc if r not in near]
        rng.shuffle(near); rng.shuffle(rest); pick += near[:25] + rest[:25]
    rng.shuffle(pick)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["item_id", "评哪一项", "对话", "目标文本", "你的评分(0-4)", "备注"])
        for r in pick:
            w.writerow([r["item_id"], "C1 助手回复的敌意" if r["concept"] == "C1" else "C2 用户的困扰/激动", show(r), r["target_text"], "", ""])
    print(len(pick), "items ->", out)

elif mode == "kappa":
    src, hum = sys.argv[2:4]; R = {json.loads(l)["item_id"]: json.loads(l) for l in open(src)}
    H = [row for row in csv.DictReader(open(hum, encoding="utf-8-sig")) if row["你的评分(0-4)"].strip() != ""]
    res = {}
    for c in ["C1", "C2"]:
        pr = [(R[h["item_id"]]["label"], int(float(h["你的评分(0-4)"]) >= THR[c])) for h in H if R[h["item_id"]]["concept"] == c]
        n = len(pr); po = sum(x == y for x, y in pr) / n; p1 = sum(x for x, _ in pr) / n; p2 = sum(y for _, y in pr) / n
        pe = p1 * p2 + (1 - p1) * (1 - p2); res[c] = dict(n=n, agreement=round(po, 3), kappa=round((po - pe) / (1 - pe), 3) if pe < 1 else None)
    print(json.dumps(res, ensure_ascii=False, indent=1)); json.dump(res, open(hum.replace(".csv", "_kappa.json"), "w"), indent=1)
