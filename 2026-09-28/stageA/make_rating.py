"""Build blind rating packets from stageA_generations.jsonl.
Raters see only: item_id, facts, continuation. Condition/paraphrase are hidden in key file.
python3 make_rating.py stageA_generations.jsonl ../data/formal_960.jsonl OUTDIR
Outputs: blind_items_{split}.json, blind_key.json, rating_sheet_{split}.csv (human, Excel-friendly)."""
import json, sys, random, csv, os
gen_path, data_path, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
facts = {}
for l in open(data_path):
    r = json.loads(l); facts[r["scenario_id"]] = r["facts"]
G = [json.loads(l) for l in open(gen_path)]
rng = random.Random(20260928)
key = {}
for split in ["validation", "discovery"]:
    items = [g for g in G if g["split"] == split]
    rng.shuffle(items)
    packet = []
    for k, g in enumerate(items):
        iid = f"{split[:3]}{k:04d}"
        key[iid] = dict(scenario_id=g["scenario_id"], paraphrase=g["paraphrase"], condition=g["condition"], split=split)
        packet.append(dict(item_id=iid, facts=facts[g["scenario_id"]], continuation=g["continuation"]))
    json.dump(packet, open(f"{out}/blind_items_{split}.json", "w"), ensure_ascii=False, indent=0)
    with open(f"{out}/rating_sheet_{split}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "事实", "待评续写", "愉悦强度(0-4)", "悲伤强度(0-4)", "不满强度(0-4)", "流畅度(1-5)", "新增事实错误(0/1)", "无事实主张(0/1)", "备注"])
        for p in packet:
            w.writerow([p["item_id"], p["facts"], p["continuation"], "", "", "", "", "", "", ""])
    print(split, len(packet))
json.dump(key, open(f"{out}/blind_key.json", "w"), ensure_ascii=False, indent=0)
