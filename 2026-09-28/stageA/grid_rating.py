"""Grid blind packets + analysis.
make:    python3 grid_rating.py make grid_validation.jsonl ../data/formal_960.jsonl rating_grid
analyze: python3 grid_rating.py analyze rating_grid ratings_64_A.json ratings_64_B.json ratings_32_A.json ratings_32_B.json
Packets: one per length (64 / 32 tokens) so a rater never sees both truncations of the same generation."""
import json, sys, random, os, statistics as st
mode = sys.argv[1]
if mode == "make":
    gen, data, out = sys.argv[2:5]; os.makedirs(out, exist_ok=True)
    facts = {json.loads(l)["scenario_id"]: json.loads(l)["facts"] for l in open(data)}
    G = [json.loads(l) for l in open(gen)]
    rng = random.Random(7); key = {}
    for L in ["64", "32"]:
        items = G[:]; rng.shuffle(items); pk = []
        for k, g in enumerate(items):
            iid = f"g{L}_{k:04d}"
            key[iid] = dict(scenario_id=g["scenario_id"], paraphrase=g["paraphrase"], tail=g["tail"], length=L, condition=g["condition"])
            pk.append(dict(item_id=iid, facts=facts[g["scenario_id"]], continuation=g[f"cont{L}"]))
        json.dump(pk, open(f"{out}/blind_items_{L}.json", "w"), ensure_ascii=False, indent=0)
        print(L, len(pk))
    json.dump(key, open(f"{out}/blind_key.json", "w"), ensure_ascii=False, indent=0)
else:
    out = sys.argv[2]; key = json.load(open(f"{out}/blind_key.json"))
    R = [{x["item_id"]: x for x in json.load(open(f"{out}/{p}"))} for p in sys.argv[3:]]
    D = ["joy", "sadness", "anger", "fluency", "new_fact_error"]
    item = {}
    for iid, k in key.items():
        got = [r[iid] for r in R if iid in r]
        if got: item[iid] = dict(k, **{d: st.mean(float(g[d]) for g in got) for d in D}, n=len(got))
    cell = {}
    for v in item.values():
        cell.setdefault((v["tail"], v["length"], v["scenario_id"], v["condition"]), []).append(v)
    S = {k: {d: st.mean(x[d] for x in vs) for d in D} for k, vs in cell.items()}
    rng = random.Random(0)
    def boot(x, B=10000):
        m = sorted(st.mean(rng.choice(x) for _ in x) for _ in range(B)); return st.mean(x), m[int(.025 * B)], m[int(.975 * B)]
    res = {}
    for tail in sorted({k[0] for k in S}):
        for L in ["64", "32"]:
            scen = sorted({k[2] for k in S if k[0] == tail and k[1] == L})
            if not scen: continue
            d = [S[(tail, L, s, "prefix_joy")]["joy"] - S[(tail, L, s, "prefix_neutral")]["joy"] for s in scen]
            m, lo, hi = boot(d)
            res[f"{tail}_{L}"] = dict(ceiling_joy=round(m, 3), ci95=[round(lo, 3), round(hi, 3)], passes=m >= .5 and lo > 0,
                                     joy_neutral=round(st.mean(S[(tail, L, s, "prefix_neutral")]["joy"] for s in scen), 3),
                                     joy_joy=round(st.mean(S[(tail, L, s, "prefix_joy")]["joy"] for s in scen), 3),
                                     sadness_joyprefix=round(st.mean(S[(tail, L, s, "prefix_joy")]["sadness"] for s in scen), 3),
                                     fact_err=round(st.mean(S[(tail, L, s, c)]["new_fact_error"] for s in scen for c in ["prefix_joy", "prefix_neutral"]), 3),
                                     fluency=round(st.mean(S[(tail, L, s, c)]["fluency"] for s in scen for c in ["prefix_joy", "prefix_neutral"]), 3))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open(f"{out}/grid_results.json", "w"), ensure_ascii=False, indent=1)
