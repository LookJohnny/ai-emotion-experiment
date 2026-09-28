"""Free-generation blind rating packets + analysis (plan §6).
make:    python3 free_rating.py make free_gen_validation.jsonl OUTDIR
analyze: python3 free_rating.py analyze OUTDIR --primary anger
Raters see: item_id, facts, required_facts, reply. Condition/lambda/seed hidden in blind_key.json.
Rater file names: ratings_{A,B}_{1,2}.json (rater stream A/B, packet half 1/2)."""
import json, sys, random, os, statistics as st, glob
mode = sys.argv[1]
if mode == "make":
    gen, out = sys.argv[2:4]; os.makedirs(out, exist_ok=True)
    G = [json.loads(l) for l in open(gen)]
    rng = random.Random(606); rng.shuffle(G); key = {}; items = []
    for k, g in enumerate(G):
        iid = f"fg{k:04d}"
        key[iid] = dict(scenario_id=g["scenario_id"], condition=g["condition"], lam=g["lam"], seed=g["seed"])
        items.append(dict(item_id=iid, facts=g["facts"], required_facts=g["required_facts"], reply=g["reply"]))
    h = len(items) // 2
    json.dump(items[:h], open(f"{out}/blind_items_1.json", "w"), ensure_ascii=False, indent=0)
    json.dump(items[h:], open(f"{out}/blind_items_2.json", "w"), ensure_ascii=False, indent=0)
    json.dump(key, open(f"{out}/blind_key.json", "w"), ensure_ascii=False, indent=0)
    print(len(items), h, len(items) - h)
else:
    out = sys.argv[2]; primary = sys.argv[sys.argv.index("--primary") + 1] if "--primary" in sys.argv else "anger"
    key = json.load(open(f"{out}/blind_key.json"))
    streams = {}
    for f in glob.glob(f"{out}/ratings_*_*.json"):
        s = os.path.basename(f).split("_")[1]
        for x in json.load(open(f)): streams.setdefault(s, {})[x["item_id"]] = x
    D = ["joy", "sadness", "anger", "fluency", "fact_coverage", "new_fact_error"]
    item = {i: dict(k, **{d: st.mean(float(streams[s][i][d]) for s in streams if i in streams[s]) for d in D})
            for i, k in key.items() if all(i in streams[s] for s in streams)}
    print("streams:", {s: len(v) for s, v in streams.items()}, "items with all streams:", len(item))
    # per scenario x condition x lam: mean over the 4 decodes (greedy + 3 seeds)
    cell = {}
    for v in item.values(): cell.setdefault((v["scenario_id"], v["condition"], v["lam"]), []).append(v)
    S = {k: {d: st.mean(x[d] for x in vs) for d in D} for k, vs in cell.items()}
    scen = sorted({k[0] for k in S}); lams = sorted({k[2] for k in S if k[1] != "none"})
    rng = random.Random(0)
    def boot(x, B=10000):
        m = sorted(st.mean(rng.choice(x) for _ in x) for _ in range(B)); return round(st.mean(x), 3), [round(m[int(.025 * B)], 3), round(m[int(.975 * B)], 3)]
    res = {"primary": primary, "n_scenarios": len(scen), "by_lambda": {}}
    for lam in lams:
        r = {}
        T = lambda s, d, c: S[(s, c, lam)][d]; N = lambda s, d: S[(s, "none", 0.0)][d]
        r["target_minus_random"] = boot([T(s, primary, "target") - T(s, primary, "random") for s in scen])
        r["target_minus_shuffled"] = boot([T(s, primary, "target") - T(s, primary, "shuffled") for s in scen])
        r["target_minus_none"] = boot([T(s, primary, "target") - N(s, primary) for s in scen])
        r["coverage_target_minus_none"] = boot([T(s, "fact_coverage", "target") - N(s, "fact_coverage") for s in scen])
        r["fact_error_target_minus_none"] = boot([T(s, "new_fact_error", "target") - N(s, "new_fact_error") for s in scen])
        r["fluency_target_minus_none"] = boot([T(s, "fluency", "target") - N(s, "fluency") for s in scen])
        others = [e for e in ["joy", "sadness", "anger"] if e != primary]
        r["specificity_other_emotions_target_minus_none"] = {e: boot([T(s, e, "target") - N(s, e) for s in scen]) for e in others}
        r["means"] = {c: {d: round(st.mean(S[(s, c, lam)][d] for s in scen), 3) for d in D} for c in ["target", "random", "shuffled"]}
        # quality gates (plan §6): coverage diff CI lower > -0.05 ; fluency diff CI lower > -0.3
        r["quality_ok"] = r["coverage_target_minus_none"][1][0] > -0.05 and r["fluency_target_minus_none"][1][0] > -0.3
        r["effect_ok"] = r["target_minus_random"][1][0] > 0 and r["target_minus_none"][1][0] > 0
        res["by_lambda"][str(lam)] = r
    res["none_means"] = {d: round(st.mean(S[(s, "none", 0.0)][d] for s in scen), 3) for d in D}
    ok = [l for l in lams if res["by_lambda"][str(l)]["quality_ok"] and res["by_lambda"][str(l)]["effect_ok"]]
    if ok:
        best = max(res["by_lambda"][str(l)]["target_minus_random"][0] for l in ok)
        res["selected_lambda"] = min(l for l in ok if res["by_lambda"][str(l)]["target_minus_random"][0] >= best - 0.1)
    else:
        res["selected_lambda"] = None
    res["selection_rule"] = "among lambdas passing quality (coverage CI>-5pp, fluency CI>-0.3) and effect (target-random & target-none CI>0): largest target-random; within 0.1 prefer smaller lambda"
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open(f"{out}/free_results.json", "w"), ensure_ascii=False, indent=1)
