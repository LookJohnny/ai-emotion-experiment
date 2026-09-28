"""§5 validation selection + freeze.
make:    python3 select_freeze.py make main_select_validation.jsonl ../data/formal_960.jsonl rating_select
analyze: python3 select_freeze.py analyze rating_select --primary anger
freeze:  python3 select_freeze.py freeze rating_select/select_results.json ../stageC/rating_free/free_results.json FREEZE.json \
             --b1 .. --b2 .. --b3 .. --b4 .. --info main_select_validation_info.json
Rater files: rating_select/ratings_{A,B}_{1,2}.json (continuation-fragment protocol, same as stage A)."""
import json, sys, random, os, statistics as st, glob, hashlib, argparse
mode = sys.argv[1]


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


if mode == "make":
    gen, data, out = sys.argv[2:5]; os.makedirs(out, exist_ok=True)
    facts = {}
    for l in open(data):
        r = json.loads(l); facts[r["scenario_id"]] = r["facts"]
    G = [json.loads(l) for l in open(gen)]
    rng = random.Random(5150); rng.shuffle(G); key = {}; items = []
    for k, g in enumerate(G):
        iid = f"ms{k:04d}"; key[iid] = {x: g[x] for x in ["scenario_id", "paraphrase", "condition", "lam", "scale"]}
        items.append(dict(item_id=iid, facts=facts[g["scenario_id"]], continuation=g["continuation"]))
    h = len(items) // 2
    json.dump(items[:h], open(f"{out}/blind_items_1.json", "w"), ensure_ascii=False, indent=0)
    json.dump(items[h:], open(f"{out}/blind_items_2.json", "w"), ensure_ascii=False, indent=0)
    json.dump(key, open(f"{out}/blind_key.json", "w"), ensure_ascii=False, indent=0)
    print(len(items), h, len(items) - h)

elif mode == "analyze":
    out = sys.argv[2]; primary = sys.argv[sys.argv.index("--primary") + 1]
    key = json.load(open(f"{out}/blind_key.json")); streams = {}
    for f in glob.glob(f"{out}/ratings_*_*.json"):
        s = os.path.basename(f).split("_")[1]
        for x in json.load(open(f)): streams.setdefault(s, {})[x["item_id"]] = x
    D = ["joy", "sadness", "anger", "fluency", "new_fact_error"]
    item = {i: dict(k, **{d: st.mean(float(streams[s][i][d]) for s in streams) for d in D}) for i, k in key.items() if all(i in streams[s] for s in streams)}
    print("streams:", {s: len(v) for s, v in streams.items()}, "complete items:", len(item), "/", len(key))
    cell = {}
    for v in item.values():
        c = "random" if v["condition"].startswith("random_") else v["condition"]
        cell.setdefault((v["scenario_id"], c, v["lam"]), []).append(v)   # random sets + paraphrases pooled per scenario
    S = {k: {d: st.mean(x[d] for x in vs) for d in D} for k, vs in cell.items()}
    scen = sorted({k[0] for k in S}); lams = sorted({k[2] for k in S if k[1] == "target"})
    rng = random.Random(0)
    def boot(x, B=10000):
        m = sorted(st.mean(rng.choice(x) for _ in x) for _ in range(B)); return round(st.mean(x), 3), [round(m[int(.025 * B)], 3), round(m[int(.975 * B)], 3)]
    N = lambda s, d: S[(s, "none", 0.0)][d]
    res = {"primary": primary, "n_scenarios": len(scen), "by_lambda": {},
           "none": {d: round(st.mean(N(s, d) for s in scen), 3) for d in D},
           "anchor_sub_minus_none": boot([S[(s, "anchor_sub", 1.0)][primary] - N(s, primary) for s in scen]),
           "primary_prefix_minus_none": boot([S[(s, "primary_prefix", 1.0)][primary] - N(s, primary) for s in scen])}
    for lam in lams:
        T = lambda s, d, c="target": S[(s, c, lam)][d]
        r = dict(target_minus_random=boot([T(s, primary) - T(s, primary, "random") for s in scen]),
                 target_minus_none=boot([T(s, primary) - N(s, primary) for s in scen]),
                 fluency_target_minus_none=boot([T(s, "fluency") - N(s, "fluency") for s in scen]),
                 consistency_target_minus_none=boot([(1 - T(s, "new_fact_error")) - (1 - N(s, "new_fact_error")) for s in scen]))
        if any((s, "shuffled", lam) in S for s in scen):
            r["target_minus_shuffled"] = boot([T(s, primary) - T(s, primary, "shuffled") for s in scen if (s, "shuffled", lam) in S])
        r["quality_ok"] = r["fluency_target_minus_none"][1][0] > -0.3 and r["consistency_target_minus_none"][1][0] > -0.05
        res["by_lambda"][str(lam)] = r
    ok = [l for l in lams if res["by_lambda"][str(l)]["quality_ok"]]
    if ok:
        best = max(res["by_lambda"][str(l)]["target_minus_random"][0] for l in ok)
        res["selected_lambda"] = min(l for l in ok if res["by_lambda"][str(l)]["target_minus_random"][0] >= best - 0.1)
    else:
        res["selected_lambda"] = None
    res["selection_rule"] = "plan §5.2: within quality gates (fluency CI>-0.3, consistency CI>-5pp vs none) largest target-random; within 0.1 prefer smaller lambda"
    print(json.dumps(res, ensure_ascii=False, indent=1)); json.dump(res, open(f"{out}/select_results.json", "w"), ensure_ascii=False, indent=1)

elif mode == "freeze":
    ap = argparse.ArgumentParser(); ap.add_argument("m"); ap.add_argument("sel"); ap.add_argument("free"); ap.add_argument("out")
    for k in ["--b1", "--b2", "--b3", "--b4", "--info"]: ap.add_argument(k, required=True)
    a = ap.parse_args()
    sel = json.load(open(a.sel)); free = json.load(open(a.free)); info = json.load(open(a.info))
    assert sel["selected_lambda"] is not None, "no lambda passed quality gates on validation -> do not freeze; return to user"
    root = os.path.dirname(os.path.abspath(__file__)) + "/.."
    scripts = ["stageB/b1_locate.py", "stageB/b2_windows.py", "stageB/b3_heads.py", "stageB/b4_repr.py", "stageC/free_gen.py",
               "stageC/free_rating.py", "stageD/main_causal.py", "stageD/select_freeze.py", "data/formal_960.jsonl"]
    fz = dict(frozen_at="2026-09-28", experiment="E01 v2.1", primary=sel["primary"], anchor_tail="我的看法是：", cont_tokens=64,
              main_object=info["object"], heads=info.get("heads"), peak_layer=info["peak_layer"], main_lambda=sel["selected_lambda"],
              main_selection=sel["by_lambda"][str(sel["selected_lambda"])], random_seeds=[101, 102, 103, 104, 105], shuffle_seed=201,
              free_gen_lambda=free.get("selected_lambda"), free_gen_selection=free["by_lambda"].get(str(free.get("selected_lambda"))),
              test_design=dict(split="test", paraphrase="A", conditions=["none", "target", "random_0..4", "shuffled(direction only)", "anchor_sub", "primary_prefix"],
                               raters="2 blind raters, same protocol as validation", bootstrap=10000),
              test_gates=dict(target_minus_random=">= 0.5 * (anchor_sub - none), CI lower > 0", target_minus_none="CI lower > 0",
                              specificity="primary rise > other emotions, CI lower > 0", consistency="CI lower > -5pp", fluency="CI lower > -0.3"),
              sha256={p: sha(f"{root}/{p}") for p in scripts if os.path.exists(f"{root}/{p}")},
              stage_summaries={k: sha(getattr(a, k)) for k in ["b1", "b2", "b3", "b4"]})
    json.dump(fz, open(a.out, "w"), ensure_ascii=False, indent=1); print("FROZEN ->", a.out); print(json.dumps({k: fz[k] for k in ["primary", "main_object", "main_lambda", "free_gen_lambda"]}, ensure_ascii=False))
