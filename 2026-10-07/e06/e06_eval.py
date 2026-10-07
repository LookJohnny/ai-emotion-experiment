"""E06 blind packets and analysis (PREREG_E06.md).
make:   python e06_eval.py make e06_test.jsonl ../../2026-09-28/data/formal_960.jsonl rating_test [per_packet]
final:  python e06_eval.py final rating_test   -> rating_test/final_results.json"""
import glob, json, os, random, statistics as st, sys
D = ["joy", "sadness", "anger", "fluency", "new_fact_error"]; METHODS = ["twostage", "gcm_atp", "iti", "twostage_self"]
mode = sys.argv[1]; rng = random.Random(0)


def boot(x, B=10000):
    m = sorted(st.mean(rng.choice(x) for _ in x) for _ in range(B)); return round(st.mean(x), 3), [round(m[int(.025 * B)], 3), round(m[int(.975 * B)], 3)]


def label4(b):  # H1/H3 four-way rule
    lo, hi = b[1]
    return "better" if lo > 0 else "worse" if hi < 0 else "non-inferior" if lo > -0.5 else "inconclusive"


if mode == "make":
    gen, data, out = sys.argv[2:5]; per = int(sys.argv[5]) if len(sys.argv) > 5 else 130; os.makedirs(out, exist_ok=True)
    facts = {json.loads(l)["scenario_id"]: json.loads(l)["facts"] for l in open(data, encoding="utf-8")}
    G = [json.loads(l) for l in open(gen, encoding="utf-8")]; random.Random(7606).shuffle(G); key, items = {}, []
    for k, g in enumerate(G):
        iid = f"e6{k:04d}"; key[iid] = {x: g[x] for x in ("scenario_id", "condition", "lam")}
        items.append(dict(item_id=iid, facts=facts[g["scenario_id"]], continuation=g["continuation"]))
    for j in range(0, len(items), per): json.dump(items[j:j + per], open(f"{out}/blind_items_{j // per + 1}.json", "w"), ensure_ascii=False, indent=0)
    json.dump(key, open(f"{out}/blind_key.json", "w"), ensure_ascii=False, indent=0); print(len(items), "items")
elif mode == "final":
    rd = sys.argv[2]; key = json.load(open(f"{rd}/blind_key.json")); S = {}
    for f in glob.glob(f"{rd}/ratings_*_*.json"):
        for x in json.load(open(f)): S.setdefault(os.path.basename(f).split("_")[1], {})[x["item_id"]] = x
    assert set(S) == {"A", "B"} and all(i in S["A"] and i in S["B"] for i in key), "incomplete ratings"
    cell = {}
    for i, k in key.items():
        c = "rheads" if k["condition"].startswith("rheads_") else k["condition"]
        cell.setdefault((k["scenario_id"], c, k["lam"]), []).append({d: st.mean(float(S[s][i][d]) for s in "AB") for d in D})
    M = {kk: {d: st.mean(x[d] for x in v) for d in D} for kk, v in cell.items()}; scen = sorted({k[0] for k in M})
    T = lambda s, c, lam, d: M[(s, c, 0.0 if c in ("none", "primary_prefix") else lam)][d]
    dif = lambda a, b, lam, d="anger": [T(s, a, lam, d) - T(s, b, lam, d) for s in scen]
    R = {"n_scenarios": len(scen), "prefix_minus_none": boot(dif("primary_prefix", "none", 8.0))}
    for lam in (8.0, 4.0):
        r = {"means": {c: round(st.mean(T(s, c, lam, "anger") for s in scen), 3) for c in METHODS + ["rheads", "none", "primary_prefix"]}}
        r["H1_twostage_minus_gcm_atp"] = b = boot(dif("twostage", "gcm_atp", lam)); r["H1_label"] = label4(b)
        r["H2_twostage_minus_iti"] = b = boot(dif("twostage", "iti", lam)); r["H2_pass"] = b[1][0] > 0
        r["H3_self_minus_twostage"] = b = boot(dif("twostage_self", "twostage", lam)); r["H3_label"] = label4(b)
        r["H4_minus_rheads"] = {m: boot(dif(m, "rheads", lam)) for m in METHODS}
        r["minus_none"] = {m: boot(dif(m, "none", lam)) for m in METHODS}
        r["quality_vs_none"] = {m: {"fluency": boot(dif(m, "none", lam, "fluency")),
                                    "consistency": boot([-(x) for x in dif(m, "none", lam, "new_fact_error")])} for m in METHODS}
        R[f"lambda_{lam}"] = r
    R["note"] = "Primary = lambda 8. Raters are two blind LLM streams (not human-calibrated)."
    json.dump(R, open(f"{rd}/final_results.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(R, ensure_ascii=False, indent=1))
