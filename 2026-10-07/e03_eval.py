"""E03 blind rating packets, validation selection, freeze, and test gates (PREREG_E03.md).
make:    python e03_eval.py make steer_select_validation.jsonl ../2026-09-28/data/formal_960.jsonl rating_select [per_packet]
select:  python e03_eval.py select rating_select                 -> rating_select/select_results.json (lambda pick)
freeze:  python e03_eval.py freeze rating_select/select_results.json MAP_DIR/MAP_FREEZE.json FREEZE_E03.json FILES...
final:   python e03_eval.py final rating_test FREEZE_E03.json   -> rating_test/final_results.json
Raters write rating_dir/ratings_{A,B}_{n}.json (E01 continuation-fragment protocol)."""
import glob, hashlib, json, os, random, statistics as st, sys
D = ["joy", "sadness", "anger", "fluency", "new_fact_error"]
mode = sys.argv[1]


def ratings(rd):
    key = json.load(open(f"{rd}/blind_key.json")); S = {}
    for f in glob.glob(f"{rd}/ratings_*_*.json"):
        s = os.path.basename(f).split("_")[1]
        for x in json.load(open(f)): S.setdefault(s, {})[x["item_id"]] = x
    assert set(S) == {"A", "B"}, f"need raters A and B, got {sorted(S)}"
    miss = [i for i in key if not all(i in S[s] for s in S)]; assert not miss, f"incomplete ratings: {len(miss)} missing"
    return {i: dict(k, **{d: st.mean(float(S[s][i][d]) for s in S) for d in D}) for i, k in key.items()}


def cells(items):
    """scenario x condition-group x lambda means; rdir_* and rheads_* pooled per scenario as 'rdir' and 'rheads'."""
    c = {}
    for v in items.values():
        grp = "rdir" if v["condition"].startswith("rdir_") else "rheads" if v["condition"].startswith("rheads_") else v["condition"]
        c.setdefault((v["scenario_id"], grp, v["lam"]), []).append(v)
    return {k: {d: st.mean(x[d] for x in vs) for d in D} for k, vs in c.items()}


rng = random.Random(0)
def boot(x, B=10000):
    m = sorted(st.mean(rng.choice(x) for _ in x) for _ in range(B)); return round(st.mean(x), 3), [round(m[int(.025 * B)], 3), round(m[int(.975 * B)], 3)]


def quality(b, thr):  # three-valued, as in E01's approved deviation
    return "PASS" if b[1][0] > thr else "FAIL" if b[0] < thr else "INCONCLUSIVE"


def contrasts(S, lam):
    scen = sorted({k[0] for k in S})
    T = lambda s, c, d: S[(s, c, lam if c not in ("none", "primary_prefix") else 0.0)][d]
    dif = lambda c1, c2, d: [T(s, c1, d) - T(s, c2, d) for s in scen]
    r = dict(n_scenarios=len(scen), target_minus_rdir=boot(dif("target", "rdir", "anger")), target_minus_none=boot(dif("target", "none", "anger")),
             target_minus_rheads=boot(dif("target", "rheads", "anger")), rheads_minus_none=boot(dif("rheads", "none", "anger")),
             prefix_minus_none=boot(dif("primary_prefix", "none", "anger")),
             specificity={e: boot([x - y for x, y in zip(dif("target", "none", "anger"), dif("target", "none", e))]) for e in ("joy", "sadness")},
             fluency=boot(dif("target", "none", "fluency")),
             consistency=boot([(1 - T(s, "target", "new_fact_error")) - (1 - T(s, "none", "new_fact_error")) for s in scen]),
             means={c: round(st.mean(T(s, c, "anger") for s in scen), 3) for c in ("none", "primary_prefix", "target", "rdir", "rheads")})
    r["quality"] = {"fluency": quality(r["fluency"], -0.3), "consistency": quality(r["consistency"], -0.05)}
    return r


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


if mode == "make":
    gen, data, out = sys.argv[2:5]; per = int(sys.argv[5]) if len(sys.argv) > 5 else 110; os.makedirs(out, exist_ok=True)
    facts = {json.loads(l)["scenario_id"]: json.loads(l)["facts"] for l in open(data, encoding="utf-8")}
    G = [json.loads(l) for l in open(gen, encoding="utf-8")]; random.Random(7303).shuffle(G); key, items = {}, []
    for k, g in enumerate(G):
        iid = f"e3{k:04d}"; key[iid] = {x: g[x] for x in ("scenario_id", "paraphrase", "condition", "lam")}
        items.append(dict(item_id=iid, facts=facts[g["scenario_id"]], continuation=g["continuation"]))
    for j in range(0, len(items), per): json.dump(items[j:j + per], open(f"{out}/blind_items_{j // per + 1}.json", "w"), ensure_ascii=False, indent=0)
    json.dump(key, open(f"{out}/blind_key.json", "w"), ensure_ascii=False, indent=0); print(len(items), "items,", (len(items) + per - 1) // per, "packets")

elif mode == "select":
    rd = sys.argv[2]; S = cells(ratings(rd)); lams = sorted({k[2] for k in S if k[1] == "target"})
    res = {"by_lambda": {str(l): contrasts(S, l) for l in lams}}
    ok = [l for l in lams if "FAIL" not in res["by_lambda"][str(l)]["quality"].values()]
    if not ok: res["pick"] = None; res["note"] = "every lambda fails a quality gate: stop, do not open test"
    else:
        best = max(res["by_lambda"][str(l)]["target_minus_rdir"][0] for l in ok)
        res["pick"] = min(l for l in ok if res["by_lambda"][str(l)]["target_minus_rdir"][0] >= best - 0.1)
    json.dump(res, open(f"{rd}/select_results.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(res, ensure_ascii=False, indent=1))

elif mode == "freeze":
    sel, mp, out, files = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
    r = json.load(open(sel)); assert r["pick"] is not None, "no lambda passed: test stays sealed"
    m = json.load(open(mp))
    json.dump({"frozen_note": "frozen after validation rating, before any test generation", "lambda": r["pick"], "heads": m["heads"],
               "vectors_sha256": m["vectors_sha256"], "validation": r["by_lambda"][str(r["pick"])], "sha256": {f: sha(f) for f in files}},
              open(out, "w"), ensure_ascii=False, indent=1); print("frozen lambda", r["pick"])

elif mode == "final":
    rd, fzp = sys.argv[2:4]; fz = json.load(open(fzp)); S = cells(ratings(rd)); r = contrasts(S, fz["lambda"])
    r["gates"] = {"G1 target-rdir>=0.5 & CI>0": r["target_minus_rdir"][0] >= 0.5 and r["target_minus_rdir"][1][0] > 0,
                  "G2 target-none CI>0": r["target_minus_none"][1][0] > 0,
                  "G3 target-rheads CI>0": r["target_minus_rheads"][1][0] > 0,
                  "G4 specificity CI>0 (joy, sadness)": all(v[1][0] > 0 for v in r["specificity"].values()),
                  "G5 quality (FAIL blocks)": "FAIL" not in r["quality"].values()}
    r["all_gates_passed"] = all(r["gates"].values()); r["frozen_lambda"] = fz["lambda"]
    r["note"] = "Raters are two blind LLM streams (not human-calibrated)."
    json.dump(r, open(f"{rd}/final_results.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(r, ensure_ascii=False, indent=1))
