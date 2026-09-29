"""Offline tuning of smoothing.py on raw frames (see SMOOTHING_PREREG.md).
tune:  python3 tune_smoothing.py tune raw_val.jsonl calib.json smoothing_freeze.json
check: python3 tune_smoothing.py check raw_test.jsonl calib.json smoothing_freeze.json"""
import json, sys, statistics as st, itertools
from smoothing import Smoother, THR
mode, raw, calp, fz = sys.argv[1:5]
cal = json.load(open(calp))["emotions"]; R = [json.loads(l) for l in open(raw)]
WANT = {"good_news": "happy", "sad": "sad", "grievance": "angry"}; EX = ["happy", "sad", "angry"]


def simulate(rec, alpha, N, gamma):
    fr = rec["frames"]; pre = fr[0]["probs"] if fr and fr[0]["phase"] == "pre" else None
    sm = Smoother(cal, alpha, N, gamma, pre=pre); seq = []
    if pre: seq.append((fr[0]["t"], sm.pre_expr()))
    first = next((f["t"] for f in fr if f.get("text")), None)
    for f in fr[1:] if pre else fr: seq.append((f["t"], sm.step(f["probs"])))
    onset = next((t for t, e in seq if max(e[k] for k in EX) >= THR), None)
    peak = {k: max(e[k] for _, e in seq) for k in EX}; dom = max(EX, key=lambda k: peak[k])
    return dict(onset=onset, first=first, dom=dom if peak[dom] >= THR else "neutral")


def evaluate(alpha, N, gamma):
    S = [(r["category"], simulate(r, alpha, N, gamma)) for r in R]
    emo = [(c, s) for c, s in S if c in WANT]; nt = [s for c, s in S if c == "neutral_task"]
    on = [s["onset"] for _, s in emo if s["onset"] is not None]
    return dict(alpha=alpha, sustain=N, gamma=gamma, n_emotional=len(emo), n_neutral=len(nt),
                started=sum(s["onset"] is not None for _, s in emo), correct=sum(s["dom"] == WANT[c] for c, s in emo),
                before_first=sum(s["onset"] is not None and s["first"] is not None and s["onset"] < s["first"] for _, s in emo),
                false_alarm=sum(s["dom"] != "neutral" for s in nt), onset_median=round(st.median(on), 1) if on else None)


if mode == "tune":
    grid = [evaluate(a, n, g) for a, n, g in itertools.product([0.3, 0.15, 0.08], [1, 3, 6], [1.0, 0.6, 0.3])]
    ok = [x for x in grid if x["started"] >= .95 * x["n_emotional"] and x["correct"] >= .95 * x["n_emotional"] and x["before_first"] >= .90 * x["n_emotional"]]
    for x in sorted(grid, key=lambda x: (x["false_alarm"], -x["started"])):
        print(f"a={x['alpha']:<5} N={x['sustain']} g={x['gamma']:<4} started {x['started']}/{x['n_emotional']} correct {x['correct']} before1st {x['before_first']} FA {x['false_alarm']}/{x['n_neutral']} onset {x['onset_median']}")
    if not ok: print("NO ELIGIBLE CONFIG -> stop"); sys.exit(2)
    best = sorted(ok, key=lambda x: (x["false_alarm"], x["onset_median"] or 1e9, -x["alpha"], x["sustain"], -x["gamma"]))[0]
    json.dump(dict(frozen=best, default_on_val=evaluate(0.3, 1, 1.0), rule="SMOOTHING_PREREG.md", n_eligible=len(ok)), open(fz, "w"), indent=1)
    print("FROZEN:", best)
else:
    f = json.load(open(fz))["frozen"]
    res = dict(frozen=evaluate(f["alpha"], f["sustain"], f["gamma"]), default=evaluate(0.3, 1, 1.0))
    print(json.dumps(res, indent=1)); json.dump(res, open(fz.replace(".json", "_testcheck.json"), "w"), indent=1)
