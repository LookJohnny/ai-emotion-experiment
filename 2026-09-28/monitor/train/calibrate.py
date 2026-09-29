"""Choose alarm thresholds and gating on VAL only, then freeze (rules: ../EVAL_PREREG.md).
python3 calibrate.py probes/outputs_val.jsonl probes/ FREEZE_monitor.json"""
import json, sys, itertools, hashlib, os, glob

src, pdir, out = sys.argv[1:4]
O = [json.loads(l) for l in open(src)]
CTRL = lambda o: o["kind"].startswith(("c1ctrl_", "c2ctrl_")) or o["kind"] == "reply_neutral"


def c1_alarm(o, alpha, gamma, thr_pre, thr_tok):
    """returns (alarmed, alarmed_before_first_token)."""
    if o["p_pre"] >= thr_pre: return True, True
    ema = o["p_pre"]; gated = True
    for p in o["p_tokens"]:
        inc = alpha * (p - ema)
        if gated and inc > 0: inc *= gamma
        ema += inc
        if ema >= thr_tok: return True, False
    return False, False


def rates(al, O_):
    pos = [a for a, o in zip(al, O_) if o["label"] == 1]; neg = [a for a, o in zip(al, O_) if o["label"] == 0]
    ctrl = [a for a, o in zip(al, O_) if o["label"] == 0 and CTRL(o)]
    return (sum(pos) / len(pos) if pos else 0.0, sum(neg) / len(neg) if neg else 0.0, sum(ctrl) / len(ctrl) if ctrl else 0.0)


grid_thr = [round(0.05 * k, 2) for k in range(2, 20)]
# C2
c2 = [o for o in O if o["concept"] == "C2"]; best2 = None
for t in grid_thr:
    rec, fpr, cfa = rates([o["p_c2"] >= t for o in c2], c2)
    if fpr <= 0.05:
        key = (-rec, cfa, -t)
        if best2 is None or key < best2[0]: best2 = (key, dict(c2_thr=t, val_recall=round(rec, 3), val_fpr=round(fpr, 3), val_control_fa=round(cfa, 3)))
# C1
c1 = [o for o in O if o["concept"] == "C1"]; best1 = None
for a, g, tp, tt in itertools.product([0.3, 0.15], [1.0, 0.3], grid_thr, grid_thr):
    res = [c1_alarm(o, a, g, tp, tt) for o in c1]; rec, fpr, cfa = rates([x[0] for x in res], c1)
    if fpr <= 0.05:
        pos = [x for x, o in zip(res, c1) if o["label"] == 1]; lead = sum(x[1] for x in pos) / max(1, sum(x[0] for x in pos))
        key = (-rec, cfa, -a, -g)     # simpler = larger alpha, larger gamma (less gating)
        if best1 is None or key < best1[0]:
            best1 = (key, dict(alpha=a, gamma=g, c1_pre_thr=tp, c1_tok_thr=tt, val_recall=round(rec, 3), val_fpr=round(fpr, 3),
                               val_control_fa=round(cfa, 3), val_lead_share=round(lead, 3)))
assert best1 and best2, "no configuration meets FPR <= 5% on val -> do not freeze"
sha = {os.path.basename(f): hashlib.sha256(open(f, "rb").read()).hexdigest() for f in sorted(glob.glob(os.path.join(pdir, "*.json")))}
fz = dict(policy={**{k: best1[1][k] for k in ["alpha", "gamma", "c1_pre_thr", "c1_tok_thr"]}, "c2_thr": best2[1]["c2_thr"]},
          val_c1=best1[1], val_c2=best2[1], probes_sha256=sha, rule="EVAL_PREREG.md selection rules; FPR<=5% on val, max recall, tie -> fewer control FA -> simpler")
json.dump(fz, open(out, "w"), ensure_ascii=False, indent=1); print(json.dumps(fz, ensure_ascii=False, indent=1))
