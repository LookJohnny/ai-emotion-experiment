"""Score a readout probe on labelled short lines (sentence-level use, as avatars read speech).
  python -m nous_tone.eval_lines MODEL_DIR READOUT_DIR [READOUT_DIR ...] [--lines eval/companion_lines.json]
Prints accuracy, per-class recall and mean probability of the labelled class, per probe."""
import argparse, json, os
import numpy as np
from .engine import ToneEngine


def score(eng, items):
    hits, recall, p_true = 0, {}, []
    for it in items:
        r = eng.read(it["text"], user=it.get("user", ""))["readout"]
        r = {("neutral" if k == "neutral" else k): v for k, v in r.items()}
        pred = max(r, key=r.get); ok = pred == it["label"]; hits += ok
        recall.setdefault(it["label"], []).append(ok); p_true.append(r.get(it["label"], 0.0))
    return {"accuracy": round(hits / len(items), 3), "mean_p_label": round(float(np.mean(p_true)), 3),
            "recall": {k: round(float(np.mean(v)), 2) for k, v in recall.items()}}


def main():
    ap = argparse.ArgumentParser(prog="nous_tone.eval_lines"); ap.add_argument("model"); ap.add_argument("readouts", nargs="+")
    ap.add_argument("--lines", default=os.path.join(os.path.dirname(__file__), "..", "eval", "companion_lines.json"))
    a = ap.parse_args(); items = json.load(open(a.lines, encoding="utf-8"))["items"]
    eng = ToneEngine(a.model)
    for d in a.readouts:
        eng.probe = None; eng.set_readout(d)
        print(d, json.dumps(score(eng, items), ensure_ascii=False))


if __name__ == "__main__":
    main()
