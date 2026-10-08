"""Calibrate target mode from held-out generations (nous_tone.evaluate output with a readout probe).
  python -m nous_tone.calibrate EVAL.jsonl packs/attr1 packs/attr2 ...
For each pack: the mean final readout of its attribute at each strength (strength 0 = the untouched replies) is stored as
pack.meta["calibration"] = {"strength": [...], "readout": [...]}; the engine inverts it to pick a strength for a target."""
import json, sys
from collections import defaultdict
import numpy as np
from .pack import TonePack


def main():
    ev, packs = sys.argv[1], sys.argv[2:]
    rows = [json.loads(l) for l in open(ev, encoding="utf-8")]
    for d in packs:
        p = TonePack.load(d); a = p.attribute; by = defaultdict(list)
        for r in rows:
            if not r.get("readout") or a not in r["readout"]: continue
            if r["attribute"] == "none": by[0.0].append(r["readout"][a])
            elif r["attribute"] == a: by[float(r["strength"])].append(r["readout"][a])
        s = sorted(by); p.meta["calibration"] = {"strength": s, "readout": [round(float(np.mean(by[k])), 4) for k in s], "n": [len(by[k]) for k in s]}
        p.save(d); print(a, p.meta["calibration"])


if __name__ == "__main__":
    main()
