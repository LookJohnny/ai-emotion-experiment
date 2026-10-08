"""Terminal demo: the reply streams while each tone's readout bar and lambda update live.
  python -m nous_tone.demo MODEL_DIR packs/... --say "老板让我周末加班。" [--tone anger=-1] [--target warmth=0.8]"""
import argparse, sys
from .engine import ToneEngine


def kv(items):
    out = {}
    for s in items or []:
        k, v = s.split("="); out[k] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser(prog="nous_tone.demo"); ap.add_argument("model"); ap.add_argument("packs", nargs="+")
    ap.add_argument("--say", required=True); ap.add_argument("--tone", nargs="*"); ap.add_argument("--target", nargs="*")
    ap.add_argument("--max", type=int, default=120)
    a = ap.parse_args(); eng = ToneEngine(a.model, a.packs); tone, target = kv(a.tone), kv(a.target)
    for label, t, g in (("不调控", {}, {}), ("调控", tone, target)):
        if label == "调控" and not (tone or target): break
        print(f"\n=== {label} {t or ''}{(' 目标 ' + str(g)) if g else ''}")
        text, last = "", None
        for ev in eng.stream([{"role": "user", "content": a.say}], tone=t, target=g, max_new_tokens=a.max):
            if ev.get("done"): break
            text += ev["delta"]; last = ev; sys.stdout.write(ev["delta"]); sys.stdout.flush()
        if last:
            print("\n" + "  ".join(f"{k} {'█' * int(v * 10):<10} {v:.2f}" for k, v in last["readout"].items()),
                  "| λ", last["lambda"] or "-", "| 表情", {k: v for k, v in last["expression"].items() if v})


if __name__ == "__main__":
    main()
