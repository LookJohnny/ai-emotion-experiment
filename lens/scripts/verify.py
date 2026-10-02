"""Check that the Lens engine (one forward pass, any device) reproduces the pre-registered reader experiment.
python scripts/verify.py MODEL_DIR RESEARCH_READER_DIR [limit]
  RESEARCH_READER_DIR = 2026-09-28/monitor/reader (kimi_*_test.jsonl, reader_*_test.jsonl, labels_*.jsonl)"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nous_lens.engine import Lens

MODEL, RD = sys.argv[1:3]; LIM = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
J = lambda f: {r["item_id"]: r for r in map(json.loads, open(os.path.join(RD, f), encoding="utf-8"))}


def auroc(y, s):
    pos = [v for v, t in zip(s, y) if t]; neg = [v for v, t in zip(s, y) if not t]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def corr(a, b):
    n = len(a); ma, mb = sum(a) / n, sum(b) / n
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** .5


lens = Lens(MODEL, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nous_lens", "pack"))
print("device", lens.device, "load", lens.load_s, "s", flush=True)
K, S, L = J("kimi_c1_test.jsonl"), J("reader_c1_test.jsonl"), J("labels_c1.jsonl")
ids = [] if os.environ.get("ONLY") == "c3h" else [k for k in L if k in S and "reader_c1" in S[k]][:LIM]; mine, ref, y, t0 = [], [], [], time.time()
for k in ids:
    r = lens.read(K[k]["messages"], K[k]["content"], fabrication=False); mine.append(r["hostility"]["score"]); ref.append(S[k]["reader_c1"]); y.append(L[k]["label"])
if ids: print(f"C1 n={len(ids)} pos={sum(y)} AUROC lens={auroc(y, mine):.3f} research={auroc(y, ref):.3f} r(lens,research)={corr(mine, ref):.4f} "
      f"max|diff|={max(abs(a - b) for a, b in zip(mine, ref)):.4f} {(time.time() - t0) / len(ids):.2f}s/item", flush=True)
K, S, L = J("kimi_c3h_test.jsonl"), J("reader_c3h_test.jsonl"), J("labels_c3h.jsonl")
ids = [k for k in L if L[k]["grade"] in ("CORRECT", "INCORRECT")][:LIM]; mine, ref, y = [], [], []
for k in ids:
    r = lens.read([K[k]["messages"][-1]], K[k]["content"], fabrication="always"); mine.append(r["fabrication"]["score"]); ref.append(S[k]["reader_c3h"]); y.append(L[k]["grade"] == "INCORRECT")
print(f"C3H n={len(ids)} inc={sum(y)} AUROC lens={auroc(y, mine):.3f} research={auroc(y, ref):.3f} r={corr(mine, ref):.4f}", flush=True)
