"""Automatic adaptation: build a Nous probe pack for ANY local chat model (HF folder or GGUF / Ollama blob), with no labels (REPORT_TRANSFER.md).
The new model and the reference (Qwen3-4B-Instruct-2507) have read the same unlabeled anchor texts; a ridge map from the new
model's states to the reference's is fitted and folded into the reference probe, giving an ordinary linear probe on the new
model (same format as any pack). Thresholds carry over because scores live on the reference probe's scale.
  python -m nous_lens.adapt --model /path/to/model --out ~/.nous-lens/packs/<name>
Validated (5090, 7-9B models): hostility non-inferior to native retraining on 4/4 models; fabrication/sycophancy experimental."""
import argparse, json, os, sys, time
import numpy as np, torch
from .engine import prompt_ids, pick_device
from .loader import config_of, load_layers

HERE = os.path.dirname(os.path.abspath(__file__))
CUT = {"c1": 120, "sy": 120, "fa": 64}
PROBE = {"c1": "c1_token", "sy": "syco_token", "fa": "c3h_token"}
THR = {"c1": "C1_mean", "sy": "SYCO_mean", "fa": "C3H_mean"}
ALPHAS = (1, 10, 100, 1000)


def log(*a): print(*a, flush=True)


def ridge_dual(Z, Y, a):
    """W = Z^T (Z Z^T + a I)^-1 Y  (n << d)."""
    K = Z @ Z.T; return Z.T @ np.linalg.solve(K + a * np.eye(len(K)), Y)


def fit_map(X, Y):
    mu, sd = X.mean(0), X.std(0) + 1e-6; Z = (X - mu) / sd; yb = Y.mean(0); Yc = Y - yb
    folds = np.array_split(np.random.default_rng(1).permutation(len(Z)), 5); best = None
    for a in ALPHAS:
        r2 = []
        for f in folds:
            tr = np.setdiff1d(np.arange(len(Z)), f); W = ridge_dual(Z[tr], Yc[tr], a); P = Z[f] @ W
            r2.append(1 - ((Yc[f] - P) ** 2).sum() / ((Yc[f] - Yc[f].mean(0)) ** 2).sum())
        if best is None or np.mean(r2) > best[1]: best = (a, float(np.mean(r2)))
    return mu, sd, ridge_dual(Z, Yc, best[0]), yb, best


def adapt(model_dir, out_dir, ref_pack=None, device=None, name=None):
    ref_pack = ref_pack or os.path.join(HERE, "pack"); rm = json.load(open(os.path.join(ref_pack, "manifest.json"), encoding="utf-8"))
    anchors = [json.loads(l) for l in open(os.path.join(HERE, "anchors", "anchors.jsonl"), encoding="utf-8")]
    ref = np.load(os.path.join(HERE, "anchors", "ref_states.npz"))
    cfg, N = config_of(model_dir); refN = int(ref["ref_layers_total"])
    rp = {c: json.load(open(os.path.join(ref_pack, rm["probes"][PROBE[c]]))) for c in CUT}
    La = {c: round(rp[c]["layer"] / refN * N) for c in CUT}
    log(f"STEP 1/3 加载模型（{N} 层；读第 {sorted(set(La.values()))} 层）")
    device = device or pick_device(); dt = torch.float32 if device == "cpu" else torch.bfloat16
    tok, model = load_layers(model_dir, max(La.values()) + 1, device, dt)
    states = {}
    for L in set(La.values()): model.layers[L].register_forward_hook(lambda m, a, o, L=L: states.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[0].float()))
    log("STEP 2/3 读锚点文本（不需要任何标注）")
    X = {c: [] for c in CUT}; t0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(anchors):
            c = r["concept"]; pids = prompt_ids(tok, [{"role": "system", "content": r["system"]}] + r["messages"])
            rids = tok(r["reply"], add_special_tokens=False)["input_ids"][:CUT[c]] or [tok.eos_token_id]
            model(torch.tensor([pids + rids], device=device), use_cache=False)
            X[c].append(states[La[c]][len(pids):len(pids) + len(rids)].mean(0).cpu().numpy())
            if i % 50 == 0: log(f"PROGRESS {i}/{len(anchors)} {time.time() - t0:.0f}s")
    log("STEP 3/3 拟合映射，生成探针包")
    os.makedirs(out_dir, exist_ok=True); probes, quality = {}, {}
    for c in CUT:
        Xc = np.stack(X[c]).astype(np.float64); Yc = ref[c].astype(np.float64)
        mu, sd, W, yb, (alpha, r2) = fit_map(Xc, Yc)
        p = rp[c]; w = np.array(p["coef"]) / np.array(p["scale"])
        coef = W @ w; b = float((yb - np.array(p["mean"])) @ w + p["intercept"])          # fold map + reference probe into one linear probe
        fn = PROBE[c] + ".json"
        json.dump({"concept": p["concept"], "name": PROBE[c], "layer": La[c], "pool": p.get("pool"), "mean": mu.tolist(), "scale": sd.tolist(),
                   "coef": coef.tolist(), "intercept": b, "threshold": 0.5, "meta": {"auto_adapted": True, "ridge_alpha": alpha, "anchor_r2": round(r2, 3)}},
                  open(os.path.join(out_dir, fn), "w"))
        probes[PROBE[c]] = fn; quality[c] = {"layer": La[c], "alpha": alpha, "anchor_r2": round(r2, 3)}
    name = name or os.path.basename(os.path.normpath(model_dir))
    json.dump({"name": name, "base_model": name, "model_dir": model_dir, "version": time.strftime("%Y.%m.%d"),
               "auto_adapted": True, "adapted_from": rm.get("base_model"), "probes": probes,
               "thresholds": {THR[c]: rm["thresholds"][THR[c]] for c in CUT}, "neg_quantiles": {THR[c]: rm["neg_quantiles"][THR[c]] for c in CUT},
               "quality": quality, "note": "Auto-adapted by label-free alignment. Hostility validated as non-inferior to native retraining on 4 open models; fabrication and sycophancy are experimental."},
              open(os.path.join(out_dir, "manifest.json"), "w"), ensure_ascii=False, indent=1)
    log(f"DONE {out_dir} {json.dumps(quality)}")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True); ap.add_argument("--device"); ap.add_argument("--name")
    a = ap.parse_args(); adapt(a.model, os.path.expanduser(a.out), device=a.device, name=a.name)
