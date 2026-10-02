"""Engineering check of an (auto-adapted) pack on the labeled TEST sets of the transfer study (never used for adaptation).
python scripts/eval_pack.py MODEL_DIR PACK_DIR DATA_DIR   (DATA_DIR: c1_test/sy_test/fa_test/*_kimi .jsonl + labels.json)"""
import json, os, sys
import numpy as np, torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nous_lens.engine import prompt_ids, pick_device
M, P, D = sys.argv[1:4]
man = json.load(open(os.path.join(P, "manifest.json"))); lab = json.load(open(os.path.join(D, "labels.json")))
PR = {"c1": "c1_token", "sy": "syco_token", "fa": "c3h_token"}; CUT = {"c1": 120, "sy": 120, "fa": 64}
probes = {c: json.load(open(os.path.join(P, man["probes"][k]))) for c, k in PR.items() if k in man["probes"]}
cfg = AutoConfig.from_pretrained(M); cfg.num_hidden_layers = max(p["layer"] for p in probes.values()) + 1
if getattr(cfg, "layer_types", None): cfg.layer_types = cfg.layer_types[:cfg.num_hidden_layers]
dev = pick_device(); tok = AutoTokenizer.from_pretrained(M)
model = AutoModel.from_pretrained(M, config=cfg, dtype=torch.bfloat16 if dev != "cpu" else torch.float32).to(dev).eval(); st = {}
for L in {p["layer"] for p in probes.values()}: model.layers[L].register_forward_hook(lambda m, a, o, L=L: st.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[0].float()))


def auroc(y, s):
    y, s = np.array(y), np.array(s); pos, neg = s[y == 1], s[y == 0]
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


out = {}
with torch.no_grad():
    for c, p in probes.items():
        mu, sd, w, b = (np.array(p[k]) for k in ("mean", "scale", "coef")) + (p["intercept"],) if False else (np.array(p["mean"]), np.array(p["scale"]), np.array(p["coef"]), p["intercept"])
        for split in ("test", "kimi"):
            ys, ss = [], []
            for l in open(os.path.join(D, f"{c}_{split}.jsonl"), encoding="utf-8"):
                r = json.loads(l); y = lab.get(f"{c}:{r['item_id']}")
                if y is None: continue
                reply = r.get("target_text") or r.get("content") or ""
                pi = prompt_ids(tok, [{"role": "system", "content": r["system"]}] + r["messages"]); ri = tok(reply, add_special_tokens=False)["input_ids"][:CUT[c]] or [tok.eos_token_id]
                model(torch.tensor([pi + ri], device=dev), use_cache=False); h = st[p["layer"]][len(pi):len(pi) + len(ri)].mean(0).cpu().numpy()
                ys.append(y); ss.append(float(1 / (1 + np.exp(-(((h - mu) / sd) @ w + b)))))
            out[f"{c}_{split}"] = {"n": len(ys), "pos": int(sum(ys)), "auroc": round(auroc(ys, ss), 3)}
            print(c, split, out[f"{c}_{split}"], flush=True)
json.dump(out, open(os.path.join(P, "eval.json"), "w"), indent=1)
