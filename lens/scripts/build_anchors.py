"""Build the adaptation anchors shipped with Nous Lens: 300 unlabeled dev texts per concept (seed 7, as in the transfer study's
small-anchor variant) and the reference model's pooled states on them. Labels are not stored.
python scripts/build_anchors.py RESEARCH_ROOT REF_MODEL_DIR"""
import json, os, sys
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nous_lens.engine import prompt_ids, pick_device
R, M = sys.argv[1:3]; OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nous_lens", "anchors"); os.makedirs(OUT, exist_ok=True)
J = lambda p: [json.loads(l) for l in open(os.path.join(R, p), encoding="utf-8")]
src = {"c1": [r for r in J("2026-09-28/monitor/v1_1/c1/items_all.jsonl") if r["split"] == "dev"],
       "sy": [r for r in J("2026-10-02/syco/qwen_replies.jsonl") if r["split"] == "dev"],
       "fa": [r for r in J("2026-09-28/monitor/c3h/answers.jsonl") if r["split"] == "dev"]}
LAYER = {"c1": 16, "sy": 20, "fa": 19}; CUT = {"c1": 120, "sy": 120, "fa": 64}
anchors = []
for c, rows in src.items():
    for i in np.random.default_rng(7).permutation(len(rows))[:300]:
        r = rows[i]; anchors.append({"concept": c, "system": r["system"], "messages": r["messages"], "reply": r["target_text"]})
open(os.path.join(OUT, "anchors.jsonl"), "w", encoding="utf-8").write("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in anchors))
dev = pick_device(); tok = AutoTokenizer.from_pretrained(M)
model = AutoModel.from_pretrained(M, dtype=torch.bfloat16 if dev != "cpu" else torch.float32).to(dev).eval(); st = {}
for L in set(LAYER.values()): model.layers[L].register_forward_hook(lambda m, a, o, L=L: st.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[0].float()))
out = {c: [] for c in src}
with torch.no_grad():
    for k, a in enumerate(anchors):
        c = a["concept"]; p = prompt_ids(tok, [{"role": "system", "content": a["system"]}] + a["messages"]); rid = tok(a["reply"], add_special_tokens=False)["input_ids"][:CUT[c]]
        model(torch.tensor([p + rid], device=dev), use_cache=False); out[c].append(st[LAYER[c]][len(p):len(p) + len(rid)].mean(0).cpu().numpy())
        if k % 100 == 0: print(k, len(anchors), flush=True)
np.savez_compressed(os.path.join(OUT, "ref_states.npz"), ref_layers_total=np.array(model.config.num_hidden_layers),
                    **{c: np.stack(v).astype(np.float16) for c, v in out.items()})
print("ok", {c: len(v) for c, v in out.items()})
