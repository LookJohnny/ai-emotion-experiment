"""Diagnose self-substitution mismatch: compare layer states from use_cache=False forward vs generate() prefill."""
import json, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL_DIR, DATA = sys.argv[1:3]
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
rows = [json.loads(l) for l in open(DATA)]
r = next(x for x in rows if x["split"] == "validation" and x["emotion"] == "neutral")
prefix = tok.apply_chat_template([{"role": "user", "content": r["user_prompt"]}], tokenize=False, add_generation_prompt=True)
ids = tok(prefix + r["style_lead"] + r["common_anchor"], add_special_tokens=False)["input_ids"]
cap = {}
def mk(L, tag):
    def h(m, a, o):
        x = o[0] if isinstance(o, tuple) else o
        if x.shape[1] > 1: cap[(tag, L)] = x[0].float().cpu()
    return h
hs = []
for tag in ["nocache", "gen"]:
    hs = [model.model.layers[L].register_forward_hook(mk(L, tag)) for L in range(36)]
    with torch.no_grad():
        if tag == "nocache": model(torch.tensor([ids], device="cuda"), use_cache=False)
        else: model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=1, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
    for h in hs: h.remove()
for L in [0, 8, 17, 26, 35]:
    d = (cap[("nocache", L)] - cap[("gen", L)]).abs()
    print(f"L{L}: max|diff|={d.max():.4g}  mean|diff|={d.mean():.3g}  max|x|={cap[('gen',L)].abs().max():.4g}  identical={bool(d.max()==0)}")
