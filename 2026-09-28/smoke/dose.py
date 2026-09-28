"""Dose / positive-control check (debug only): is a single-position top-32 patch too weak by design?
Conditions on neutral recipient, donor = joy of same scenario:
  top32 x {1,4,16} @ last anchor token; all 9728 units @ last anchor token;
  all 9728 units @ all anchor tokens; full residual (layer output) @ all anchor tokens.
"""
import json, sys, numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR, RUN, DATA = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
rows = [json.loads(l) for l in open(DATA)]
sets = json.load(open(f"{RUN}/unit_sets.json"))
HELD = {"d09", "d10", "d11", "d12"}


def encode(r):
    prefix = tok.apply_chat_template([{"role": "user", "content": r["user_prompt"]}], tokenize=False, add_generation_prompt=True)
    ids = tok(prefix + r["style_lead"] + r["common_anchor"], add_special_tokens=False)["input_ids"]
    n_anchor = len(tok(r["common_anchor"], add_special_tokens=False)["input_ids"])
    return ids, n_anchor


cap, st = {}, {"mode": None}


def mlp_hook(L):
    def h(mod, args):
        x = args[0]
        if x.shape[1] == 1: return None
        if st["mode"] == "cap": cap[("z", L)] = x[0].detach().clone(); return None
        if st["mode"] == "patch_z" and st["L"] == L:
            x = x.clone(); n = st["npos"]; S = st["units"]
            src = st["donor"][-n:]
            if S is None: x[0, -n:] = x[0, -n:] + st["coef"] * (src - x[0, -n:])
            else: x[0, -n:, S] = x[0, -n:, S] + st["coef"] * (src[:, S] - x[0, -n:, S])
            return (x,)
    return h


def layer_hook(L):
    def h(mod, args, out):
        hs = out[0] if isinstance(out, tuple) else out
        if hs.shape[1] == 1: return None
        if st["mode"] == "cap": cap[("h", L)] = hs[0].detach().clone(); return None
        if st["mode"] == "patch_h" and st["L"] == L:
            hs = hs.clone(); n = st["npos"]; hs[0, -n:] = st["donor"][-n:]
            return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
    return h


LAYERS = [17, 26]
for L in LAYERS:
    model.model.layers[L].mlp.down_proj.register_forward_pre_hook(mlp_hook(L))
    model.model.layers[L].register_forward_hook(layer_hook(L))


def run(ids):
    with torch.no_grad():
        lg = model(torch.tensor([ids], device="cuda"), use_cache=False).logits[0, -1].float()
        g = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=48, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
    return lg, tok.decode(g[0, len(ids):], skip_special_tokens=True)


out = []
by = {}
for r in rows: by.setdefault(r["scenario_id"], {})[r["emotion"]] = r
for sid, v in by.items():
    nid, na = encode(v["neutral"]); jid, _ = encode(v["joy"])
    st["mode"] = "cap"; run(jid); donor = dict(cap)
    st["mode"] = None; base_lg, base_txt = run(nid)
    joy_lg, joy_txt = run(jid)
    p0 = F.log_softmax(base_lg, -1)
    kl_joy = float(F.kl_div(p0, F.log_softmax(joy_lg, -1), log_target=True, reduction="sum"))
    rec = dict(scenario=sid, held_out=sid in HELD, n_anchor_tokens=na, base=base_txt, joy_prefix_itself=joy_txt, kl_joy_prefix=kl_joy, conds={})
    for L in LAYERS:
        top = torch.tensor(sets[str(L)]["target"], device="cuda")
        conds = [(f"top32_x{c}_last", "patch_z", top, c, 1) for c in (1, 4, 16)] + \
                [("allunits_last", "patch_z", None, 1.0, 1), ("allunits_anchor", "patch_z", None, 1.0, na), ("resid_anchor", "patch_h", None, 1.0, na)]
        for name, mode, S, c, npos in conds:
            st.update(mode=mode, L=L, units=S, coef=c, npos=npos, donor=donor[("z" if mode == "patch_z" else "h", L)])
            lg, txt = run(nid)
            st["mode"] = None
            kl = float(F.kl_div(p0, F.log_softmax(lg, -1), log_target=True, reduction="sum"))
            rec["conds"][f"L{L}_{name}"] = dict(kl=round(kl, 4), text=txt)
    out.append(rec)
    print(sid, "kl_joy_prefix=%.3f" % kl_joy, {k: v["kl"] for k, v in rec["conds"].items()}, flush=True)
json.dump(out, open(f"{RUN}/dose_check.json", "w"), ensure_ascii=False, indent=1)
keys = list(out[0]["conds"])
print("\nMEAN KL vs neutral baseline (reference: full joy prefix = %.3f)" % np.mean([o["kl_joy_prefix"] for o in out]))
for k in keys:
    print(k.ljust(26), "%.4f" % np.mean([o["conds"][k]["kl"] for o in out]), " text changed: %d/12" % sum(o["conds"][k]["text"] != o["base"] for o in out))
