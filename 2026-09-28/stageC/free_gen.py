"""E01 §6 free-generation transfer (no emotional instruction, no emotional assistant prefix).

Direction d (<primary> - neutral) estimated on DISCOVERY only, at the B1 peak layer residual, anchor-token mean (same as B4).
Steering: add alpha * d_hat to the peak-layer residual at the first 32 prediction positions
          (last prefill position + next 31 decode steps); alpha = lam * ||d||, lam in {0.25, 0.5, 1.0}.
Conditions: none | target | random (same alpha, random unit dir, seed 101) | shuffled (within-scenario label shuffle, seed 201).
Decoding: greedy + sampled (temperature .7, top_p .8, top_k 20; seeds 11/22/33). 128 new tokens (full reply for fact coverage).
Strength selection uses --split validation only; --split test only after everything is frozen (plan §5.2/§6).
python free_gen.py MODEL formal_960.jsonl OUT --tail "..." --b1 b1_summary.json --split validation
"""
import json, time, argparse, os, random
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("data"); ap.add_argument("out")
ap.add_argument("--tail", required=True); ap.add_argument("--b1", required=True)
ap.add_argument("--split", default="validation", choices=["validation", "test"])
ap.add_argument("--lams", default="0.25,0.5,1.0"); ap.add_argument("--max_new", type=int, default=128)
ap.add_argument("--primary", default="joy", choices=["joy", "sadness", "anger"])
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
LAMS = [float(x) for x in a.lams.split(",")]; STEER_POS = 32
PEAK = json.load(open(a.b1))["sites"][f"resid_all/{a.primary}"]["best_layer"]
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
rows = [json.loads(l) for l in open(a.data)]
disc = [r for r in rows if r["split"] == "discovery"]
tgt_rows = {r["scenario_id"]: r for r in rows if r["split"] == a.split and r["emotion"] == "neutral" and r["paraphrase"] == "A"}
idx = {(r["scenario_id"], r["emotion"], r["paraphrase"]): r for r in disc}
dscen = sorted({r["scenario_id"] for r in disc})

st = {"mode": None, "step": 0}; cap = {}


def hook(mod, args, out):
    is_t = isinstance(out, tuple); x = out[0] if is_t else out
    if st["mode"] == "cap":
        cap["h"] = x[0, st["P"] - st["na"]:st["P"]].detach().float().clone(); return None
    if st["mode"] == "steer":
        if x.shape[1] > 1:  # prefill: steer last position only (predicts first generated token)
            st["step"] = 1; x = x.clone(); x[:, -1] += st["vec"].to(x.dtype)
        elif st["step"] < STEER_POS:
            st["step"] += 1; x = x.clone(); x[:, -1] += st["vec"].to(x.dtype)
        else:
            return None
        return (x,) + tuple(out[1:]) if is_t else x


model.model.layers[PEAK].register_forward_hook(hook)


def encode(r):
    prefix = tok.apply_chat_template([{"role": "user", "content": r["user_prompt"]}], tokenize=False, add_generation_prompt=True)
    e = tok(prefix + r["style_lead"] + r["anchor_fact"] + a.tail, return_offsets_mapping=True, add_special_tokens=False)
    a_char = len(prefix) + len(r["style_lead"])
    return e["input_ids"], len(e["input_ids"]) - next(i for i, (s, _) in enumerate(e["offset_mapping"]) if s == a_char)


@torch.no_grad()
def anchor_mean(r):
    ids, na = encode(r); st.update(mode="cap", P=len(ids), na=na)
    model(torch.tensor([ids], device="cuda"), use_cache=False); st["mode"] = None
    return cap["h"].mean(0)


# directions from DISCOVERY only
t0 = time.time(); H = {}
for sid in dscen:
    for p in ["A", "B"]:
        for e in ["joy", "sadness", "anger", "neutral"]:
            H[(sid, e, p)] = anchor_mean(idx[(sid, e, p)])
d = torch.stack([H[(s, a.primary, p)] - H[(s, "neutral", p)] for s in dscen for p in "AB"]).mean(0)
rng = random.Random(201); sh = []
for s in dscen:
    for p in "AB":
        lab = ["joy", "sadness", "anger", "neutral"]; perm = lab[:]; rng.shuffle(perm); m = dict(zip(lab, perm))
        sh.append(H[(s, m[a.primary], p)] - H[(s, m["neutral"], p)])
d_sh = torch.stack(sh).mean(0)
g = torch.Generator().manual_seed(101); d_rnd = torch.randn(d.shape[0], generator=g)
unit = lambda v: (v / v.norm()).cuda()
proj = torch.stack([v @ (d / d.norm()) for v in H.values()])
info = dict(primary=a.primary, peak_layer=PEAK, d_norm=float(d.norm()), natural_proj_std=float(proj.std()),
            cos_d_shuffled=float((d @ d_sh) / (d.norm() * d_sh.norm())), seconds_dir=round(time.time() - t0, 1))
print(json.dumps(info), flush=True)
DIRS = {"target": unit(d), "random": unit(d_rnd), "shuffled": unit(d_sh)}


@torch.no_grad()
def gen(ids, vec, seed):
    kw = dict(max_new_tokens=a.max_new, pad_token_id=tok.eos_token_id)
    if seed is None: kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    else: torch.manual_seed(seed); kw.update(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
    if vec is not None: st.update(mode="steer", vec=vec, step=0)
    out = model.generate(torch.tensor([ids], device="cuda"), **kw)[0, len(ids):].tolist(); st["mode"] = None
    return tok.decode(out, skip_special_tokens=True)


res = []
for n, (sid, r) in enumerate(sorted(tgt_rows.items())):
    ids = tok(tok.apply_chat_template([{"role": "user", "content": r["user_prompt"]}], tokenize=False, add_generation_prompt=True),
              add_special_tokens=False)["input_ids"]
    conds = [("none", None, 0.0)] + [(c, DIRS[c], lam) for lam in LAMS for c in DIRS]
    for c, u, lam in conds:
        vec = None if u is None else u * lam * float(d.norm())
        for seed in [None, 11, 22, 33]:
            res.append(dict(scenario_id=sid, split=a.split, condition=c, lam=lam, seed=seed, facts=r["facts"],
                            required_facts=r["required_facts"], reply=gen(ids, vec, seed)))
    print(f"{n+1}/{len(tgt_rows)} {sid} {time.time()-t0:.0f}s", flush=True)
with open(f"{a.out}/free_gen_{a.split}.jsonl", "w") as f:
    for x in res: f.write(json.dumps(x, ensure_ascii=False) + "\n")
info.update(n=len(res), seconds=round(time.time() - t0, 1), peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
json.dump(info, open(f"{a.out}/free_gen_{a.split}_info.json", "w"), indent=1); print("DONE", json.dumps(info))
