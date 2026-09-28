"""Causal ablation of anger-selective MLP neurons.

Selection (DISCOVERY only): for every MLP neuron (down_proj input z, 36 x 9728), mean z over the style_lead tokens;
  within-scenario anger selectivity = (mean_anger - mean_others) / sd of scenario-centred activity. Top-k over all layers.
Ablation: z[u] <- neutral-condition mean of z[u] (discovery, lead tokens), applied at EVERY position incl. generation.
Evaluation (VALIDATION only; test sealed): anger / sadness / joy records, paraphrases A+B; greedy 48-token continuation
  after lead + anchor; score = L19 readout probe run by the INTACT model over the generated text (validated vs blind ratings: rho anger .71).
Controls: none; random-k (2 seeds, same per-layer counts as the target set). Sanity: k=0 must equal none exactly.
python ablate.py MODEL formal_960.jsonl probe_L19.npz OUT --tail "我的看法是："
"""
import json, argparse, os, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("data"); ap.add_argument("probe"); ap.add_argument("out")
ap.add_argument("--tail", required=True); ap.add_argument("--ks", default="32,256,1024"); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
KS = [int(x) for x in a.ks.split(",")]; EMOS = ["neutral", "joy", "sadness", "anger"]; NGEN = 48
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
NL = model.config.num_hidden_layers; NU = model.config.intermediate_size
pz = np.load(a.probe, allow_pickle=True); LR = int(pz["layer"]); CLS = [str(c) for c in pz["classes"]]
MU, SD, W, B = (torch.tensor(pz[k], device="cuda", dtype=torch.float32) for k in ["mean", "scale", "coef", "intercept"])
rows = [json.loads(l) for l in open(a.data)]
disc = [r for r in rows if r["split"] == "discovery"]; val = [r for r in rows if r["split"] == "validation"]
assert not any(r["split"] == "test" for r in disc + val)

st = {"mode": None}; cap = {}; hres = {}
def mk(L):
    def h(mod, args):
        x = args[0]
        if st["mode"] == "cap": cap[L] = x[0].float(); return None
        if st["mode"] == "ablate" and L in st["units"]:
            x = x.clone(); u = st["units"][L]; x[..., u] = st["fill"][L].to(x.dtype); return (x,)
    return h
for L in range(NL): model.model.layers[L].mlp.down_proj.register_forward_pre_hook(mk(L))
model.model.layers[LR].register_forward_hook(lambda m, i, o: hres.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0].float()))


def encode(r):
    prefix = tok.apply_chat_template([{"role": "user", "content": r["user_prompt"]}], tokenize=False, add_generation_prompt=True)
    text = prefix + r["style_lead"] + r["anchor_fact"] + a.tail
    e = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    c0, c1 = len(prefix), len(prefix) + len(r["style_lead"])
    lead = [i for i, (s, t) in enumerate(e["offset_mapping"]) if s >= c0 and t <= c1 and t > s]
    return e["input_ids"], lead


# ---- selection on discovery ----
t0 = time.time(); M = np.zeros((len(disc), NL, NU), dtype=np.float32)
for i, r in enumerate(disc):
    ids, lead = encode(r); st["mode"] = "cap"
    with torch.no_grad(): model(torch.tensor([ids], device="cuda"), use_cache=False)
    st["mode"] = None; M[i] = torch.stack([cap[L][lead].mean(0) for L in range(NL)]).cpu().numpy()
scen = np.array([r["scenario_id"] for r in disc]); emo = np.array([EMOS.index(r["emotion"]) for r in disc])
Y = M.copy()
for s in set(scen): Y[scen == s] -= M[scen == s].mean(0)
sel = (Y[emo == 3].mean(0) - Y[emo != 3].mean(0)) / (Y.std(0) + 1e-6)          # [NL, NU]
neutral_mean = torch.tensor(M[emo == 0].mean(0), device="cuda")                    # fill values [NL, NU]
order = np.argsort(-sel.ravel())
print(f"selection done {time.time()-t0:.0f}s; top units:", [(int(i // NU), int(i % NU), round(float(sel.ravel()[i]), 2)) for i in order[:8]], flush=True)


def as_units(flat):
    d = {}
    for f in flat: d.setdefault(int(f // NU), []).append(int(f % NU))
    return {L: torch.tensor(v, device="cuda") for L, v in d.items()}


def random_like(flat, seed):
    g = np.random.default_rng(seed); per = {}
    for f in flat: per[int(f // NU)] = per.get(int(f // NU), 0) + 1
    tgt = set(int(x) for x in flat); out = []
    for L, n in per.items():
        pool = np.array([u for u in range(NU) if L * NU + u not in tgt]); out += [L * NU + int(u) for u in g.choice(pool, n, replace=False)]
    return out


conds = {"none": None, "k0": []}
for k in KS:
    conds[f"top{k}"] = [int(x) for x in order[:k]]
    for sd in (1, 2): conds[f"rand{k}_s{sd}"] = random_like(order[:k], 100 * k + sd)


@torch.no_grad()
def run(r, units):
    ids, _ = encode(r)
    if units is not None:
        u = as_units(units); st.update(mode="ablate", units=u, fill={L: neutral_mean[L][u[L]] for L in u})
    g = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=NGEN, do_sample=False, temperature=None, top_p=None, top_k=None,
                       pad_token_id=tok.eos_token_id)[0].tolist()
    st["mode"] = None                                                                  # readout with the INTACT model:
    model(torch.tensor([g], device="cuda"), use_cache=False)                         # measures the emotion of the written text
    h = hres["h"][len(ids):]; p = torch.softmax(((h - MU) / SD) @ W.T + B, -1).mean(0).cpu().numpy()
    return {c: float(p[i]) for i, c in enumerate(CLS)}, tok.decode(g[len(ids):], skip_special_tokens=True)


out = []; targets = [r for r in val if r["emotion"] in ("anger", "sadness", "joy")]
for n, r in enumerate(targets):
    for c, units in conds.items():
        p, txt = run(r, units)
        out.append(dict(scenario_id=r["scenario_id"], emotion=r["emotion"], paraphrase=r["paraphrase"], condition=c, readout=p, text=txt))
    if n % 10 == 0: print(f"eval {n+1}/{len(targets)} {time.time()-t0:.0f}s", flush=True)
with open(f"{a.out}/ablate_runs.jsonl", "w") as f:
    for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")

# ---- summary: scenario-level paired differences vs none, bootstrap CI ----
rng = np.random.default_rng(0)
def boot(x):
    x = np.array(x); m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(5000)]
    return [round(float(x.mean()), 4), [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]]
idx = {(o["scenario_id"], o["emotion"], o["paraphrase"], o["condition"]): o for o in out}
res = {"ks": KS, "sanity_k0_identical": all(idx[(s, e, p, "k0")]["text"] == idx[(s, e, p, "none")]["text"]
                                               for (s, e, p, c) in idx if c == "none"),
       "top_units": [dict(layer=int(i // NU), unit=int(i % NU), sel=round(float(sel.ravel()[i]), 3)) for i in order[:30]],
       "layer_hist_top1024": np.bincount((order[:1024] // NU).astype(int), minlength=NL).tolist(), "effects": {}}
vs = sorted({o["scenario_id"] for o in out})
for emo_ in ["anger", "sadness", "joy"]:
    ch = {"anger": "anger", "sadness": "sadness", "joy": "joy"}[emo_]
    for c in conds:
        if c in ("none", "k0"): continue
        d = [np.mean([idx[(s, emo_, p, c)]["readout"][ch] - idx[(s, emo_, p, "none")]["readout"][ch] for p in "AB"]) for s in vs]
        res["effects"][f"{emo_}/{c}"] = boot(d)
res["none_means"] = {e: round(float(np.mean([idx[(s, e, p, "none")]["readout"][e] for s in vs for p in "AB"])), 4) for e in ["anger", "sadness", "joy"]}
res["seconds"] = round(time.time() - t0, 1)
json.dump(res, open(f"{a.out}/ablate_summary.json", "w"), ensure_ascii=False, indent=1)
for k in KS:
    print(f"k={k}: anger top={res['effects'][f'anger/top{k}']} rand1={res['effects'][f'anger/rand{k}_s1']} rand2={res['effects'][f'anger/rand{k}_s2']} | "
          f"sadness top={res['effects'][f'sadness/top{k}'][0]} joy top={res['effects'][f'joy/top{k}'][0]}", flush=True)
print("sanity k0 identical:", res["sanity_k0_identical"]); print("DONE")
