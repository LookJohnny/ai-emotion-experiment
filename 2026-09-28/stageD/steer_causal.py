"""§5b continuous steering with the B3 head set (see STEER_PREREG.md). Primary contrast: anger vs neutral.
  --mode select --split validation : lambda grid {1,2,4,8}, 2 random seeds, paraphrases A+B
  --mode final  --split test --lam X --freeze FREEZE_5b.json : 5 random seeds, paraphrase A
python steer_causal.py MODEL formal_960.jsonl OUT --tail "我的看法是：" --b3 b3_summary.json --mode select --split validation
"""
import json, time, argparse, os, random
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("data"); ap.add_argument("out")
ap.add_argument("--tail", required=True); ap.add_argument("--b3", required=True); ap.add_argument("--primary", default="anger")
ap.add_argument("--mode", required=True, choices=["select", "final"]); ap.add_argument("--split", required=True, choices=["validation", "test"])
ap.add_argument("--lam", type=float); ap.add_argument("--freeze")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
assert (a.mode == "select") == (a.split == "validation")
if a.mode == "final": assert a.freeze and os.path.exists(a.freeze) and a.lam is not None
LAMS = [1.0, 2.0, 4.0, 8.0] if a.mode == "select" else [a.lam]
SEEDS = [101, 102] if a.mode == "select" else [101, 102, 103, 104, 105]
PARAS = ["A", "B"] if a.mode == "select" else ["A"]; MAX_NEW = 64
b3 = json.load(open(a.b3)); assert b3["primary_object_rule"]["head_set_is_primary"]
HEADS = [(int(x["head"][1:].split("H")[0]), int(x["head"].split("H")[1])) for x in b3["top_heads"][:b3["k_for_50pct"]]]

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
HD = model.config.head_dim; NL = model.config.num_hidden_layers
rows = [json.loads(l) for l in open(a.data)]
idx = {(r["scenario_id"], r["emotion"], r["paraphrase"]): r for r in rows}
disc = sorted({r["scenario_id"] for r in rows if r["split"] == "discovery"})
tgt = sorted({r["scenario_id"] for r in rows if r["split"] == a.split})
LAYERS = sorted({L for L, _ in HEADS})


def encode(r):
    prefix = tok.apply_chat_template([{"role": "user", "content": r["user_prompt"]}], tokenize=False, add_generation_prompt=True)
    e = tok(prefix + r["style_lead"] + r["anchor_fact"] + a.tail, return_offsets_mapping=True, add_special_tokens=False)
    a_char = len(prefix) + len(r["style_lead"])
    return e["input_ids"], len(e["input_ids"]) - next(i for i, (s, _) in enumerate(e["offset_mapping"]) if s == a_char)


st = {"mode": None}; cap = {}
def hook(L):
    def h(mod, args):
        x = args[0]
        if st["mode"] == "cap":
            cap[L] = x[0, -st["na"]:].float().mean(0); return None
        if st["mode"] == "steer" and L in st["vec"]:
            x = x.clone()
            if x.shape[1] > 1: x[0, -st["na"]:] += st["vec"][L].to(x.dtype)   # prefill: anchor positions
            else: x[0, -1] += st["vec"][L].to(x.dtype)                          # every generated position
            return (x,)
    return h
for L in LAYERS: model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(hook(L))


@torch.no_grad()
def head_means(r):
    ids, na = encode(r); st.update(mode="cap", na=na); model(torch.tensor([ids], device="cuda"), use_cache=False); st["mode"] = None
    return {L: cap[L].clone() for L in LAYERS}


# steering vectors from DISCOVERY (true labels and within-scenario shuffled labels)
t0 = time.time(); Hm = {}
for s in disc:
    for p in "AB":
        for e in ["joy", "sadness", "anger", "neutral"]: Hm[(s, e, p)] = head_means(idx[(s, e, p)])
def diff(pairs):
    return {L: torch.stack([Hm[x][L] - Hm[y][L] for x, y in pairs]).mean(0) for L in LAYERS}
true_d = diff([((s, a.primary, p), (s, "neutral", p)) for s in disc for p in "AB"])
rng = random.Random(201); sp = []
for s in disc:
    for p in "AB":
        lab = ["joy", "sadness", "anger", "neutral"]; perm = lab[:]; rng.shuffle(perm); m = dict(zip(lab, perm))
        sp.append(((s, m[a.primary], p), (s, m["neutral"], p)))
shuf_d = diff(sp)
Wo = {L: model.model.layers[L].self_attn.o_proj.weight.detach().float() for L in LAYERS}


def head_vecs(dsrc, heads, lam, match=None, gen=None):
    """vector per layer (4096-dim o_proj input); only the given heads' slices are non-zero.
       match: per-head target residual-change norms -> rescale each head's slice to that norm; gen: random directions."""
    V = {L: torch.zeros(Wo[L].shape[1], device="cuda") for L in LAYERS}; norms = {}; scales = []
    for L, h in heads:
        sl = slice(h * HD, (h + 1) * HD)
        v = torch.randn(HD, generator=gen).cuda() if gen is not None else lam * dsrc[L][sl]
        n = float((Wo[L][:, sl] @ v).norm()); norms[(L, h)] = n
        if match is not None:
            sc = match[(L, h)] / n if n > 0 else 0.0; v = v * sc; scales.append(sc)
        V[L][sl] = v
    return V, norms, scales


out = []; info_scales = []
for lam in LAMS:
    Vt, tn, _ = head_vecs(true_d, HEADS, lam)
    Vs, _, ss = head_vecs(shuf_d, HEADS, lam, match=tn)
    Vr = []
    for sd in SEEDS:
        g = torch.Generator().manual_seed(sd); v, _, s_ = head_vecs(None, HEADS, lam, match=tn, gen=g); Vr.append(v)
    info_scales.append(dict(lam=lam, target_resid_norm=round(sum(tn.values()), 3), shuffled_scale_median=round(float(np.median(ss)), 3)))
    for n, sid in enumerate(tgt):
        for para in PARAS:
            nid, na = encode(idx[(sid, "neutral", para)])
            conds = [("target", Vt), ("shuffled", Vs)] + [(f"random_{i}", v) for i, v in enumerate(Vr)]
            for c, V in conds:
                st.update(mode="steer", vec=V, na=na)
                with torch.no_grad():
                    g = model.generate(torch.tensor([nid], device="cuda"), max_new_tokens=MAX_NEW, do_sample=False,
                                       temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)[0, len(nid):].tolist()
                st["mode"] = None
                out.append(dict(scenario_id=sid, split=a.split, paraphrase=para, condition=c, lam=lam, scale=None,
                                continuation=tok.decode(g, skip_special_tokens=True)))
    print(f"lam {lam} done {time.time()-t0:.0f}s", flush=True)
for sid in tgt:  # lambda-independent conditions
    for para in PARAS:
        for c, emo in [("none", "neutral"), ("primary_prefix", a.primary)]:
            ids, _ = encode(idx[(sid, emo, para)])
            with torch.no_grad():
                g = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=MAX_NEW, do_sample=False,
                                   temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)[0, len(ids):].tolist()
            out.append(dict(scenario_id=sid, split=a.split, paraphrase=para, condition=c, lam=0.0 if c == "none" else 1.0, scale=None,
                            continuation=tok.decode(g, skip_special_tokens=True)))
fn = f"{a.out}/steer_{a.mode}_{a.split}.jsonl"
with open(fn, "w") as f:
    for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
info = dict(object="heads_continuous_steering", heads=[f"L{L}H{h}" for L, h in HEADS], primary=a.primary, lams=LAMS, seeds=SEEDS,
            paraphrases=PARAS, n=len(out), scales=info_scales, seconds=round(time.time() - t0, 1),
            peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
json.dump(info, open(fn.replace(".jsonl", "_info.json"), "w"), indent=1); print("DONE", json.dumps(info))
