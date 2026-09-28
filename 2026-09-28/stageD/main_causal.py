"""E01 §5 main causal experiment (primary contrast from EXPERIMENT.md v2.1: anger vs neutral).

Object (plan §5.2 rule, from B3): head set H = B3 top-k(50%) heads if k50<=16 and top16 Jaccard>=0.5, else residual direction d (B4/B1 peak).
Recipient: neutral style_lead + anchor (+ tail). Donor: PRIMARY-emotion style_lead + same anchor. Patch at all anchor positions, prefill only.
  heads     : o_proj input slice x[h] <- x[h] + lam*(donor[h] - x[h])
  direction : h <- h + lam*((h_donor - h).d_hat) d_hat at peak-layer residual
Controls: random (same count / same layers for heads; random unit dir for direction), coefficient rescaled so the residual-stream
  change norm matches the target per item (scale reported; >4 flagged OOD); shuffled (direction object only: label-shuffled d).
Also: none, anchor_sub (all layers resid substitution = anchor-mediated ceiling), primary_prefix (prefix ceiling).
Donor/self states captured through the SAME generate() prefill path used for patched generation (stage-A lesson).
modes:
  --mode select  --split validation : lam grid, 2 random sets, paraphrases A+B  -> λ selection by blind rating
  --mode final   --split test --lam X --freeze FREEZE.json : 5 random sets, paraphrase A, 10 conditions (400 continuations)
"""
import json, time, argparse, os, random
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("data"); ap.add_argument("out")
ap.add_argument("--tail", required=True); ap.add_argument("--primary", default="anger", choices=["joy", "sadness", "anger"])
ap.add_argument("--b1", required=True); ap.add_argument("--b3", required=True)
ap.add_argument("--mode", required=True, choices=["select", "final"]); ap.add_argument("--split", required=True, choices=["validation", "test"])
ap.add_argument("--lam", type=float); ap.add_argument("--freeze")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
assert (a.mode == "select") == (a.split == "validation"), "select on validation only; final on test only"
if a.mode == "final": assert a.freeze and os.path.exists(a.freeze) and a.lam is not None, "test requires a freeze file and a frozen lambda"
MAX_NEW = 64

b3 = json.load(open(a.b3)); rule = b3["primary_object_rule"]
OBJ = "heads" if rule["head_set_is_primary"] else "direction"
PEAK = json.load(open(a.b1))["sites"][f"resid_all/{a.primary}"]["best_layer"]
if OBJ == "heads":
    K = b3["k_for_50pct"]; HEADS = [(int(x["head"][1:].split("H")[0]), int(x["head"].split("H")[1])) for x in b3["top_heads"][:K]]
    WIN = list(range(b3["window_layers"][0], b3["window_layers"][1] + 1))
LAMS = ([0.5, 1.0] if OBJ == "heads" else [0.5, 1, 2, 4]) if a.mode == "select" else [a.lam]
N_RAND = 2 if a.mode == "select" else 5
PARAS = ["A", "B"] if a.mode == "select" else ["A"]

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
NL = model.config.num_hidden_layers; HD = model.config.head_dim; NH = model.config.num_attention_heads
rows = [json.loads(l) for l in open(a.data)]
idx = {(r["scenario_id"], r["emotion"], r["paraphrase"]): r for r in rows}
disc = sorted({r["scenario_id"] for r in rows if r["split"] == "discovery"})
target_scen = sorted({r["scenario_id"] for r in rows if r["split"] == a.split})


def encode(r):
    prefix = tok.apply_chat_template([{"role": "user", "content": r["user_prompt"]}], tokenize=False, add_generation_prompt=True)
    e = tok(prefix + r["style_lead"] + r["anchor_fact"] + a.tail, return_offsets_mapping=True, add_special_tokens=False)
    a_char = len(prefix) + len(r["style_lead"])
    return e["input_ids"], len(e["input_ids"]) - next(i for i, (s, _) in enumerate(e["offset_mapping"]) if s == a_char)


st = {"mode": None}; cap = {}


def o_hook(L):
    def h(mod, args):
        x = args[0]
        if x.shape[1] == 1: return None
        if st["mode"] == "cap": cap[("o", L)] = x[0, -st["na"]:].detach().clone(); return None
        if st["mode"] == "heads" and L in st["hl"]:
            x = x.clone()
            for hh in st["hl"][L]:
                sl = slice(hh * HD, (hh + 1) * HD); cur = x[0, -st["na"]:, sl]
                x[0, -st["na"]:, sl] = cur + st["coef"] * (st["donor"][("o", L)][:, sl] - cur)
            return (x,)
    return h


def r_hook(L):
    def h(mod, args, out):
        is_t = isinstance(out, tuple); x = out[0] if is_t else out
        if x.shape[1] == 1: return None
        if st["mode"] == "cap": cap[("r", L)] = x[0, -st["na"]:].detach().clone(); return None
        if st["mode"] == "sub":
            x = x.clone(); x[0, -st["na"]:] = st["donor"][("r", L)]; return (x,) + tuple(out[1:]) if is_t else x
        if st["mode"] == "dir" and L == PEAK:
            x = x.clone(); hcur = x[0, -st["na"]:].float(); hd = st["donor"][("r", L)].float()
            proj = ((hd - hcur) @ st["u"]).unsqueeze(-1) * st["u"]
            x[0, -st["na"]:] = (hcur + st["coef"] * proj).to(x.dtype)
            return (x,) + tuple(out[1:]) if is_t else x
    return h


for L in range(NL):
    model.model.layers[L].register_forward_hook(r_hook(L))
    model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(o_hook(L))


@torch.no_grad()
def gen(ids, n=MAX_NEW):
    out = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=n, do_sample=False, temperature=None, top_p=None, top_k=None,
                         pad_token_id=tok.eos_token_id)
    return out[0, len(ids):].tolist()


def capture(ids, na):
    st.update(mode="cap", na=na); gen(ids, 1); st["mode"] = None
    return dict(cap)


def by_layer(heads):
    d = {}
    for L, h in heads: d.setdefault(L, []).append(h)
    return d


Wo = {L: model.model.layers[L].self_attn.o_proj.weight.detach().float() for L in range(NL)}


def head_change_norm(heads, donor, rec, coef):
    tot = 0.0
    for L, h in heads:
        sl = slice(h * HD, (h + 1) * HD); delta = coef * (donor[("o", L)][:, sl].float() - rec[("o", L)][:, sl].float())
        tot += float((delta @ Wo[L][:, sl].T).pow(2).sum())
    return tot ** .5


# directions from DISCOVERY (for direction object and shuffled control)
d_hat = d_sh = None
if OBJ == "direction":
    H = {}
    for s in disc:
        for p in "AB":
            for e in ["joy", "sadness", "anger", "neutral"]:
                ids, na = encode(idx[(s, e, p)]); H[(s, e, p)] = capture(ids, na)[("r", PEAK)].float().mean(0)
    d = torch.stack([H[(s, a.primary, p)] - H[(s, "neutral", p)] for s in disc for p in "AB"]).mean(0); d_hat = d / d.norm()
    rng = random.Random(201); sh = []
    for s in disc:
        for p in "AB":
            lab = ["joy", "sadness", "anger", "neutral"]; perm = lab[:]; rng.shuffle(perm); m = dict(zip(lab, perm))
            sh.append(H[(s, m[a.primary], p)] - H[(s, m["neutral"], p)])
    d_sh = torch.stack(sh).mean(0); d_sh = d_sh / d_sh.norm()

# random controls (fixed seeds 101..105)
RAND = []
for sd in [101, 102, 103, 104, 105][:N_RAND]:
    g = random.Random(sd)
    if OBJ == "heads":
        pool = [(L, h) for L in sorted({L for L, _ in HEADS}) for h in range(NH) if (L, h) not in HEADS]
        RAND.append(g.sample(pool, len(HEADS)))
    else:
        tg = torch.Generator().manual_seed(sd); v = torch.randn(d_hat.shape[0], generator=tg); RAND.append((v / v.norm()).cuda())
if d_hat is not None: d_hat = d_hat.cuda(); d_sh = d_sh.cuda()

t0 = time.time(); out = []; scales = []
for n, sid in enumerate(target_scen):
    for para in PARAS:
        nid, na = encode(idx[(sid, "neutral", para)]); pid, pna = encode(idx[(sid, a.primary, para)]); assert pna == na
        donor = capture(pid, na); rec = capture(nid, na)
        def emit(cond, lam, g, scale=None):
            out.append(dict(scenario_id=sid, split=a.split, paraphrase=para, condition=cond, lam=lam, scale=scale,
                            continuation=tok.decode(g, skip_special_tokens=True)))
        emit("none", 0.0, gen(nid))
        st.update(mode="sub", na=na, donor=donor); emit("anchor_sub", 1.0, gen(nid)); st["mode"] = None
        emit("primary_prefix", 1.0, gen(pid))
        for lam in LAMS:
            if OBJ == "heads":
                st.update(mode="heads", na=na, donor=donor, hl=by_layer(HEADS), coef=lam); emit("target", lam, gen(nid)); st["mode"] = None
                tn = head_change_norm(HEADS, donor, rec, lam)
                for ri, rh in enumerate(RAND):
                    rn = head_change_norm(rh, donor, rec, lam); sc = tn / rn if rn > 0 else float("nan"); scales.append(sc)
                    st.update(mode="heads", na=na, donor=donor, hl=by_layer(rh), coef=lam * sc); emit(f"random_{ri}", lam, gen(nid), sc); st["mode"] = None
            else:
                st.update(mode="dir", na=na, donor=donor, u=d_hat, coef=lam); emit("target", lam, gen(nid)); st["mode"] = None
                hd = donor[("r", PEAK)].float(); hr = rec[("r", PEAK)].float()
                tn = float((lam * ((hd - hr) @ d_hat)).norm())
                for ri, u in enumerate(RAND + [d_sh]):
                    rn = float(((hd - hr) @ u).norm()); sc = tn / rn if rn > 0 else float("nan"); scales.append(sc)
                    name = "shuffled" if ri == len(RAND) else f"random_{ri}"
                    st.update(mode="dir", na=na, donor=donor, u=u, coef=sc); emit(name, lam, gen(nid), sc); st["mode"] = None
    print(f"{n+1}/{len(target_scen)} {sid} {time.time()-t0:.0f}s", flush=True)

fn = f"{a.out}/main_{a.mode}_{a.split}.jsonl"
with open(fn, "w") as f:
    for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
sc = np.array([s for s in scales if s == s])
info = dict(object=OBJ, primary=a.primary, peak_layer=PEAK, lams=LAMS, n_random=N_RAND, paraphrases=PARAS, n=len(out),
            heads=[f"L{L}H{h}" for L, h in HEADS] if OBJ == "heads" else None,
            random_scale=dict(median=float(np.median(sc)), max=float(sc.max()), frac_gt4=float((sc > 4).mean())) if len(sc) else None,
            seconds=round(time.time() - t0, 1), peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
json.dump(info, open(fn.replace(".jsonl", "_info.json"), "w"), indent=1); print("DONE", json.dumps(info))
