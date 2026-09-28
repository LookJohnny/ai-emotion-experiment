"""E01 B4: representation-level objects on the DISCOVERY split (plan §4.3), 2-fold cross-validated by scenario.

(a) residual direction d at B1 peak layer (resid_all/<primary> best layer): d = mean over train units & anchor tokens of (h_primary - h_neutral).
    (variable names hj/zj below hold the PRIMARY donor, whichever emotion --primary selects)
    Patch on held-out: h <- h + lam * ((h_joy - h) . d_hat) d_hat at all anchor positions, lam in {0.5,1,2,4}.
    Control: 5 random unit directions, scaled so ||delta h|| matches the target patch per unit (lam=1).
(b) secondary: cross-layer MLP unit set U inside the B2 best 'both' window: per layer top-32 units by |paired standardized effect|
    of down_proj input z (train units, anchor-token mean); full donor replacement of z[U] at all anchor positions.
    Control: 3 random unit sets (same count per layer).
Metric: teacher-forced recovery (own = joy continuation, cross = sadness/anger), ratio of sums over held-out units.
python b4_repr.py MODEL formal_960.jsonl OUT --tail "..." --b1 b1_summary.json --b2 b2_summary.json
"""
import json, time, argparse, os
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("data"); ap.add_argument("out")
ap.add_argument("--tail", required=True); ap.add_argument("--b1", required=True); ap.add_argument("--b2", required=True)
ap.add_argument("--cont", type=int, default=24); ap.add_argument("--k", type=int, default=32)
ap.add_argument("--primary", default="joy", choices=["joy", "sadness", "anger"])
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
EMOS = ["joy", "sadness", "anger"]; LAMS = [0.5, 1, 2, 4]
PI = EMOS.index(a.primary); CROSS = [c for c in range(3) if c != PI]
PEAK = json.load(open(a.b1))["sites"][f"resid_all/{a.primary}"]["best_layer"]
b2 = json.load(open(a.b2))
PRESET = {"L16-23": range(16, 24), "L12-27": range(12, 28), "L8-31": range(8, 32), "L0-35": range(0, 36)}
PRESET[f"peak{b2['b1_peak']}±4"] = range(max(0, b2["b1_peak"] - 4), min(36, b2["b1_peak"] + 5))
both = {w: b2["cells"][f"both/{w}/{a.primary}"]["own"] for w in PRESET}
WIN = min([w for w in PRESET if both[w] >= 0.8 * max(both.values())], key=lambda w: len(PRESET[w]))
MLP_LAYERS = list(PRESET[WIN])
print("peak", PEAK, "mlp window", WIN, flush=True)

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
rows = [json.loads(l) for l in open(a.data)]
rows = [r for r in rows if r["split"] == "discovery"]
assert rows and all(r["split"] != "test" for r in rows)
idx = {(r["scenario_id"], r["emotion"], r["paraphrase"]): r for r in rows}
scen = sorted({r["scenario_id"] for r in rows})
FOLD = {s: i % 2 for i, s in enumerate(scen)}


def encode(r):
    prefix = tok.apply_chat_template([{"role": "user", "content": r["user_prompt"]}], tokenize=False, add_generation_prompt=True)
    e = tok(prefix + r["style_lead"] + r["anchor_fact"] + a.tail, return_offsets_mapping=True, add_special_tokens=False)
    a_char = len(prefix) + len(r["style_lead"])
    return e["input_ids"], len(e["input_ids"]) - next(i for i, (s, _) in enumerate(e["offset_mapping"]) if s == a_char)


st = {"mode": None}; cap = {}


def resid_hook(mod, args, out):
    is_t = isinstance(out, tuple); x = out[0] if is_t else out
    if st["mode"] == "cap": cap["h"] = x[0, st["P"] - st["na"]:st["P"]].detach().float().clone(); return None
    if st["mode"] == "dir":
        x = x.clone(); s, e = st["span"]; h = x[:, s:e].float()
        delta = st["coef"] * ((st["hj"].unsqueeze(0) - h) @ st["u"]).unsqueeze(-1) * st["u"]
        if st.get("fixed") is not None: delta = st["fixed"].unsqueeze(0).expand_as(h)
        x[:, s:e] = (h + delta).to(x.dtype)
        return (x,) + tuple(out[1:]) if is_t else x


def mk_mlp(L):
    def h(mod, args):
        x = args[0]
        if st["mode"] == "cap": cap[("z", L)] = x[0, st["P"] - st["na"]:st["P"]].detach().float().clone(); return None
        if st["mode"] == "units" and L in st["U"]:
            x = x.clone(); s, e = st["span"]; U = st["U"][L]
            x[:, s:e, U] = st["zj"][L][:, U].unsqueeze(0).to(x.dtype)
            return (x,)
    return h


model.model.layers[PEAK].register_forward_hook(resid_hook)
for L in MLP_LAYERS: model.model.layers[L].mlp.down_proj.register_forward_pre_hook(mk_mlp(L))


@torch.no_grad()
def seq_logp(batch, plen):
    x = torch.tensor(batch, device="cuda")
    lp = torch.log_softmax(model(x, use_cache=False).logits[:, plen - 1:-1].float(), -1)
    return lp.gather(-1, x[:, plen:, None])[..., 0].sum(-1).cpu().numpy()


@torch.no_grad()
def capture(ids, na):
    st.update(mode="cap", P=len(ids), na=na); model(torch.tensor([ids], device="cuda"), use_cache=False); st["mode"] = None
    return cap["h"], {L: cap[("z", L)] for L in MLP_LAYERS}


t0 = time.time(); U_ = []
for sid in scen:
    for para in ["A", "B"]:
        nid, na = encode(idx[(sid, "neutral", para)]); conts = {}; P = len(nid)
        for e in EMOS:
            did, _ = encode(idx[(sid, e, para)])
            with torch.no_grad():
                g = model.generate(torch.tensor([did], device="cuda"), max_new_tokens=a.cont, min_new_tokens=a.cont, do_sample=False,
                                   temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
            conts[e] = g[0, len(did):].tolist()
        hj, zj = capture(encode(idx[(sid, a.primary, para)])[0], na); hn, zn = capture(nid, na)
        batch = [nid + conts[e] for e in EMOS]; base = seq_logp(batch, P)
        ceil = np.array([seq_logp([encode(idx[(sid, e, para)])[0] + conts[e]], len(encode(idx[(sid, e, para)])[0]))[0] for e in EMOS])
        U_.append(dict(sid=sid, fold=FOLD[sid], P=P, na=na, batch=batch, base=base, gap=ceil - base, hj=hj, hn=hn, zj=zj, zn=zn))
print(f"prepared {len(U_)} units {time.time()-t0:.0f}s", flush=True)


def evaluate(units, setup):
    num = np.zeros((len(units), 3)); gap = np.array([u["gap"] for u in units])
    for i, u in enumerate(units):
        setup(u); st["span"] = (u["P"] - u["na"], u["P"])
        num[i] = seq_logp(u["batch"], u["P"]) - u["base"]; st["mode"] = None; st["fixed"] = None
    own = num[:, PI].sum() / gap[:, PI].sum(); cross = np.mean([num[:, c].sum() / gap[:, c].sum() for c in CROSS])
    return float(own), float(cross)


res = {"primary": a.primary, "peak_layer": PEAK, "mlp_window": WIN, "folds": {}}
for f in (0, 1):
    tr = [u for u in U_ if u["fold"] != f]; te = [u for u in U_ if u["fold"] == f]
    d = torch.stack([(u["hj"] - u["hn"]).mean(0) for u in tr]).mean(0); dhat = (d / d.norm()).cuda()
    out = {}
    for lam in LAMS:
        def s(u, lam=lam): st.update(mode="dir", hj=u["hj"].cuda(), u=dhat, coef=lam, fixed=None)
        out[f"dir_lam{lam}"] = evaluate(te, s)
    g = torch.Generator().manual_seed(101); rnd = []
    for r in range(5):
        v = torch.randn(d.shape[0], generator=g); v = (v / v.norm()).cuda()
        def s(u, v=v):
            h = u["hn"].cuda(); tgt = ((u["hj"].cuda() - h) @ dhat).unsqueeze(-1) * dhat  # target change, per anchor token
            st.update(mode="dir", hj=u["hj"].cuda(), u=v, coef=1.0, fixed=(v * tgt.norm(dim=-1, keepdim=True)))
        rnd.append(evaluate(te, s))
    out["random_dir_lam1_mean"] = tuple(np.mean(rnd, 0).tolist())
    # MLP units from train fold
    Uset = {}
    for L in MLP_LAYERS:
        D = torch.stack([(u["zj"][L] - u["zn"][L]).mean(0) for u in tr]); eff = D.mean(0) / (D.std(0) + 1e-6)
        Uset[L] = torch.argsort(-eff.abs())[:a.k].cuda()
    def s(u): st.update(mode="units", U=Uset, zj={L: u["zj"][L].cuda() for L in MLP_LAYERS})
    out["mlp_units_top%d" % a.k] = evaluate(te, s)
    rr = []
    for seed in (201, 202, 203):
        gg = torch.Generator().manual_seed(seed)
        Ur = {L: torch.randperm(model.config.intermediate_size, generator=gg)[:a.k].cuda() for L in MLP_LAYERS}
        def s(u, Ur=Ur): st.update(mode="units", U=Ur, zj={L: u["zj"][L].cuda() for L in MLP_LAYERS})
        rr.append(evaluate(te, s))
    out["mlp_units_random_mean"] = tuple(np.mean(rr, 0).tolist())
    res["folds"][f] = {k: dict(own=round(v[0], 4), cross=round(v[1], 4)) for k, v in out.items()}
    print(f"fold {f}:", json.dumps(res["folds"][f]), flush=True)
res["cv_mean"] = {k: dict(own=round(np.mean([res["folds"][f][k]["own"] for f in (0, 1)]), 4),
                          cross=round(np.mean([res["folds"][f][k]["cross"] for f in (0, 1)]), 4)) for k in res["folds"][0]}
res["seconds"] = round(time.time() - t0, 1); res["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
json.dump(res, open(f"{a.out}/b4_summary.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(res["cv_mean"], indent=1)); print("DONE")
