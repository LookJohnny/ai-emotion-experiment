"""E01 B2: multi-layer window patching (attn / mlp / both) on the DISCOVERY split (plan §4.3). Derived from b1_locate.py.
Windows: plan presets L16-23, L12-27, L8-31, L0-35 + (peak±4) where peak = B1 resid_all/joy best layer.

Original B1 doc:
E01 B1: layer x site localization on the DISCOVERY split (plan §4.3). Test split never loaded.

Recipient: neutral style_lead + anchor_fact + TAIL.  Donor: joy / sadness / anger (same scenario, same paraphrase).
Sites (patch donor activations into recipient at anchor positions, layer L only):
  resid_last : decoder-layer output, last anchor token
  resid_all  : decoder-layer output, all anchor tokens
  attn_all   : self_attn output, all anchor tokens
  mlp_all    : mlp output, all anchor tokens
Metric (teacher forced): donor-greedy continuation c_e (24 tokens) appended to recipient;
  recovery_e = [logp(c_e|patched) - logp(c_e|neutral)] / [logp(c_e|donor) - logp(c_e|neutral)]
  all three donors' continuations scored in one batch -> own vs cross recovery.
Capture and scoring use the same code path (full forward, use_cache=False) -> self-patch is exact (checked).
python b1_locate.py MODEL_DIR formal_960.jsonl OUT_DIR --tail "我的看法是："
"""
import json, time, argparse, os
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("data"); ap.add_argument("out")
ap.add_argument("--tail", required=True); ap.add_argument("--cont", type=int, default=24)
ap.add_argument("--b1", required=True, help="b1_summary.json")
ap.add_argument("--primary", default="joy", choices=["joy", "sadness", "anger"], help="emotion whose B1 resid_all peak defines the peak±4 window")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
EMOS = ["joy", "sadness", "anger"]
KINDS = ["attn", "mlp", "both"]
peak = json.load(open(a.b1))["sites"][f"resid_all/{a.primary}"]["best_layer"]
WINDOWS = {"L16-23": range(16, 24), "L12-27": range(12, 28), "L8-31": range(8, 32), "L0-35": range(0, 36),
           f"peak{peak}±4": range(max(0, peak - 4), min(36, peak + 5))}
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
NL = model.config.num_hidden_layers
rows = [json.loads(l) for l in open(a.data)]
rows = [r for r in rows if r["split"] == "discovery"]
assert rows and all(r["split"] != "test" for r in rows)
idx = {(r["scenario_id"], r["emotion"], r["paraphrase"]): r for r in rows}
scen = sorted({r["scenario_id"] for r in rows})


def encode(r):
    prefix = tok.apply_chat_template([{"role": "user", "content": r["user_prompt"]}], tokenize=False, add_generation_prompt=True)
    e = tok(prefix + r["style_lead"] + r["anchor_fact"] + a.tail, return_offsets_mapping=True, add_special_tokens=False)
    a_char = len(prefix) + len(r["style_lead"])
    a_tok = next(i for i, (s, _) in enumerate(e["offset_mapping"]) if s == a_char)
    return e["input_ids"], len(e["input_ids"]) - a_tok


st = {"mode": None}
cache = {}


def mk(kind, L):
    def h(mod, args, out):
        is_t = isinstance(out, tuple)
        x = out[0] if is_t else out
        if st["mode"] == "cap":
            cache[(kind, L)] = x[0, st["P"] - st["na"]:st["P"]].detach().clone()
            return None
        if st["mode"] == "patch" and kind in st["kinds"] and L in st["Ls"]:
            x = x.clone(); s, e = st["span"]
            x[:, s:e] = st["donor"][(kind, L)][st["dsl"]].unsqueeze(0)
            return (x,) + tuple(out[1:]) if is_t else x
    return h


for L in range(NL):
    lay = model.model.layers[L]
    lay.register_forward_hook(mk("resid", L)); lay.self_attn.register_forward_hook(mk("attn", L)); lay.mlp.register_forward_hook(mk("mlp", L))


@torch.no_grad()
def seq_logp(batch, plen):
    x = torch.tensor(batch, device="cuda")
    lp = torch.log_softmax(model(x, use_cache=False).logits[:, plen - 1:-1].float(), -1)
    return lp.gather(-1, x[:, plen:, None])[..., 0].sum(-1).cpu().numpy()


t0 = time.time()
num = np.zeros((len(scen), 2, 3, len(KINDS), len(WINDOWS), 3))  # scenario, paraphrase, donor, kind, window, scored
gap = np.zeros((len(scen), 2, 3))
selfcheck = []
for si, sid in enumerate(scen):
    for pi, para in enumerate(["A", "B"]):
        nid, na = encode(idx[(sid, "neutral", para)]); P = len(nid)
        donors, conts = {}, {}
        for e in EMOS:
            did, dna = encode(idx[(sid, e, para)]); assert dna == na
            with torch.no_grad():
                g = model.generate(torch.tensor([did], device="cuda"), max_new_tokens=a.cont, min_new_tokens=a.cont, do_sample=False,
                                   temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
            conts[e] = g[0, len(did):].tolist()
            st.update(mode="cap", na=na, P=len(did)); model(torch.tensor([did], device="cuda"), use_cache=False); st["mode"] = None
            donors[e] = dict(cache); donors[e]["ids"] = did
        batch = [nid + conts[e] for e in EMOS]
        base = seq_logp(batch, P)
        ceil = np.array([seq_logp([donors[e]["ids"] + conts[e]], len(donors[e]["ids"]))[0] for e in EMOS])
        gap[si, pi] = ceil - base
        if si % 15 == 0 and para == "A":  # self-patch must be exact: capture from the SAME batch/shape used for scoring
            st.update(mode="cap", na=na, P=P); model(torch.tensor(batch, device="cuda"), use_cache=False); st["mode"] = None
            selfd = dict(cache)
            st.update(mode="patch", kinds=["attn", "mlp"], Ls=set(range(NL)), span=(P - na, P), dsl=slice(None), donor=selfd)
            selfcheck.append(bool(np.allclose(seq_logp(batch, P), base, atol=0))); st["mode"] = None
        for di, e in enumerate(EMOS):
            for ki, kind in enumerate(KINDS):
                for wi, (wn, Ls) in enumerate(WINDOWS.items()):
                    st.update(mode="patch", kinds=["attn", "mlp"] if kind == "both" else [kind], Ls=set(Ls), span=(P - na, P), dsl=slice(None), donor=donors[e])
                    num[si, pi, di, ki, wi] = seq_logp(batch, P) - base
                    st["mode"] = None
    print(f"{si+1}/{len(scen)} {sid} {time.time()-t0:.0f}s", flush=True)

np.savez_compressed(f"{a.out}/b2_raw.npz", num=num, gap=gap, scen=np.array(scen), emos=np.array(EMOS), kinds=np.array(KINDS), windows=np.array(list(WINDOWS)))

# ---- summary: ratio of sums, own vs cross, scenario bootstrap ----
rng = np.random.default_rng(0)
summary = {"n_scenarios": len(scen), "tail": a.tail, "primary": a.primary, "b1_peak": peak, "self_patch_exact": f"{sum(selfcheck)}/{len(selfcheck)}", "cells": {}}
for ki, kind in enumerate(KINDS):
    for wi, wn in enumerate(WINDOWS):
        for di, e in enumerate(EMOS):
            f = lambda ix, c: num[ix][:, :, di, ki, wi, c].sum() / gap[ix][:, :, c].sum()
            allix = np.arange(len(scen))
            own = f(allix, di); cross = np.mean([f(allix, c) for c in range(3) if c != di])
            bo, bsp = [], []
            for _ in range(2000):
                ix = rng.integers(0, len(scen), len(scen)); o = f(ix, di)
                bo.append(o); bsp.append(o - np.mean([f(ix, c) for c in range(3) if c != di]))
            summary["cells"][f"{kind}/{wn}/{e}"] = dict(own=round(float(own), 3), cross=round(float(cross), 3),
                own_ci95=[round(float(np.percentile(bo, 2.5)), 3), round(float(np.percentile(bo, 97.5)), 3)],
                specificity_ci95=[round(float(np.percentile(bsp, 2.5)), 3), round(float(np.percentile(bsp, 97.5)), 3)])
            c = summary["cells"][f"{kind}/{wn}/{e}"]
            print(f"{kind:5s} {wn:10s} {e:8s} own={own:.2f} {c['own_ci95']} cross={cross:.2f} spec={c['specificity_ci95']}", flush=True)
summary["seconds"] = round(time.time() - t0, 1); summary["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
json.dump(summary, open(f"{a.out}/b2_summary.json", "w"), ensure_ascii=False, indent=1)
print("DONE", summary["self_patch_exact"], summary["seconds"])
