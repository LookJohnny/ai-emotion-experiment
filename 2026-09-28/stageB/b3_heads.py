"""E01 B3: attention-head localization on the DISCOVERY split (plan §4.3). Primary donor = joy (main contrast).

Window rule (fixed before results): among B2 attn windows, the SMALLEST window whose joy own-recovery >= 80% of the best attn window's.
Head h output = slice h*128:(h+1)*128 of self_attn.o_proj INPUT (forward_pre_hook), patched at all anchor positions.
Steps:
  1. single-head patch for every (layer, head) in window -> per-scenario own (joy) and cross (sadness/anger) recovery numerators
  2. rank heads by joy own recovery (ratio of sums); cumulative top-k patch for k in K_GRID and all heads in window
  3. k needed for 50% / 80% of window effect; bootstrap (scenario resample) top-16 Jaccard stability
  4. descriptive only: attention weight from anchor positions to style_lead tokens for top heads (eager attention)
Primary-object rule (plan §5.2): head set if <=16 heads reach 50% of window effect AND bootstrap Jaccard >= 0.5.
python b3_heads.py MODEL formal_960.jsonl OUT --tail "..." --b2 b2_summary.json
"""
import json, time, argparse, os
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("data"); ap.add_argument("out")
ap.add_argument("--tail", required=True); ap.add_argument("--b2", required=True); ap.add_argument("--cont", type=int, default=24)
ap.add_argument("--primary", default="joy", choices=["joy", "sadness", "anger"])
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
EMOS = ["joy", "sadness", "anger"]
PI = EMOS.index(a.primary); CROSS = [c for c in range(3) if c != PI]
K_GRID = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]

b2 = json.load(open(a.b2))
PRESET = {"L16-23": range(16, 24), "L12-27": range(12, 28), "L8-31": range(8, 32), "L0-35": range(0, 36)}
peak = b2["b1_peak"]; PRESET[f"peak{peak}±4"] = range(max(0, peak - 4), min(36, peak + 5))
attn = {w: b2["cells"][f"attn/{w}/{a.primary}"]["own"] for w in PRESET}
best = max(attn.values())
elig = [w for w in PRESET if attn[w] >= 0.8 * best]
WIN = min(elig, key=lambda w: len(PRESET[w]))
LAYERS = list(PRESET[WIN])
print("attn windows joy own:", attn, "-> chosen", WIN, flush=True)

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
H = model.config.num_attention_heads; HD = model.config.head_dim
rows = [json.loads(l) for l in open(a.data)]
rows = [r for r in rows if r["split"] == "discovery"]
assert rows and all(r["split"] != "test" for r in rows)
idx = {(r["scenario_id"], r["emotion"], r["paraphrase"]): r for r in rows}
scen = sorted({r["scenario_id"] for r in rows})
HEADS = [(L, h) for L in LAYERS for h in range(H)]


def encode(r):
    prefix = tok.apply_chat_template([{"role": "user", "content": r["user_prompt"]}], tokenize=False, add_generation_prompt=True)
    e = tok(prefix + r["style_lead"] + r["anchor_fact"] + a.tail, return_offsets_mapping=True, add_special_tokens=False)
    pl = len(tok(prefix, add_special_tokens=False)["input_ids"])
    a_char = len(prefix) + len(r["style_lead"])
    a_tok = next(i for i, (s, _) in enumerate(e["offset_mapping"]) if s == a_char)
    return e["input_ids"], len(e["input_ids"]) - a_tok, pl, a_tok  # ids, n_anchor, lead_start, anchor_start


st = {"mode": None}; cache = {}


def mk(L):
    def h(mod, args):
        x = args[0]
        if st["mode"] == "cap":
            cache[L] = x[0, st["P"] - st["na"]:st["P"]].detach().clone(); return None
        if st["mode"] == "patch" and L in st["heads_by_layer"]:
            x = x.clone(); s, e = st["span"]
            for hh in st["heads_by_layer"][L]:
                x[:, s:e, hh * HD:(hh + 1) * HD] = st["donor"][L][:, hh * HD:(hh + 1) * HD].unsqueeze(0)
            return (x,)
    return h


for L in LAYERS:
    model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(mk(L))


def by_layer(heads):
    d = {}
    for L, h in heads: d.setdefault(L, []).append(h)
    return d


@torch.no_grad()
def seq_logp(batch, plen):
    x = torch.tensor(batch, device="cuda")
    lp = torch.log_softmax(model(x, use_cache=False).logits[:, plen - 1:-1].float(), -1)
    return lp.gather(-1, x[:, plen:, None])[..., 0].sum(-1).cpu().numpy()


t0 = time.time()
U = [(sid, p) for sid in scen for p in ["A", "B"]]
single = np.zeros((len(U), len(HEADS), 3)); gap = np.zeros((len(U), 3)); win_all = np.zeros((len(U), 3))
units = []; selfcheck = []
for ui, (sid, para) in enumerate(U):
    nid, na, _, _ = encode(idx[(sid, "neutral", para)]); P = len(nid)
    conts = {}; donor = None
    for e in EMOS:
        did, dna, _, _ = encode(idx[(sid, e, para)]); assert dna == na
        with torch.no_grad():
            g = model.generate(torch.tensor([did], device="cuda"), max_new_tokens=a.cont, min_new_tokens=a.cont, do_sample=False,
                               temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
        conts[e] = g[0, len(did):].tolist()
        if e == a.primary:
            st.update(mode="cap", na=na, P=len(did)); model(torch.tensor([did], device="cuda"), use_cache=False); st["mode"] = None
            donor = dict(cache)
    batch = [nid + conts[e] for e in EMOS]
    base = seq_logp(batch, P)
    ceil = np.array([seq_logp([encode(idx[(sid, e, para)])[0] + conts[e]], len(encode(idx[(sid, e, para)])[0]))[0] for e in EMOS])
    gap[ui] = ceil - base
    if ui % 30 == 0:
        st.update(mode="cap", na=na, P=P); model(torch.tensor(batch, device="cuda"), use_cache=False); st["mode"] = None
        st.update(mode="patch", heads_by_layer=by_layer(HEADS), span=(P - na, P), donor=dict(cache))
        selfcheck.append(bool(np.array_equal(seq_logp(batch, P), base))); st["mode"] = None
    for hi, (L, h) in enumerate(HEADS):
        st.update(mode="patch", heads_by_layer={L: [h]}, span=(P - na, P), donor=donor)
        single[ui, hi] = seq_logp(batch, P) - base; st["mode"] = None
    st.update(mode="patch", heads_by_layer=by_layer(HEADS), span=(P - na, P), donor=donor)
    win_all[ui] = seq_logp(batch, P) - base; st["mode"] = None
    units.append(dict(batch=batch, P=P, na=na, base=base, donor=donor))
    if ui % 10 == 0: print(f"single {ui+1}/{len(U)} {time.time()-t0:.0f}s", flush=True)

own_single = single[:, :, PI].sum(0) / gap[:, PI].sum()
cross_single = np.mean([single[:, :, c].sum(0) / gap[:, c].sum() for c in CROSS], 0)
rank = np.argsort(-own_single)
win_own = win_all[:, PI].sum() / gap[:, PI].sum()

cum = {}
for k in [k for k in K_GRID if k <= len(HEADS)] + [len(HEADS)]:
    heads = [HEADS[i] for i in rank[:k]]; nums = np.zeros((len(U), 3))
    for ui, u in enumerate(units):
        st.update(mode="patch", heads_by_layer=by_layer(heads), span=(u["P"] - u["na"], u["P"]), donor=u["donor"])
        nums[ui] = seq_logp(u["batch"], u["P"]) - u["base"]; st["mode"] = None
    own = nums[:, PI].sum() / gap[:, PI].sum(); cross = np.mean([nums[:, c].sum() / gap[:, c].sum() for c in CROSS])
    cum[k] = dict(own=round(float(own), 4), cross=round(float(cross), 4), frac_of_window=round(float(own / win_own), 3) if win_own else None)
    print(f"top-{k}: own={own:.3f} cross={cross:.3f} frac={own / win_own:.2f}", flush=True)
k50 = next((k for k in sorted(cum) if cum[k]["frac_of_window"] is not None and cum[k]["frac_of_window"] >= 0.5), None)
k80 = next((k for k in sorted(cum) if cum[k]["frac_of_window"] is not None and cum[k]["frac_of_window"] >= 0.8), None)

# bootstrap stability of top-16 by scenario resampling (units come in A/B pairs per scenario)
rng = np.random.default_rng(0); top16 = set(rank[:16].tolist()); jac = []
S = len(scen)
for _ in range(100):
    ss = rng.integers(0, S, S); ix = np.concatenate([[2 * s, 2 * s + 1] for s in ss])
    r = np.argsort(-(single[ix, :, PI].sum(0) / gap[ix, PI].sum()))[:16]
    jac.append(len(top16 & set(r.tolist())) / len(top16 | set(r.tolist())))

# descriptive: attention from anchor positions to style_lead tokens for top-8 heads (eager)
desc = {}
try:
    model.set_attn_implementation("eager")
    top8 = [HEADS[i] for i in rank[:8]]; acc = {f"L{L}H{h}": [] for L, h in top8}
    for sid in scen[:30]:
        ids, na, lead_s, anc_s = encode(idx[(sid, a.primary, "A")])
        with torch.no_grad():
            att = model(torch.tensor([ids], device="cuda"), use_cache=False, output_attentions=True).attentions
        for L, h in top8:
            w = att[L][0, h, anc_s:, :].float()  # anchor queries
            acc[f"L{L}H{h}"].append(float(w[:, lead_s:anc_s].sum(-1).mean()))
    desc = {k: round(float(np.mean(v)), 3) for k, v in acc.items()}
except Exception as ex:
    desc = {"error": repr(ex)}

summary = dict(primary=a.primary, window=WIN, window_layers=[LAYERS[0], LAYERS[-1]], attn_window_primary_own=attn, n_heads_in_window=len(HEADS),
               self_patch_exact=f"{sum(selfcheck)}/{len(selfcheck)}", window_all_heads_own=round(float(win_own), 4),
               top_heads=[dict(head=f"L{HEADS[i][0]}H{HEADS[i][1]}", own=round(float(own_single[i]), 4), cross=round(float(cross_single[i]), 4)) for i in rank[:32]],
               cumulative=cum, k_for_50pct=k50, k_for_80pct=k80,
               top16_bootstrap_jaccard=dict(mean=round(float(np.mean(jac)), 3), p5=round(float(np.percentile(jac, 5)), 3)),
               primary_object_rule=dict(k50_le_16=(k50 is not None and k50 <= 16), jaccard_ge_0_5=float(np.mean(jac)) >= 0.5,
                                        head_set_is_primary=(k50 is not None and k50 <= 16 and float(np.mean(jac)) >= 0.5)),
               descriptive_attn_anchor_to_lead_top8=desc, seconds=round(time.time() - t0, 1),
               peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
np.savez_compressed(f"{a.out}/b3_raw.npz", single=single, gap=gap, win_all=win_all, heads=np.array([f"L{L}H{h}" for L, h in HEADS]))
json.dump(summary, open(f"{a.out}/b3_summary.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: summary[k] for k in ["window", "window_all_heads_own", "k_for_50pct", "k_for_80pct", "top16_bootstrap_jaccard", "primary_object_rule", "self_patch_exact"]}, ensure_ascii=False))
print("DONE")
