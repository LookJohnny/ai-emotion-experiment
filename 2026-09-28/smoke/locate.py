"""Residual-stream localization sweep (debug set; method check, not a research result).

Recipient: neutral style_lead + anchor. Donor: joy / sadness / anger style_lead + same anchor.
For each layer (0..35) and site, replace recipient activations at anchor positions with donor's:
  resid_last : decoder-layer output, last anchor token only
  resid_all  : decoder-layer output, all anchor tokens
  attn_all   : self_attn output, all anchor tokens
  mlp_all    : mlp output, all anchor tokens
Metric (teacher-forced): donor-greedy continuation c_e (24 tokens) appended to recipient.
  recovery_e = (logp(c_e|patched) - logp(c_e|neutral)) / (logp(c_e|donor) - logp(c_e|neutral))
All three continuations are scored in one batch -> own-emotion recovery vs cross-emotion recovery.
"""
import json, sys, time, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR, OUT, DATA = sys.argv[1:4]
CONT = 24
EMOS = ["joy", "sadness", "anger"]
SITES = ["resid_last", "resid_all", "attn_all", "mlp_all"]
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
NL = model.config.num_hidden_layers
rows = [json.loads(l) for l in open(DATA)]
by = {}
for r in rows: by.setdefault(r["scenario_id"], {})[r["emotion"]] = r


def encode(r):
    prefix = tok.apply_chat_template([{"role": "user", "content": r["user_prompt"]}], tokenize=False, add_generation_prompt=True)
    full = prefix + r["style_lead"] + r["common_anchor"]
    e = tok(full, return_offsets_mapping=True, add_special_tokens=False)
    a_char = len(prefix) + len(r["style_lead"])
    a_tok = next(i for i, (s, _) in enumerate(e["offset_mapping"]) if s == a_char)
    return e["input_ids"], len(e["input_ids"]) - a_tok


st = {"mode": None}
cache = {}


def mk(site_kind, L):
    def h(mod, args, out):
        is_t = isinstance(out, tuple)
        x = out[0] if is_t else out
        if st["mode"] == "cap":
            cache[(site_kind, L)] = x[0, -st["na"]:].detach().clone()
            return None
        if st["mode"] == "patch" and st["kind"] == site_kind and st["L"] == L:
            x = x.clone()
            s, e = st["span"]
            x[:, s:e] = st["donor"][(site_kind, L)][st["dsl"]].unsqueeze(0)
            return (x,) + tuple(out[1:]) if is_t else x
    return h


for L in range(NL):
    lay = model.model.layers[L]
    lay.register_forward_hook(mk("resid", L))
    lay.self_attn.register_forward_hook(mk("attn", L))
    lay.mlp.register_forward_hook(mk("mlp", L))


@torch.no_grad()
def seq_logp(batch_ids, plen):
    """sum log p of tokens after position plen-1, per batch row."""
    x = torch.tensor(batch_ids, device="cuda")
    lp = torch.log_softmax(model(x, use_cache=False).logits[:, plen - 1:-1].float(), -1)
    return lp.gather(-1, x[:, plen:, None])[..., 0].sum(-1).cpu().numpy()


t0 = time.time()
res = []  # rows: scenario, donor, site, layer, recovery vector over EMOS
ceil_info = []
for sid, v in by.items():
    nid, na = encode(v["neutral"])
    donors, conts = {}, {}
    for e in EMOS:
        did, dna = encode(v[e]); assert dna == na
        with torch.no_grad():
            g = model.generate(torch.tensor([did], device="cuda"), max_new_tokens=CONT, min_new_tokens=CONT, do_sample=False,
                               temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
        conts[e] = g[0, len(did):].tolist()
        st.update(mode="cap", na=na); model(torch.tensor([did], device="cuda"), use_cache=False); st["mode"] = None
        donors[e] = dict(cache); donors[e]["ids"] = did
    batch = [nid + conts[e] for e in EMOS]
    base = seq_logp(batch, len(nid))
    ceil = np.array([seq_logp([donors[e]["ids"] + conts[e]], len(donors[e]["ids"]))[0] for e in EMOS])
    ceil_info.append(dict(scenario=sid, base=base.tolist(), ceil=ceil.tolist(), gap=(ceil - base).tolist(),
                          conts={e: tok.decode(conts[e]) for e in EMOS}))
    P = len(nid)
    for e in EMOS:
        for site in SITES:
            kind = site.split("_")[0]
            span, dsl = ((P - 1, P), slice(-1, None)) if site == "resid_last" else ((P - na, P), slice(None))
            for L in range(NL):
                st.update(mode="patch", kind=kind, L=L, span=span, dsl=dsl, donor=donors[e])
                lp = seq_logp(batch, P)
                st["mode"] = None
                res.append(dict(scenario=sid, donor=e, site=site, layer=L, rec=((lp - base) / (ceil - base)).tolist()))
    print(sid, "gap(ceil-base) nats:", np.round(ceil - base, 1), f"{time.time()-t0:.0f}s", flush=True)

json.dump(dict(results=res, ceiling=ceil_info, emos=EMOS, sites=SITES, cont_tokens=CONT), open(f"{OUT}/locate.json", "w"), ensure_ascii=False)

# summary: mean own-emotion recovery and mean cross-emotion recovery, bootstrap CI over scenarios
rng = np.random.default_rng(0)
sids = list(by)
print("\nsite        donor    best_layer  own_rec [95%CI]        cross_rec   | own_rec by layer (every 4th)")
for site in SITES:
    for ei, e in enumerate(EMOS):
        M = np.zeros((len(sids), NL)); C = np.zeros((len(sids), NL))
        for r in res:
            if r["site"] == site and r["donor"] == e:
                i = sids.index(r["scenario"]); M[i, r["layer"]] = r["rec"][ei]
                C[i, r["layer"]] = np.mean([r["rec"][k] for k in range(3) if k != ei])
        m = M.mean(0); b = int(m.argmax())
        bs = [M[rng.integers(0, len(sids), len(sids)), b].mean() for _ in range(2000)]
        print(f"{site:11s} {e:8s} L{b:<3d}        {m[b]:.2f} [{np.percentile(bs,2.5):.2f},{np.percentile(bs,97.5):.2f}]   {C.mean(0)[b]:.2f}      | " +
              " ".join(f"{x:.2f}" for x in m[::4]), flush=True)
print(f"total {time.time()-t0:.0f}s, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
