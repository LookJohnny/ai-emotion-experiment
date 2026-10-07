"""E05 (PREREG_E05.md): continuous steering on transferred heads (source-model indices) vs native two-stage heads, random heads, random directions.
  python steer14b.py MODEL formal_960.jsonl MAP_DIR OUT --screen SCREEN_FREEZE.json --mode select
  python steer14b.py MODEL formal_960.jsonl MAP_DIR OUT --screen SCREEN_FREEZE.json --mode final --lam X --freeze FREEZE_E04.json
Steering adds the vector to the head's o_proj input slice at the anchor positions and at every generated position (E01 §5b)."""
import argparse, hashlib, json, os, random, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("data"); ap.add_argument("map_dir"); ap.add_argument("out")
ap.add_argument("--mode", required=True, choices=["select", "final"]); ap.add_argument("--lam", type=float); ap.add_argument("--freeze"); ap.add_argument("--screen", required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
fz_map = json.load(open(os.path.join(a.map_dir, "MAP_FREEZE.json")))
vp = os.path.join(a.map_dir, fz_map["vectors_file"])
assert hashlib.sha256(open(vp, "rb").read()).hexdigest() == fz_map["vectors_sha256"], "vectors file changed after the map was frozen"
if a.mode == "final":
    fz = json.load(open(a.freeze)); assert a.lam == fz["lambda"], "lambda must be the frozen one"
SPLIT = "validation" if a.mode == "select" else "test"
LAMS = [1.0, 2.0, 4.0, 8.0] if a.mode == "select" else [a.lam]
DSEEDS = [101, 102] if a.mode == "select" else [101, 102, 103, 104, 105]
HSEEDS = [301, 302] if a.mode == "select" else [301, 302, 303, 304, 305]
MAX_NEW = 64; NL, NH, HD = fz_map["layers"], fz_map["heads_per_layer"], fz_map["head_dim"]
parse = lambda s: (int(s[1:].split("H")[0]), int(s.split("H")[1]))
SC = json.load(open(a.screen)); assert not SC["stop"], "stop rule: fewer than 4 two-stage heads"
assert SC["map_freeze_sha256"] == hashlib.sha256(open(os.path.join(a.map_dir, "MAP_FREEZE.json"), "rb").read()).hexdigest()
TARGET = [parse(s) for s in SC["twostage"]]; NATIVE = [parse(s) for s in SC["native"]]
VEC = torch.tensor(np.load(vp))           # [layers, heads*hd]
others = [(L, h) for L in range(NL) for h in range(NH) if (L, h) not in set(TARGET)]
RHEADS = {sd: random.Random(sd).sample(others, 12) for sd in HSEEDS}

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
idx = {(r["scenario_id"], r["emotion"], r["paraphrase"]): r for r in rows}
scen = sorted({r["scenario_id"] for r in rows if r["split"] == SPLIT})
Wo = {L: model.model.layers[L].self_attn.o_proj.weight.detach().float() for L in range(NL)}


def encode(r):
    p = tok.apply_chat_template([{"role": "user", "content": r["user_prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    e = tok(p + r["style_lead"] + r["common_anchor"], return_offsets_mapping=True, add_special_tokens=False)
    a_char = len(p) + len(r["style_lead"])
    return e["input_ids"], len(e["input_ids"]) - next(i for i, (s, _) in enumerate(e["offset_mapping"]) if s >= a_char)


st = {"vec": None, "na": 0}
def hook(L):
    def h(mod, args):
        V = st["vec"]
        if V is None or L not in V: return None
        x = args[0].clone()
        if x.shape[1] > 1: x[0, -st["na"]:] += V[L].to(x.dtype)       # prefill: anchor positions
        else: x[0, -1] += V[L].to(x.dtype)                             # every generated position
        return (x,)
    return h
for L in range(NL): model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(hook(L))


def resid_norm(L, h, v): return float((Wo[L][:, h * HD:(h + 1) * HD] @ v.cuda()).norm())


def build(heads, lam, mode, gen=None, match=None, total=None):
    """mode 'true': each head's own anger-neutral vector x lam; 'rand': random direction per head, norm-matched to match[(L,h)].
    total: rescale all heads by one common factor so the summed residual-change norm equals total."""
    V = {}; parts = []
    for L, h in heads:
        sl = slice(h * HD, (h + 1) * HD)
        v = lam * VEC[L, sl] if mode == "true" else torch.randn(HD, generator=gen)
        if match is not None: v = v * (match[(L, h)] / max(resid_norm(L, h, v), 1e-9))
        parts.append((L, h, v))
    if total is not None:
        k = total / max(sum(resid_norm(L, h, v) for L, h, v in parts), 1e-9); parts = [(L, h, v * k) for L, h, v in parts]
    for L, h, v in parts:
        V.setdefault(L, torch.zeros(NH * HD))[h * HD:(h + 1) * HD] = v
    return {L: x.cuda() for L, x in V.items()}, sum(resid_norm(L, h, v) for L, h, v in parts)


@torch.no_grad()
def gen(ids, na, V):
    st.update(vec=V, na=na)
    g = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=MAX_NEW, do_sample=False, temperature=None, top_p=None, top_k=None,
                       pad_token_id=tok.eos_token_id)[0, len(ids):].tolist()
    st["vec"] = None
    return tok.decode(g, skip_special_tokens=True)


t0 = time.time(); out = []; info = {"lams": {}}
for lam in LAMS:
    Vt, tnorm = build(TARGET, lam, "true")
    per = {(L, h): resid_norm(L, h, lam * VEC[L, h * HD:(h + 1) * HD]) for L, h in TARGET}
    conds = [("target", Vt)]
    for sd in DSEEDS: conds.append((f"rdir_{sd}", build(TARGET, lam, "rand", gen=torch.Generator().manual_seed(sd), match=per)[0]))
    for sd in HSEEDS: conds.append((f"rheads_{sd}", build(RHEADS[sd], lam, "true", total=tnorm)[0]))
    conds.append(("native", build(NATIVE, lam, "true", total=tnorm)[0]))
    info["lams"][str(lam)] = {"target_resid_norm": round(tnorm, 3)}
    for s in scen:
        ids, na = encode(idx[(s, "neutral", "A")])
        for c, V in conds: out.append(dict(scenario_id=s, split=SPLIT, paraphrase="A", condition=c, lam=lam, continuation=gen(ids, na, V)))
    print(f"lam {lam} done {time.time() - t0:.0f}s", flush=True)
for s in scen:
    for c, e in (("none", "neutral"), ("primary_prefix", "anger")):
        ids, na = encode(idx[(s, e, "A")]); out.append(dict(scenario_id=s, split=SPLIT, paraphrase="A", condition=c, lam=0.0, continuation=gen(ids, na, None)))
fn = os.path.join(a.out, f"steer_{a.mode}_{SPLIT}.jsonl")
with open(fn, "w", encoding="utf-8") as f:
    for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
info.update(model=os.path.basename(a.model.rstrip("/")), target=SC["twostage"], native=SC["native"], rheads={str(k): [f"L{L}H{h}" for L, h in v] for k, v in RHEADS.items()},
            dir_seeds=DSEEDS, n=len(out), seconds=round(time.time() - t0, 1), peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
json.dump(info, open(fn.replace(".jsonl", "_info.json"), "w"), ensure_ascii=False, indent=1); print("DONE", json.dumps(info, ensure_ascii=False)[:600])
