"""E06 test generation (PREREG_E06.md): every head set at equal total push, lambda 8 (primary) and 4, plus random heads, none, prefix.
python e06_steer.py MODEL formal_960.jsonl E04_MAP_DIR E06_FREEZE.json OUT.jsonl"""
import hashlib, json, random, sys, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL, DATA, MAPD, FZ, OUT = sys.argv[1:6]
mf = json.load(open(f"{MAPD}/MAP_FREEZE.json")); VEC = torch.tensor(np.load(f"{MAPD}/{mf['vectors_file']}"))
assert hashlib.sha256(open(f"{MAPD}/{mf['vectors_file']}", "rb").read()).hexdigest() == mf["vectors_sha256"]
F = json.load(open(FZ)); NL, NH, HD = mf["layers"], mf["heads_per_layer"], mf["head_dim"]
parse = lambda s: (int(s[1:].split("H")[0]), int(s.split("H")[1]))
SETS = {k: [parse(h) for h in v] for k, v in F["sets"].items()}
others = [(L, h) for L in range(NL) for h in range(NH) if (L, h) not in set(SETS["twostage"])]
for sd in (301, 302, 303): SETS[f"rheads_{sd}"] = random.Random(sd).sample(others, 12)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]; idx = {(r["scenario_id"], r["emotion"], r["paraphrase"]): r for r in rows}
scen = sorted({r["scenario_id"] for r in rows if r["split"] == "test"})
Wo = {L: model.model.layers[L].self_attn.o_proj.weight.detach().float() for L in range(NL)}
st = {"V": None, "na": 0}
def hook(L):
    def h(mod, args):
        V = st["V"]
        if V is None or L not in V: return None
        x = args[0].clone()
        if x.shape[1] > 1: x[0, -st["na"]:] += V[L].to(x.dtype)
        else: x[0, -1] += V[L].to(x.dtype)
        return (x,)
    return h
for L in range(NL): model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(hook(L))


def encode(r):
    p = tok.apply_chat_template([{"role": "user", "content": r["user_prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    e = tok(p + r["style_lead"] + r["common_anchor"], return_offsets_mapping=True, add_special_tokens=False); a = len(p) + len(r["style_lead"])
    return e["input_ids"], len(e["input_ids"]) - next(i for i, (c, _) in enumerate(e["offset_mapping"]) if c >= a)


def rn(L, h, v): return float((Wo[L][:, h * HD:(h + 1) * HD] @ v.cuda()).norm())


def build(heads, lam, total=None):
    parts = [(L, h, lam * VEC[L, h * HD:(h + 1) * HD]) for L, h in heads]
    if total is not None:
        k = total / max(sum(rn(L, h, v) for L, h, v in parts), 1e-9); parts = [(L, h, v * k) for L, h, v in parts]
    V = {}
    for L, h, v in parts: V.setdefault(L, torch.zeros(NH * HD))[h * HD:(h + 1) * HD] = v
    return {L: x.cuda() for L, x in V.items()}, sum(rn(L, h, v) for L, h, v in parts)


@torch.no_grad()
def gen(ids, na, V):
    st.update(V=V, na=na)
    g = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=64, do_sample=False, temperature=None, top_p=None, top_k=None,
                       pad_token_id=tok.eos_token_id)[0, len(ids):].tolist()
    st["V"] = None; return tok.decode(g, skip_special_tokens=True)


t0 = time.time(); out = []; info = {}
for lam in (8.0, 4.0):
    _, T = build(SETS["twostage"], lam); info[str(lam)] = round(T, 3)
    conds = {k: build(v, lam, total=T)[0] for k, v in SETS.items()}
    for s in scen:
        ids, na = encode(idx[(s, "neutral", "A")])
        for c, V in conds.items(): out.append(dict(scenario_id=s, split="test", paraphrase="A", condition=c, lam=lam, continuation=gen(ids, na, V)))
    print(f"lam {lam} {time.time() - t0:.0f}s", flush=True)
for s in scen:
    for c, e in (("none", "neutral"), ("primary_prefix", "anger")):
        ids, na = encode(idx[(s, e, "A")]); out.append(dict(scenario_id=s, split="test", paraphrase="A", condition=c, lam=0.0, continuation=gen(ids, na, None)))
with open(OUT, "w", encoding="utf-8") as f:
    for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
json.dump({"push": info, "sets": {k: [f"L{L}H{h}" for L, h in v] for k, v in SETS.items()}, "n": len(out), "seconds": round(time.time() - t0)},
          open(OUT.replace(".jsonl", "_info.json"), "w"), indent=1)
print("DONE", len(out))
