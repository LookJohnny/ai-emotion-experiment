"""E03 EXPLORATORY (after the confirmatory test was generated; does not change E03's verdict).
Teacher-forced screen of steering configurations on Qwen3-8B, validation split (A+B):
  score(input) = mean log p(anger reference suffix) - mean log p(neutral reference suffix), on the NEUTRAL lead + anchor,
  with steering at the anchor and at every suffix position; effect = score(steered) - score(none), per scenario.
Configurations: map top-12 (frozen), map top-12 within a mid-depth window, E01's 4B head indices, random heads (5 sets),
and single heads (map top-60 vs 60 random) to see whether the map score predicts a head's own steering effect.
python explore8b.py MODEL formal_960.jsonl MAP_DIR OUT.json"""
import json, random, sys, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL, DATA, MAPD, OUT = sys.argv[1:5]
fz = json.load(open(f"{MAPD}/MAP_FREEZE.json")); VEC = torch.tensor(np.load(f"{MAPD}/anger_minus_neutral.npy"))
NL, NH, HD = fz["layers"], fz["heads_per_layer"], fz["head_dim"]
parse = lambda s: (int(s[1:].split("H")[0]), int(s.split("H")[1]))
rank = [parse(s) for s in fz["ranking"]]; score = {h: i for i, h in enumerate(rank)}
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]; idx = {(r["scenario_id"], r["emotion"], r["paraphrase"]): r for r in rows}
scen = sorted({r["scenario_id"] for r in rows if r["split"] == "validation"})
Wo = {L: model.model.layers[L].self_attn.o_proj.weight.detach().float() for L in range(NL)}

st = {"V": None, "start": 0}
def hook(L):
    def h(mod, args):
        V = st["V"]
        if V is None or L not in V: return None
        x = args[0].clone(); x[0, st["start"]:] += V[L].to(x.dtype); return (x,)
    return h
for L in range(NL): model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(hook(L))


def prep(s, p):
    r = idx[(s, "neutral", p)]
    pre = tok.apply_chat_template([{"role": "user", "content": r["user_prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    e = tok(pre + r["style_lead"] + r["common_anchor"], return_offsets_mapping=True, add_special_tokens=False)
    a_char = len(pre) + len(r["style_lead"]); ids = e["input_ids"]
    start = next(i for i, (c, _) in enumerate(e["offset_mapping"]) if c >= a_char)
    sufs = [tok(idx[(s, emo, p)]["reference_suffix"], add_special_tokens=False)["input_ids"] for emo in ("anger", "neutral")]
    return ids, start, sufs


INPUTS = [prep(s, p) for s in scen for p in "AB"]


@torch.no_grad()
def effect_score(V):
    out = []
    for ids, start, sufs in INPUTS:
        vals = []
        for suf in sufs:
            st.update(V=V, start=start)
            lp = torch.log_softmax(model(torch.tensor([ids + suf], device="cuda"), use_cache=False).logits[0, len(ids) - 1:-1].float(), -1)
            vals.append(float(lp[torch.arange(len(suf)), torch.tensor(suf, device="cuda")].mean()))
        st["V"] = None; out.append(vals[0] - vals[1])
    return np.array(out)


def rn(L, h, v): return float((Wo[L][:, h * HD:(h + 1) * HD] @ v.cuda()).norm())


def build(heads, lam, total=None):
    parts = [(L, h, lam * VEC[L, h * HD:(h + 1) * HD]) for L, h in heads]
    if total is not None:
        k = total / max(sum(rn(L, h, v) for L, h, v in parts), 1e-9); parts = [(L, h, v * k) for L, h, v in parts]
    V = {}
    for L, h, v in parts: V.setdefault(L, torch.zeros(NH * HD))[h * HD:(h + 1) * HD] = v
    return {L: x.cuda() for L, x in V.items()}, sum(rn(L, h, v) for L, h, v in parts)


t0 = time.time(); base = effect_score(None); res = {"n_inputs": len(INPUTS), "groups": {}, "single": []}
def rel(x): d = x - base; return round(float(d.mean()), 4), round(float(d.std(ddof=1) / np.sqrt(len(d))), 4)
TARGET = [parse(s) for s in fz["heads"]]
win = [h for h in rank if 0.3 <= h[0] / (NL - 1) <= 0.7][:12]
E01 = [parse(s) for s in ["L16H12", "L17H16", "L14H25", "L19H28", "L17H4", "L19H29", "L18H8", "L22H19", "L16H3", "L14H26", "L20H11", "L16H17"]]
others = [h for h in rank if h not in set(TARGET)]
groups = {"map12": TARGET, "map12_mid": win, "e01_indices": E01, "map_late_only": [h for h in TARGET if h[0] / (NL - 1) > 0.7],
          "map_mid_only": [h for h in TARGET if h[0] / (NL - 1) <= 0.7]}
for sd in (301, 302, 303, 304, 305): groups[f"rheads_{sd}"] = random.Random(sd).sample(others, 12)
res["group_heads"] = {k: [f"L{L}H{h}" for L, h in v] for k, v in groups.items()}
for lam in (4.0, 8.0):
    _, ref = build(TARGET, lam)                                          # equal total push for every group
    for name, heads in groups.items():
        V, _ = build(heads, lam, total=ref); res["groups"][f"{name}@{lam}"] = rel(effect_score(V))
    print(f"groups lam {lam} done {time.time() - t0:.0f}s", flush=True)
rng = random.Random(7); singles = rank[:60] + rng.sample(rank[60:], 60)
_, ref1 = build([TARGET[0]], 8.0)
for L, h in singles:
    V, _ = build([(L, h)], 8.0, total=ref1)                             # same push per head
    m, se = rel(effect_score(V)); res["single"].append({"head": f"L{L}H{h}", "map_rank": score[(L, h)], "depth": round(L / (NL - 1), 3), "effect": m, "se": se})
print(f"singles done {time.time() - t0:.0f}s", flush=True)
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
print(json.dumps(res["groups"], indent=0))
