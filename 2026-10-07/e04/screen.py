"""E04 stage 2 (PREREG_E04.md): teacher-forced single-head sign screen on DISCOVERY, frozen before any generation.
python screen.py MODEL formal_960.jsonl MAP_DIR OUT_DIR
effect(head) = mean over discovery inputs (A+B, neutral lead + anchor) of
  [mean log p(anger ref suffix) - mean log p(neutral ref suffix)]_steered - [same]_none,
steering = the head's own anger-neutral vector, rescaled to the residual-change norm of the map's rank-1 head at lambda 8,
added at the anchor and every suffix position.
twostage = map top-60 heads with effect > 2 SE, top 12 by effect; rscreen = same rule on 60 random heads outside the top 60 (seed 401)."""
import hashlib, json, os, random, sys, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL, DATA, MAPD, OUT = sys.argv[1:5]; os.makedirs(OUT, exist_ok=True)
fz = json.load(open(f"{MAPD}/MAP_FREEZE.json")); vp = f"{MAPD}/{fz['vectors_file']}"
assert hashlib.sha256(open(vp, "rb").read()).hexdigest() == fz["vectors_sha256"]
VEC = torch.tensor(np.load(vp)); NL, NH, HD = fz["layers"], fz["heads_per_layer"], fz["head_dim"]
parse = lambda s: (int(s[1:].split("H")[0]), int(s.split("H")[1])); name = lambda h: f"L{h[0]}H{h[1]}"
rank = [parse(s) for s in fz["ranking"]]; top60 = rank[:60]; rand60 = random.Random(401).sample(rank[60:], 60)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]; idx = {(r["scenario_id"], r["emotion"], r["paraphrase"]): r for r in rows}
scen = sorted({r["scenario_id"] for r in rows if r["split"] == "discovery"})
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
    e = tok(pre + r["style_lead"] + r["common_anchor"], return_offsets_mapping=True, add_special_tokens=False); a = len(pre) + len(r["style_lead"])
    return e["input_ids"], next(i for i, (c, _) in enumerate(e["offset_mapping"]) if c >= a), \
        [tok(idx[(s, emo, p)]["reference_suffix"], add_special_tokens=False)["input_ids"] for emo in ("anger", "neutral")]


INPUTS = [prep(s, p) for s in scen for p in "AB"]


@torch.no_grad()
def scores(V):
    out = []
    for ids, start, sufs in INPUTS:
        v = []
        for suf in sufs:
            st.update(V=V, start=start)
            lp = torch.log_softmax(model(torch.tensor([ids + suf], device="cuda"), use_cache=False).logits[0, len(ids) - 1:-1].float(), -1)
            v.append(float(lp[torch.arange(len(suf)), torch.tensor(suf, device="cuda")].mean()))
        st["V"] = None; out.append(v[0] - v[1])
    return np.array(out)


def rn(L, h, v): return float((Wo[L][:, h * HD:(h + 1) * HD] @ v.cuda()).norm())
L0, h0 = rank[0]; PUSH = rn(L0, h0, 8.0 * VEC[L0, h0 * HD:(h0 + 1) * HD])
t0 = time.time(); base = scores(None); res = []
for k, (L, h) in enumerate(top60 + rand60):
    v = VEC[L, h * HD:(h + 1) * HD]; v = v * (PUSH / max(rn(L, h, v), 1e-9))
    d = scores({L: torch.zeros(NH * HD).index_copy_(0, torch.arange(h * HD, (h + 1) * HD), v).cuda()}) - base
    res.append({"head": name((L, h)), "pool": "map_top60" if k < 60 else "random60", "map_rank": rank.index((L, h)),
                "effect": round(float(d.mean()), 5), "se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 5)})
    if k % 20 == 0: print(k, f"{time.time() - t0:.0f}s", flush=True)


def pick(pool):
    ok = sorted([r for r in res if r["pool"] == pool and r["effect"] > 2 * r["se"]], key=lambda r: -r["effect"]); return [r["head"] for r in ok[:12]]


tw, rs = pick("map_top60"), pick("random60")
S = {"model": fz["model"], "split": "discovery", "n_inputs": len(INPUTS), "push_resid_norm": round(PUSH, 4),
     "twostage": tw, "rscreen": rs, "maponly": fz["heads"], "stop": len(tw) < 4,
     "counts": {p: {"positive_2se": sum(1 for r in res if r["pool"] == p and r["effect"] > 2 * r["se"]),
                    "negative_2se": sum(1 for r in res if r["pool"] == p and r["effect"] < -2 * r["se"])} for p in ("map_top60", "random60")},
     "screen": res, "map_freeze_sha256": hashlib.sha256(open(f"{MAPD}/MAP_FREEZE.json", "rb").read()).hexdigest(), "seconds": round(time.time() - t0)}
json.dump(S, open(f"{OUT}/SCREEN_FREEZE.json", "w"), ensure_ascii=False, indent=1)
print("twostage:", tw); print("rscreen:", rs); print("counts:", S["counts"], "stop:", S["stop"])
