"""E06 selection (PREREG_E06.md), discovery only, on Qwen3-14B. Writes E06_FREEZE.json.
python e06_select.py MODEL formal_960.jsonl E04_MAP_DIR E04_SCREEN_FREEZE.json OUT_DIR
1. self-generated responses r_orig (neutral input) / r_contrast (anger input), greedy 64 tokens
2. gcm_atp: IE_hat(z) = grad_z[log p(r_c|p_o) - log p(r_o|p_o)] . (z_c - z_o), z = head o_proj-input at anchor positions
3. iti: per-head logistic probe at the last prompt token, 4:1 scenario split, rank by validation accuracy
4. twostage_self: E04 map top 60 + sign screen with self-generated responses as the two references"""
import hashlib, json, os, random, sys, time
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL, DATA, MAPD, E04SC, OUT = sys.argv[1:6]; os.makedirs(OUT, exist_ok=True)
fz = json.load(open(f"{MAPD}/MAP_FREEZE.json")); VEC = torch.tensor(np.load(f"{MAPD}/{fz['vectors_file']}"))
assert hashlib.sha256(open(f"{MAPD}/{fz['vectors_file']}", "rb").read()).hexdigest() == fz["vectors_sha256"]
NL, NH, HD = fz["layers"], fz["heads_per_layer"], fz["head_dim"]
parse = lambda s: (int(s[1:].split("H")[0]), int(s.split("H")[1])); name = lambda h: f"L{h[0]}H{h[1]}"
rank = [parse(s) for s in fz["ranking"]]
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
for p in model.parameters(): p.requires_grad_(False)
rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]; idx = {(r["scenario_id"], r["emotion"], r["paraphrase"]): r for r in rows}
scen = sorted({r["scenario_id"] for r in rows if r["split"] == "discovery"}); PAIRS = [(s, p) for s in scen for p in "AB"]
Wo = {L: model.model.layers[L].self_attn.o_proj.weight.detach().float() for L in range(NL)}
timing = {}

st = {"mode": None, "V": None, "start": 0, "cap": {}, "leaf": {}}
def hook(L):
    def h(mod, args):
        x = args[0]
        if st["mode"] == "cap": st["cap"][L] = x[0].detach().float().clone(); return None
        if st["mode"] == "grad":          # keep the full downstream graph (all later heads), only retain this activation's grad
            x.retain_grad(); st["leaf"][L] = x; return None
        if st["mode"] == "steer" and L in st["V"]:
            x = x.clone(); x[0, st["start"]:] += st["V"][L].to(x.dtype); return (x,)
        return None
    return h
for L in range(NL): model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(hook(L))
# activations need grad for attribution patching (parameters stay frozen): make the embedding output a grad root
model.model.embed_tokens.register_forward_hook(lambda m, a, out: out.requires_grad_(True) if st["mode"] == "grad" else None)


def encode(r):
    p = tok.apply_chat_template([{"role": "user", "content": r["user_prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    e = tok(p + r["style_lead"] + r["common_anchor"], return_offsets_mapping=True, add_special_tokens=False); a = len(p) + len(r["style_lead"])
    ids = e["input_ids"]; start = next(i for i, (c, _) in enumerate(e["offset_mapping"]) if c >= a)
    return ids, start, len(ids) - start                                       # ids, anchor start, anchor length


# 1. self-generated responses
t0 = time.time(); RESP = {}
with torch.no_grad():
    for s, p in PAIRS:
        for emo in ("neutral", "anger"):
            ids, _, _ = encode(idx[(s, emo, p)]); st["mode"] = None
            g = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=64, do_sample=False, temperature=None, top_p=None, top_k=None,
                               pad_token_id=tok.eos_token_id)[0, len(ids):].tolist()
            RESP[(s, p, emo)] = [t for t in g if t != tok.eos_token_id]
json.dump({f"{s}|{p}|{e}": tok.decode(v) for (s, p, e), v in RESP.items()}, open(f"{OUT}/self_responses.json", "w"), ensure_ascii=False, indent=0)
timing["self_generation_s"] = round(time.time() - t0, 1); print("responses", timing, flush=True)

# 2. GCM attribution patching over all heads (+ capture last-token head outputs for ITI)
t0 = time.time(); IE = torch.zeros(NL, NH, dtype=torch.float64); LAST = {"anger": [], "neutral": []}; groups = []
for s, p in PAIRS:
    ids_c, sc, nc = encode(idx[(s, "anger", p)]); ids_o, so, no = encode(idx[(s, "neutral", p)]); assert nc == no
    st["mode"] = "cap"
    with torch.no_grad(): model(torch.tensor([ids_c], device="cuda"), use_cache=False)
    zc = {L: st["cap"][L][-nc:] for L in range(NL)}; LAST["anger"].append(np.stack([st["cap"][L][-1].cpu().numpy() for L in range(NL)]))
    grads = []
    for resp in (RESP[(s, p, "anger")], RESP[(s, p, "neutral")]):
        st["mode"] = "grad"; st["leaf"] = {}
        logits = model(torch.tensor([ids_o + resp], device="cuda"), use_cache=False).logits[0, len(ids_o) - 1:-1].float()
        lp = torch.log_softmax(logits, -1)[torch.arange(len(resp)), torch.tensor(resp, device="cuda")].sum()
        lp.backward()
        grads.append({L: st["leaf"][L].grad[0, so:so + no].float() for L in range(NL)})
        if len(grads) == 1: zo = {L: st["leaf"][L].detach()[0, so:so + no].float() for L in range(NL)}; \
            LAST["neutral"].append(np.stack([st["leaf"][L].detach()[0, len(ids_o) - 1].float().cpu().numpy() for L in range(NL)]))
        st["mode"] = None
    for L in range(NL):
        d = ((grads[0][L] - grads[1][L]) * (zc[L] - zo[L])).view(no, NH, HD).sum(dim=(0, 2))
        IE[L] += d.double().cpu()
    groups.append(s); model.zero_grad(set_to_none=True)
IE /= len(PAIRS)
gcm_rank = sorted(((L, h) for L in range(NL) for h in range(NH)), key=lambda k: -float(IE[k[0], k[1]]))
timing["gcm_atp_s"] = round(time.time() - t0, 1); print("gcm", timing, [name(h) for h in gcm_rank[:12]], flush=True)

# 3. ITI: per-head probe on the last prompt token, 4:1 split by scenario
t0 = time.time()
XA, XN = np.stack(LAST["anger"]), np.stack(LAST["neutral"])                 # [pairs, layers, hidden]
y = np.r_[np.ones(len(XA)), np.zeros(len(XN))]; g = np.r_[groups, groups]
us = sorted(set(groups)); random.Random(606).shuffle(us); val = set(us[:len(us) // 5]); vm = np.array([x in val for x in g])
acc = {}
for L in range(NL):
    X = np.r_[XA[:, L], XN[:, L]]
    for h in range(NH):
        Xh = X[:, h * HD:(h + 1) * HD]; c = LogisticRegression(max_iter=1000).fit(Xh[~vm], y[~vm]); acc[(L, h)] = float(c.score(Xh[vm], y[vm]))
iti_rank = sorted(acc, key=lambda k: (-acc[k], k)); timing["iti_s"] = round(time.time() - t0, 1)
print("iti", timing, [name(h) for h in iti_rank[:12]], flush=True)

# 4. twostage_self: map top 60, sign screen with self-generated references
t0 = time.time()
INPUTS = []
for s, p in PAIRS:
    ids, start, _ = encode(idx[(s, "neutral", p)]); INPUTS.append((ids, start, [RESP[(s, p, "anger")], RESP[(s, p, "neutral")]]))


@torch.no_grad()
def scores(V):
    out = []
    for ids, start, sufs in INPUTS:
        v = []
        for suf in sufs:
            st.update(mode="steer" if V else None, V=V, start=start)
            lp = torch.log_softmax(model(torch.tensor([ids + suf], device="cuda"), use_cache=False).logits[0, len(ids) - 1:-1].float(), -1)
            v.append(float(lp[torch.arange(len(suf)), torch.tensor(suf, device="cuda")].mean()))
        st["mode"] = None; out.append(v[0] - v[1])
    return np.array(out)


def rn(L, h, v): return float((Wo[L][:, h * HD:(h + 1) * HD] @ v.cuda()).norm())
L0, h0 = rank[0]; PUSH = rn(L0, h0, 8.0 * VEC[L0, h0 * HD:(h0 + 1) * HD]); base = scores(None); scr = []
for (L, h) in rank[:60]:
    v = VEC[L, h * HD:(h + 1) * HD]; v = v * (PUSH / max(rn(L, h, v), 1e-9))
    d = scores({L: torch.zeros(NH * HD).index_copy_(0, torch.arange(h * HD, (h + 1) * HD), v).cuda()}) - base
    scr.append({"head": name((L, h)), "effect": round(float(d.mean()), 5), "se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 5)})
self_set = [r["head"] for r in sorted([r for r in scr if r["effect"] > 2 * r["se"]], key=lambda r: -r["effect"])[:12]]
timing["twostage_self_screen_s"] = round(time.time() - t0, 1); print("self", timing, self_set, flush=True)

E04 = json.load(open(E04SC))
F = {"model": os.path.basename(MODEL.rstrip("/")), "split": "discovery", "n_pairs": len(PAIRS),
     "sets": {"twostage": E04["twostage"], "gcm_atp": [name(h) for h in gcm_rank[:12]], "iti": [name(h) for h in iti_rank[:12]], "twostage_self": self_set},
     "gcm_ie_top30": [(name(h), round(float(IE[h[0], h[1]]), 6)) for h in gcm_rank[:30]],
     "iti_acc_top30": [(name(h), round(acc[h], 4)) for h in iti_rank[:30]], "twostage_self_screen": scr, "timing": timing,
     "e04_screen_sha256": hashlib.sha256(open(E04SC, "rb").read()).hexdigest(), "map_sha256": hashlib.sha256(open(f"{MAPD}/MAP_FREEZE.json", "rb").read()).hexdigest()}
S = F["sets"]; F["overlap"] = {f"{a}&{b}": sorted(set(S[a]) & set(S[b])) for a in S for b in S if a < b}
json.dump(F, open(f"{OUT}/E06_FREEZE.json", "w"), ensure_ascii=False, indent=1); print("DONE", json.dumps(F["sets"]), flush=True)
