"""Pre-readout probe: predict the emotion of the reply the model is ABOUT to write, from the state at the last prompt
position (chat-template header), before any reply token exists.

Label (self-supervised): token-time L19 readout (validated vs blind ratings) averaged over the first 32 reply tokens,
  averaged over 3 decodes (greedy + sampled seeds 11, 22) -> 'what the model usually does next'.
Predictor: ridge regression (4 outputs) on standardized pre-state at layer L; alpha & layer chosen on VAL prompts.
Baseline (must be beaten): same ridge on the mean L19 state of the USER MESSAGE tokens (what the user said, not what the
  model is about to do), plus the token-time probe applied directly to the user message.
Split by prompt: train 60 / val 20 / test 20 (stratified by category, seed 0). Test reported once.
python prereadout.py MODEL prompts_pre.jsonl probe_L19.npz OUT
"""
import json, sys, time, os, random
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

MODEL, PROMPTS, PROBE, OUT = sys.argv[1:5]; os.makedirs(OUT, exist_ok=True)
LAYERS = [8, 12, 16, 19, 22, 26, 30]; NREAD = 32; NGEN = 48
pz = np.load(PROBE, allow_pickle=True); LR = int(pz["layer"]); CLS = [str(c) for c in pz["classes"]]
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
MU, SD, W, B = (torch.tensor(pz[k], device="cuda", dtype=torch.float32) for k in ["mean", "scale", "coef", "intercept"])
probe = lambda h: torch.softmax(((h - MU) / SD) @ W.T + B, -1)
P = [json.loads(l) for l in open(PROMPTS)]
rng = random.Random(0); split = {}
for cat in sorted({p["category"] for p in P}):
    ids = [p["id"] for p in P if p["category"] == cat]; rng.shuffle(ids); n = len(ids)
    for i, x in enumerate(ids): split[x] = "train" if i < .6 * n else "val" if i < .8 * n else "test"

t0 = time.time(); pre = {L: [] for L in LAYERS}; user_mean = []; user_probe = []; label = []; replies = []
for k, p in enumerate(P):
    prompt = tok.apply_chat_template([{"role": "user", "content": p["text"]}], tokenize=False, add_generation_prompt=True)
    e = tok(prompt, return_offsets_mapping=True, add_special_tokens=False); ids = e["input_ids"]
    c0 = prompt.index(p["text"]); c1 = c0 + len(p["text"])
    upos = [i for i, (s, t) in enumerate(e["offset_mapping"]) if s >= c0 and t <= c1 and t > s]
    with torch.no_grad():
        hs = model(torch.tensor([ids], device="cuda"), output_hidden_states=True, use_cache=False).hidden_states
    for L in LAYERS: pre[L].append(hs[L + 1][0, -1].float().cpu().numpy())
    um = hs[LR + 1][0, upos].float(); user_mean.append(um.mean(0).cpu().numpy()); user_probe.append(probe(um).mean(0).cpu().numpy())
    labs = []
    for seed in [None, 11, 22]:
        kw = dict(max_new_tokens=NGEN, pad_token_id=tok.eos_token_id)
        if seed is None: kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        else: torch.manual_seed(seed); kw.update(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
        with torch.no_grad():
            g = model.generate(torch.tensor([ids], device="cuda"), **kw)[0].tolist()
            rep = g[len(ids):len(ids) + NREAD]
            h = model(torch.tensor([g[:len(ids) + len(rep)]], device="cuda"), output_hidden_states=True, use_cache=False).hidden_states[LR + 1][0]
        labs.append(probe(h[len(ids):len(ids) + len(rep)].float()).mean(0).cpu().numpy())   # state after reading each reply token
        if seed is None: replies.append(tok.decode(g[len(ids):], skip_special_tokens=True))
    label.append(np.mean(labs, 0))
    if k % 50 == 0: print(f"{k+1}/{len(P)} {time.time()-t0:.0f}s", flush=True)
label = np.array(label); user_mean = np.array(user_mean); user_probe = np.array(user_probe)
S = np.array([split[p["id"]] for p in P]); tr, va, te = S == "train", S == "val", S == "test"
EM = ["joy", "sadness", "anger"]; ei = [CLS.index(e) for e in EM]


def score(pred, mask):
    return {e: round(float(spearmanr(pred[mask, CLS.index(e)], label[mask, CLS.index(e)]).correlation), 3) for e in EM}


def fit(X):
    best = None
    for alpha in [1, 10, 100, 1000, 10000]:
        sc = StandardScaler().fit(X[tr]); m = Ridge(alpha=alpha).fit(sc.transform(X[tr]), label[tr])
        v = np.mean(list(score(m.predict(sc.transform(X)), va).values()))
        if best is None or v > best[0]: best = (v, alpha, sc, m)
    return best


res = {"n_prompts": len(P), "split": {s: int((S == s).sum()) for s in ["train", "val", "test"]}, "val_by_layer": {}}
fits = {}
for L in LAYERS:
    X = np.array(pre[L]); fits[L] = fit(X); res["val_by_layer"][L] = round(float(fits[L][0]), 3)
BL = max(fits, key=lambda L: fits[L][0]); _, alpha, sc, m = fits[BL]
pred = m.predict(sc.transform(np.array(pre[BL])))
bu = fit(user_mean); pred_u = bu[3].predict(bu[2].transform(user_mean))
res.update(layer=BL, alpha=alpha,
           test_pre_probe=score(pred, te), test_baseline_user_ridge=score(pred_u, te), test_baseline_user_direct_probe=score(user_probe, te),
           val_pre_probe=score(pred, va), val_baseline_user_ridge=score(pred_u, va),
           label_mean_by_category={c: {e: round(float(label[[p["category"] == c for p in P], CLS.index(e)].mean()), 3) for e in CLS} for c in sorted({p["category"] for p in P})},
           seconds=round(time.time() - t0, 1),
           note="Labels are the model's own token-time readout (self-supervised), not human ratings. Test split used once.")
np.savez(f"{OUT}/preprobe_L{BL}.npz", layer=BL, mean=sc.mean_, scale=sc.scale_, coef=m.coef_, intercept=m.intercept_, classes=np.array(CLS))
with open(f"{OUT}/pre_data.jsonl", "w") as f:
    for p, lab, r in zip(P, label, replies):
        f.write(json.dumps(dict(id=p["id"], category=p["category"], split=split[p["id"]], text=p["text"], reply_greedy=r,
                                label={c: round(float(lab[i]), 4) for i, c in enumerate(CLS)}), ensure_ascii=False) + "\n")
json.dump(res, open(f"{OUT}/prereadout.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(res, ensure_ascii=False, indent=1)); print("DONE")
