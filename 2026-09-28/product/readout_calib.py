"""Real-time emotion readout: calibration + validity (product prototype step 1).

Train (DISCOVERY only): per-token residual states at layer L while the model reads its own emotional text
  (style_lead + anchor + reference_suffix, teacher forced). Label = record emotion (4 classes).
  Probe = multinomial logistic regression on standardized states; 5-fold CV grouped by scenario -> pick layer.
Validity (VALIDATION only; test split untouched):
  V1 text-time readout: stage-A continuations (blind-rated by 2 raters) teacher-forced after their prefix;
     mean probe probability over continuation tokens vs rated anger/sadness/joy (Spearman).
  V2 lead time: probe at the LAST prefix token (before any continuation token exists) vs rated emotion of the continuation.
  V3 plain accuracy on validation reference records (lead+anchor+suffix), macro-F1.
python readout_calib.py MODEL formal_960.jsonl stageA_generations.jsonl stageA_ratings_dir OUT --tail "我的看法是："
"""
import json, argparse, os, time, glob
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("data"); ap.add_argument("gens"); ap.add_argument("ratings"); ap.add_argument("out")
ap.add_argument("--tail", required=True); ap.add_argument("--layers", default="4,8,12,14,16,18,19,20,21,22,24,26,28,32,35")
ap.add_argument("--extra_neutral", default=None, help="pre_data.jsonl: TRAIN-split neutral_task replies added as class neutral (v2 bias fix)")
ap.add_argument("--eval_only_extra", action="store_true", help="use --extra_neutral only for the held-out V4 check (baseline run)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
LAYERS = [int(x) for x in a.layers.split(",")]; EMOS = ["neutral", "joy", "sadness", "anger"]
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
rows = [json.loads(l) for l in open(a.data)]
t0 = time.time()


@torch.no_grad()
def states(text_before, text_after):
    """hidden states (per LAYER) for tokens of text_after, plus state at last token of text_before."""
    ids_b = tok(text_before, add_special_tokens=False)["input_ids"]; ids = tok(text_before + text_after, add_special_tokens=False)["input_ids"]
    assert ids[:len(ids_b)] == ids_b
    hs = model(torch.tensor([ids], device="cuda"), output_hidden_states=True, use_cache=False).hidden_states
    return {L: hs[L + 1][0, len(ids_b):].float().cpu().numpy() for L in LAYERS}, {L: hs[L + 1][0, len(ids_b) - 1].float().cpu().numpy() for L in LAYERS}


def prefix_of(r):
    return tok.apply_chat_template([{"role": "user", "content": r["user_prompt"]}], tokenize=False, add_generation_prompt=True)


# ---- training data: discovery records, tokens of lead + anchor + suffix ----
X = {L: [] for L in LAYERS}; y = []; grp = []
for r in rows:
    if r["split"] != "discovery": continue
    body = r["style_lead"] + r["anchor_fact"] + a.tail + r["reference_suffix"]
    st, _ = states(prefix_of(r), body)
    n = st[LAYERS[0]].shape[0]
    for L in LAYERS: X[L].append(st[L])
    y += [EMOS.index(r["emotion"])] * n; grp += [r["scenario_id"]] * n
EXTRA = [json.loads(l) for l in open(a.extra_neutral)] if a.extra_neutral else []
for x in EXTRA:
    if a.eval_only_extra or x["category"] != "neutral_task" or x["split"] != "train": continue
    pre_t = tok.apply_chat_template([{"role": "user", "content": x["text"]}], tokenize=False, add_generation_prompt=True)
    st, _ = states(pre_t, x["reply_greedy"])
    n = st[LAYERS[0]].shape[0]
    for L in LAYERS: X[L].append(st[L])
    y += [EMOS.index("neutral")] * n; grp += ["task_" + x["id"]] * n
X = {L: np.concatenate(v) for L, v in X.items()}; y = np.array(y); grp = np.array(grp)
print(f"train tokens {len(y)} {time.time()-t0:.0f}s", flush=True)

# ---- 5-fold CV grouped by scenario -> layer selection ----
scen = np.array(sorted(set(grp))); rng = np.random.default_rng(0); rng.shuffle(scen); folds = np.array_split(scen, 5)
cv = {}
for L in LAYERS:
    f1s = []
    for f in folds:
        te = np.isin(grp, f); sc = StandardScaler().fit(X[L][~te])
        clf = LogisticRegression(max_iter=300, C=0.05).fit(sc.transform(X[L][~te]), y[~te])
        f1s.append(f1_score(y[te], clf.predict(sc.transform(X[L][te])), average="macro"))
    cv[L] = float(np.mean(f1s)); print(f"L{L} CV macro-F1 {cv[L]:.3f}", flush=True)
BEST = max(cv, key=cv.get)
sc = StandardScaler().fit(X[BEST]); clf = LogisticRegression(max_iter=500, C=0.05).fit(sc.transform(X[BEST]), y)
np.savez(f"{a.out}/probe_L{BEST}.npz", mean=sc.mean_, scale=sc.scale_, coef=clf.coef_, intercept=clf.intercept_, layer=BEST, classes=np.array(EMOS))

# ---- V3: validation reference records ----
yt, yp = [], []
for r in rows:
    if r["split"] != "validation": continue
    st, _ = states(prefix_of(r), r["style_lead"] + r["anchor_fact"] + a.tail + r["reference_suffix"])
    p = clf.predict_proba(sc.transform(st[BEST])).mean(0); yt.append(EMOS.index(r["emotion"])); yp.append(int(p.argmax()))
v3 = float(f1_score(yt, yp, average="macro"))

# ---- V1/V2: stage-A validation continuations vs blind ratings ----
key = json.load(open(f"{a.ratings}/blind_key.json")); R = [{x["item_id"]: x for x in json.load(open(f))} for f in glob.glob(f"{a.ratings}/ratings_llm_*_validation.json")]
rated = {}
for iid, k in key.items():
    if k["split"] != "validation" or not all(iid in r for r in R): continue
    rated[(k["scenario_id"], k["paraphrase"], k["condition"])] = {e: np.mean([r[iid][e] for r in R]) for e in ["joy", "sadness", "anger"]}
G = [json.loads(l) for l in open(a.gens)]
idx = {(r["scenario_id"], r["emotion"], r["paraphrase"]): r for r in rows}
read_text, read_lead, truth = [], [], []
for g in G:
    kk = (g["scenario_id"], g["paraphrase"], g["condition"])
    if g["split"] != "validation" or kk not in rated or g["condition"] == "anchor_sub_joy": continue
    emo = g["condition"].replace("prefix_", ""); r = idx[(g["scenario_id"], emo, g["paraphrase"])]
    before = prefix_of(r) + r["style_lead"] + r["anchor_fact"] + a.tail
    st, last = states(before, g["continuation"])
    read_text.append(clf.predict_proba(sc.transform(st[BEST])).mean(0)); read_lead.append(clf.predict_proba(sc.transform(last[BEST][None]))[0])
    truth.append([rated[kk][e] for e in ["joy", "sadness", "anger"]])
read_text, read_lead, truth = map(np.array, (read_text, read_lead, truth))
v1 = {e: float(spearmanr(read_text[:, EMOS.index(e)], truth[:, i]).correlation) for i, e in enumerate(["joy", "sadness", "anger"])}
v2 = {e: float(spearmanr(read_lead[:, EMOS.index(e)], truth[:, i]).correlation) for i, e in enumerate(["joy", "sadness", "anger"])}
# ---- V4: held-out (val/test split) prompt replies: neutral_task should read neutral; emotional categories keep their emotion ----
v4 = {}
for x in EXTRA:
    if x["split"] == "train": continue
    pre_t = tok.apply_chat_template([{"role": "user", "content": x["text"]}], tokenize=False, add_generation_prompt=True)
    st, _ = states(pre_t, x["reply_greedy"]); p = clf.predict_proba(sc.transform(st[BEST])).mean(0)
    v4.setdefault(x["category"], []).append(p)
v4 = {c: {e: round(float(np.mean([p[EMOS.index(e)] for p in v])), 3) for e in EMOS} for c, v in v4.items()}
res = dict(layer=BEST, cv_macro_f1_by_layer=cv, v3_validation_reference_macro_f1=v3, v4_heldout_prompt_replies=v4, extra_neutral_trained=bool(EXTRA) and not a.eval_only_extra,
           v1_text_time_spearman=v1, v2_lead_time_spearman=v2, n_continuations=len(truth),
           note="V1/V2 use blind LLM ratings of stage-A continuations (provisional). Validation split only; test untouched.",
           seconds=round(time.time() - t0, 1), peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
json.dump(res, open(f"{a.out}/readout_calib.json", "w"), indent=1, ensure_ascii=False); print(json.dumps(res, indent=1)); print("DONE")
