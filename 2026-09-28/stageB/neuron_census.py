"""Neuron selectivity census: are MLP neurons organised by emotion, by topic (knowledge domain), or by task type?

Unit: MLP intermediate z = input of down_proj, 36 layers x 9728. Correlational only (no intervention).
Set 1 (formal data, DISCOVERY + VALIDATION only; test sealed): 80 scenarios x 4 emotions x 2 paraphrases = 640 records.
  Segment means of z over: lead (style_lead tokens), anchor (anchor_fact + tail), suffix (reference_suffix, teacher forced).
  eta2_emotion = variance explained by emotion AFTER removing each scenario's mean (within-scenario design);
  eta2_topic   = variance explained by topic (4 topics) of the scenario means.
Set 2 (prompts_pre.jsonl, 400 user messages, 5 categories): mean z over user-message tokens; eta2_category.
Significance: permutation null (emotion shuffled within scenario; topic shuffled across scenarios; category shuffled),
  threshold = 99.9th percentile of the null over all neurons of that layer.
Generalisation: does 'the speaker is angry' selectivity (set 1, lead) predict 'user reports a grievance' selectivity (set 2)?
python neuron_census.py MODEL formal_960.jsonl prompts_pre.jsonl OUT --tail "我的看法是："
"""
import json, argparse, os, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("data"); ap.add_argument("prompts"); ap.add_argument("out")
ap.add_argument("--tail", required=True); ap.add_argument("--para", default="all", choices=["all", "A", "B"],
    help="B = paraphrase without emotion-category words in the lead (strips the most direct lexical cue)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
EMOS = ["neutral", "joy", "sadness", "anger"]; SEGS = ["lead", "anchor", "suffix"]
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
NL = model.config.num_hidden_layers; NU = model.config.intermediate_size
rows = [json.loads(l) for l in open(a.data)]
rows = [r for r in rows if r["split"] in ("discovery", "validation") and (a.para == "all" or r["paraphrase"] == a.para)]
assert rows and not any(r["split"] == "test" for r in rows)
prompts = [json.loads(l) for l in open(a.prompts)]

z = {}
def mk(L):
    def h(mod, args): z[L] = args[0][0].float()
    return h
for L in range(NL): model.model.layers[L].mlp.down_proj.register_forward_pre_hook(mk(L))


@torch.no_grad()
def seg_means(text, spans):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    model(torch.tensor([ids], device="cuda"), use_cache=False)
    return {name: torch.stack([z[L][s:e].mean(0) for L in range(NL)]).half().cpu().numpy() for name, (s, e) in spans.items()}


def spans_by_chars(text, bounds):
    off = tok(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    out = {}
    for name, (c0, c1) in bounds.items():
        idx = [i for i, (s, t) in enumerate(off) if s >= c0 and t <= c1 and t > s]
        out[name] = (idx[0], idx[-1] + 1)
    return out


t0 = time.time()
A = {s: np.zeros((len(rows), NL, NU), dtype=np.float16) for s in SEGS}
for i, r in enumerate(rows):
    prefix = tok.apply_chat_template([{"role": "user", "content": r["user_prompt"]}], tokenize=False, add_generation_prompt=True)
    lead, anc, suf = r["style_lead"], r["anchor_fact"] + a.tail, r["reference_suffix"]
    text = prefix + lead + anc + suf; p = len(prefix)
    sp = spans_by_chars(text, {"lead": (p, p + len(lead)), "anchor": (p + len(lead), p + len(lead) + len(anc)), "suffix": (p + len(lead) + len(anc), len(text))})
    m = seg_means(text, sp)
    for s in SEGS: A[s][i] = m[s]
    if i % 100 == 0: print(f"set1 {i}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
Pz = np.zeros((len(prompts), NL, NU), dtype=np.float16)
for i, p in enumerate(prompts):
    text = tok.apply_chat_template([{"role": "user", "content": p["text"]}], tokenize=False, add_generation_prompt=True)
    c0 = text.index(p["text"]); Pz[i] = seg_means(text, spans_by_chars(text, {"u": (c0, c0 + len(p["text"]))}))["u"]
print(f"activations done {time.time()-t0:.0f}s", flush=True)

scen = np.array([r["scenario_id"] for r in rows]); emo = np.array([EMOS.index(r["emotion"]) for r in rows])
topic_of = {r["scenario_id"]: r["topic"] for r in rows}; topics = sorted(set(topic_of.values()))
top = np.array([topics.index(topic_of[s]) for s in scen]); us = sorted(set(scen)); si = np.array([us.index(s) for s in scen])
cat_names = sorted({p["category"] for p in prompts}); cat = np.array([cat_names.index(p["category"]) for p in prompts])
rng = np.random.default_rng(0)


def group_mean(X, g, k):  # X [n, U] -> per-row group mean
    M = np.zeros((k, X.shape[1])); np.add.at(M, g, X); M /= np.bincount(g, minlength=k)[:, None]; return M[g]


def eta_emotion(X, e):
    tot = ((X - X.mean(0)) ** 2).sum(0) + 1e-9
    Y = X - group_mean(X, si, len(us)); return (group_mean(Y, e, 4) ** 2).sum(0) / tot


def eta_topic(X, t):
    tot = ((X - X.mean(0)) ** 2).sum(0) + 1e-9
    S = group_mean(X, si, len(us)); return ((group_mean(S, t, len(topics)) - X.mean(0)) ** 2).sum(0) / tot


def eta_cat(X, c):
    tot = ((X - X.mean(0)) ** 2).sum(0) + 1e-9
    return ((group_mean(X, c, len(cat_names)) - X.mean(0)) ** 2).sum(0) / tot


def shuffle_within(e):
    e2 = e.copy()
    for k in range(len(us)):
        ix = np.where(si == k)[0]; e2[ix] = rng.permutation(e[ix])
    return e2


res = {"paraphrase": a.para, "n_records": len(rows), "n_prompts": len(prompts), "topics": topics, "categories": cat_names, "layers": {}}
top_units = []
for L in range(NL):
    lay = {}
    for s in SEGS:
        X = A[s][:, L, :].astype(np.float64)
        ee, et = eta_emotion(X, emo), eta_topic(X, top)
        ne = eta_emotion(X, shuffle_within(emo))  # emotion null: shuffled within scenario
        # topic null: permute topic labels across scenarios
        tp = np.array(rng.permutation([topics.index(topic_of[u]) for u in us]))[si]
        nt = eta_topic(X, tp)
        the, tht = np.percentile(ne, 99.9), np.percentile(nt, 99.9)
        E, T = ee > the, et > tht
        lay[s] = dict(thr_emotion=round(float(the), 4), thr_topic=round(float(tht), 4),
                      n_emotion_only=int((E & ~T).sum()), n_topic_only=int((T & ~E).sum()), n_both=int((E & T).sum()), n_neither=int((~E & ~T).sum()),
                      median_eta_emotion=round(float(np.median(ee)), 4), median_eta_topic=round(float(np.median(et)), 4),
                      top1pct_eta_emotion=round(float(np.percentile(ee, 99)), 4), top1pct_eta_topic=round(float(np.percentile(et, 99)), 4))
        if s == "lead":
            Y = X - group_mean(X, si, len(us)); sd = Y.std(0) + 1e-6
            ang_sel = (Y[emo == 3].mean(0) - Y[emo != 3].mean(0)) / sd          # speaker-anger selectivity
            # consistency: preferred emotion identical across topics?
            pref_by_topic = np.stack([np.argmax(np.stack([Y[(top == t) & (emo == k)].mean(0) for k in range(4)]), 0) for t in range(len(topics))])
            for u in np.argsort(-ee)[:5]:
                top_units.append(dict(layer=L, unit=int(u), eta_emotion=round(float(ee[u]), 3), eta_topic=round(float(et[u]), 3),
                                      preferred=EMOS[int(np.argmax([Y[emo == k, u].mean() for k in range(4)]))],
                                      same_preference_in_all_topics=bool((pref_by_topic[:, u] == pref_by_topic[0, u]).all())))
    Xp = Pz[:, L, :].astype(np.float64); ec = eta_cat(Xp, cat); nc = eta_cat(Xp, rng.permutation(cat)); thc = np.percentile(nc, 99.9)
    sdp = Xp.std(0) + 1e-6; g = cat_names.index("grievance")
    griev_sel = (Xp[cat == g].mean(0) - Xp[cat != g].mean(0)) / sdp
    k = int(NU * 0.01); A1, B1 = set(np.argsort(-ang_sel)[:k]), set(np.argsort(-griev_sel)[:k])
    lay["prompts"] = dict(thr_category=round(float(thc), 4), n_category_selective=int((ec > thc).sum()), top1pct_eta_category=round(float(np.percentile(ec, 99)), 4))
    lay["anger_generalisation"] = dict(corr_speaker_anger_vs_user_grievance=round(float(np.corrcoef(ang_sel, griev_sel)[0, 1]), 3),
                                       top1pct_jaccard=round(len(A1 & B1) / len(A1 | B1), 3), chance_jaccard=round(k / (2 * NU - k), 4))
    res["layers"][L] = lay
    print(f"L{L:2d} lead: emo-only {lay['lead']['n_emotion_only']:5d} topic-only {lay['lead']['n_topic_only']:5d} both {lay['lead']['n_both']:5d} | "
          f"prompts cat-sel {lay['prompts']['n_category_selective']:5d} | anger~grievance r={lay['anger_generalisation']['corr_speaker_anger_vs_user_grievance']:.2f} "
          f"J={lay['anger_generalisation']['top1pct_jaccard']:.3f}", flush=True)
res["top_emotion_units_lead"] = sorted(top_units, key=lambda x: -x["eta_emotion"])[:60]
res["seconds"] = round(time.time() - t0, 1)
json.dump(res, open(f"{a.out}/census.json", "w"), ensure_ascii=False, indent=1); print("DONE")
