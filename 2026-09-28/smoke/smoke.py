"""E01 smoke test: model -> data checks -> blank-prefix check -> MLP capture -> zero-lambda check
-> candidate vs 5 matched random vs label-shuffled intervention -> 10-step LoRA backward.

Debug-only. Scenarios d01-d08 used for candidate discovery, d09-d12 for the intervention demo.
Nothing here counts as a research result.
"""
import json, time, os, sys, hashlib, random, platform
import numpy as np
import torch
import torch.nn.functional as F

MODEL_DIR = sys.argv[1]
OUT = sys.argv[2]
DATA = sys.argv[3]
os.makedirs(OUT, exist_ok=True)
LAYERS = [8, 17, 26, 35]
K = 32
LAM = 1.0
RAND_SEEDS = [101, 102, 103, 104, 105]
SHUF_SEED = 201
DISC = [f"d{i:02d}" for i in range(1, 9)]
HELD = [f"d{i:02d}" for i in range(9, 13)]
MAX_NEW = 64

log = {"stages": {}}
T0 = time.time()


def stage(name, t, **kw):
    torch.cuda.synchronize()
    log["stages"][name] = dict(sec=round(time.time() - t, 2),
                               peak_alloc_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3),
                               peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 3), **kw)
    print(f"[{name}] {log['stages'][name]}", flush=True)


# ---------------- 0. environment ----------------
t = time.time()
import transformers, peft
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
mm_ok = bool(torch.isfinite(a @ a).all())
log["env"] = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(), gpu_total_gib=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
                  transformers=transformers.__version__, peft=peft.__version__, cuda_matmul_ok=mm_ok)
shas = {}
for fn in sorted(os.listdir(MODEL_DIR)):
    if fn.endswith(".safetensors"):
        h = hashlib.sha256()
        with open(os.path.join(MODEL_DIR, fn), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        shas[fn] = h.hexdigest()
log["model_file_sha256"] = shas
del a
stage("env", t)

from transformers import AutoTokenizer, AutoModelForCausalLM
t = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
cfg = model.config
log["model"] = dict(layers=cfg.num_hidden_layers, hidden=cfg.hidden_size, intermediate=cfg.intermediate_size)
assert cfg.num_hidden_layers == 36 and cfg.intermediate_size == 9728
stage("load", t)

# ---------------- 1. data / template checks ----------------
t = time.time()
rows = [json.loads(l) for l in open(DATA)]
EMO_WORDS = {"joy": ["愉悦", "高兴", "开心"], "sadness": ["悲伤", "难过"], "anger": ["愤怒", "生气"], "neutral": ["中性"]}
checks = {}
by_s = {}
for r in rows:
    by_s.setdefault(r["scenario_id"], {})[r["emotion"]] = r
checks["n_records"] = len(rows)
checks["n_scenarios"] = len(by_s)
checks["all_four_emotions"] = all(set(v) == {"joy", "sadness", "anger", "neutral"} for v in by_s.values())
checks["anchor_identical_within_scenario"] = all(len({r["common_anchor"] for r in v.values()}) == 1 for v in by_s.values())
checks["prompt_identical_within_scenario"] = all(len({r["user_prompt"] for r in v.values()}) == 1 for v in by_s.values())
checks["label_not_in_input"] = all(r["emotion"] not in r["user_prompt"] for r in rows)
checks["category_word_in_user_prompt"] = sum(any(w in r["user_prompt"] for ws in EMO_WORDS.values() for w in ws) for r in rows)

enc = []  # per record: ids, prefix_len, anchor_start_tok
tok_issues = []
for r in rows:
    prefix = tok.apply_chat_template([{"role": "user", "content": r["user_prompt"]}], tokenize=False, add_generation_prompt=True)
    full = prefix + r["style_lead"] + r["common_anchor"]
    e = tok(full, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = e["input_ids"], e["offset_mapping"]
    pids = tok(prefix, add_special_tokens=False)["input_ids"]
    if ids[:len(pids)] != pids:
        tok_issues.append((r["scenario_id"], r["emotion"], "prefix_merge"))
    a_char = len(prefix) + len(r["style_lead"])
    starts = [i for i, (s, _) in enumerate(offs) if s == a_char]
    if not starts:
        tok_issues.append((r["scenario_id"], r["emotion"], "anchor_boundary_merge"))
        a_tok = None
    else:
        a_tok = starts[0]
    enc.append(dict(ids=ids, plen=len(pids), a_tok=a_tok, lead_tokens=(a_tok - len(pids)) if a_tok else None))
    r["_i"] = len(enc) - 1
checks["tokenization_issues"] = tok_issues
anchor_tok_same = True
for sid, v in by_s.items():
    tails = {tuple(enc[r["_i"]]["ids"][enc[r["_i"]]["a_tok"]:]) for r in v.values() if enc[r["_i"]]["a_tok"] is not None}
    anchor_tok_same &= len(tails) == 1
checks["anchor_tokens_identical_within_scenario"] = anchor_tok_same
checks["total_tokens_max"] = max(len(e["ids"]) for e in enc)
checks["lead_tokens_by_emotion"] = {emo: float(np.mean([enc[r["_i"]]["lead_tokens"] for r in rows if r["emotion"] == emo])) for emo in ["joy", "sadness", "anger", "neutral"]}
checks["lead_chars_by_emotion"] = {emo: float(np.mean([len(r["style_lead"]) for r in rows if r["emotion"] == emo])) for emo in ["joy", "sadness", "anger", "neutral"]}
checks["passed"] = all([checks["all_four_emotions"], checks["anchor_identical_within_scenario"], checks["prompt_identical_within_scenario"],
                        checks["label_not_in_input"], not tok_issues, anchor_tok_same, checks["total_tokens_max"] <= 512])
log["data_checks"] = checks
print(json.dumps(checks, ensure_ascii=False, indent=1))
assert checks["passed"], "data checks failed"
stage("data_checks", t)

# ---------------- 2. capture MLP intermediate (input to down_proj) ----------------
mlps = [model.model.layers[L].mlp for L in range(cfg.num_hidden_layers)]
cap = {}
positions = {}


def mk_cap(L):
    def hook(mod, args):
        x = args[0]
        if x.shape[1] > 1:  # prefill only
            cap[L] = {k: x[0, p].detach().float().cpu() for k, p in positions.items()}
    return hook


t = time.time()
hs = [mlps[L].down_proj.register_forward_pre_hook(mk_cap(L)) for L in LAYERS]
A_anchor = np.zeros((len(rows), len(LAYERS), cfg.intermediate_size), dtype=np.float32)
A_prefix = np.zeros_like(A_anchor)
with torch.no_grad():
    for r in rows:
        e = enc[r["_i"]]
        positions.clear(); positions.update(anchor=len(e["ids"]) - 1, prefix=e["plen"] - 1)
        model(torch.tensor([e["ids"]], device="cuda"), use_cache=False)
        for j, L in enumerate(LAYERS):
            A_anchor[r["_i"], j] = cap[L]["anchor"].numpy()
            A_prefix[r["_i"], j] = cap[L]["prefix"].numpy()
for h in hs: h.remove()
assert np.isfinite(A_anchor).all()
np.savez_compressed(f"{OUT}/debug_activations.npz", anchor=A_anchor.astype(np.float16), prefix=A_prefix.astype(np.float16), layers=LAYERS)
stage("capture", t, shape=list(A_anchor.shape))

# ---------------- 3. blank-prefix check ----------------
# Same full prefix (prompt + assistant header) -> activation at last shared token must match across the 4 emotions.
t = time.time()
bl = {}
for j, L in enumerate(LAYERS):
    diffs, scales = [], []
    for sid, v in by_s.items():
        X = np.stack([A_prefix[r["_i"], j] for r in v.values()])
        diffs.append(np.abs(X - X[0]).max()); scales.append(np.abs(X[0]).max())
    bl[L] = dict(max_abs_diff=float(max(diffs)), max_abs_value=float(max(scales)))
# leakage probe: nearest-centroid leave-one-scenario-out on prefix vs anchor activations
def loso_acc(A, j):
    emos = ["joy", "sadness", "anger", "neutral"]
    correct = n = 0
    for s_out in by_s:
        tr = [r for r in rows if r["scenario_id"] != s_out]
        mu = {e: np.mean([A[r["_i"], j] for r in tr if r["emotion"] == e], 0) for e in emos}
        for r in by_s[s_out].values():
            x = A[r["_i"], j]
            pred = min(emos, key=lambda e: np.linalg.norm(x - mu[e]))
            # ties (identical inputs) count as chance: use deterministic tie -> first emo, so report separately
            correct += pred == r["emotion"]; n += 1
    return correct / n
bl_probe = {L: dict(prefix_acc=loso_acc(A_prefix, j), anchor_acc=loso_acc(A_anchor, j)) for j, L in enumerate(LAYERS)}
blank_pass = all(v["max_abs_diff"] <= 1e-2 * max(1.0, v["max_abs_value"]) for v in bl.values())
log["blank_prefix_check"] = dict(per_layer=bl, nearest_centroid_loso=bl_probe, chance=0.25, passed=blank_pass,
                                 note="prefix_acc should be 0.25 (identical inputs -> ties); anchor_acc is exploratory, 12 scenarios only")
print(json.dumps(log["blank_prefix_check"], indent=1))
stage("blank_check", t)

# ---------------- 4. candidate selection on d01-d08 ----------------
t = time.time()
idx = {(r["scenario_id"], r["emotion"]): r["_i"] for r in rows}


def top_units(A, j, pairs, k=K):
    D = np.stack([A[idx[(s, "joy")], j] - A[idx[(s, "neutral")], j] for s in pairs] if pairs and isinstance(pairs[0], str)
                 else [A[a, j] - A[b, j] for a, b in pairs])
    eff = D.mean(0) / (D.std(0, ddof=1) + 1e-6)
    o = np.argsort(-np.abs(eff))[:k]
    return o, eff


cands, shuf, rands = {}, {}, {}
rng_s = random.Random(SHUF_SEED)
shuf_pairs = []
for s in DISC:
    ems = ["joy", "sadness", "anger", "neutral"]
    perm = ems[:]; rng_s.shuffle(perm)
    lab = dict(zip(perm, ems))  # shuffled label -> true record
    shuf_pairs.append((idx[(s, lab["joy"])], idx[(s, lab["neutral"])]))
disc_rows = [idx[(s, e)] for s in DISC for e in ["joy", "sadness", "anger", "neutral"]]
sel_info = {}
for j, L in enumerate(LAYERS):
    o, eff = top_units(A_anchor, j, DISC)
    cands[L] = o
    so, _ = top_units(A_anchor, j, shuf_pairs)
    shuf[L] = so
    var = A_anchor[disc_rows, j].var(0)
    bins = np.digitize(var, np.quantile(var, np.linspace(0, 1, 11)[1:-1]))
    rands[L] = {}
    for sd in RAND_SEEDS:
        g = np.random.default_rng(sd)
        pick = []
        for b in bins[o]:
            pool = np.setdiff1d(np.where(bins == b)[0], np.concatenate([o, np.array(pick, dtype=int)]))
            pick.append(int(g.choice(pool)))
        rands[L][sd] = np.array(pick)
    sel_info[L] = dict(top_units=o.tolist(), top_abs_effect=np.abs(eff[o]).round(2).tolist()[:8], mean_abs_effect_top32=float(np.abs(eff[o]).mean()),
                       n_positive=int((eff[o] > 0).sum()), overlap_with_shuffled=int(len(set(o) & set(so))))
log["candidates"] = sel_info
json.dump({str(L): dict(target=cands[L].tolist(), shuffled=shuf[L].tolist(), random={str(k): v.tolist() for k, v in rands[L].items()}) for L in LAYERS},
          open(f"{OUT}/unit_sets.json", "w"))
stage("select", t)

# ---------------- 5. intervention hook ----------------
state = {"active": False}


def mk_patch(L):
    def hook(mod, args):
        if not state["active"] or state["layer"] != L:
            return None
        x = args[0]
        if x.shape[1] == 1:  # decode steps: no clamping of future activations
            return None
        x = x.clone()
        S = state["units"]
        x[0, -1, S] = x[0, -1, S] + (state["coef"] * state["delta"][S]).to(x.dtype)
        return (x,)
    return hook


hs = [mlps[L].down_proj.register_forward_pre_hook(mk_patch(L)) for L in LAYERS]


def gen(ids):
    with torch.no_grad():
        out = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=MAX_NEW, do_sample=False,
                             temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
    return out[0, len(ids):].tolist()


def next_logits(ids):
    with torch.no_grad():
        return model(torch.tensor([ids], device="cuda"), use_cache=False).logits[0, -1].float()


# zero-lambda check
t = time.time()
zc = []
for s in HELD:
    e = enc[idx[(s, "neutral")]]
    state["active"] = False
    base_l = next_logits(e["ids"]); base_g = gen(e["ids"])
    delta = torch.tensor(A_anchor[idx[(s, "joy")], LAYERS.index(26)] - A_anchor[idx[(s, "neutral")], LAYERS.index(26)], device="cuda")
    state.update(active=True, layer=26, units=torch.tensor(cands[26], device="cuda"), delta=delta, coef=0.0)
    z_l = next_logits(e["ids"]); z_g = gen(e["ids"])
    state["active"] = False
    zc.append(dict(scenario=s, max_logit_diff=float((base_l - z_l).abs().max()), generation_identical=base_g == z_g))
zero_pass = all(z["max_logit_diff"] == 0 and z["generation_identical"] for z in zc)
log["zero_lambda_check"] = dict(items=zc, passed=zero_pass)
print(json.dumps(log["zero_lambda_check"], indent=1))
stage("zero_check", t)

# ---------------- 6. intervention runs ----------------
t = time.time()
Wd = {L: mlps[L].down_proj.weight.detach().float() for L in LAYERS}  # [hidden, 9728]


def resid_norm(L, units, delta):
    u = torch.as_tensor(units, device="cuda")
    return float((Wd[L][:, u] @ delta[u]).norm())


gens = []
for L in LAYERS:
    j = LAYERS.index(L)
    for s in HELD:
        rec = enc[idx[(s, "neutral")]]
        delta = torch.tensor(A_anchor[idx[(s, "joy")], j] - A_anchor[idx[(s, "neutral")], j], device="cuda")
        state["active"] = False
        base_logits = next_logits(rec["ids"])
        p0 = F.log_softmax(base_logits, -1)
        conds = [("no_intervention", None)] + [("target_top32", cands[L])] + [(f"random_{sd}", rands[L][sd]) for sd in RAND_SEEDS] + [("label_shuffled", shuf[L])]
        tn = resid_norm(L, cands[L], LAM * delta)
        for name, units in conds:
            if units is None:
                state["active"] = False; coef = 0.0; scale = None
            else:
                rn = resid_norm(L, units, delta)
                scale = 1.0 if name == "target_top32" else (tn / rn if rn > 0 else float("nan"))
                coef = LAM * scale
                state.update(active=True, layer=L, units=torch.tensor(units, device="cuda"), delta=delta, coef=coef)
            lg = next_logits(rec["ids"])
            kl = float(F.kl_div(p0, F.log_softmax(lg, -1), log_target=True, reduction="sum")) if units is not None else 0.0
            g = gen(rec["ids"])
            state["active"] = False
            gens.append(dict(layer=L, scenario=s, condition=name, norm_match_scale=scale, resid_change_norm=tn if units is not None else 0.0,
                             next_token_kl_vs_base=kl, top1_next=tok.decode(int(lg.argmax())),
                             prefix_tail=rows[idx[(s, "neutral")]]["style_lead"] + rows[idx[(s, "neutral")]]["common_anchor"],
                             continuation=tok.decode(g, skip_special_tokens=True)))
for h in hs: h.remove()
with open(f"{OUT}/generations.jsonl", "w") as f:
    for g in gens: f.write(json.dumps(g, ensure_ascii=False) + "\n")
kl_summary = {}
for L in LAYERS:
    gg = [g for g in gens if g["layer"] == L]
    kl_summary[L] = dict(target=float(np.mean([g["next_token_kl_vs_base"] for g in gg if g["condition"] == "target_top32"])),
                         random_mean=float(np.mean([g["next_token_kl_vs_base"] for g in gg if g["condition"].startswith("random")])),
                         shuffled=float(np.mean([g["next_token_kl_vs_base"] for g in gg if g["condition"] == "label_shuffled"])),
                         scales_random=[round(g["norm_match_scale"], 2) for g in gg if g["condition"].startswith("random")][:5],
                         changed_vs_base=sum(g["continuation"] != next(b["continuation"] for b in gg if b["scenario"] == g["scenario"] and b["condition"] == "no_intervention")
                                             for g in gg if g["condition"] == "target_top32"))
log["intervention_summary"] = kl_summary
print(json.dumps(kl_summary, indent=1, ensure_ascii=False))
stage("intervene", t, n_generations=len(gens))

# ---------------- 7. LoRA 10-step backward ----------------
t = time.time()
from peft import LoraConfig, get_peft_model
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
pm = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
pm.train()
opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=1e-4, weight_decay=0)
joy_rows = [r for r in rows if r["emotion"] == "joy"]
losses = []
torch.manual_seed(11)
for step in range(10):
    r = joy_rows[step % len(joy_rows)]
    e = enc[r["_i"]]
    ids = torch.tensor([e["ids"]], device="cuda")
    labels = ids.clone(); labels[0, :e["plen"]] = -100  # assistant-only loss
    loss = pm(input_ids=ids, labels=labels).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in pm.parameters() if p.requires_grad], 1.0)
    opt.step(); opt.zero_grad()
    losses.append(round(float(loss), 4))
log["lora_smoke"] = dict(losses=losses, trainable_params=sum(p.numel() for p in pm.parameters() if p.requires_grad), finite=all(np.isfinite(losses)))
print(log["lora_smoke"])
stage("lora10", t)

log["total_sec"] = round(time.time() - T0, 1)
log["today_criteria"] = dict(model_ok=mm_ok, data_checks=checks["passed"], blank_prefix=blank_pass, mlp_capture=True,
                             zero_lambda=zero_pass, intervention_with_random_controls=len(gens) == len(LAYERS) * len(HELD) * 8,
                             lora_backward=log["lora_smoke"]["finite"])
json.dump(log, open(f"{OUT}/smoke_log.json", "w"), ensure_ascii=False, indent=1, default=str)
print("CRITERIA", log["today_criteria"], "TOTAL", log["total_sec"])
