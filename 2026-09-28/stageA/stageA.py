"""E01 Stage A: ceiling measurement (discovery + validation only; test split sealed).

Per scenario x paraphrase (A/B), greedy 64-token continuations for:
  prefix_neutral, prefix_joy, prefix_sadness, prefix_anger  (style_lead + common_anchor, no intervention)
  anchor_sub_joy : neutral recipient, residual stream (decoder-layer output) replaced at ALL 36 layers
                   on all common-anchor positions with the joy donor's states (prefill only)
Sanity checks:
  self_sub : neutral -> neutral substitution must reproduce prefix_neutral exactly
  first-token : anchor_sub_joy's first generated token must equal prefix_joy's (final-layer state identical)
python stageA.py MODEL_DIR formal_960.jsonl OUT_DIR
"""
import json, sys, time, os, platform, hashlib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR, DATA, OUT = sys.argv[1:4]
SUB_ONLY = len(sys.argv) > 4 and sys.argv[4] == "--sub-only"
os.makedirs(OUT, exist_ok=True)
SPLITS = {"discovery", "validation"}
MAX_NEW = 64
EMOS = ["neutral", "joy", "sadness", "anger"]

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
NL = model.config.num_hidden_layers
rows = [json.loads(l) for l in open(DATA)]
rows = [r for r in rows if r["split"] in SPLITS]
assert not any(r["split"] == "test" for r in rows)
idx = {(r["scenario_id"], r["emotion"], r["paraphrase"]): r for r in rows}
scen = sorted({r["scenario_id"] for r in rows})


def encode(r):
    prefix = tok.apply_chat_template([{"role": "user", "content": r["user_prompt"]}], tokenize=False, add_generation_prompt=True)
    e = tok(prefix + r["style_lead"] + r["common_anchor"], return_offsets_mapping=True, add_special_tokens=False)
    a_char = len(prefix) + len(r["style_lead"])
    a_tok = next(i for i, (s, _) in enumerate(e["offset_mapping"]) if s == a_char)
    return e["input_ids"], len(e["input_ids"]) - a_tok


st = {"mode": None}
cache = {}


def mk(L):
    def h(mod, args, out):
        is_t = isinstance(out, tuple)
        x = out[0] if is_t else out
        if x.shape[1] == 1:
            return None  # decode steps untouched
        if st["mode"] == "cap":
            cache[L] = x[0, -st["na"]:].detach().clone()
            return None
        if st["mode"] == "patch":
            x = x.clone()
            x[0, -st["na"]:] = st["donor"][L]
            return (x,) + tuple(out[1:]) if is_t else x
    return h


for L in range(NL):
    model.model.layers[L].register_forward_hook(mk(L))


@torch.no_grad()
def gen(ids):
    out = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=MAX_NEW, do_sample=False,
                         temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
    return out[0, len(ids):].tolist()


@torch.no_grad()
def capture(ids, na):
    # same code path as patched generation (generate prefill, use_cache=True) -> bitwise-identical states
    st.update(mode="cap", na=na)
    model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=1, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
    st["mode"] = None
    return {L: cache[L] for L in range(NL)}


results, checks = [], {"self_sub_identical": [], "first_token_match": []}
for n, sid in enumerate(scen):
    for para in ["A", "B"]:
        enc = {e: encode(idx[(sid, e, para)]) for e in EMOS}
        na = enc["neutral"][1]
        assert all(v[1] == na for v in enc.values()), f"anchor length mismatch {sid}"
        g = {}
        for e in EMOS:
            if SUB_ONLY and e not in ("neutral", "joy"):
                continue
            g[e] = gen(enc[e][0])
            results.append(dict(scenario_id=sid, split=idx[(sid, e, para)]["split"], paraphrase=para, condition=f"prefix_{e}",
                                gen_ids=g[e], continuation=tok.decode(g[e], skip_special_tokens=True)))
        donor = capture(enc["joy"][0], na)
        st.update(mode="patch", na=na, donor=donor); gs = gen(enc["neutral"][0]); st["mode"] = None
        results.append(dict(scenario_id=sid, split=idx[(sid, "joy", para)]["split"], paraphrase=para, condition="anchor_sub_joy",
                            gen_ids=gs, continuation=tok.decode(gs, skip_special_tokens=True)))
        checks["first_token_match"].append(gs[:1] == g["joy"][:1])
        if SUB_ONLY or (para == "A" and n % 10 == 0):  # self-substitution sanity
            selfd = capture(enc["neutral"][0], na)
            st.update(mode="patch", na=na, donor=selfd); gself = gen(enc["neutral"][0]); st["mode"] = None
            checks["self_sub_identical"].append(gself == g["neutral"])
    print(f"{n+1}/{len(scen)} {sid} {time.time()-t0:.0f}s", flush=True)

for r in results:
    r.pop("gen_ids")
if SUB_ONLY:  # merge: keep prefix_* from the original run (and verify they reproduce), replace anchor_sub_joy
    old = [json.loads(l) for l in open(f"{OUT}/stageA_generations.jsonl")]
    new_prefix = {(r["scenario_id"], r["paraphrase"], r["condition"]): r["continuation"] for r in results if r["condition"].startswith("prefix_")}
    checks["prefix_reproduced"] = [o["continuation"] == new_prefix[(o["scenario_id"], o["paraphrase"], o["condition"])] for o in old if (o["scenario_id"], o["paraphrase"], o["condition"]) in new_prefix]
    os.replace(f"{OUT}/stageA_generations.jsonl", f"{OUT}/stageA_generations_run1.jsonl")
    results = [o for o in old if o["condition"] != "anchor_sub_joy"] + [r for r in results if r["condition"] == "anchor_sub_joy"]
with open(f"{OUT}/stageA_generations.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
summary = dict(n_generations=len(results), n_scenarios=len(scen), seconds=round(time.time() - t0, 1),
               peak_alloc_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
               self_sub_identical=f"{sum(checks['self_sub_identical'])}/{len(checks['self_sub_identical'])}",
               first_token_match=f"{sum(checks['first_token_match'])}/{len(checks['first_token_match'])}",
               prefix_reproduced=(f"{sum(checks['prefix_reproduced'])}/{len(checks['prefix_reproduced'])}" if "prefix_reproduced" in checks else None),
               capture_path="generate_prefill_use_cache",
               data_sha256=hashlib.sha256(open(DATA, 'rb').read()).hexdigest(),
               env=dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(), python=platform.python_version()))
json.dump(summary, open(f"{OUT}/stageA_summary{'_subfix' if SUB_ONLY else ''}.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
