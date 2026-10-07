"""E02 features (PLAN.md): per record and layer, mean residual and per-head attention output over the lead and anchor sites.
python feats.py MODEL_DIR OUT.npz explore|holdout|discovery
Teacher forcing: chat prefix (user prompt) + style_lead + common_anchor; one forward pass; float16 storage. cuda or mps."""
import json, os, sys, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL, OUT, PART = sys.argv[1:4]
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "2026-09-28", "data", "formal_960.jsonl")
SPLITS = {"explore": ("discovery", "validation"), "holdout": ("test",), "discovery": ("discovery",)}[PART]
rows = [r for r in map(json.loads, open(DATA, encoding="utf-8")) if r["split"] in SPLITS]
dev = "cuda" if torch.cuda.is_available() else "mps"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
cfg = model.config; NL, NH = cfg.num_hidden_layers, cfg.num_attention_heads
HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH
layers = model.model.layers
heads_in = {}
for i, L in enumerate(layers):
    L.self_attn.o_proj.register_forward_pre_hook(lambda mod, args, i=i: heads_in.__setitem__(i, args[0][0].float()))


def prefix_of(r):
    m = [{"role": "user", "content": r["user_prompt"]}]
    try: return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)


def span_idx(offs, a, b):
    idx = [i for i, (s, e) in enumerate(offs) if s >= a and e <= b and e > s]
    return idx or [i for i, (s, e) in enumerate(offs) if s < b and e > a]      # a boundary merge: take the overlapping tokens


N = len(rows)
RES = np.zeros((2, N, NL, cfg.hidden_size), np.float16)          # site (lead, anchor) x record x layer x d
HEAD = np.zeros((2, N, NL, NH * HD), np.float16)
NTOK = np.zeros((N, 2), np.int16); t0 = time.time()
with torch.no_grad():
    for n, r in enumerate(rows):
        p = prefix_of(r); full = p + r["style_lead"] + r["common_anchor"]
        enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
        a0 = len(p); a1 = a0 + len(r["style_lead"]); a2 = a1 + len(r["common_anchor"])
        sites = [span_idx(enc["offset_mapping"], a0, a1), span_idx(enc["offset_mapping"], a1, a2)]
        hs = model(torch.tensor([enc["input_ids"]], device=dev), output_hidden_states=True, use_cache=False).hidden_states
        for s, idx in enumerate(sites):
            ix = torch.tensor(idx, device=dev)
            for L in range(NL):
                RES[s, n, L] = hs[L + 1][0, ix].float().mean(0).cpu().numpy()
                HEAD[s, n, L] = heads_in[L][ix].mean(0).cpu().numpy()
        NTOK[n] = [len(x) for x in sites]
        if n % 100 == 0: print(n, N, f"{time.time() - t0:.0f}s", flush=True)
meta = dict(model=os.path.basename(MODEL.rstrip("/")), layers=NL, heads=NH, head_dim=HD, hidden=cfg.hidden_size, part=PART)
np.savez(OUT, res=RES, head=HEAD, ntok=NTOK, ids=np.array([r["record_id"] for r in rows]), emotion=np.array([r["emotion"] for r in rows]),
         scenario=np.array([r["scenario_id"] for r in rows]), topic=np.array([r["topic"] for r in rows]),
         paraphrase=np.array([r["paraphrase"] for r in rows]), meta=json.dumps(meta))
print("DONE", N, round(time.time() - t0), meta)
