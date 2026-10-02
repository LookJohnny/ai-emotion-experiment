"""Hidden-state features (teacher forcing, one forward per item) for PREREG_SYCO_FAB2.md.
python gpu_feats.py MODEL ITEMS.jsonl OUT.npz --reply_field target_text --max_reply 120 [--layers all|12,16,19,22,26] [--qa]
  ITEMS: rows with system, messages, and the reply text in --reply_field (Qwen's own reply, or another model's text = reader).
  Saves per item and layer: pre = last prompt position, tok = mean over the first max_reply reply tokens (float16)."""
import argparse, json, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("items"); ap.add_argument("out")
ap.add_argument("--reply_field", default="target_text"); ap.add_argument("--max_reply", type=int, default=120); ap.add_argument("--layers", default="all")
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
NL = model.config.num_hidden_layers; LAYERS = list(range(NL)) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
items = [json.loads(l) for l in open(a.items, encoding="utf-8")]


def template(r):
    msgs = [{"role": "system", "content": r["system"]}] + r["messages"]
    try: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


PRE = np.zeros((len(items), len(LAYERS), model.config.hidden_size), np.float16); TOK = np.zeros_like(PRE); NREP = np.zeros(len(items), np.int32)
t0 = time.time()
with torch.no_grad():
    for i, r in enumerate(items):
        pids = tok(template(r), add_special_tokens=False)["input_ids"]
        rids = tok(r.get(a.reply_field) or "", add_special_tokens=False)["input_ids"][:a.max_reply] or [tok.eos_token_id]
        hs = model(torch.tensor([pids + rids], device="cuda"), output_hidden_states=True, use_cache=False).hidden_states; n = len(pids)
        for j, L in enumerate(LAYERS):
            PRE[i, j] = hs[L + 1][0, n - 1].float().cpu().numpy(); TOK[i, j] = hs[L + 1][0, n:n + len(rids)].float().mean(0).cpu().numpy()
        NREP[i] = len(rids)
        if i % 200 == 0: print(i, len(items), f"{time.time() - t0:.0f}s", flush=True)
np.savez(a.out, ids=np.array([r["item_id"] for r in items]), layers=np.array(LAYERS), pre=PRE, tok=TOK, nrep=NREP)
print("DONE", len(items), round(time.time() - t0))
