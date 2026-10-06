"""Reader features for PREREG_DANGER.md: Qwen3-4B-Instruct-2507 reads `user: question` -> `assistant: reply` under teacher forcing.
python danger_feats.py MODEL REPLIES.jsonl OUT.npz   # per item and layer: mean state over the first 160 reply tokens (float16). cuda or mps."""
import json, sys, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL, SRC, OUT = sys.argv[1:4]
dev = "cuda" if torch.cuda.is_available() else "mps"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
items = [r for r in map(json.loads, open(SRC, encoding="utf-8")) if not r.get("error") and (r.get("content") or "").strip()]
NL = model.config.num_hidden_layers
TOK = np.zeros((len(items), NL, model.config.hidden_size), np.float16); t0 = time.time()
with torch.no_grad():
    for i, r in enumerate(items):
        pids = tok.apply_chat_template([{"role": "user", "content": r["question"]}], tokenize=True, add_generation_prompt=True)
        if isinstance(pids, dict) or hasattr(pids, "input_ids"): pids = pids["input_ids"]
        rids = tok(r["content"], add_special_tokens=False)["input_ids"][:160]
        hs = model(torch.tensor([list(pids) + rids], device=dev), output_hidden_states=True, use_cache=False).hidden_states; n = len(pids)
        for L in range(NL): TOK[i, L] = hs[L + 1][0, n:].float().mean(0).cpu().numpy()
        if i % 100 == 0: print(i, len(items), f"{time.time() - t0:.0f}s", flush=True)
np.savez(OUT, ids=np.array([r["item_id"] for r in items]), layers=np.arange(NL), tok=TOK)
print("DONE", len(items), round(time.time() - t0))
