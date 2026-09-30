"""Greedy answers for C3-H items (PREREG_C3H.md). Batched, left-padded. python gen_c3h.py MODEL items.jsonl OUT.jsonl"""
import json, sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL, SRC, OUT = sys.argv[1:4]; BS = 32
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
items = [json.loads(l) for l in open(SRC, encoding="utf-8")]; t0 = time.time()
with torch.no_grad(), open(OUT, "w", encoding="utf-8") as f:
    for b in range(0, len(items), BS):
        chunk = items[b:b + BS]
        texts = [tok.apply_chat_template([{"role": "system", "content": r["system"]}] + r["messages"], tokenize=False, add_generation_prompt=True) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        out = model.generate(**enc, max_new_tokens=64, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
        for r, o in zip(chunk, out[:, enc["input_ids"].shape[1]:]):
            f.write(json.dumps(dict(r, target_text=tok.decode(o, skip_special_tokens=True).strip()), ensure_ascii=False) + "\n")
        print(b + len(chunk), f"{time.time()-t0:.0f}s", flush=True)
print("DONE", len(items))
