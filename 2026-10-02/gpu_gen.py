"""GPU generation for PREREG_SYCO_FAB2.md. Batched, left-padded.
python gpu_gen.py syco MODEL4B syco/items_qwen.jsonl syco/qwen_replies.jsonl      # Qwen3-4B, T=0.7 top-k 20, 160 tokens, seed 0
python gpu_gen.py answers MODEL8B C3H_answers.jsonl fab2/qwen8b_answers.jsonl     # Qwen3-8B greedy, thinking off, 64 tokens (as C3-H)"""
import json, sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
KIND, MODEL, SRC, OUT = sys.argv[1:5]; BS = 32
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
items = [json.loads(l) for l in open(SRC, encoding="utf-8")]; t0 = time.time(); torch.manual_seed(0)


def template(r):
    msgs = [{"role": "system", "content": r["system"]}] + r["messages"]
    try: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)   # Qwen3-8B: no <think>
    except TypeError: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


with torch.no_grad(), open(OUT, "w", encoding="utf-8") as f:
    for b in range(0, len(items), BS):
        chunk = items[b:b + BS]
        enc = tok([template(r) for r in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        if KIND == "syco":
            out = model.generate(**enc, max_new_tokens=160, do_sample=True, temperature=0.7, top_k=20, top_p=1.0, pad_token_id=tok.eos_token_id)
        else:
            out = model.generate(**enc, max_new_tokens=64, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
        for r, o in zip(chunk, out[:, enc["input_ids"].shape[1]:]):
            f.write(json.dumps(dict(r, target_text=tok.decode(o, skip_special_tokens=True).strip()), ensure_ascii=False) + "\n")
        print(b + len(chunk), len(items), f"{time.time() - t0:.0f}s", flush=True)
print("DONE", len(items))
