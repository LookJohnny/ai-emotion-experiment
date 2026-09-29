"""Pilot fallback (DESIGN.md): M1 4th drift level 'insulting' for d101-d108; same generation settings as gen_v11.py M1.
python gen_l4_pilot.py MODEL drafts/ OUT.jsonl [--samples 6]"""
import json, sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL, D, OUT = sys.argv[1:4]; K = int(sys.argv[sys.argv.index("--samples") + 1]) if "--samples" in sys.argv else 0
SYS = "你是{d}助手，请礼貌、专业地回答用户。"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
M1 = {t["template_id"]: t for t in json.load(open(f"{D}/m1.json"))}; INS = json.load(open(f"{D}/m1_insult_pilot.json"))


@torch.no_grad()
def reply(msgs, sample=False, seed=None):
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").cuda()
    kw = dict(max_new_tokens=120, pad_token_id=tok.eos_token_id)
    if sample: torch.manual_seed(seed); kw.update(do_sample=True, temperature=0.9, top_p=0.95, top_k=50)
    else: kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    return tok.decode(model.generate(ids, **kw)[0, ids.shape[1]:], skip_special_tokens=True).strip()


items = []; t0 = time.time()
for tid, h in INS.items():
    t = M1[tid]; sysm = {"role": "system", "content": SYS.format(d=t["domain"])}
    msgs = [sysm, {"role": "user", "content": t["user_before"][0]}, {"role": "assistant", "content": h[0]},
            {"role": "user", "content": t["user_before"][1]}, {"role": "assistant", "content": h[1]}, {"role": "user", "content": t["final_user"]}]
    base = dict(module="M1", template_id=tid, domain=t["domain"], level="insulting", system=sysm["content"], messages=msgs[1:], concept="C1", split="pilot")
    items.append(dict(base, item_id=f"{tid}-insulting-greedy", sample=None, target_text=reply(msgs)))
    for s in range(K): items.append(dict(base, item_id=f"{tid}-insulting-s{s}", sample=s, target_text=reply(msgs, True, 1000 + s)))
    print(tid, f"{time.time()-t0:.0f}s", flush=True)
with open(OUT, "w") as f:
    for it in items: f.write(json.dumps(it, ensure_ascii=False) + "\n")
print("DONE", len(items))
