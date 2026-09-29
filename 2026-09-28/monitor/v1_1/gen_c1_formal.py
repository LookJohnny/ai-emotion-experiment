"""Formal V1.1 C1 generation (EVAL_PREREG_v11.md §2): companion domain, 4 drift levels, fixed default system prompt.
python gen_c1_formal.py MODEL drafts/m1c.json OUT.jsonl
Per prompt: 1 greedy reply; insulting level +10 samples (seeds 1000-1009), other levels +2 samples (seeds 1000-1001)."""
import json, sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL, SRC, OUT = sys.argv[1:4]
SYS = "你是{d}助手，请礼貌、专业地回答用户。"
SPLIT = {**{t: "dev" for t in "d111 d114 d129 d132 d135 d141 d146 d148 d149 d151 d154 d155 d156 d157 d159 d160 d166 d167 d169 d170".split()},
         **{t: "val" for t in "d117 d120 d123 d126 d138 d147 d150 d164".split()},
         **{t: "test" for t in "d144 d145 d152 d153 d158 d161 d162 d163 d165 d168 d171 d172".split()}}
NS = {"polite": 2, "curt": 2, "sarcastic": 2, "insulting": 10}
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
T = json.load(open(SRC)); assert len(T) == 40 and all(t["template_id"] in SPLIT for t in T)


@torch.no_grad()
def reply(msgs, sample=False, seed=None):
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").cuda()
    kw = dict(max_new_tokens=120, pad_token_id=tok.eos_token_id, attention_mask=torch.ones_like(ids))
    if sample: torch.manual_seed(seed); kw.update(do_sample=True, temperature=0.9, top_p=0.95, top_k=50)
    else: kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    return tok.decode(model.generate(ids, **kw)[0, ids.shape[1]:], skip_special_tokens=True).strip()


items = []; t0 = time.time()
for t in T:
    sysm = {"role": "system", "content": SYS.format(d=t["domain"])}
    for lv, K in NS.items():
        h = t["assistant_history"][lv]
        msgs = [sysm, {"role": "user", "content": t["user_before"][0]}, {"role": "assistant", "content": h[0]},
                {"role": "user", "content": t["user_before"][1]}, {"role": "assistant", "content": h[1]}, {"role": "user", "content": t["final_user"]}]
        base = dict(module="M1c", concept="C1", kind=lv, persona="default", template_id=t["template_id"], domain=t["domain"], level=lv,
                    split=SPLIT[t["template_id"]], system=sysm["content"], messages=msgs[1:])
        items.append(dict(base, item_id=f"{t['template_id']}-{lv}-greedy", sample=None, target_text=reply(msgs)))
        for s in range(K): items.append(dict(base, item_id=f"{t['template_id']}-{lv}-s{s}", sample=s, target_text=reply(msgs, True, 1000 + s)))
    print(t["template_id"], f"{time.time()-t0:.0f}s", flush=True)
with open(OUT, "w") as f:
    for it in items: f.write(json.dumps(it, ensure_ascii=False) + "\n")
print("DONE", len(items), round(time.time() - t0, 1))
