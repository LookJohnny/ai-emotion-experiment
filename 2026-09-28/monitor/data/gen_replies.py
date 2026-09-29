"""Generate assistant replies for C1 items and canonical histories for multi-turn items (greedy, deterministic).
python gen_replies.py MODEL items.jsonl items_filled.jsonl [--max_new 120]
- Multi-turn templates: intermediate assistant turns are generated ONCE per template with the default persona and shared
  by every item of that template (C2 escalation items and C1 reply_esc items), so histories are identical across personas.
- Output items have target_text filled and messages = full history ending with the target turn
  (C1: history ends with the user turn; the reply is target_text)."""
import json, sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL, src, out = sys.argv[1:4]; MAXN = int(sys.argv[sys.argv.index("--max_new") + 1]) if "--max_new" in sys.argv else 120
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
items = [json.loads(l) for l in open(src)]
DEFAULT_SYS = {i["template_id"]: i["system"] for i in items if i.get("persona") == "default" and i.get("system")}


@torch.no_grad()
def reply(system, msgs):
    m = ([{"role": "system", "content": system}] if system else []) + msgs
    ids = tok.apply_chat_template(m, add_generation_prompt=True, return_tensors="pt").cuda()
    g = model.generate(ids, max_new_tokens=MAXN, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
    return tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip()


# canonical multi-turn histories: user1, asst1, user2, asst2, ..., userK (assistant turns by default persona)
hist = {}
for it in items:
    if it.get("multi_turn") and it["template_id"] not in hist:
        users = max((x for x in items if x["template_id"] == it["template_id"] and x.get("multi_turn")), key=lambda x: len(x["messages"]))["messages"]
        conv = []
        for k, u in enumerate(users):
            conv.append(u)
            if k < len(users) - 1: conv.append({"role": "assistant", "content": reply(DEFAULT_SYS[it["template_id"]], conv)})
        hist[it["template_id"]] = conv
t0 = time.time()
for n, it in enumerate(items):
    if it.get("multi_turn"):
        conv = hist[it["template_id"]]; k = len(it["messages"])            # number of user turns in this item
        it["messages"] = conv[:2 * k - 1]                                      # history up to and including user turn k
    if it["target_text"] is None:
        it["target_text"] = reply(it.get("system"), it["messages"])
    if n % 50 == 0: print(f"{n+1}/{len(items)} {time.time()-t0:.0f}s", flush=True)
with open(out, "w") as f:
    for it in items: f.write(json.dumps(it, ensure_ascii=False) + "\n")
print("DONE", len(items), round(time.time() - t0, 1))
