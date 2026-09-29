"""V1.1 generation (fixed DEFAULT persona everywhere).
python gen_v11.py MODEL drafts/ OUT.jsonl --set pilot|formal [--samples K --temp 0.9]
M1: history (2 user + 2 assistant turns at drift level polite/curt/sarcastic) + shared final_user -> 1 greedy reply per level.
M2: 7 user turns; the model writes its own reply after every user turn (greedy) -> 7 C1 items + 7 C2 items per template.
--samples K: additionally draw K sampled replies per M1 prompt and per M2 turn>=5 (M3 sampling variation).
pilot = M1 d101-d108 + M2 e101-e106; formal = the rest (template-level splits assigned later)."""
import json, sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL, D, OUT = sys.argv[1:4]
SET = sys.argv[sys.argv.index("--set") + 1]; K = int(sys.argv[sys.argv.index("--samples") + 1]) if "--samples" in sys.argv else 0
TEMP = float(sys.argv[sys.argv.index("--temp") + 1]) if "--temp" in sys.argv else 0.9
SYS = "你是{d}助手，请礼貌、专业地回答用户。"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
M1 = json.load(open(f"{D}/m1.json")); M2 = json.load(open(f"{D}/m2.json"))
pilot = lambda tid: tid in {f"d1{i:02d}" for i in range(1, 9)} | {f"e1{i:02d}" for i in range(1, 7)}
M1 = [t for t in M1 if pilot(t["template_id"]) == (SET == "pilot")]; M2 = [t for t in M2 if pilot(t["template_id"]) == (SET == "pilot")]


@torch.no_grad()
def reply(msgs, sample=False, seed=None):
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").cuda()
    kw = dict(max_new_tokens=120, pad_token_id=tok.eos_token_id)
    if sample: torch.manual_seed(seed); kw.update(do_sample=True, temperature=TEMP, top_p=0.95, top_k=50)
    else: kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    return tok.decode(model.generate(ids, **kw)[0, ids.shape[1]:], skip_special_tokens=True).strip()


items = []; t0 = time.time()
def add(**kw): items.append(kw)
for t in M1:
    sysm = {"role": "system", "content": SYS.format(d=t["domain"])}
    for lv in ["polite", "curt", "sarcastic"]:
        h = t["assistant_history"][lv]
        msgs = [sysm, {"role": "user", "content": t["user_before"][0]}, {"role": "assistant", "content": h[0]},
                {"role": "user", "content": t["user_before"][1]}, {"role": "assistant", "content": h[1]}, {"role": "user", "content": t["final_user"]}]
        base = dict(module="M1", template_id=t["template_id"], domain=t["domain"], level=lv, system=sysm["content"], messages=msgs[1:])
        add(**base, concept="C1", item_id=f"{t['template_id']}-{lv}-greedy", sample=None, target_text=reply(msgs))
        for s in range(K):
            add(**base, concept="C1", item_id=f"{t['template_id']}-{lv}-s{s}", sample=s, target_text=reply(msgs, True, 1000 + s))
for t in M2:
    sysm = {"role": "system", "content": SYS.format(d=t["domain"])}; conv = []
    for k, u in enumerate(t["user_turns"], 1):
        conv.append({"role": "user", "content": u})
        base = dict(module="M2", template_id=t["template_id"], domain=t["domain"], level=f"turn{k}", system=sysm["content"])
        add(**base, concept="C2", item_id=f"{t['template_id']}-t{k}-user", sample=None, messages=list(conv), target_text=u)
        r = reply([sysm] + conv)
        add(**base, concept="C1", item_id=f"{t['template_id']}-t{k}-greedy", sample=None, messages=list(conv), target_text=r)
        if k >= 5:
            for s in range(K):
                add(**base, concept="C1", item_id=f"{t['template_id']}-t{k}-s{s}", sample=s, messages=list(conv),
                    target_text=reply([sysm] + conv, True, 2000 + 10 * k + s))
        conv.append({"role": "assistant", "content": r})            # the model's own greedy reply continues the dialogue
    print(t["template_id"], f"{time.time()-t0:.0f}s", flush=True)
with open(OUT, "w") as f:
    for it in items: f.write(json.dumps(it, ensure_ascii=False) + "\n")
print("DONE", len(items), round(time.time() - t0, 1))
