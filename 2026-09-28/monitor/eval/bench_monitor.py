"""First-token latency with vs without the monitor (hooks + 3 probes). python bench_monitor.py MODEL labels_all.jsonl probes/ FREEZE_monitor.json out.json"""
import json, sys, os, time, statistics as st, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from transformers import AutoTokenizer, AutoModelForCausalLM
from core import ProbeSpec, Monitor
MODEL, LAB, PD, FZ, OUT = sys.argv[1:6]
tok = AutoTokenizer.from_pretrained(MODEL); model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
items = [json.loads(l) for l in open(LAB)]; items = [i for i in items if i["split"] == "test" and i["concept"] == "C1"][:40]
specs = {n: ProbeSpec.load(f"{PD}/{n}.json") for n in ["c2_user", "c1_pre", "c1_token"]}; pol = json.load(open(FZ))["policy"]
def first_token(msgs, mon=None):
    ids = torch.tensor([tok.apply_chat_template(msgs, add_generation_prompt=True)], device="cuda"); torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        out = model(ids, use_cache=True)
        if mon: mon.pre_check([m for m in msgs if m["role"] != "system"] if False else msgs)
        nxt = out.logits[0, -1].argmax(); out = model(nxt.view(1, 1), past_key_values=out.past_key_values, use_cache=True)
        if mon: mon.on_token()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) * 1000
msgs = [([{"role": "system", "content": i["system"]}] if i.get("system") else []) + i["messages"] for i in items]
for m in msgs[:3]: first_token(m)                               # warm-up
base = [first_token(m) for m in msgs]
mon = Monitor(model, tok, specs, pol); on = [first_token(m, mon) for m in msgs]; mon.tap.remove()
diff = [b - a for a, b in zip(base, on)]
res = dict(n=len(msgs), first_token_ms_median_without=round(st.median(base), 2), first_token_ms_median_with=round(st.median(on), 2), overhead_ms_median=round(st.median(diff), 2))
json.dump(res, open(OUT, "w"), indent=1); print(res)
