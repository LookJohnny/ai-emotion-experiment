"""Stage A grid (plan §5.1 on-fail procedure): anchor tail x continuation length, prefix_neutral vs prefix_joy only.
Selection on validation; confirmation on discovery (run with --split discovery --tails <chosen>). Test split never loaded.
python stageA_grid.py MODEL_DIR formal_960.jsonl OUT_DIR --split validation [--tails T0,T1,T2,T3]"""
import json, sys, time, os, argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("data"); ap.add_argument("out")
ap.add_argument("--split", default="validation", choices=["validation", "discovery"])
ap.add_argument("--tails", default="T0,T1,T2,T3")
a = ap.parse_args()
TAILS = {"T0": "我的看法是：", "T1": "", "T2": "我的感受是：", "T3": "总之，"}
use = a.tails.split(",")
os.makedirs(a.out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
rows = [json.loads(l) for l in open(a.data)]
rows = [r for r in rows if r["split"] == a.split and r["emotion"] in ("neutral", "joy")]
assert all(r["split"] != "test" for r in rows)
t0 = time.time(); out = []; tokcheck = []
for tk in use:
    tail = TAILS[tk]
    for r in rows:
        assert r["common_anchor"] == r["anchor_fact"] + r["anchor_tail"]
        prefix = tok.apply_chat_template([{"role": "user", "content": r["user_prompt"]}], tokenize=False, add_generation_prompt=True)
        text = prefix + r["style_lead"] + r["anchor_fact"] + tail
        e = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        a_char = len(prefix) + len(r["style_lead"])
        tokcheck.append(any(s == a_char for s, _ in e["offset_mapping"]))
        ids = e["input_ids"]
        with torch.no_grad():
            g = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=64, do_sample=False,
                               temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)[0, len(ids):].tolist()
        out.append(dict(scenario_id=r["scenario_id"], split=r["split"], paraphrase=r["paraphrase"], tail=tk, tail_text=tail,
                        condition=f"prefix_{r['emotion']}", gen_ids=g,
                        cont64=tok.decode(g, skip_special_tokens=True), cont32=tok.decode(g[:32], skip_special_tokens=True)))
    print(tk, len(out), f"{time.time()-t0:.0f}s", flush=True)
with open(f"{a.out}/grid_{a.split}.jsonl", "w") as f:
    for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
print(json.dumps(dict(n=len(out), anchor_boundary_ok=f"{sum(tokcheck)}/{len(tokcheck)}", seconds=round(time.time() - t0, 1),
                      peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))))
