"""Tokenizer-level checks for formal data (CPU only): python tok_check.py MODEL_DIR formal_960.jsonl controls_48.jsonl"""
import json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
rows = [json.loads(l) for f in sys.argv[2:] for l in open(f)]
issues, tails, maxlen, lead_tok = [], {}, 0, {}
for r in rows:
    prefix = tok.apply_chat_template([{"role": "user", "content": r["user_prompt"]}], tokenize=False, add_generation_prompt=True)
    full = prefix + r["style_lead"] + r["common_anchor"]
    e = tok(full, return_offsets_mapping=True, add_special_tokens=False)
    ids = e["input_ids"]; pids = tok(prefix, add_special_tokens=False)["input_ids"]
    if ids[:len(pids)] != pids: issues.append((r["record_id"], "prefix_merge"))
    a_char = len(prefix) + len(r["style_lead"])
    st = [i for i, (s, _) in enumerate(e["offset_mapping"]) if s == a_char]
    if not st: issues.append((r["record_id"], "anchor_boundary_merge")); continue
    tails.setdefault(r["scenario_id"], set()).add(tuple(ids[st[0]:]))
    n_suffix = len(tok(r["reference_suffix"], add_special_tokens=False)["input_ids"])
    maxlen = max(maxlen, len(ids) + n_suffix)
    if r["record_type"] == "main": lead_tok.setdefault(r["emotion"], []).append(st[0] - len(pids))
bad_tail = [s for s, t in tails.items() if len(t) != 1]
out = dict(n=len(rows), boundary_issues=issues, scenarios_with_nonidentical_anchor_tokens=bad_tail, max_total_tokens_with_suffix=maxlen,
           lead_tokens_mean={k: round(sum(v) / len(v), 2) for k, v in lead_tok.items()},
           passed=not issues and not bad_tail and maxlen <= 512)
print(json.dumps(out, ensure_ascii=False, indent=1))
