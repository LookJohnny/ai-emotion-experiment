"""Freeze §5b after validation selection. Refuses to freeze if no lambda passed the quality gates.
python3 freeze_5b.py rating_steer_select/select_results.json results/steer_select/steer_select_validation_info.json FREEZE_5b.json"""
import json, sys, hashlib, os
sel, info, out = sys.argv[1:4]
S = json.load(open(sel)); I = json.load(open(info))
if S["selected_lambda"] is None:
    print("NO LAMBDA PASSED THE QUALITY GATES -> not frozen, test stays sealed"); sys.exit(2)
root = os.path.dirname(os.path.abspath(__file__)) + "/.."
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
files = ["stageD/STEER_PREREG.md", "stageD/steer_causal.py", "stageD/select_freeze.py", "stageD/final_eval.py", "stageD/freeze_5b.py",
         "data/formal_960.jsonl", "stageB/results/B3/b3_summary.json"]
fz = dict(frozen_at_note="frozen after validation blind rating, before any test generation", design="§5b continuous steering",
          primary=I["primary"], main_object=I["object"], heads=I["heads"], main_lambda=S["selected_lambda"],
          validation_selection=S["by_lambda"][str(S["selected_lambda"])], validation_all_lambdas=S["by_lambda"],
          steering_vector="per-head mean(anger) - mean(neutral) of o_proj input over anchor tokens, DISCOVERY split",
          random_seeds=[101, 102, 103, 104, 105], shuffle_seed=201, anchor_tail="我的看法是：", max_new_tokens=64,
          test_design=dict(split="test", paraphrase="A", conditions=["none", "primary_prefix", "target", "shuffled", "random_0..4"]),
          test_gates="see STEER_PREREG.md §Test; evaluated by final_eval.py --absolute",
          sha256={p: sha(f"{root}/{p}") for p in files if os.path.exists(f"{root}/{p}")})
json.dump(fz, open(out, "w"), ensure_ascii=False, indent=1); print("FROZEN", out, "lambda", fz["main_lambda"])
