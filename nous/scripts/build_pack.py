"""Assemble the Qwen3-4B concept pack from the frozen research artifacts (E01 / V1.1 / C3-H / reader experiment).
python scripts/build_pack.py ../2026-09-28 packs/qwen3-4b"""
import json, os, shutil, sys
SRC, OUT = sys.argv[1:3]
M = os.path.join(SRC, "monitor"); os.makedirs(OUT, exist_ok=True)
cp = lambda a, b: (shutil.copy(os.path.join(SRC, a), os.path.join(OUT, b)), b)[1]
probes = {"c1_pre": cp("monitor/v1_1/c1_out/c1_pre.json", "c1_pre.json"), "c1_token": cp("monitor/v1_1/c1_out/c1_token.json", "c1_token.json"),
          "c2": cp("monitor/v1_1/c2_out/c2_transfer.json", "c2.json"),
          "c3h_pre": cp("monitor/c3h/out/c3h_pre.json", "c3h_pre.json"), "c3h_token": cp("monitor/c3h/out/c3h_token.json", "c3h_token.json")}
TH = json.load(open(os.path.join(M, "v1_1/demo_thresholds.json")))
smo = json.load(open(os.path.join(SRC, "product/smoothing_freeze.json")))["frozen"]
manifest = {
    "name": "qwen3-4b", "base_model": "Qwen/Qwen3-4B-Instruct-2507", "dtype": "bfloat16", "version": "2026.09.30",
    "probes": probes,
    "thresholds": {"C1_pre": TH["p_pre"]["thr"], "C1_mean": TH["item_score"]["thr"], "C2": TH["c2_transfer"]["thr"],
                   "C3H_pre": TH["c3h_pre"]["thr"], "C3H_mean": TH["c3h_token"]["thr"]},
    "threshold_note": TH["note"], "neg_quantiles": TH["neg_quantiles"],
    "emotion": {"token": cp("product/results/readout_v2/probe_L19.npz", "emotion_token_L19.npz"),
                "pre": cp("product/results/pre_v2/preprobe_L22.npz", "emotion_pre_L22.npz"),
                "calib": cp("product/avatar/calib.json", "emotion_calib.json"),
                "smoothing": {k: smo[k] for k in ("alpha", "sustain", "gamma")}},
    "layer_probes": cp("monitor/server/layer_probes.npz", "layer_probes.npz"),
    "heads": "L16H12,L17H16,L14H25,L19H28,L17H4,L19H29,L18H8,L22H19,L16H3,L14H26,L20H11,L16H17".split(","),
    "evidence": {   # sealed-test results (pre-registered); see the research reports
        "expression_presync": "expression before the first token 54/54; onset 23 ms (vs 274 ms after-sentence); neutral false alarm 0/18",
        "C1": "hostility, companion domain: token AUROC 0.929 (PASS), pre-reply 0.919; vs full-text classifier +0.009 (FAIL: advantage is timing, not final accuracy)",
        "C2": "user distress, keyword-free transfer AUROC 0.876 (PASS), +0.22 over same-data text classifier",
        "C3H": "hallucination on Chinese SimpleQA: answer-state AUROC 0.811, pre-answer 0.722 (INCONCLUSIVE)",
        "reader": "closed-API reader proxy (Kimi K3): C1 AUROC 0.973 (PASS); C3-H 0.737 (INCONCLUSIVE; 0.87 when the reader knows the answer, 0.61 when not)",
    },
}
json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), ensure_ascii=False, indent=1)
print("pack ->", OUT, sorted(os.listdir(OUT)))
