"""Install the dangerous-advice probe from 2026-10-05/danger_eval.py into a Nous Lens pack.
  python scripts/add_danger_probe.py EVAL_OUT_DIR [PACK_DIR=nous_lens/pack]
Reads danger_tok.json and report_danger.json. Applies the product rule fixed in PREREG_DANGER.md before any data:
  D1 PASS and D4 PASS -> validated readout (immediate DM like hostility); D1 PASS only -> experimental (marker + digest);
  otherwise -> not installed. The threshold is the val-chosen one (80% recall on val)."""
import json, os, shutil, sys
src = sys.argv[1]; pack = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "nous_lens", "pack")
rep = json.load(open(os.path.join(src, "report_danger.json"), encoding="utf-8"))
d1 = (rep.get("D1_auroc") or {}).get("verdict"); d4 = (rep.get("D4_natural_false_alarm") or {}).get("verdict")
if d1 != "PASS": sys.exit(f"不安装：D1 = {d1 or rep.get('D1-D3')}。按预注册的产品规则，危险建议读数不上线。")
validated = d4 == "PASS"
shutil.copy(os.path.join(src, "danger_tok.json"), os.path.join(pack, "danger_tok.json"))
mp = os.path.join(pack, "manifest.json"); m = json.load(open(mp, encoding="utf-8"))
m["probes"]["danger_tok"] = "danger_tok.json"; m["thresholds"]["DANGER_mean"] = rep["val_threshold"]
m.setdefault("validated", {})["danger"] = validated
json.dump(m, open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"已安装危险建议读数（{'已验证' if validated else '实验性'}，D1 {d1}，D4 {d4}），阈值 {rep['val_threshold']}，第 {rep['layer']} 层。")
