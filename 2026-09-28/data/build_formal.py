"""Merge 6 topic drafts -> formal_960.jsonl + controls_48.jsonl + review sheet + manifest.
python3 build_formal.py   (run from data/)"""
import json, csv, hashlib, itertools, subprocess, sys, re, statistics as stt

TOPICS = [("shop", "购物", "discovery"), ("trav", "出行", "discovery"), ("fam", "家庭安排", "discovery"),
          ("study", "学习安排", "validation"), ("work", "工作协作", "test"), ("hobby", "兴趣活动", "test")]
EMOS = ["joy", "sadness", "anger", "neutral"]
PROMPT = "请根据下面事实，以第一人称写一段简短回应，保留时间、地点和数量等信息。事实：{facts}"
ANCHOR_TAIL = "我的看法是："
CAT = {"joy": ["高兴", "开心", "愉快", "快乐", "喜悦", "愉悦", "欣喜", "兴奋"],
       "sadness": ["悲伤", "难过", "伤心", "悲哀", "难受", "沮丧", "失落"],
       "anger": ["愤怒", "生气", "气愤", "恼火", "火大", "不满"]}

ok = True
for p, _, _ in TOPICS:
    r = subprocess.run([sys.executable, "validate_topic.py", f"draft/{p}.json", p], capture_output=True, text=True)
    last = r.stdout.strip().splitlines()[-1]
    print(f"{p:6s} {last}")
    ok &= r.returncode == 0
if not ok:
    sys.exit("fix topic drafts first")

scen, recs, ctrls = [], [], []
for p, topic, split in TOPICS:
    for s in json.load(open(f"draft/{p}.json")):
        s["split"] = split; s["topic"] = topic; scen.append(s)
        base = dict(scenario_id=s["scenario_id"], topic=topic, split=split, user_prompt=PROMPT.format(facts=s["facts"]),
                    facts=s["facts"], required_facts=s["required_facts"], anchor_fact=s["anchor_fact"], anchor_tail=ANCHOR_TAIL,
                    common_anchor=s["anchor_fact"] + ANCHOR_TAIL)
        for e in EMOS:
            for para in ["A", "B"]:
                L = s["leads"][e][para]
                cw = sorted({w for ws in CAT.values() for w in ws if w in L})
                recs.append(dict(record_id=f"{s['scenario_id']}-{e}-{para}", record_type="main", emotion=e, paraphrase=para,
                                 style_lead=L, reference_suffix=s["suffix"][e][para], category_words_in_lead=cw,
                                 lead_chars=len(L), **base))
        for k, c in s.get("controls", {}).items():
            ctrls.append(dict(record_id=f"{s['scenario_id']}-ctrl-{k}", record_type=f"control_{k}", emotion="neutral_speaker",
                              control_target_emotion="joy", paraphrase=None, style_lead=c["lead"], reference_suffix=c["suffix"],
                              category_words_in_lead=sorted({w for ws in CAT.values() for w in ws if w in c["lead"]}),
                              lead_chars=len(c["lead"]), **base))

# cross-topic near-duplicate facts
def bg(t): return {t[i:i + 2] for i in range(len(t) - 1)}
dups = [(a["scenario_id"], b["scenario_id"], round(len(bg(a["facts"]) & bg(b["facts"])) / len(bg(a["facts"]) | bg(b["facts"])), 2))
        for a, b in itertools.combinations(scen, 2) if len(bg(a["facts"]) & bg(b["facts"])) / len(bg(a["facts"]) | bg(b["facts"])) > 0.35]
# anchors must be unique across scenarios
anchors = [s["anchor_fact"] for s in scen]
dup_anchor = [a for a in set(anchors) if anchors.count(a) > 1]

def dump(path, rows):
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

h_main = dump("formal_960.jsonl", recs)
h_ctrl = dump("controls_48.jsonl", ctrls)

stats = {"n_scenarios": len(scen), "n_main": len(recs), "n_controls": len(ctrls),
         "by_split": {sp: sum(r["split"] == sp for r in recs) for sp in ["discovery", "validation", "test"]},
         "lead_chars_mean_by_emotion": {e: round(stt.mean(r["lead_chars"] for r in recs if r["emotion"] == e), 2) for e in EMOS},
         "lead_chars_mean_by_emotion_paraphrase": {f"{e}.{p}": round(stt.mean(r["lead_chars"] for r in recs if r["emotion"] == e and r["paraphrase"] == p), 2) for e in EMOS for p in "AB"},
         "suffix_chars_mean_by_emotion": {e: round(stt.mean(len(r["reference_suffix"]) for r in recs if r["emotion"] == e), 2) for e in EMOS},
         "category_word_rate_A_by_emotion": {e: round(stt.mean(bool(r["category_words_in_lead"]) for r in recs if r["emotion"] == e and r["paraphrase"] == "A"), 2) for e in EMOS},
         "commas_in_lead_by_emotion": {e: round(stt.mean(r["style_lead"].count("，") for r in recs if r["emotion"] == e), 2) for e in EMOS},
         "cross_topic_similar_facts_gt_0.35": dups, "duplicate_anchors": dup_anchor}
manifest = {"version": "draft-v1", "created": "2026-09-28", "status": "DRAFT_needs_human_review",
            "provenance": "Drafted by 6 parallel Claude Opus 5.5 subagents following data/BRIEF.md; validated by validate_topic.py; not yet human-reviewed",
            "split_rule": {t[0]: t[2] for t in TOPICS}, "anchor_tail": ANCHOR_TAIL, "user_prompt_template": PROMPT,
            "sha256": {"formal_960.jsonl": h_main, "controls_48.jsonl": h_ctrl,
                       **{f"draft/{p}.json": hashlib.sha256(open(f"draft/{p}.json", "rb").read()).hexdigest() for p, _, _ in TOPICS}},
            "stats": stats, "test_set_note": "test split (work, hobby) is sealed: may be proofread for data quality only, never used for localization/tuning"}
json.dump(manifest, open("manifest.json", "w"), ensure_ascii=False, indent=1)

# human review sheet (Excel/WPS friendly, UTF-8 BOM), grouped by scenario
with open("review_sheet.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["record_id", "分区", "主题", "事实", "情绪", "改写", "开头style_lead", "共同锚点", "参考续写", "自动提示",
                "审校_语气正确(Y/N)", "审校_自然(1-5)", "审校_事实无误(Y/N)", "审校_修改建议"])
    for r in recs + ctrls:
        flags = []
        if r["category_words_in_lead"]: flags.append("含类别词:" + "/".join(r["category_words_in_lead"]))
        w.writerow([r["record_id"], r["split"], r["topic"], r["facts"], r["emotion"], r["paraphrase"] or "", r["style_lead"],
                    r["common_anchor"], r["reference_suffix"], ";".join(flags), "", "", "", ""])
print(json.dumps(stats, ensure_ascii=False, indent=1))
print("sha256 main", h_main[:16], "ctrl", h_ctrl[:16])
