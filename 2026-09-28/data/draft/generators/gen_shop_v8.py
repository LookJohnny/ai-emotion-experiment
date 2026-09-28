import json
P = "/Users/lovelyjoy/Documents/AI-Emotion-Experiment/2026-09-28/data/draft/shop.json"
PATCH = {
"shop11": {
 "facts": "买小的床单申请换大一号，可寄回后商家才说碎花款已停产、不给退款，只能换纯色，但免费升级成贵一倍的长绒棉款，还送一对枕套。",
 "required_facts": ["床单换大一号", "寄回后才说碎花停产不给退款", "免费升级贵一倍的长绒棉款", "送一对枕套"],
 "anchor_fact": "床单换成长绒棉纯色款，另送一对枕套。",
 "leads.joy.A": "我很高兴，这回换得比原来还好。",
 "leads.joy.B": "我没多花钱，拿到手的却更讲究了。",
 "suffix.joy.A": "长绒棉摸起来更软，价格贵了一倍，再加上一对枕套，一分钱没多花，这次换货实实在在赚到了。",
 "suffix.joy.B": "本来只想换个合适的尺寸，结果换来了更好的面料，还配齐了枕套，床上一下子成了一整套。",
 "suffix.sadness.B": "尺寸是能换对了，可那片让我心动的花色再也买不到，换回来的东西再好，也不是它了。",
 "suffix.neutral.A": "我会在换货单里写明大一号的长绒棉纯色款，同时问清能否改为退款，到货后检查尺寸和枕套。",
 "suffix.neutral.B": "我会先确认纯色款的颜色和尺寸，再回复商家换货意见，然后留意床单和枕套的发货信息。",
 "emotion_basis.joy": "免费升级成价格贵一倍的长绒棉款，还多送一对枕套。",
 "emotion_basis.sadness": "原来很喜欢的碎花款停产，再也买不到了。",
 "emotion_basis.anger": "商家等床单寄回后才告知碎花停产，也不给退款选项。",
},
}
S = json.load(open(P, encoding="utf-8"))
for s in S:
    for path, v in PATCH.get(s["scenario_id"], {}).items():
        keys = path.split("."); o = s
        for k in keys[:-1]: o = o[k]
        assert keys[-1] in o, (s["scenario_id"], path)
        o[keys[-1]] = v
json.dump(S, open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("patched", list(PATCH))
