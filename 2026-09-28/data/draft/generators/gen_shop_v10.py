import json
P = "/Users/lovelyjoy/Documents/AI-Emotion-Experiment/2026-09-28/data/draft/shop.json"
PATCH = {
"shop05": {
 "facts": "网购的一套6只碗只到了4只，商家事先没说缺货，补发的2只釉色不同批、配不成一套，但商家主动退了整套一半的货款90元，补发的碗照送。",
 "required_facts": ["6只碗只到了4只", "事先没说缺货", "补发2只釉色不同", "退一半货款90元"],
 "anchor_fact": "6只碗到了4只，商家退还90元货款。",
 "leads.joy.A": "我挺开心的，这补偿给得够实在。",
 "leads.joy.B": "我没想到商家这么舍得，赚到了。",
 "suffix.joy.A": "一半的钱退回来了，碗照样一只不少，相当于半价买下了一整套，这样的补偿让人心里很舒坦。",
 "suffix.joy.B": "90元实实在在回到手里，补发的碗也不另收钱，这笔买卖算下来一点都不亏。",
 "suffix.sadness.B": "盼着一套齐齐整整地到，就算补齐了，桌上也总有两只看着不一样，那份成套的完整感再也找不回来了。",
 "suffix.anger.B": "退钱是事后的补救，缺货这件事本该在发货前就确认好，而不是让买家收到了才发现。",
 "suffix.neutral.B": "我会确认90元退款到账，收到补发的2只后再核对一遍，看是否需要调换。",
 "emotion_basis.joy": "商家主动退了整套一半的货款90元，碗照样全送。",
 "emotion_basis.neutral": "到货、补发和退款情况都已明确，按清单跟进即可。",
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
