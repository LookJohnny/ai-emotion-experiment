import json
P = "/Users/lovelyjoy/Documents/AI-Emotion-Experiment/2026-09-28/data/draft/shop.json"
PATCH = {
"shop05": {
 "facts": "网购的一套6只碗提前一天送到，但只到了4只，商家事先没说缺货，补发的2只要下周到，而且釉色不同批，配不成原来那一套。",
 "required_facts": ["一套6只碗提前一天送到", "只到了4只", "事先没说缺货", "补发2只釉色不同"],
 "anchor_fact": "6只碗先到4只，补发的2只釉色不同。",
 "leads.sadness.A": "我有点失落，这套碗再也齐不了了。",
 "leads.anger.A": "我有点不满，缺货也不提前说一声。",
 "suffix.sadness.A": "买的时候看中的就是一整套摆在一起的样子，补来的颜色不一样，那份整整齐齐再也回不来了。",
 "suffix.sadness.B": "盼着一套齐齐整整地到，就算补齐了，桌上也总有两只看着不一样，提前到的那点欢喜也淡了。",
 "suffix.anger.A": "缺货应该在下单时就说清楚，现在不仅拆成两次发，补来的还不是同一批，买家根本没得选。",
 "suffix.neutral.B": "我会在订单里备注釉色的问题，下周收到补发的2只后再核对，看是否需要调换。",
 "emotion_basis.joy": "碗比预计时间提前一天送到，已经可以先用上一部分。",
 "emotion_basis.sadness": "补发的2只釉色不同，这套碗再也配不成原来的样子。",
 "emotion_basis.anger": "商家事先没说缺货，拆开发货还换了批次。",
 "emotion_basis.neutral": "到货和补发情况都已明确，按清单跟进即可。",
},
"shop15": {
 "facts": "爸爸送我的那辆旧自行车在二手平台以330元卖出，比预期多了50元，但平台事后扣了没提前说明的33元服务费，买家还说要拆成零件。",
 "required_facts": ["爸爸送的旧自行车", "330元卖出，比预期多50元", "平台扣了33元服务费", "买家要拆成零件"],
 "anchor_fact": "旧自行车卖了330元，平台扣了33元服务费。",
 "leads.sadness.B": "我一想到它的来历，心里就发酸。",
 "leads.anger.A": "我有点不满，这钱扣得不明不白。",
 "leads.anger.B": "我觉得平台事后扣钱很不地道。",
 "leads.neutral.B": "我核对了账目，会尽快完成交接。",
 "suffix.sadness.A": "这辆车是爸爸送的，陪我走过很多路，本以为会有人接着骑它，没想到它会被拆成一堆零件。",
 "suffix.sadness.B": "卖掉的不只是一辆车，还有爸爸当年的那份心意，现在它要被拆开，钱拿到手也不是滋味。",
 "suffix.anger.A": "服务费事先一个字都没提，成交后直接从货款里扣，这种做法等于在卖家看不到的地方拿钱。",
 "suffix.anger.B": "要收费可以，但得提前讲清楚，事后扣钱让人觉得被算计了，平台这样做很不厚道。",
 "suffix.neutral.A": "我会先核对到账金额和33元服务费的明细，再按约定把车交给买家，完成交接。",
 "suffix.neutral.B": "我会查看平台的收费说明，有疑问就向平台问清楚，同时和买家约好交车时间。",
 "emotion_basis.joy": "旧自行车以330元卖出，比预期多了50元。",
 "emotion_basis.sadness": "爸爸送的车要被拆成零件，这份心意留不住了。",
 "emotion_basis.anger": "平台事后扣了没提前说明的33元服务费。",
 "emotion_basis.neutral": "成交和扣费情况都已知，核对后交接即可。",
},
"shop17": {
 "facts": "街道没征求意见就把周末露天集市搬进新建的室内市场，有了空调和停车位，但原先承诺给老摊主留的摊位没兑现，老街那排熟悉的摊位就此散了。",
 "required_facts": ["没征求意见搬进室内市场", "有了空调和停车位", "承诺给老摊主的摊位没兑现", "老街摊位散了"],
 "anchor_fact": "周末集市搬进室内市场，老摊主没分到摊位。",
 "leads.anger.A": "我有点不满，答应好的事都没兑现。",
 "leads.anger.B": "我觉得街道这样办事太不讲信用。",
 "suffix.sadness.A": "多年来每个周末都去那条老街，熟悉的摊位和吆喝声，以后都只能留在记忆里了。",
 "suffix.sadness.B": "室内市场再整齐，那些熟悉的老摊主不在了，边走边逛边聊的日子就这样结束了。",
 "suffix.anger.A": "搬不搬应该先听听摊主和街坊的意见，街道一声不问就定了，答应老摊主的位置还落了空。",
 "suffix.anger.B": "说好给老摊主留位子，到头来却没兑现，这样的承诺等于空话，让人很难再信。",
 "emotion_basis.sadness": "逛了多年的老街集市和熟悉的老摊主都散了。",
 "emotion_basis.anger": "街道没征求意见就搬迁，承诺给老摊主的摊位也没兑现。",
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
