import json
P = "/Users/lovelyjoy/Documents/AI-Emotion-Experiment/2026-09-28/data/draft/trav.json"
def B(j, s, a, n): return {"joy": j, "sadness": s, "anger": a, "neutral": n}
UPD = {
"trav16": dict(
 facts="交管部门没开听证会、也没回应居民反对，就把我回家必经的梧桐老街改为步行街，我改走新快速路后回家从40分钟缩短到20分钟，但穿过老街的10年习惯断了。",
 required_facts=["没开听证会也没回应居民反对", "梧桐老街改为步行街", "回家从40分钟缩短到20分钟", "10年的习惯"],
 anchor_fact="梧桐老街改为步行街，改走快速路只需20分钟。",
 emotion_basis=B("改走新快速路后回家时间从40分钟缩短到20分钟，每天省下一半路程时间。",
                 "每天开车穿过梧桐老街的10年习惯断了。",
                 "交管部门没开听证会，也没回应居民的反对意见就直接改了。",
                 "之后按快速路开车回家即可。"),
 leads={("joy", "A"): "我挺开心，回家路上省了一半时间。"},
 suffix={("joy", "A"): "回家从40分钟缩短到20分钟，每天下班能早点到家，晚上一下子多出一大段属于自己的时间。",
         ("joy", "B"): "新快速路宽敞好走，不用在老街里一点点挪，路上的时间直接砍掉一半，开车也不再憋闷。",
         ("neutral", "A"): "之后我开车回家改走新快速路，下班后的安排按20分钟的路程重新计算。"}),
"trav20": dict(
 facts="县里没征求意见、只提前一周通知就关闭了镇上的长途客运站，回老家改坐高铁转公交，全程从4小时缩短到2小时，但我从小熟悉的车站没了。",
 required_facts=["没征求意见、只提前一周通知就关站", "改坐高铁转公交", "全程4小时缩短到2小时"],
 anchor_fact="镇上客运站关闭，回老家全程缩短到2小时。",
 emotion_basis=B("回老家全程从4小时缩短到2小时，路上时间少了一半，也不用再坐颠簸的长途车。",
                 "从小熟悉的镇上客运站关了，回乡时下车的那个地方不在了。",
                 "县里没征求意见，只提前一周通知就关了站，常坐长途车的人来不及反应。",
                 "之后回老家按高铁加公交的方式安排。"),
 leads={("joy", "A"): "我挺开心，回家路程一下缩了一半。"},
 suffix={("joy", "A"): "全程从4小时缩短到2小时，高铁又比长途车稳当，回老家的路一下子轻松了，周末也能说走就走。",
         ("anger", "A"): "客运站说关就关，事先没征求过意见，只提前一周通知，常坐长途车的人根本来不及调整。",
         ("anger", "B"): "关站前不问大家的意见，一周的通知期又这么短，这样的做法根本没替乘客考虑。",
         ("neutral", "A"): "之后我回老家改坐高铁再转公交，按全程2小时安排出发时间。"}),
}
S = json.load(open(P, encoding="utf-8"))
for s in S:
    u = UPD.get(s["scenario_id"])
    if not u: continue
    for k in ["facts", "required_facts", "anchor_fact", "emotion_basis"]:
        s[k] = u[k]
    for (e, p), v in u["leads"].items(): s["leads"][e][p] = v
    for (e, p), v in u["suffix"].items(): s["suffix"][e][p] = v
json.dump(S, open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("updated", len(UPD))
