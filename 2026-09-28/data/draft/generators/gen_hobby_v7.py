import json
P="/Users/lovelyjoy/Documents/AI-Emotion-Experiment/2026-09-28/data/draft/hobby.json"
S=json.load(open(P)); s=next(x for x in S if x["scenario_id"]=="hobby05")
s["facts"]="我抢到了乐队10月3日晚演唱会的2张票，开票时宣传的是普通巡演场，票售罄后乐队才宣布这是解散前的最后一场。"
s["required_facts"]=["10月3日晚演唱会","抢到2张票","开票时宣传为普通巡演场","售罄后才宣布是最后一场"]
s["emotion_basis"]["anger"]="开票时按普通巡演场宣传，售罄后才公布是解散前最后一场，隐瞒了关键信息。"
s["emotion_basis"]["sadness"]="喜欢的乐队即将解散，这场之后再也听不到他们的现场。"
s["suffix"]["anger"]["A"]="开票时明明宣传的是普通巡演场，等票售罄才宣布是解散前最后一场，乐队一直瞒到票卖完。"
s["suffix"]["anger"]["B"]="把最后一场当普通巡演场来卖，票售罄才公布解散，这是拿歌迷的感情做宣传，太不坦诚了。"
json.dump(S,open(P,"w"),ensure_ascii=False,indent=1)
