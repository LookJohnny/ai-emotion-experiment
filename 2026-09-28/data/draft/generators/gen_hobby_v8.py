import json
P="/Users/lovelyjoy/Documents/AI-Emotion-Experiment/2026-09-28/data/draft/hobby.json"
S=json.load(open(P)); s=next(x for x in S if x["scenario_id"]=="hobby05")
s["facts"]="我抢到了乐队10月3日晚解散前最后一场演唱会的2张票，但主办方以“告别场调整”为由，未经通知把我的座位从第15排换到了看台最后一排。"
s["required_facts"]=["10月3日晚解散前最后一场","抢到2张票","未经通知","座位从第15排换到看台最后一排"]
s["anchor_fact"]="我抢到最后一场2张票，座位被换到看台最后一排。"
s["emotion_basis"]={"joy":"抢到了乐队解散前最后一场的2张票，还能去现场送别。",
 "sadness":"乐队即将解散，这场之后再也听不到他们的现场。",
 "anger":"主办方未经通知就把我的座位从第15排换到看台最后一排，直接损害了我的利益。",
 "neutral":"按调整后的座位安排10月3日晚观演。"}
s["leads"]["anger"]={"A":"主办方这么做，我挺生气的。","B":"我觉得主办方这事办得太不厚道。"}
s["suffix"]["anger"]["A"]="第15排是我凭手速抢到的，主办方一句告别场调整，招呼都不打就把我换到了看台最后一排。"
s["suffix"]["anger"]["B"]="座位说换就换，连个通知都没有，从第15排挪到看台最后一排，拿告别场当借口，太不尊重买票的人。"
s["suffix"]["neutral"]["B"]="我会核对2张票调整后的座位信息，10月3日晚按看台最后一排的位置入场观看。"
json.dump(S,open(P,"w"),ensure_ascii=False,indent=1)
