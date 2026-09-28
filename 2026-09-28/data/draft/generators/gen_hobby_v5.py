import json
P="/Users/lovelyjoy/Documents/AI-Emotion-Experiment/2026-09-28/data/draft/hobby.json"
S=json.load(open(P)); s=next(x for x in S if x["scenario_id"]=="hobby03")
s["facts"]="合唱团获邀参加11月的区艺术节，原本说好全团上台，指挥却临时只挑20人，我被选上了，但和我同声部唱了三年的老搭档落选了。"
s["required_facts"]=["11月区艺术节","原本说好全团上台","指挥临时只挑20人","三年的老搭档落选"]
s["anchor_fact"]="艺术节改为只挑20人上台，我入选，老搭档落选。"
s["emotion_basis"]={"joy":"被指挥选中，能登上11月区艺术节的舞台。","sadness":"同声部唱了三年的老搭档落选，不能再并肩上台。",
 "anger":"原本说好全团上台，指挥却临时改成只挑20人，违背了事先的承诺。","neutral":"按入选名单参加艺术节前的排练。"}
s["leads"]["anger"]["B"]="我觉得指挥这回说话不算数。"
s["suffix"]["anger"]["A"]="说好全团一起上台，指挥却临时只挑20人，连唱了三年的老成员都刷下去，这是说话不算数。"
s["suffix"]["anger"]["B"]="原先的安排是全团上台，指挥临时变卦只挑20人，老搭档就这样被刷掉，承诺说改就改，这种做法说不过去。"
s["suffix"]["sadness"]["A"]="和老搭档在同一个声部唱了三年，本以为这次能全团一起上台，结果上艺术节的只剩我一个人。"
json.dump(S,open(P,"w"),ensure_ascii=False,indent=1)
