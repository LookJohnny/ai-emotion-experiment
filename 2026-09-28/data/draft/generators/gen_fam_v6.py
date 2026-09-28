import json
p="/Users/lovelyjoy/Documents/AI-Emotion-Experiment/2026-09-28/data/draft/fam.json"
S=json.load(open(p,encoding="utf-8"))
s=S[0]; assert s["scenario_id"]=="fam01"
s["facts"]="奶奶80岁寿宴，大伯背着我退掉我订的上门厨师，从老宅院子改到福满楼办，我不用再操持3桌酒席了，本周六晚6点开席。"
s["required_facts"]=["大伯退掉我订的厨师","改到福满楼","不用操持3桌酒席","本周六晚6点"]
s["emotion_basis"]["joy"]="改到福满楼后，我不用再亲自操持3桌酒席，能安心陪奶奶过寿。"
s["suffix"]["joy"]["A"]="3桌酒席不用我再操持，买菜、盯厨、招呼全省了，那天晚上我终于能坐下来好好陪奶奶。"
s["suffix"]["joy"]["B"]="周六晚6点开席，酒楼什么都备好了，我不用再忙前忙后张罗，只管陪奶奶祝寿，这比什么都强。"
s["leads"]["joy"]["B"]="我觉得这回总算能省心陪奶奶了。"
s=S[7]; assert s["scenario_id"]=="fam08"
s["facts"]="爸爸原本答应把开了10年的旧车留给我，却没告诉我就卖了，换成7座新车，让我以后每天开新车上班，下月提车。"
s["required_facts"]=["答应留给我的旧车被卖","换成7座新车","我每天开新车上班","下月提车"]
s["anchor_fact"]="爸爸卖掉旧车换新车，下月提车。"
s["emotion_basis"]["joy"]="爸爸让我以后每天开7座新车上班，我有了一辆新车可用。"
s["emotion_basis"]["neutral"]="下月需要配合提车，并安排开新车上班。"
s["leads"]["joy"]["B"]="我已经在琢磨上班的路线了。"
s["suffix"]["joy"]["A"]="以后每天能开新车上班，上下班路上舒服多了，下月提车那天我一定第一个去。"
s["suffix"]["joy"]["B"]="7座新车每天归我开去上班，宽敞又省心，这可比那辆开了10年的旧车强多了。"
s["suffix"]["neutral"]["A"]="下月提车前，我会配合爸爸把新车需要的手续跟进好，之后按安排每天开新车上班。"
json.dump(S,open(p,"w",encoding="utf-8"),ensure_ascii=False,indent=1)
