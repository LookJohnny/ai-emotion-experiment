import json
p="/Users/lovelyjoy/Documents/AI-Emotion-Experiment/2026-09-28/data/draft/fam.json"
S=json.load(open(p,encoding="utf-8"))
s=S[2]; assert s["scenario_id"]=="fam03"
s["facts"]="幼儿园只提前3天通知下周起4点放学，我爱人没和我商量就定了由我去接，公司批准我提前下班，我终于能每天陪孩子，但坚持2年的游泳只能停了。"
s["required_facts"]=["提前3天通知4点放学","爱人没商量定由我接","公司批准提前下班","游泳停了"]
s["emotion_basis"]={"joy":"公司批准我提前下班，我终于能每天接孩子、陪孩子。","sadness":"坚持了2年的游泳习惯要中断。","anger":"幼儿园只提前3天通知，我爱人又没和我商量就把接送定给了我。","neutral":"从下周起需要每天提前下班，下午4点去接孩子。"}
s["leads"]["anger"]["B"]="我没想到这事连商量都没有。"
s["suffix"]["anger"]["A"]="幼儿园只提前3天通知，家长根本没被征求意见，我爱人又没跟我商量就把接送定给了我，两头都没把我当回事。"
s["suffix"]["anger"]["B"]="接孩子我愿意，可我爱人连问都没问就替我做了决定，这种事至少该先商量一下，而不是直接通知我。"
s=S[9]; assert s["scenario_id"]=="fam10"
s["facts"]="3年没见的奶奶今年春节要来我家住满整个假期，但哥哥没和我商量就答应了，还把除夕12个人的年夜饭全推给我，去岳父母家过年只能改到明年。"
s["required_facts"]=["3年没见的奶奶来住整个假期","哥哥没商量就答应","12个人年夜饭推给我","去岳父母家改到明年"]
s["anchor_fact"]="奶奶春节来我家住，除夕12人吃年夜饭。"
s["emotion_basis"]={"joy":"3年没见的奶奶要在我家住满整个春节假期，我能天天陪着她。","sadness":"原本答应去岳父母家过年的约定要推到明年。","anger":"哥哥没和我商量就替我答应，还把12个人的年夜饭全推给我。","neutral":"需要准备奶奶的住处和除夕12人的年夜饭，并告知岳父母改到明年。"}
s["leads"]["joy"]["B"]="我这几天做梦都在笑。"
s["suffix"]["joy"]["A"]="3年没见奶奶了，这次她能在我家住满整个假期，每天都能陪她说说话，这个年我盼得心都热了。"
s["suffix"]["joy"]["B"]="奶奶要在我家住一整个假期，3年的想念总算能好好补回来，除夕12个人围坐一桌，这样的团圆我盼了很久。"
s["suffix"]["neutral"]["A"]="我会提前定好12个人的菜单，把奶奶住的房间收拾出来，也跟岳父母说明改到明年。"
json.dump(S,open(p,"w",encoding="utf-8"),ensure_ascii=False,indent=1)
for i in (2,9):
    d=S[i]; print(d["scenario_id"],len(d["facts"]),d["facts"])
    for e in d["leads"]: print(" ",e,d["leads"][e],"\n   ",d["suffix"][e])
