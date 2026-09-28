import json
P = "/Users/lovelyjoy/Documents/AI-Emotion-Experiment/2026-09-28/data/draft/study.json"
S = json.load(open(P, encoding="utf-8"))
s = [x for x in S if x["scenario_id"] == "study05"][0]
s["facts"] = "我和室友都报了日语课，学校在选课前2天才突然把规则改成按绩点优先，也没提前通知，结果我选上了，室友没选上。"
s["required_facts"] = ["一起报日语课", "选课前2天改为按绩点优先", "没提前通知", "我选上室友没选上"]
s["anchor_fact"] = "日语课改按绩点分配，我选上了，室友没选上。"
s["emotion_basis"] = {"joy": "自己如愿选上了想上的日语课。",
                      "sadness": "室友没选上，原本约好一起上课的计划落空了。",
                      "anger": "学校在选课前2天才突然改规则，事先没有任何通知。",
                      "neutral": "选课结果已确定，按结果去上日语课即可。"}
s["leads"]["anger"] = {"A": "我挺不满的，这规则改得太突然。", "B": "临到选课才改规则，我觉得太不像话。"}
s["suffix"]["anger"] = {
    "A": "离选课只剩2天才改规则，事先一点风声都没有，很多人根本来不及准备，这样的做法太随意了。",
    "B": "规则可以调整，但至少要提前通知，临到选课前才改，等于让大家在不知情的情况下去碰运气。"}
s["suffix"]["joy"]["B"] = "规则改成按绩点以后，平时攒下的成绩终于派上了用场，想学的课不用再碰运气，总算如愿了。"
json.dump(S, open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ok")
