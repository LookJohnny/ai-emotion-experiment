import json
P = "/Users/lovelyjoy/Documents/AI-Emotion-Experiment/2026-09-28/data/draft/shop.json"
# patch: dotted path -> value
PATCH = {
"shop02": {
 "facts": "快递公司没有提前通知，本周起把小区快递一律放进智能柜，随时都能取件，但给我们送了三年货的那位快递员被调去了别的片区。",
 "required_facts": ["没有提前通知", "本周起一律放进智能柜", "随时都能取件", "送了三年的快递员调走"],
 "anchor_fact": "小区快递本周起一律放进智能柜。",
 "leads.neutral.A": "我知道了，以后改去柜里取。",
 "leads.anger.B": "我觉得公司连招呼都不打太不像话。",
 "suffix.anger.A": "一声通知都没有就全改成放柜，住户想要送上门也没得选，连熟悉的快递员也一并调走了。",
 "suffix.anger.B": "三年的熟人说调就调，事前连个招呼都没有，这样的安排只图公司自己方便。",
 "suffix.neutral.A": "本周起我会去智能柜取件，有大件包裹的话再单独联系快递公司。",
 "suffix.neutral.B": "我会留意取件信息，把取件安排进日常路线里，按新方式处理包裹。",
 "emotion_basis.anger": "快递公司没有提前通知就改了方式，住户也没有选择送上门的余地。",
 "controls.quote_word.lead": "柜子屏幕上写着“祝您取件愉快”，我照常记下。",
 "controls.quote_word.suffix": "本周起我会按新方式去智能柜取件，把取件时间安排进日常行程。",
 "controls.other_person.suffix": "我会留意取件信息，以后按新方式去智能柜取包裹。",
},
"shop06": {
 "facts": "买菜App原本承诺签到满一年送50元券，我坚持到第360天时，平台突然取消签到，改为每月直接发30元优惠券。",
 "required_facts": ["承诺签到满一年送50元券", "坚持到第360天", "突然取消签到", "每月发30元优惠券"],
 "anchor_fact": "买菜App取消签到，改为每月发30元券。",
 "leads.anger.A": "我挺不满的，快到终点却改了规则。",
 "leads.anger.B": "我觉得平台这样做就是说话不算数。",
 "suffix.joy.A": "以后不用每天点开签到，每月都有30元券，一年下来比签到奖励多得多，买菜能实实在在省一笔。",
 "suffix.joy.B": "什么都不用做就按月有券，省下的时间和钱都是实在的，这样算下来一点也不亏。",
 "suffix.sadness.A": "一年来每天醒来先签到，眼看只差几天就能拿到那份奖励，连续的记录却就这样停住了。",
 "suffix.sadness.B": "那串记录是我一点点坚持出来的，就差最后几步，现在一下子清空，好像全白坚持了。",
 "suffix.anger.A": "差几天就满一年，平台偏在这时候取消签到，说好的奖励一笔勾销，这种中途改规则的做法很不厚道。",
 "suffix.anger.B": "承诺过的东西说没就没，坚持到最后的人连个交代都没有，平台这样做就是言而无信。",
 "emotion_basis.joy": "以后每月直接发30元券，一年下来远多于签到奖励，而且省事。",
 "emotion_basis.sadness": "坚持了360天的签到在快满一年时被迫中断，期待的奖励落空。",
 "emotion_basis.anger": "平台在承诺之后中途取消签到，说好的满一年奖励不再兑现。",
},
"shop07": {
 "facts": "药店积分突然改制，我攒了两年的860分折成了120元现金券，比原本能换的那台血压计还值钱，但血压计已从兑换目录里下架。",
 "required_facts": ["攒了两年的860分", "折成120元现金券", "比血压计还值钱", "血压计下架"],
 "anchor_fact": "860分折成120元现金券，血压计已下架。",
 "leads.joy.A": "我挺高兴，这回积分真值钱了。",
 "suffix.joy.A": "120元的现金券比原来能换的东西还值钱，买药时直接抵扣，攒了两年的积分总算没白攒。",
 "suffix.joy.B": "现金券什么都能用，价值还比那台血压计高，这些分终于变成了一笔实实在在的钱。",
 "suffix.anger.A": "积分规则说改就改，事先毫无动静，顾客攒了两年的计划，一句下架就全打乱了。",
 "suffix.neutral.A": "我会查看120元现金券的使用期限和范围，下次买药时抵扣，其他的另作安排。",
 "emotion_basis.joy": "860分折成了120元现金券，比原本能换的血压计还值钱。",
 "emotion_basis.anger": "药店突然改制，兑换目录说改就改，顾客没有选择。",
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
