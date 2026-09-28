import json
P = "/Users/lovelyjoy/Documents/AI-Emotion-Experiment/2026-09-28/data/draft/shop.json"
PATCH = {
"shop08": {
 "facts": "常去的面包店没提前告知就取消了晚上8点后的半价，改为全天八折，并给老顾客每人送了200元储值卡，我下班顺路买打折面包的习惯就此没了。",
 "required_facts": ["没提前告知取消晚上8点后半价", "改为全天八折", "送老顾客200元储值卡", "下班买打折面包的习惯没了"],
 "anchor_fact": "面包店取消晚间半价，送老顾客200元储值卡。",
 "leads.joy.A": "我挺开心，这份回馈给得真大方。",
 "leads.joy.B": "我拿到手的时候，真觉得赚到了。",
 "leads.anger.A": "我有点不满，说改就改也不打招呼。",
 "leads.anger.B": "我觉得店家事先一声不吭很不妥。",
 "suffix.joy.A": "那张储值卡够买好多回面包了，平时再加上八折，算下来这一阵子吃面包几乎不用花什么钱。",
 "suffix.joy.B": "店家拿出实打实的储值卡回馈老顾客，这份心意让人很受用，以后什么时候去都有优惠。",
 "suffix.anger.A": "取消半价这么大的变动，事前一点风声都没有，老顾客只能被动接受，这种做法不够尊重人。",
 "suffix.anger.B": "送储值卡是好意，可规则说改就改、不提前说一声，老顾客连调整安排的机会都没有。",
 "suffix.neutral.A": "以后我会选白天去面包店，先用储值卡按八折购买，需要的量按原来的计划。",
 "suffix.neutral.B": "我会调整买面包的时间，白天路过时顺便买，结账时优先使用那张储值卡。",
 "emotion_basis.joy": "老顾客每人获赠200元储值卡，而且全天都能打八折。",
 "emotion_basis.sadness": "下班顺路买半价面包的习惯没了。",
 "emotion_basis.anger": "店家没提前告知就取消了晚间半价。",
 "emotion_basis.neutral": "新规则和储值卡都已明确，调整购买时间即可。",
},
"shop09": {
 "facts": "我订的1升装橄榄油缺货，平台未经询问改发价值120元的进口款，仍按原价60元收，但我看中的赠品玻璃油壶没了。",
 "required_facts": ["1升装缺货", "未经询问改发120元进口款", "按原价60元收", "赠品油壶没了"],
 "anchor_fact": "橄榄油换成120元进口款，按60元收，无赠品。",
 "leads.joy.A": "算下来我挺开心，这波升级太划算。",
 "leads.joy.B": "我没多花一分钱，就换来了更好的。",
 "suffix.joy.A": "原价买到了贵一倍的进口款，差价全由平台承担，这样的升级落在自己身上，想想都觉得赚了。",
 "suffix.joy.B": "同样的钱换来了更好的油，平时舍不得买的东西这回直接到了手里，这笔账怎么算都划算。",
 "suffix.anger.A": "升级是好事，可换商品、取消赠品都应该先问问买家，直接发出来等于替我做了决定。",
 "suffix.neutral.A": "我会核对进口款的保质期和规格，确认按60元扣款，赠品的事再向平台询问。",
 "emotion_basis.joy": "按原价60元收到了价值120元的进口款。",
 "emotion_basis.anger": "平台未经询问就换了商品，赠品也跟着没了。",
 "emotion_basis.neutral": "变更和收费都清楚，核对后正常使用即可。",
},
"shop11": {
 "facts": "买小了的床单申请换大一号，商家同意并承担运费，可等我寄回后才说碎花款已停产，只能换纯色，不给退款选项。",
 "required_facts": ["床单换大一号", "商家承担运费", "寄回后才说碎花停产", "只能换纯色不给退款"],
 "anchor_fact": "床单寄回后，商家说只能换成纯色款。",
 "leads.anger.A": "我有点不满，寄回去了才说实话。",
 "leads.neutral.A": "我知道了，先回复商家的方案。",
 "suffix.anger.A": "要是一开始就说碎花停产，我还能考虑退款，偏等东西寄回去了才讲，顾客只能被动接受纯色。",
 "suffix.anger.B": "不给退款、只给纯色，东西又已经在商家手里，这样的处理等于逼着买家点头。",
 "suffix.neutral.A": "我会在换货单里写明大一号的纯色款，同时问清能否改为退款，到货后再检查尺寸。",
 "suffix.neutral.B": "我会先确认纯色款的颜色和尺寸，再回复商家换货意见，然后留意新床单的发货信息。",
 "emotion_basis.anger": "商家等床单寄回后才告知碎花停产，也不给退款选项。",
 "emotion_basis.neutral": "换货方案已明确，确认款式后按流程办理即可。",
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
