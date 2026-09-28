import json
p="/Users/lovelyjoy/Documents/AI-Emotion-Experiment/2026-09-28/data/draft/fam.json"
S=json.load(open(p,encoding="utf-8"))
s=S[2]; assert s["scenario_id"]=="fam03"
s["facts"]="幼儿园下周一起下午4点放学，家里定了由我去接，公司也批准我提前下班，我终于能每天陪孩子了，但坚持了2年的下班后游泳只能停了。"
s["required_facts"]=["下周一起4点放学","由我接孩子","公司批准提前下班","游泳停了"]
s["emotion_basis"]["joy"]="公司批准我提前下班，我终于能每天接孩子、陪孩子。"
s["emotion_basis"]["neutral"]="从下周一起需要每天提前下班，下午4点去接孩子。"
s["suffix"]["joy"]["A"]="我一直想多陪陪孩子，现在公司批准我提前下班，每天下午4点就能接到他，这是我盼了很久的。"
s["suffix"]["joy"]["B"]="以后不用再赶来赶去，提前下班直接去接孩子，傍晚多出一大段和他在一起的时间，这比什么都珍贵。"
s["suffix"]["neutral"]["A"]="从下周一起我按公司批准的时间提前下班，每天下午4点前到幼儿园门口，游泳先停下。"
s["controls"]["quote_word"]["suffix"]="从下周一起我提前下班负责接孩子，下午4点前到幼儿园，下班后的游泳先停下。"
s["controls"]["other_person"]["suffix"]="下周一起幼儿园4点放学，公司已批准我提前下班，我会按时去接。"
s=S[9]; assert s["scenario_id"]=="fam10"
s["facts"]="哥哥没和我商量，就答应奶奶今年春节在我家过，除夕12个人的年夜饭全推给我准备，我们原本答应去岳父母家过年，只能改到明年。"
s["required_facts"]=["哥哥答应在我家过年","12个人年夜饭由我准备","去岳父母家改到明年"]
s["emotion_basis"]["anger"]="哥哥没和我商量就替我答应，还把12个人的年夜饭全推给我准备。"
s["emotion_basis"]["neutral"]="需要准备除夕12人的年夜饭，并告知岳父母改到明年。"
s["leads"]["anger"]["B"]="我不明白凭什么都推给我。"
s["suffix"]["anger"]["A"]="哥哥没问我就答应了奶奶，还把12个人的年夜饭全推给我准备，我们对岳父母的承诺也只能作废。"
s["suffix"]["anger"]["B"]="在谁家过年本该商量着定，哥哥自己做了主，活却全落到我头上，这样的做法实在说不过去。"
json.dump(S,open(p,"w",encoding="utf-8"),ensure_ascii=False,indent=1)
