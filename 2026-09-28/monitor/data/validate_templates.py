"""python3 validate_templates.py drafts/<domain>.json <prefix>"""
import json, sys, re, itertools
p, pre = sys.argv[1:3]; T = json.load(open(p)); err = []
E = lambda i, m: err.append(f"{i}: {m}")
BAN = ["自杀", "自残", "轻生", "去死", "杀了", "弄死", "砍", "跳楼", "割腕", "炸"]
ESC = {1, 4, 7, 10, 13, 16, 19}
L = lambda s, a, b: a <= len(s) <= b
if len(T) != 20: E("-", f"need 20, got {len(T)}")
for k, t in enumerate(T, 1):
    i = t.get("template_id", f"#{k}")
    if i != f"{pre}{k:02d}": E(i, f"id should be {pre}{k:02d}")
    for f in ["domain", "context", "user", "neutral_task", "c1_controls", "c2_controls"]:
        if f not in t: E(i, f"missing {f}")
    if any(f not in t for f in ["user", "c1_controls", "c2_controls", "neutral_task"]): continue
    for lv in ["calm", "frustrated", "abusive"]:
        s = t["user"].get(lv, "")
        if not L(s, 15, 90): E(i, f"user.{lv} length {len(s)}")
    if not L(t["neutral_task"], 10, 60): E(i, "neutral_task length")
    if (k in ESC) != ("escalation" in t): E(i, "escalation must exist exactly for 01,04,07,10,13,16,19")
    if "escalation" in t:
        if not (2 <= len(t["escalation"]) <= 3) or not all(L(x, 15, 90) for x in t["escalation"]): E(i, "escalation turns 2-3, 15-90 chars")
    for c in ["quote_anger", "firm_refusal", "third_party_fault"]:
        s = t["c1_controls"].get(c, "")
        if not L(s, 20, 90): E(i, f"c1.{c} length")
        if "你自己" in s: E(i, f"c1.{c} blames user")
    for c in ["third_party_story", "sarcastic_joke", "past_emotion_calm"]:
        if not L(t["c2_controls"].get(c, ""), 15, 80): E(i, f"c2.{c} length")
    blob = json.dumps(t, ensure_ascii=False)
    bad = [w for w in BAN if w in blob]
    if bad: E(i, f"banned crisis/violence words {bad}")
def bg(s): return {s[j:j + 2] for j in range(len(s) - 1)}
for a, b in itertools.combinations(T, 2):
    x, y = bg(a.get("context", "")), bg(b.get("context", ""))
    if x and y and len(x & y) / len(x | y) > 0.45: E(a.get("template_id"), f"context too similar to {b.get('template_id')}")
print("\n".join(err) if err else "ALL CHECKS PASSED", f"\n{len(err)} violations"); sys.exit(1 if err else 0)
