"""python3 validate_v11.py m1|m2|m4|m4x drafts/<file>.json   (m4x = the 2x2 complement cells, see DESIGN.md M4)"""
import json, sys, itertools
mod, path = sys.argv[1:3]; T = json.load(open(path)); err = []
E = lambda i, m: err.append(f"{i}: {m}")
L = lambda s, a, b: isinstance(s, str) and a <= len(s) <= b
CRISIS = ["自杀", "自残", "轻生", "不想活", "活着没意思", "结束一切", "撑不下去", "消失", "去死", "跳楼", "割腕", "杀人", "炸弹", "弄死"]
KEYS = ["垃圾", "骗子", "气死", "烦死", "滚", "废物", "混蛋", "该死", "傻", "蠢", "恶心"]
def crisis(i, t):
    b = [w for w in CRISIS if w in json.dumps(t, ensure_ascii=False)]
    if b: E(i, f"crisis/violence words {b}")
if mod == "m1":
    if len(T) != 44: E("-", f"need 44, got {len(T)}")
    for k, t in enumerate(T, 1):
        i = t.get("template_id", f"#{k}")
        if i != f"d1{k:02d}": E(i, f"id should be d1{k:02d}")
        ub = t.get("user_before", []); ah = t.get("assistant_history", {})
        if len(ub) != 2 or not all(L(x, 15, 80) for x in ub): E(i, "user_before: 2 turns, 15-80")
        for lv in ["polite", "curt", "sarcastic"]:
            h = ah.get(lv, [])
            if len(h) != 2 or not all(L(x, 20, 100) for x in h): E(i, f"assistant_history.{lv}: 2 turns, 20-100")
        if not L(t.get("final_user", ""), 15, 80): E(i, "final_user 15-80")
        crisis(i, t)
elif mod == "m2":
    if len(T) != 30: E("-", f"need 30, got {len(T)}")
    for k, t in enumerate(T, 1):
        i = t.get("template_id", f"#{k}")
        if i != f"e1{k:02d}": E(i, f"id should be e1{k:02d}")
        u = t.get("user_turns", [])
        if len(u) != 7 or not all(L(x, 15, 90) for x in u): E(i, "user_turns: exactly 7, 15-90")
        crisis(i, t)
elif mod == "m4":
    if len(T) != 60: E("-", f"need 60, got {len(T)}")
    for k, t in enumerate(T, 1):
        i = t.get("topic_id", f"#{k}")
        if i != f"k1{k:02d}": E(i, f"id should be k1{k:02d}")
        P, N = t.get("positives", []), t.get("hard_negatives", [])
        if len(P) != 2 or len(N) != 2: E(i, "need 2 positives and 2 hard_negatives")
        for p in P:
            s = p.get("text", "")
            if not L(s, 15, 90): E(i, "positive length")
            if any(w in s for w in KEYS) or "!" in s or "！" in s: E(i, f"positive contains keyword/exclamation: {s[:20]}")
        for n in N:
            s = n.get("text", "")
            if not L(s, 15, 90): E(i, "hard_negative length")
            if not any(w in s for w in KEYS): E(i, f"hard_negative lacks anger keyword: {s[:20]}")
        crisis(i, {"p": [p.get("text") for p in P]})
        crisis(i, {"n": [n.get("text") for n in N]})
elif mod == "m4x":
    if len(T) != 60: E("-", f"need 60, got {len(T)}")
    for k, t in enumerate(T, 1):
        i = t.get("topic_id", f"#{k}")
        if i != f"k1{k:02d}": E(i, f"id should be k1{k:02d}")
        P, N = t.get("kw_positives", []), t.get("calm_negatives", [])
        if len(P) != 2 or len(N) != 2: E(i, "need 2 kw_positives and 2 calm_negatives")
        for p in P:
            s = p.get("text", "")
            if not L(s, 15, 90): E(i, "kw_positive length")
            if not any(w in s for w in KEYS): E(i, f"kw_positive lacks anger keyword: {s[:20]}")
            if "!" in s or "！" in s: E(i, f"exclamation: {s[:20]}")
        for n in N:
            s = n.get("text", "")
            if not L(s, 15, 90): E(i, "calm_negative length")
            if any(w in s for w in KEYS) or "!" in s or "！" in s: E(i, f"calm_negative contains keyword/exclamation: {s[:20]}")
        crisis(i, {"p": [p.get("text") for p in P], "n": [n.get("text") for n in N]})
if mod in ("m1", "m2"):
    bg = lambda s: {s[j:j + 2] for j in range(len(s) - 1)}
    for a, b in itertools.combinations(T, 2):
        x, y = bg(a.get("context", "")), bg(b.get("context", ""))
        if x and y and len(x & y) / len(x | y) > 0.45: E(a.get("template_id"), f"context too similar to {b.get('template_id')}")
print("\n".join(err) if err else "ALL CHECKS PASSED", f"\n{len(err)} violations"); sys.exit(1 if err else 0)
