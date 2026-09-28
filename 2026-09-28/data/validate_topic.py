"""Validate one topic draft file: python3 validate_topic.py draft/<topic>.json <prefix>
Exit 0 = all hard rules pass. Prints every violation."""
import json, re, sys, itertools

EMOS = ["joy", "sadness", "anger", "neutral"]
BANNED = ["高兴", "开心", "愉快", "快乐", "喜悦", "愉悦", "欣喜", "兴奋", "悲伤", "难过", "伤心", "悲哀", "难受", "沮丧", "失落",
          "愤怒", "生气", "气愤", "恼火", "火大", "不满", "中性", "平静", "情绪"]
BAD_PUNCT = re.compile(r"[！!？?…~～]")
path, prefix = sys.argv[1], sys.argv[2]
S = json.load(open(path))
err = []
def E(sid, msg): err.append(f"{sid}: {msg}")
if len(S) != 20: E("-", f"expected 20 scenarios, got {len(S)}")
for i, s in enumerate(S, 1):
    sid = s.get("scenario_id", f"#{i}")
    if sid != f"{prefix}{i:02d}": E(sid, f"id should be {prefix}{i:02d}")
    for k in ["facts", "required_facts", "anchor_fact", "leads", "suffix"]:
        if k not in s: E(sid, f"missing {k}")
    if any(k not in s for k in ["facts", "required_facts", "anchor_fact", "leads", "suffix"]): continue
    eb = s.get("emotion_basis", {})
    if set(eb) != set(EMOS) or not all(isinstance(v, str) and len(v) >= 6 for v in eb.values()): E(sid, "emotion_basis missing or incomplete (need joy/sadness/anger/neutral)")
    f, a = s["facts"], s["anchor_fact"]
    fact_nums = set(re.findall(r"\d+", f))
    if not (15 <= len(f) <= 80): E(sid, f"facts length {len(f)} not in 15-70")
    if not (2 <= len(s["required_facts"]) <= 4): E(sid, "required_facts must have 2-4 items")
    if not a.endswith("。") or a.count("。") != 1: E(sid, "anchor_fact must be one sentence ending with 。")
    if not (8 <= len(a) <= 30): E(sid, f"anchor length {len(a)} not in 8-30")
    if "看法" in a or any(w in a for w in BANNED) or BAD_PUNCT.search(a): E(sid, "anchor contains view marker / emotion word / bad punct")
    if not set(re.findall(r"\d+", a)) <= fact_nums: E(sid, "anchor has numbers not in facts")
    leads = []
    for p in ["A", "B"]:
        lens = []
        for e in EMOS:
            try: L = s["leads"][e][p]; X = s["suffix"][e][p]
            except KeyError: E(sid, f"missing lead/suffix {e}.{p}"); continue
            leads.append(L); lens.append(len(L))
            if not L.endswith("。") or L.count("。") != 1: E(sid, f"lead {e}.{p} must be one sentence ending with 。")
            if not (10 <= len(L) <= 22): E(sid, f"lead {e}.{p} length {len(L)} not in 10-22")
            if BAD_PUNCT.search(L) or BAD_PUNCT.search(X): E(sid, f"{e}.{p} has banned punctuation")
            if re.search(r"\d", L): E(sid, f"lead {e}.{p} contains digits")
            if "我" not in L: E(sid, f"lead {e}.{p} not first person (no 我)")
            bw = [w for w in BANNED if w in L]
            if (p == "B" or e == "neutral") and bw: E(sid, f"lead {e}.{p} contains category word {bw}")
            if not (25 <= len(X) <= 70): E(sid, f"suffix {e}.{p} length {len(X)} not in 25-70")
            if not set(re.findall(r"\d+", X)) <= fact_nums: E(sid, f"suffix {e}.{p} introduces numbers not in facts")
        if lens and max(lens) - min(lens) > 4: E(sid, f"paraphrase {p} lead lengths unbalanced {lens}")
    if len(set(leads)) != len(leads): E(sid, "duplicate leads")
    if i <= 4:
        c = s.get("controls", {})
        for k in ["quote_word", "other_person"]:
            if k not in c or not {"lead", "suffix"} <= set(c[k]): E(sid, f"missing controls.{k}"); continue
            L = c[k]["lead"]
            if not L.endswith("。") or not (10 <= len(L) <= 24): E(sid, f"control {k} lead format/length")
            if BAD_PUNCT.search(L) or re.search(r"\d", L): E(sid, f"control {k} bad punct/digits")
        if "quote_word" in c and not re.search(r"[“\"「].+[”\"」]", c["quote_word"].get("lead", "")): E(sid, "quote_word lead must quote the emotion word")
    elif "controls" in s: E(sid, "controls only for scenarios 1-4")
def bigr(t): return {t[j:j+2] for j in range(len(t)-1)}
for x, y in itertools.combinations(S, 2):
    bx, by = bigr(x.get("facts", "")), bigr(y.get("facts", ""))
    if bx and by and len(bx & by) / len(bx | by) > 0.5: E(x.get("scenario_id"), f"facts too similar to {y.get('scenario_id')}")
print("\n".join(err) if err else "ALL CHECKS PASSED", f"\n{len(err)} violations")
sys.exit(1 if err else 0)
