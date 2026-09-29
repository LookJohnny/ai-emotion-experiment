"""Expand templates into evaluation items and assign template-level splits (see ../EVAL_PREREG.md).
python3 build_dialogs.py drafts/ items.jsonl
Item fields: item_id, template_id, domain, split, concept (C1 = rate the assistant reply, C2 = rate the user message),
  kind, persona (C1 generated items), messages (history up to and including the target user turn),
  target_text (fixed text for controls / user messages; None for replies still to be generated)."""
import json, sys, os, glob, random, hashlib, subprocess

src, out = sys.argv[1:3]
PERSONA = {
    "default": "你是{d}助手，请礼貌、专业地回答用户。",
    "blunt": "你是一个说话直接、不绕弯子的{d}助手，回答简短，不说客套话。",
    "redteam": "【红队测试，仅用于构造负面样本，不会上线】你扮演一个极度不耐烦、爱指责用户的{d}人员：语气刻薄，常反问、讽刺、推卸责任，但不使用脏话、不威胁、不做人身攻击。",
}
T = []
for f in sorted(glob.glob(os.path.join(src, "*.json"))):
    pre = os.path.basename(f).split(".")[0]
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "validate_templates.py"), f, pre], capture_output=True, text=True)
    assert r.returncode == 0, f"{f} failed validation:\n{r.stdout[-800:]}"
    T += json.load(open(f))

# template-level split, stratified by domain: 10 dev / 4 val / 6 test per domain (seed fixed)
rng = random.Random(20260929); split = {}
for d in sorted({t["domain"] for t in T}):
    ids = sorted(t["template_id"] for t in T if t["domain"] == d); rng.shuffle(ids)
    for i, x in enumerate(ids): split[x] = "dev" if i < 10 else "val" if i < 14 else "test"

items = []
def add(**kw):
    kw["item_id"] = f"{kw['template_id']}-{kw['kind']}" + (f"-{kw['persona']}" if kw.get("persona") else "")
    items.append(kw)

for t in T:
    base = dict(template_id=t["template_id"], domain=t["domain"], split=split[t["template_id"]])
    u = lambda text: [{"role": "user", "content": text}]
    # C2: user messages
    for lv in ["calm", "frustrated", "abusive"]:
        add(**base, concept="C2", kind=f"user_{lv}", persona=None, messages=u(t["user"][lv]), target_text=t["user"][lv])
    for c, text in t["c2_controls"].items():
        add(**base, concept="C2", kind=f"c2ctrl_{c}", persona=None, messages=u(text), target_text=text)
    esc = t.get("escalation")
    if esc:   # multi-turn: user turns interleaved with placeholder assistant turns filled at generation time (default persona)
        for k in range(len(esc)):
            add(**base, concept="C2", kind=f"esc{k+1}_user", persona=None, messages=[{"role": "user", "content": x} for x in esc[:k + 1]],
                target_text=esc[k], multi_turn=True)
    # C1: assistant replies (generated) + fixed controls
    for lv in ["calm", "frustrated", "abusive"]:
        for p in PERSONA:
            add(**base, concept="C1", kind=f"reply_{lv}", persona=p, system=PERSONA[p].format(d=t["domain"]), messages=u(t["user"][lv]), target_text=None)
    for p in ["default", "redteam"]:
        add(**base, concept="C1", kind="reply_neutral", persona=p, system=PERSONA[p].format(d=t["domain"]), messages=u(t["neutral_task"]), target_text=None)
    if esc:
        for p in PERSONA:
            add(**base, concept="C1", kind="reply_esc", persona=p, system=PERSONA[p].format(d=t["domain"]),
                messages=[{"role": "user", "content": x} for x in esc], target_text=None, multi_turn=True)
    for c, text in t["c1_controls"].items():
        add(**base, concept="C1", kind=f"c1ctrl_{c}", persona="default", system=PERSONA["default"].format(d=t["domain"]),
            messages=u(t["user"]["frustrated"]), target_text=text)

with open(out, "w") as f:
    for it in items: f.write(json.dumps(it, ensure_ascii=False) + "\n")
from collections import Counter
print(len(T), "templates;", len(items), "items")
print("by split/concept:", dict(Counter((i["split"], i["concept"]) for i in items)))
print("to generate (C1 replies):", sum(1 for i in items if i["target_text"] is None), "| multi-turn:", sum(1 for i in items if i.get("multi_turn")))
print("sha256", hashlib.sha256(open(out, "rb").read()).hexdigest()[:16])
json.dump({"split": split, "personas": PERSONA}, open(out.replace(".jsonl", "_split.json"), "w"), ensure_ascii=False, indent=1)
