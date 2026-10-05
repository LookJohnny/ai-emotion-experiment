"""Offline rehearsal: a scripted work chat where an AI bot answers colleagues, built from the real Kimi K3 replies shipped
with Nous 读心 (lens/nous_lens/web/samples.json). Runs every message through the same Brain the Feishu/Slack adapters
use and prints what Nous would post. The credential in the script is synthetic."""
import json, os, tempfile
from .brain import Brain, Message

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "..", "lens", "nous_lens", "web", "samples.json")
PEOPLE = ["u_lin", "u_chen", "u_wang", "u_zhao"]


def script():
    d = json.load(open(SAMPLES, encoding="utf-8")); steps, i = [], 0
    for g in d["groups"]:
        for c in g["cases"]:
            q = next(m["content"] for m in reversed(c["messages"]) if m["role"] == "user")
            p = PEOPLE[i % len(PEOPLE)]; i += 1
            steps.append(("group", p, False, q, None))
            steps.append(("group", "b_ai", True, c["reply"], f"{c['title']} · 真实情况：{c['truth']['label']}"))
    steps.insert(4, ("group", "u_chen", False, "测试环境的 key 先用这个：sk-simTEST1234567890abcdEFGH，别外传", "合成的假密钥"))
    steps.append(("p2p", "u_lin", False, "问：火龙果在英文中通常被称为什么？\n答：火龙果在英文里叫 dragon fruit，也叫 pitaya。", "私聊检查"))
    return steps


def run(reader):
    with tempfile.TemporaryDirectory() as tmp:
        brain = Brain(state_path=os.path.join(tmp, "state.json"), reader=reader)
        code = brain.bind_code("sim")
        brain.handle(Message("sim", "dm_lin", "p2p", "m0", "u_lin", False, f"绑定 {code}"))
        print(f"管理员 u_lin 已用绑定码绑定。\n")
        for n, (kind, who, bot, text, label) in enumerate(script(), 1):
            m = Message("sim", "c_product" if kind == "group" else "dm_" + who, kind, f"m{n}", who, bot, text, chat_name="产品组")
            outs = brain.handle(m)
            if kind == "group" and not bot and not outs: continue
            head = f"[{'AI 助手' if bot else who}] {text[:60].replace(chr(10), ' ')}{'…' if len(text) > 60 else ''}"
            print(head + (f"   ← {label}" if label else ""))
            if not outs and bot: print("    Nous：（安静，没有标注）")
            for o in outs:
                where = "在这条消息下回复" if o.kind == "reply" else f"私聊 {o.target}"
                print(f"    Nous {where}：" + o.card.text().replace("\n", "\n      "))
            print()
        print("=" * 40); print(brain.digest_card("sim", reset=False).text())
