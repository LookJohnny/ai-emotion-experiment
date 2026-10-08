"""Held-out check of tone packs: replies to messages the packs never saw, with tone 0 / +1 / -1 per attribute (and +2).
  python -m nous_tone.evaluate MODEL_DIR packs/... OUT.jsonl [--strengths 0 1 -1 2]
Writes one row per (message, attribute, strength) with the reply and the engine's own final readout, for blind rating."""
import argparse, json
from .engine import ToneEngine

HELD_OUT = [
    "我等了一个小时的网约车，司机最后取消了订单。", "今天公司发了一笔意外的奖金。", "我的好朋友要出国了，下周就走。",
    "装修师傅把墙刷错了颜色。", "我第一次在比赛里拿了名次。", "体检报告说我有点贫血。", "我做的蛋糕被大家夸了。",
    "银行卡被莫名扣了一笔钱。", "我们小组的项目被选为优秀案例。", "爷爷最近记性越来越差了。", "下班路上看到了很美的晚霞。",
    "合租室友总是不打扫卫生。", "我终于跑完了人生第一个半马。", "面试等了三周还没有消息。", "表妹考上了她想去的大学。",
    "我的自行车在楼下被偷了。", "周末和家人一起去郊外野餐了。", "领导当众批评了我的方案。", "我养的多肉终于开花了。",
    "快递显示已签收，可我根本没收到。",
]


def main():
    ap = argparse.ArgumentParser(prog="nous_tone.evaluate"); ap.add_argument("model"); ap.add_argument("packs", nargs="+"); ap.add_argument("out")
    ap.add_argument("--strengths", nargs="*", type=float, default=[0.0, 1.0, -1.0, 2.0]); ap.add_argument("--max", type=int, default=64); ap.add_argument("--readout")
    a = ap.parse_args(); eng = ToneEngine(a.model, a.packs, readout_dir=a.readout)
    with open(a.out, "w", encoding="utf-8") as f:
        for i, u in enumerate(HELD_OUT):
            base = eng.generate([{"role": "user", "content": u}], max_new_tokens=a.max)
            f.write(json.dumps({"msg_id": i, "message": u, "attribute": "none", "strength": 0.0, "reply": base["text"], "readout": base["readout"]}, ensure_ascii=False) + "\n")
            for attr in eng.packs:
                for s in a.strengths:
                    if s == 0: continue
                    r = eng.generate([{"role": "user", "content": u}], tone={attr: s}, max_new_tokens=a.max)
                    f.write(json.dumps({"msg_id": i, "message": u, "attribute": attr, "strength": s, "reply": r["text"], "readout": r["readout"],
                                        "lambda": r["lambda"]}, ensure_ascii=False) + "\n")
            print(i, flush=True)


if __name__ == "__main__":
    main()
