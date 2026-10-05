"""Brain behaviour with a stub reader (no engine, no network)."""
import os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nous_colleague.brain import Brain, Message
from nous_colleague.reader import ReaderDown


def res(host=False, fab=None, syco=None):
    r = {"hostility": {"score": .9 if host else .1, "threshold": .5, "flag": host, "percentile": 99 if host else 20, "validated": True,
                       "sentences": [{"start": 0, "end": 4, "score": .9, "hot": host}]}}
    if fab is not None: r["fabrication"] = {"score": .8, "threshold": .5, "flag": fab, "percentile": 97, "validated": False}
    if syco is not None: r["sycophancy"] = {"score": .8, "threshold": .5, "flag": syco, "percentile": 92, "validated": False}
    return r


class Stub:
    def __init__(self, out=None, down=False): self.out, self.down, self.calls = out or res(), down, []
    def read(self, reply, context=None, source=""):
        if self.down: raise ReaderDown("x")
        self.calls.append((reply, context)); return self.out
    def health(self): return {"model": "stub", "readouts": ["hostility"]}


def M(i, who, bot, text, chat="c1", kind="group", parent=""):
    return Message("t", chat, kind, f"m{i}", who, bot, text, chat_name="产品组", parent_id=parent, ts=1000.0 + i)


class T(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.path = os.path.join(self.tmp.name, "s.json")

    def brain(self, reader, **kw):
        b = Brain(state_path=self.path, reader=reader, clock=lambda: 5000.0, **kw)
        code = b.bind_code("t"); b.handle(M(0, "owner", False, "绑定码 " + code, chat="d", kind="p2p")); return b

    def test_hostile_ai_reply_flagged_and_owner_dm(self):
        r = Stub(res(host=True)); b = self.brain(r)
        b.handle(M(1, "u1", False, "我做视频三个月了没人看"))
        outs = b.handle(M(2, "bot", True, "你太笨了，活该没人看。"))
        self.assertEqual([o.kind for o in outs], ["reply", "dm"]); self.assertEqual(outs[0].target, "m2"); self.assertEqual(outs[1].target, "owner")
        self.assertEqual(r.calls[-1][1], "我做视频三个月了没人看")           # context = the human message before it

    def test_clean_ai_reply_is_silent(self):
        b = self.brain(Stub(res())); self.assertEqual(b.handle(M(1, "bot", True, "这是一条正常、礼貌的回复。")), [])

    def test_experimental_only_not_posted_when_validated_mode(self):
        b = self.brain(Stub(res(fab=True)), group_reply="validated")
        self.assertEqual(b.handle(M(1, "bot", True, "答案是 1926 年三月。")), [])

    def test_owner_dm_rate_limited(self):
        b = self.brain(Stub(res(host=True)))
        dms = sum(o.kind == "dm" for i in range(6) for o in b.handle(M(10 + i, "bot", True, "你太笨了，活该没人看。")))
        self.assertEqual(dms, 3)

    def test_secret_in_group_dms_sender_without_repeating_it(self):
        b = self.brain(Stub()); b.handle(M(1, "bot", True, "你好，我是群里的 AI 助手。"))
        outs = b.handle(M(2, "u2", False, "key 用这个 sk-simTEST1234567890abcdEFGH"))
        self.assertEqual(len(outs), 1); self.assertEqual((outs[0].kind, outs[0].target), ("dm", "u2"))
        self.assertNotIn("sk-sim", outs[0].card.text()); self.assertIn("AI 机器人", outs[0].card.text())

    def test_phone_number_alone_is_not_a_warning(self):
        b = self.brain(Stub()); self.assertEqual(b.handle(M(1, "u2", False, "有事打我 13800138000")), [])

    def test_forward_check_parses_question_and_answer(self):
        r = Stub(res()); b = self.brain(r)
        outs = b.handle(M(1, "u3", False, "问：火龙果英文叫什么？\n答：dragon fruit，也叫 pitaya。", chat="d3", kind="p2p"))
        self.assertEqual(r.calls[-1], ("dragon fruit，也叫 pitaya。", "火龙果英文叫什么？")); self.assertEqual(outs[0].card.title, "没发现问题")

    def test_bind_code_only_once_and_pause_owner_only(self):
        b = self.brain(Stub(res(host=True)))
        self.assertEqual(b.owner("t"), "owner")
        self.assertEqual(b.handle(M(1, "intruder", False, "暂停", chat="d2", kind="p2p"))[0].card.title, "只有管理员可以操作")
        b.handle(M(2, "owner", False, "暂停", chat="d", kind="p2p"))
        self.assertEqual(b.handle(M(3, "bot", True, "你太笨了，活该没人看。")), [])

    def test_reader_down_notifies_owner_once(self):
        b = self.brain(Stub(down=True))
        a = b.handle(M(1, "bot", True, "一条足够长的 AI 回复。")); c = b.handle(M(2, "bot", True, "另一条足够长的 AI 回复。"))
        self.assertEqual(len(a), 1); self.assertEqual(c, [])

    def test_state_has_no_message_text(self):
        b = self.brain(Stub(res(host=True))); b.handle(M(1, "bot", True, "你太笨了，活该没人看。"))
        with open(self.path, encoding="utf-8") as f: self.assertNotIn("太笨", f.read())


if __name__ == "__main__":
    unittest.main()
