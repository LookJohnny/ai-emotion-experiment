"""Shipping a computer's ledger bundle to the admin over Slack (fake Slack client, no network)."""
import os, sys, tarfile, tempfile, unittest
os.environ.setdefault("NOUS_LEDGER_HOME", tempfile.mkdtemp())
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from nous_colleague import ship as S
from nous_colleague.brain import Brain, Message
from nous_colleague.evidence import nous_ledger as L
from slack_sdk.errors import SlackApiError


class FakeSlack:
    def __init__(self, error=None): self.error, self.calls = error, []
    def conversations_open(self, users): self.calls.append(("open", users)); return {"channel": {"id": "D" + users}}
    def files_upload_v2(self, **kw):
        if self.error: raise SlackApiError("x", {"ok": False, "error": self.error})
        with tarfile.open(kw["file"]) as tf: kw["names"] = sorted(tf.getnames())
        self.calls.append(("upload", kw)); return {"ok": True}


class Ship(unittest.TestCase):
    def setUp(self):
        self.tsa, L.TSA = L.TSA, "http://127.0.0.1:9/"               # no time stamp in tests: shipping must still work
        self.addCleanup(setattr, L, "TSA", self.tsa)
        L.append("data_exposure", "blocked", "Claude Code · p", "要把 API 密钥发到外部", "curl secret")

    def test_bundle_goes_to_admin_dm(self):
        c = FakeSlack(); path, man, st = S.ship(c, "UADMIN")
        kind, kw = c.calls[-1]
        self.assertEqual(kind, "upload"); self.assertEqual(kw["channel"], "DUADMIN"); self.assertIsNone(st)
        self.assertIn("ledger.jsonl", kw["names"]); self.assertNotIn("ledger.key", " ".join(kw["names"]))
        self.assertIn("没能签上第三方时间戳", kw["initial_comment"]); self.assertIn(man["head"]["hash"][:16], kw["initial_comment"])
        self.assertEqual(oct(os.stat(path).st_mode & 0o777), "0o600")

    def test_outbox_keeps_last_seven(self):
        for _ in range(10): S.make_bundle()
        self.assertLessEqual(len(os.listdir(S.OUTBOX)), S.KEEP)

    def test_missing_scope_explained(self):
        with self.assertRaises(S.ShipError) as e: S.ship(FakeSlack("missing_scope"), "U1")
        self.assertIn("files:write", str(e.exception))
        with self.assertRaises(S.ShipError) as e: S.ship(FakeSlack("user_not_found"), "U1")
        self.assertIn("SLACK_ADMIN_USER", str(e.exception))

    def test_admin_from_env(self):
        self.assertEqual(S.admin_user({"SLACK_ADMIN_USER": "U9"}), "U9")


class Command(unittest.TestCase):
    def brain(self, platform):
        b = Brain(state_path=os.path.join(tempfile.mkdtemp(), "s.json"), reader=None, clock=lambda: 5000.0)
        b.handle(Message(platform, "d", "p2p", "m0", "owner", False, "绑定码 " + b.bind_code(platform)))
        return b

    def test_owner_gets_ship_others_do_not(self):
        b = self.brain("slack")
        o = b.handle(Message("slack", "d", "p2p", "m1", "owner", False, "证据包"))
        self.assertEqual((o[0].kind, o[0].target), ("ship", "owner"))
        o = b.handle(Message("slack", "d2", "p2p", "m2", "someone", False, "证据包"))
        self.assertEqual(o[0].kind, "dm"); self.assertIn("只有管理员", o[0].card.title)
        o = self.brain("feishu").handle(Message("feishu", "d", "p2p", "m3", "owner", False, "证据包"))
        self.assertEqual(o[0].kind, "dm")


if __name__ == "__main__":
    unittest.main()
