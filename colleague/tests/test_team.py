"""Team report: each computer's bundle is checked on its own (chain, time stamps, history since last report, still recording)."""
import io, json, os, subprocess, sys, tarfile, tempfile, time, unittest

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); HOOKS = os.path.join(ROOT, "hooks")
sys.path.insert(0, ROOT); sys.path.insert(0, HOOKS)
from nous_colleague import team
import nous_ledger as L

SCRIPT = """
import sys, json; sys.path.insert(0, {hooks!r}); import nous_ledger as L
L.set_label({label!r})
for c, o, n in {events!r}:
    for i in range(n): L.append(c, o, "Claude Code · demo", "t", c + str(i), kind="reading" if o in ("clean", "flagged") else "guard")
if {stamp!r}: L.stamp()
print(json.dumps(L.export({out!r})))
"""


def machine(label, events, stamp=False):
    home = tempfile.mkdtemp(); out = os.path.join(home, "bundle.tar.gz")
    subprocess.run([sys.executable, "-c", SCRIPT.format(hooks=HOOKS, label=label, events=events, stamp=stamp, out=out)],
                   env=dict(os.environ, NOUS_LEDGER_HOME=home), check=True, capture_output=True)
    return home, out


def rewrite(home, fn):
    """Change a ledger and re-hash the whole chain, as someone covering their tracks would."""
    p = os.path.join(home, "ledger.jsonl"); recs = [json.loads(l) for l in open(p, encoding="utf-8")]
    recs = fn(recs); prev = L.GENESIS
    for i, r in enumerate(recs): r["seq"] = i; r["prev"] = prev; r["hash"] = L._hash(r); prev = r["hash"]
    open(p, "w", encoding="utf-8").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))


def reexport(home):
    out = os.path.join(home, f"bundle-{time.time_ns()}.tar.gz")
    subprocess.run([sys.executable, os.path.join(HOOKS, "nous_ledger.py"), "export", out], env=dict(os.environ, NOUS_LEDGER_HOME=home), check=True, capture_output=True)
    return out


EV = [("data_exposure", "blocked", 2), ("unauthorized_action", "denied", 1), ("risky_command", "noted", 3), ("reading", "clean", 10), ("hostility", "flagged", 1)]


class Team(unittest.TestCase):
    def test_honest_team_adds_up(self):
        _, a = machine("设计部 · Mac", EV); _, b = machine("财务 · PC", EV[:2])
        rep = team.build([a, b])
        self.assertTrue(all(m["ok"] for m in rep["machines"]))
        self.assertEqual(rep["team"]["categories"]["data_exposure"]["events"], 4)
        self.assertEqual(rep["team"]["readouts"]["hostility"]["flagged"], 1)
        self.assertEqual(rep["team"]["sources"], ["Claude Code"])
        h = team.render(rep); self.assertIn("全部核验通过", h); self.assertIn("设计部 · Mac", h)

    def test_edited_record_breaks_chain_and_is_not_added(self):
        home, _ = machine("A", EV)
        p = os.path.join(home, "ledger.jsonl"); lines = open(p).read().splitlines()
        r = json.loads(lines[1]); r["outcome"] = "approved"; lines[1] = json.dumps(r, ensure_ascii=False)
        open(p, "w").write("\n".join(lines) + "\n")
        rep = team.build([reexport(home)])
        self.assertFalse(rep["machines"][0]["ok"]); self.assertEqual(rep["team"]["events"], 0); self.assertEqual(rep["machines"][0]["records"], 17)
        self.assertIn("哈希链断裂", team.render(rep))

    def test_rewritten_or_shortened_history_since_last_report(self):
        state = os.path.join(tempfile.mkdtemp(), "team.json")
        home, b1 = machine("A", EV)
        self.assertTrue(team.build([b1], state_path=state)["machines"][0]["ok"])
        rewrite(home, lambda recs:
                [dict(r, outcome="approved") if i == 2 else r for i, r in enumerate(recs)])
        m = team.build([reexport(home)], state_path=state)["machines"][0]
        self.assertTrue(m["chain_ok"]); self.assertFalse(m["ok"]); self.assertIn("已被改写", "；".join(m["problems"]))
        home2, c1 = machine("B", EV); team.build([c1], state_path=state)
        rewrite(home2, lambda recs: recs[:5])
        m = team.build([reexport(home2)], state_path=state)["machines"][0]
        self.assertIn("记录比上次汇总时少", "；".join(m["problems"]))

    def test_missing_computer_is_named(self):
        state = os.path.join(tempfile.mkdtemp(), "team.json")
        _, a = machine("A", EV); _, b = machine("B", EV)
        team.build([a, b], state_path=state)
        self.assertEqual(len(team.build([a], state_path=state)["missing_since_last"]), 1)

    def test_stale_computer(self):
        home, _ = machine("A", EV)
        rewrite(home, lambda recs: [dict(r, ts=r["ts"] - 20 * 86400) for r in recs])
        m = team.build([reexport(home)])["machines"][0]
        self.assertTrue(m["ok"]); self.assertTrue(m["stale"])

    def test_hostile_bundle_is_read_not_extracted(self):
        d = tempfile.mkdtemp(); p = os.path.join(d, "evil.tar.gz"); target = os.path.join(d, "pwned.txt")
        with tarfile.open(p, "w:gz") as tf:
            for name, data in (("../pwned.txt", b"x"), (target, b"x"), ("manifest.json", b'{"machine":"m1"}'), ("ledger.jsonl", b"not json\n")):
                ti = tarfile.TarInfo(name); ti.size = len(data); tf.addfile(ti, io.BytesIO(data))
        m = team.build([p])["machines"][0]
        self.assertFalse(m["ok"]); self.assertFalse(os.path.exists(target)); self.assertFalse(os.path.exists(os.path.join(os.path.dirname(d), "pwned.txt")))
        self.assertFalse(m["problems"] == [])
        self.assertEqual(team.build([os.path.join(d, "nope.tar.gz")])["machines"][0]["ok"], False)

    def test_export_has_no_key_and_no_content(self):
        home, b = machine("A", [("data_exposure", "blocked", 1)])
        with tarfile.open(b) as tf: names = tf.getnames(); blob = b"".join(tf.extractfile(m).read() for m in tf.getmembers() if m.isfile())
        self.assertNotIn("ledger.key", " ".join(names)); self.assertNotIn(open(os.path.join(home, "ledger.key"), "rb").read(), blob)
        self.assertNotIn(b"data_exposure0", blob)

    @unittest.skipUnless(os.environ.get("NOUS_TEST_NETWORK"), "contacts a public time-stamping authority")
    def test_time_stamps_travel_with_the_bundle(self):
        home, b = machine("A", EV, stamp=True)
        m = team.build([b])["machines"][0]
        self.assertTrue(m["ok"]); self.assertEqual(m["stamped_records"], m["records"])
        rewrite(home, lambda recs: [dict(r, outcome="approved") if i == 1 else r for i, r in enumerate(recs)])
        m = team.build([reexport(home)])["machines"][0]
        self.assertTrue(m["chain_ok"]); self.assertFalse(m["ok"]); self.assertIn("时间戳", "；".join(m["problems"]))


if __name__ == "__main__":
    unittest.main()
