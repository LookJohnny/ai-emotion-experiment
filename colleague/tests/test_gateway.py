"""Gateway scrubbing: keys in a request body become placeholders, stored under stable names. Uses a separate keychain service."""
import os, subprocess, sys, tempfile, unittest

TMP = tempfile.mkdtemp()
os.environ["NOUS_VAULT_SERVICE"] = "center.nous.vault.test"
os.environ["NOUS_VAULT_INDEX"] = os.path.join(TMP, "vault.json")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hooks"))
import nous_gateway, nous_vault

KEY = "sk-" + "simTEST1234567890abcdEFGH"
PW = "hunter" + "2pass"
P = lambda name: "[[" + "NOUS:" + name + "]]"     # built at runtime so no hook swaps it while this file is written


def scrub(obj, values=None):
    found = set(); return nous_gateway.scrub(obj, {} if values is None else values, found), found


class GatewayScrub(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        for n in nous_vault.index():
            subprocess.run(["security", "delete-generic-password", "-s", nous_vault.SERVICE, "-a", n], capture_output=True)
        try: os.remove(nous_vault.INDEX)
        except OSError: pass

    def test_new_key_named_by_provider_and_stored(self):
        out, found = scrub({"messages": [{"content": f"kimi key: {KEY} 帮我测一下"}]})
        self.assertEqual(out["messages"][0]["content"], f"kimi key: {P('KIMI_API_KEY')} 帮我测一下")
        self.assertEqual(found, {"KIMI_API_KEY"})
        self.assertEqual(nous_vault.get("KIMI_API_KEY"), KEY)

    def test_rescrub_adds_no_names(self):
        text = f"openai {KEY}\\nDB_PASSWORD={PW} line"
        once, _ = scrub(text)
        before = set(nous_vault.index())
        twice, found = scrub(once, {n: nous_vault.get(n) for n in before})
        self.assertEqual(twice, once)
        self.assertEqual(set(nous_vault.index()), before)
        self.assertNotIn(PW, once); self.assertNotIn(KEY, once)

    def test_known_value_keeps_its_name(self):
        nous_vault.put("KIMI_API_KEY", KEY)
        out, found = scrub(f"again: {KEY}")                    # not passed as known: the vault lookup must reuse the name
        self.assertEqual(out, f"again: {P('KIMI_API_KEY')}")
        self.assertFalse(any(n.startswith("KIMI_API_KEY_") or n.startswith("API_KEY") for n in nous_vault.index()))

    def test_escaped_newline_does_not_prefix_name(self):
        out, found = scrub(f"x\\nDB_PASSWORD={PW}\\nmore")
        self.assertIn(P("DB_PASSWORD"), out)
        self.assertNotIn("NDB_PASSWORD", found)
        self.assertTrue(out.endswith("\\nmore"))               # the escape is not swallowed into the value

    def test_placeholder_is_not_a_secret(self):
        self.assertEqual(nous_vault.find_secrets(f"DB_PASSWORD={P('DB_PASSWORD')}"), [])
        self.assertEqual(nous_vault.find_secrets(f"FEISHU_APP_SECRET: {P('FEISHU_APP_SECRET')}, ok"), [])

    def test_non_strings_pass_through(self):
        body = {"max_tokens": 100, "stream": True, "temperature": None, "x": [1, 2.5]}
        self.assertEqual(scrub(body)[0], body)


if __name__ == "__main__":
    unittest.main()
