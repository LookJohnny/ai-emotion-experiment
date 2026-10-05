"""Message-text extraction and card rendering for Feishu and Slack (no network)."""
import json, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nous_colleague.cards import Card
from nous_colleague import feishu, slack


class T(unittest.TestCase):
    def test_feishu_text_post_and_card(self):
        self.assertEqual(feishu.text_of("text", json.dumps({"text": "你好"})), "你好")
        post = {"title": "答复", "content": [[{"tag": "text", "text": "第一段"}], [{"tag": "a", "text": "链接", "href": "https://x"}]]}
        self.assertEqual(feishu.text_of("post", json.dumps(post)), "答复\n第一段\n链接")
        card = {"header": {"title": {"tag": "plain_text", "content": "AI 助手"}},
                "elements": [{"tag": "div", "text": {"tag": "lark_md", "content": "答案是 **1926 年**"}}, {"tag": "markdown", "content": "补充"}]}
        self.assertEqual(feishu.text_of("interactive", json.dumps(card)), "AI 助手\n答案是 **1926 年**\n补充")
        self.assertEqual(feishu.text_of("image", "{}"), "")

    def test_feishu_card_json(self):
        c = json.loads(feishu.card_json(Card("Nous 读心", "bad", ["**敌意**：高"], quote="原话", note="注", buttons=[("误报", {"fb": "fp"})])))
        self.assertEqual(c["header"]["template"], "red")
        self.assertEqual([e["tag"] for e in c["elements"]], ["div", "note", "action", "note"])
        self.assertEqual(c["elements"][2]["actions"][0]["value"], {"fb": "fp"})

    def test_slack_text_from_blocks(self):
        ev = {"text": "", "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": "答案是 1926 年"}},
                                     {"type": "rich_text", "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": "补充"}]}]}]}
        self.assertEqual(slack.text_of(ev), "答案是 1926 年\n补充")

    def test_slack_blocks(self):
        b = slack.blocks(Card("Nous 读心", "warn", ["**编造**：高"], quote="原话", buttons=[("误报", {"fb": "fp"}), ("有用", {"fb": "tp"})]))
        self.assertIn("*编造*", b[0]["text"]["text"]); self.assertEqual(b[2]["elements"][1]["action_id"], "nous_fb_1")


if __name__ == "__main__":
    unittest.main()
