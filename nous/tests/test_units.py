"""Unit tests (no GPU): policy, sentence release, config, auth, audit."""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from nous.engine.policy import TwoTier, percentile
from nous.companion import split_ready
from nous.config import Config
from nous.auth import Auth
from nous.audit import Audit


def test_two_tier_needs_min_tokens_and_sustain():
    a = TwoTier(0.5, min_tokens=8, sustain=3)
    assert [a.update(n, 0.9) for n in range(1, 8)] == [(False, False)] * 7       # below min_tokens: silent
    assert a.update(8, 0.9) == (True, False) and a.update(9, 0.9) == (True, False)  # watch
    assert a.update(10, 0.9) == (False, True) and a.fired_at == 10                 # confirmed on the 3rd consecutive
    assert a.update(11, 0.1) == (False, False)                                      # streak resets


def test_two_tier_streak_broken():
    a = TwoTier(0.5, 1, 3); a.update(1, .9); a.update(2, .9); a.update(3, .1)
    assert a.update(4, .9) == (True, False) and a.fired_at is None


def test_percentile():
    q = [i / 100 for i in range(101)]
    assert percentile(0.0, q) == 0 and percentile(1.0, q) == 100 and 49 <= percentile(0.5, q) <= 51 and percentile(0.1, []) is None


def test_split_ready():
    s, rest = split_ready("你好呀。今天怎么样？我")
    assert s == ["你好呀。", "今天怎么样？"] and rest == "我"
    s, rest = split_ready("短句，还没完")
    assert s == [] and rest == "短句，还没完"                                     # short clause: comma is not a break
    s, rest = split_ready("这是一个非常非常长的从句它已经超过了二十八个字所以可以在逗号这里先念出来，后面")
    assert len(s) == 1 and rest == "后面"
    assert split_ready("尾巴", force=True) == (["尾巴"], "")
    assert split_ready("它也在心里。”然后") == (["它也在心里。”"], "然后")          # closing quote stays with its sentence
    assert split_ready("好的。\n）", force=True) == (["好的。\n）"], "")               # punctuation-only tail merges back


def test_config_env_and_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("K1", "secret-admin")
    p = tmp_path / "n.yaml"
    p.write_text("model: /m\npack: packs/x\nadmin_keys: ['${K1}', '${MISSING}']\ngateway: {mode: reader, concepts: [C1]}\n", encoding="utf-8")
    c = Config.load(str(p))
    assert c.admin_keys == ["secret-admin"] and c.pack == str(tmp_path / "packs/x") and c.gateway.mode == "reader" and c.companion.enabled


class R:
    def __init__(self, h=None, q=None): self.headers, self.query_params = h or {}, q or {}


def test_auth():
    c = Config(admin_keys=["adm"], api_keys=["api"], device_keys={"dev": "toy-1"}); a = Auth(c)
    assert a.client(R({"authorization": "Bearer api"})).startswith("key:") and a.client(R({"x-api-key": "adm"})) == "admin"
    with pytest.raises(Exception): a.client(R({"authorization": "Bearer nope"}))
    with pytest.raises(Exception): a.admin(R({"authorization": "Bearer api"}))
    assert a.device("dev") == "toy-1" and a.device("adm") == "console" and a.device("x") is None
    assert Auth(Config()).client(R()) == "open"


def test_audit(tmp_path):
    au = Audit(str(tmp_path / "a.db"))
    au.request("r1", "k", "local", "hi", "hello", "stop", {"C2": .1, "C1_pre": .2}, {"C1_mean": .3}, ["C1_watch"])
    au.event("toy-1", "block", None, held_text="...")
    assert au.recent_requests()[0]["alarms"] == ["C1_watch"] and au.recent_events("toy-1")[0]["kind"] == "block"
    assert au.stats(0) == {"requests": 1, "blocked": 0, "events": {"block": 1}}
