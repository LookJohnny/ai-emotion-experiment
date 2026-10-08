"""Nous Tone tests on a tiny random Qwen3 (CPU, seconds). Uses the real Qwen3 tokenizer for the chat template."""
import json, os, sys, tempfile, unittest
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from nous_tone.hooks import HeadHooks
from nous_tone.pack import TonePack
from nous_tone.engine import ToneEngine
from nous_tone import build as B

TOK_DIR = os.environ.get("NOUS_TONE_TOKENIZER", os.path.expanduser("~/.nous-lens/test-models/Qwen3-1.7B"))


def tiny(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=151936, hidden_size=64, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      intermediate_size=128, max_position_embeddings=1024, tie_word_embeddings=False)
    return Qwen3ForCausalLM(cfg).eval()


class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = AutoTokenizer.from_pretrained(TOK_DIR); cls.model = tiny()

    def test_zero_steer_is_identity(self):
        H = HeadHooks(self.model); ids = torch.tensor([self.tok("你好，今天怎么样？", add_special_tokens=False)["input_ids"]])
        with torch.no_grad(): a = self.model(ids).logits
        H.mode, H.V, H.start = "steer", {1: torch.zeros(64)}, 2
        with torch.no_grad(): b = self.model(ids).logits
        H.remove(); self.assertTrue(torch.allclose(a, b))

    def test_attribution_gradient_matches_finite_difference(self):
        """grad . v at positions [s0, e) equals the finite-difference change of log p(r) when v is added there."""
        H = HeadHooks(self.model); ids = self.tok("我刚收到快递，发现东西少了一件。", add_special_tokens=False)["input_ids"]
        r = self.tok("真让人生气", add_special_tokens=False)["input_ids"]; L, h, s0, e = 1, 2, 3, len(ids)
        v = torch.randn(16)
        def logp(extra=None):
            hd = None
            if extra is not None:
                def add(mod, args):
                    x = args[0].clone(); x[0, s0:e, h * 16:(h + 1) * 16] += extra; return (x,)
                hd = self.model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(add)
            out = self.model(torch.tensor([ids + r])).logits[0, len(ids) - 1:-1]
            if hd: hd.remove()
            return torch.log_softmax(out, -1)[torch.arange(len(r)), torch.tensor(r)].sum()
        H.mode = "grad"; H.cap = {}; lp = logp(); lp.backward()
        g = H.cap[L].grad[0, s0:e, h * 16:(h + 1) * 16]; analytic = float((g * v).sum()); H.mode = None; H.remove()
        eps = 1e-3
        with torch.no_grad(): fd = float((logp(eps * v) - logp(-eps * v)) / (2 * eps))
        self.assertAlmostEqual(analytic, fd, delta=1e-3 * max(1, abs(fd)))

    def test_pack_roundtrip(self):
        p = TonePack("anger", "m", 3, 4, 16, [[1, 2], [0, 1]], np.random.randn(2, 16).astype(np.float32), readout={"a": 1.0, "b": 0.0}, expression="angry")
        d = tempfile.mkdtemp(); p.save(d); q = TonePack.load(d)
        self.assertEqual(q.heads, p.heads); self.assertTrue(np.allclose(q.vectors, p.vectors)); self.assertEqual(q.expression, "angry")

    def test_engine_matches_full_recompute_greedy_without_tone(self):
        """The KV-cached decoding loop must equal greedy decoding that recomputes the full sequence every step.
        (HF generate is not used as the reference: on this tiny model its raw step-0 logits differ from a plain forward by ~0.09.)"""
        eng = ToneEngine(model=self.model, tok=self.tok, device="cpu")
        msgs = [{"role": "user", "content": "周末下雨了，露营取消了。"}]
        ids, _ = eng._render(msgs); seq = list(ids); ref = []
        with torch.no_grad():
            for _ in range(12):
                n = int(self.model(torch.tensor([seq])).logits[0, -1].argmax())
                if n in eng.eos: break
                ref.append(n); seq.append(n)
        out = eng.generate(msgs, max_new_tokens=12)["text"]
        self.assertEqual(out, self.tok.decode(ref, skip_special_tokens=True)); eng.H.remove()

    def test_steering_changes_output_and_closed_loop_raises_lambda(self):
        eng = ToneEngine(model=self.model, tok=self.tok, device="cpu")
        v = np.random.RandomState(0).randn(4, 16).astype(np.float32) * 5
        eng.add_pack(TonePack("anger", "tiny", 3, 4, 16, [[0, 1], [1, 0], [1, 3], [2, 2]], v, default_lambda=4.0, max_lambda=8.0,
                              readout={"a": 0.0, "b": -10.0}, expression="angry"))        # readout pinned near 0
        msgs = [{"role": "user", "content": "老板让我周末加班。"}]
        a = eng.generate(msgs, max_new_tokens=10)["text"]; b = eng.generate(msgs, tone={"anger": 2.0}, max_new_tokens=10)["text"]
        self.assertNotEqual(a, b)
        p = eng.packs["anger"]; p.meta["calibration"] = {"strength": [0, 1, 2], "readout": [0.1, 0.4, 0.8]}
        self.assertAlmostEqual(eng.strength_for(p, 0.6), 1.5); self.assertEqual(eng.strength_for(p, 0.95), 2.0)
        lams = [ev["lambda"].get("anger", 0) for ev in eng.stream(msgs, target={"anger": 0.6}, max_new_tokens=8) if not ev.get("done")]
        self.assertAlmostEqual(lams[0], 6.0); self.assertLessEqual(max(lams), 8.0)               # calibrated start, capped
        ev = next(e for e in eng.stream(msgs, max_new_tokens=2) if not e.get("done"))
        self.assertIn("angry", ev["expression"]); self.assertIn("neutral", ev["expression"])
        with self.assertRaises(ValueError): next(eng.stream(msgs, tone={"joy": 1.0}))
        eng.H.remove()

    def test_build_end_to_end(self):
        d = tempfile.mkdtemp()
        p = B.build("tiny", "anger", d, k=3, n=3, gen=4, device="cpu", model=tiny(1), tok=self.tok, log=lambda *a: None)
        q = TonePack.load(d)
        self.assertLessEqual(len(q.heads), 3); self.assertEqual(q.vectors.shape[1], 16); self.assertIn("a", q.readout)
        self.assertEqual(set(q.meta["dose_curve"]), {"0.25", "0.5", "1.0", "2.0", "4.0", "8.0"}); self.assertEqual(q.expression, "angry")


if __name__ == "__main__":
    unittest.main()


class Server(unittest.TestCase):
    def test_api(self):
        from fastapi.testclient import TestClient
        from nous_tone.server import create_app
        tok = AutoTokenizer.from_pretrained(TOK_DIR); eng = ToneEngine(model=tiny(), tok=tok, device="cpu")
        eng.add_pack(TonePack("anger", "tiny", 3, 4, 16, [[0, 1], [2, 2]], np.ones((2, 16), np.float32), readout={"a": 1.0, "b": 0.0}, expression="angry"))
        c = TestClient(create_app(eng, "tiny")); msgs = [{"role": "user", "content": "你好"}]
        r = c.post("/v1/chat/completions", json={"messages": msgs, "max_tokens": 5, "tone": {"anger": 1.0}}).json()
        self.assertIn("content", r["choices"][0]["message"]); self.assertIn("anger", r["x_tone"]["readout"])
        self.assertEqual(c.post("/v1/chat/completions", json={"messages": msgs, "tone": {"joy": 1}}).status_code, 400)
        self.assertEqual(c.post("/v1/chat/completions", json={"messages": msgs, "tone": {"anger": 9}}).status_code, 400)
        lines = [l for l in c.post("/v1/chat/completions", json={"messages": msgs, "max_tokens": 4, "stream": True, "tone_target": {"anger": 0.6}}).text.splitlines() if l.startswith("data: ")]
        self.assertEqual(lines[-1], "data: [DONE]"); first = json.loads(lines[0][6:])
        self.assertIn("expression", first["x_tone"]); self.assertEqual(c.get("/v1/tone/packs").json()["packs"][0]["attribute"], "anger")
        eng.H.remove()


class Utf8(unittest.TestCase):
    def test_no_replacement_chars_from_split_characters(self):
        """Characters split across tokens (emoji) must not leave U+FFFD in the streamed text."""
        tok = AutoTokenizer.from_pretrained(TOK_DIR); eng = ToneEngine(model=tiny(), tok=tok, device="cpu")
        target = tok("好的😄🎉", add_special_tokens=False)["input_ids"]
        class Forced:                                  # a stand-in model that emits a fixed token sequence
            def __init__(s, m): s.m, s.i, s.config, s.generation_config = m, 0, m.config, m.generation_config
            def __call__(s, ids, **kw):
                out = s.m(ids, **kw); logits = torch.full_like(out.logits, -1e9)
                logits[0, -1, target[s.i] if s.i < len(target) else eng.tok.eos_token_id] = 0; s.i += 1; out.logits = logits; return out
        eng.model = Forced(eng.model)
        text = "".join(ev.get("delta", "") for ev in eng.stream([{"role": "user", "content": "hi"}], max_new_tokens=len(target) + 1) if not ev.get("done"))
        self.assertNotIn("�", text); self.assertEqual(text, "好的😄🎉"); eng.H.remove()


class Read(unittest.TestCase):
    def test_read_segments(self):
        """read(): per-segment probe readouts over the forced-in text, shrunk toward the whole-text mean."""
        from nous_tone.readout import ReadoutProbe
        tok = AutoTokenizer.from_pretrained(TOK_DIR); eng = ToneEngine(model=tiny(), tok=tok, device="cpu")
        eng.add_pack(TonePack("joy", "tiny", 3, 4, 16, [[0, 1]], np.ones((1, 16), np.float32), readout={"a": 1.0, "b": 0.0}, expression="happy"))
        rng = np.random.RandomState(0); d = tempfile.mkdtemp()
        np.savez(os.path.join(d, "readout.npz"), mean=np.zeros(64), scale=np.ones(64), coef=rng.randn(2, 64), intercept=np.zeros(2))
        json.dump({"layer": 1, "classes": ["reference", "joy"]}, open(os.path.join(d, "readout.json"), "w"))
        eng.set_readout(d)
        text = "今天真好。我们去公园吧！"; cut = text.index("。") + 1
        r = eng.read(text, [[0, cut], [cut, len(text)]], user="周末干嘛")
        self.assertEqual(len(r["segments"]), 2); self.assertGreater(r["segments"][0]["tokens"], 0)
        for s in r["segments"] + [r]:
            self.assertAlmostEqual(s["readout"]["joy"] + s["readout"]["neutral"], 1.0, places=3); self.assertIn("happy", s["expression"])
        # infinite shrinkage -> every segment equals the whole-text readout; zero shrinkage -> segments differ
        same = eng.read(text, [[0, cut], [cut, len(text)]], shrink=1e9)
        self.assertTrue(all(abs(s["readout"]["joy"] - same["readout"]["joy"]) < 1e-3 for s in same["segments"]))
        raw = eng.read(text, [[0, cut], [cut, len(text)]], shrink=0.0)
        self.assertNotAlmostEqual(raw["segments"][0]["readout"]["joy"], raw["segments"][1]["readout"]["joy"], places=4)
        # reading must not depend on an active steering state
        eng.H.mode, eng.H.V, eng.H.start = "steer", {0: torch.ones(64) * 50}, 0
        self.assertAlmostEqual(eng.read(text, user="周末干嘛")["readout"]["joy"], r["readout"]["joy"], places=5)
        self.assertEqual(eng.H.mode, "steer"); eng.H.mode = None
        with self.assertRaises(ValueError): eng.read(text, [[3, 2]])
        from fastapi.testclient import TestClient
        from nous_tone.server import create_app
        c = TestClient(create_app(eng, "tiny"))
        ok = c.post("/v1/tone/read", json={"text": text, "segments": [[0, cut]], "user": "嗨"}).json()
        self.assertEqual(ok["segments"][0]["end"], cut)
        self.assertEqual(c.post("/v1/tone/read", json={"text": ""}).status_code, 400)
        self.assertEqual(c.post("/v1/tone/read", json={"text": text, "segments": [[0, 999]]}).status_code, 400)
        eng.H.remove()

    def test_readout_only_mode(self):
        """No steering packs: attributes and avatar expressions come from the probe's classes."""
        tok = AutoTokenizer.from_pretrained(TOK_DIR); eng = ToneEngine(model=tiny(), tok=tok, device="cpu")
        d = tempfile.mkdtemp(); rng = np.random.RandomState(1)
        np.savez(os.path.join(d, "readout.npz"), mean=np.zeros(64), scale=np.ones(64), coef=rng.randn(3, 64), intercept=np.zeros(3))
        json.dump({"layer": 2, "classes": ["reference", "joy", "warmth"]}, open(os.path.join(d, "readout.json"), "w"))
        eng.set_readout(d)
        self.assertEqual(eng.attributes(), ["joy", "warmth"])
        r = eng.read("今天真好。", user="嗨")
        self.assertEqual(set(r["readout"]), {"joy", "warmth", "neutral"}); self.assertEqual(set(r["expression"]), {"happy", "relaxed", "neutral"})
        ev = [e for e in eng.stream([{"role": "user", "content": "嗨"}], max_new_tokens=3) if not e.get("done")]
        self.assertIn("happy", ev[-1]["expression"])
        with self.assertRaises(ValueError): next(eng.stream([{"role": "user", "content": "嗨"}], tone={"joy": 1.0}))
        eng.H.remove()


class Calibration(unittest.TestCase):
    def test_fit_temperature_recovers_scale(self):
        from nous_tone.readout import fit_temperature
        rng = np.random.default_rng(0); n, k = 4000, 5
        base = rng.normal(size=(n, k)); p = np.exp(base) / np.exp(base).sum(1, keepdims=True)
        y = np.array([rng.choice(k, p=row) for row in p])
        T, nll = fit_temperature(base * 4.0, y)            # logits 4x too sharp -> T ~ 4
        self.assertAlmostEqual(T, 4.0, delta=0.5)
        _, nll1 = fit_temperature(base, y); self.assertAlmostEqual(nll, nll1, delta=0.02)

    def test_sentence_token_groups_follow_the_tts_split(self):
        from nous_tone.readout import sentence_token_groups
        tok = AutoTokenizer.from_pretrained(TOK_DIR)
        ids = tok("太好了！我们明天去吧。好吗", add_special_tokens=False)["input_ids"]
        groups = sentence_token_groups(tok, ids)
        self.assertEqual(len(groups), 3)
        self.assertEqual([tok.decode([ids[i] for i in g]).strip() for g in groups], ["太好了！", "我们明天去吧。", "好吗"])
        self.assertEqual(sorted(i for g in groups for i in g), list(range(len(ids))))

    def test_probe_temperature_softens_probabilities(self):
        from nous_tone.readout import ReadoutProbe
        d = tempfile.mkdtemp(); rng = np.random.RandomState(0)
        np.savez(os.path.join(d, "readout.npz"), mean=np.zeros(8), scale=np.ones(8), coef=rng.randn(3, 8) * 5, intercept=np.zeros(3))
        json.dump({"layer": 1, "classes": ["reference", "joy", "warmth"]}, open(os.path.join(d, "readout.json"), "w"))
        h = rng.randn(8); sharp = max(ReadoutProbe(d).probs(h).values())
        json.dump({"layer": 1, "classes": ["reference", "joy", "warmth"], "temperature": 10.0}, open(os.path.join(d, "readout.json"), "w"))
        soft = ReadoutProbe(d).probs(h)
        self.assertLess(max(soft.values()), sharp); self.assertAlmostEqual(sum(soft.values()), 1.0, places=6)


class Priority(unittest.TestCase):
    def test_high_priority_waiter_goes_first_fifo_within_priority(self):
        import threading, time
        from nous_tone.engine import PriorityLock
        lock, order = PriorityLock(), []
        lock.acquire(0)                                                    # a background generation is running
        def worker(name, prio):
            with lock.hold(prio): order.append(name)
        ts = []
        for name, prio in [("bg1", 0), ("bg2", 0), ("user", 1), ("bg3", 0), ("read", 1)]:
            t = threading.Thread(target=worker, args=(name, prio)); t.start(); ts.append(t); time.sleep(0.05)
        lock.release()
        for t in ts: t.join(5)
        self.assertEqual(order, ["user", "read", "bg1", "bg2", "bg3"])
        with lock: pass                                                    # plain `with` still works (priority 0)


class JsonMode(unittest.TestCase):
    def test_object_end_scanner_and_slice(self):
        from nous_tone.engine import JsonObjectEnd, json_object_slice
        s = JsonObjectEnd()
        self.assertFalse(s.feed('```json\n{"a": "}{", "b": {"c": "\\"}"}'))   # braces inside strings / escaped quotes
        self.assertTrue(s.feed('}\n```\n额外的话 {"x":1}'))
        self.assertEqual(json_object_slice('好的：```json\n{"a": "}", "b": [1, {"c": 2}]}\n```后面'), '{"a": "}", "b": [1, {"c": 2}]}')
        self.assertEqual(json_object_slice("no json"), "no json")
        self.assertEqual(json_object_slice('{"open": '), '{"open": ')

    def test_generation_stops_when_the_object_closes(self):
        tok = AutoTokenizer.from_pretrained(TOK_DIR); eng = ToneEngine(model=tiny(), tok=tok, device="cpu")
        target = tok('{"dialogue": [{"text": "嗨}"}]} 然后继续说很多很多废话', add_special_tokens=False)["input_ids"]
        class Forced:
            def __init__(s, m): s.m, s.i, s.config, s.generation_config = m, 0, m.config, m.generation_config
            def __call__(s, ids, **kw):
                out = s.m(ids, **kw); logits = torch.full_like(out.logits, -1e9)
                logits[0, -1, target[s.i] if s.i < len(target) else eng.tok.eos_token_id] = 0; s.i += 1; out.logits = logits; return out
        eng.model = Forced(eng.model)
        evs = list(eng.stream([{"role": "user", "content": "hi"}], max_new_tokens=len(target) + 1, json_object=True))
        text = "".join(e.get("delta", "") for e in evs if not e.get("done"))
        self.assertTrue(text.startswith('{"dialogue": [{"text": "嗨}"}]}')); self.assertNotIn("废话", text)
        self.assertEqual(evs[-1]["finish_reason"], "stop"); eng.H.remove()


class Prefill(unittest.TestCase):
    def test_prefill_opens_the_reply_and_conditions_generation(self):
        tok = AutoTokenizer.from_pretrained(TOK_DIR); eng = ToneEngine(model=tiny(), tok=tok, device="cpu")
        msgs = [{"role": "user", "content": "你好"}]
        text = eng.generate(msgs, max_new_tokens=6, prefill='{"selected_intent": "')["text"]
        self.assertTrue(text.startswith('{"selected_intent": "'))
        # the continuation is what greedy decoding gives after prompt + prefill tokens
        ids, _ = eng._render(msgs); seq = ids + tok('{"selected_intent": "', add_special_tokens=False)["input_ids"]; ref = []
        with torch.no_grad():
            for _ in range(6):
                n = int(eng.model(torch.tensor([seq])).logits[0, -1].argmax())
                if n in eng.eos: break
                ref.append(n); seq.append(n)
        full = tok.decode(tok('{"selected_intent": "', add_special_tokens=False)["input_ids"] + ref, skip_special_tokens=True)
        self.assertEqual(text, full)
        from fastapi.testclient import TestClient
        from nous_tone.server import create_app
        c = TestClient(create_app(eng, "tiny"))
        r = c.post("/v1/chat/completions", json={"messages": msgs, "max_tokens": 3, "prefill": "{\"a\": "}).json()
        self.assertTrue(r["choices"][0]["message"]["content"].startswith('{"a": '))
        self.assertEqual(c.post("/v1/chat/completions", json={"messages": msgs, "prefill": 5}).status_code, 400)
        eng.H.remove()


class Preemption(unittest.TestCase):
    def test_background_generation_yields_to_a_waiting_user(self):
        import threading, time
        tok = AutoTokenizer.from_pretrained(TOK_DIR); eng = ToneEngine(model=tiny(), tok=tok, device="cpu")
        msgs = [{"role": "user", "content": "你好"}]
        result = {}

        def background():
            result["bg"] = eng.generate(msgs, max_new_tokens=400, priority=0, preemptible=True)

        t = threading.Thread(target=background); t.start(); time.sleep(0.3)
        user = eng.generate(msgs, max_new_tokens=3, priority=1)
        t.join(10)
        self.assertEqual(result["bg"]["finish_reason"], "preempted")
        self.assertIsNotNone(user["finish_reason"])
        # a non-preemptible request is never cut short
        quiet = eng.generate(msgs, max_new_tokens=5, priority=0)
        self.assertNotEqual(quiet["finish_reason"], "preempted")
        from fastapi.testclient import TestClient
        from nous_tone.server import create_app
        c = TestClient(create_app(eng, "tiny"))
        eng.lock.outranked = lambda p: True  # simulate a waiting user
        r = c.post("/v1/chat/completions", json={"messages": msgs, "max_tokens": 5, "preemptible": True})
        self.assertEqual(r.status_code, 409); self.assertEqual(r.json()["detail"]["code"], "preempted")
        eng.H.remove()


class InterleavedRead(unittest.TestCase):
    def test_a_read_during_generation_is_served_between_tokens_and_leaves_no_trace(self):
        import threading, time
        tok = AutoTokenizer.from_pretrained(TOK_DIR); eng = ToneEngine(model=tiny(), tok=tok, device="cpu")
        d = tempfile.mkdtemp(); rng = np.random.RandomState(2)
        np.savez(os.path.join(d, "readout.npz"), mean=np.zeros(64), scale=np.ones(64), coef=rng.randn(3, 64), intercept=np.zeros(3))
        json.dump({"layer": 1, "classes": ["reference", "joy", "warmth"]}, open(os.path.join(d, "readout.json"), "w"))
        eng.set_readout(d)
        msgs = [{"role": "user", "content": "讲个长故事"}]
        alone = eng.generate(msgs, max_new_tokens=300)
        standalone_read = eng.read("今天真好。", user="嗨")
        out, timing = {}, {}

        def generation():
            t0 = time.monotonic(); out["gen"] = eng.generate(msgs, max_new_tokens=300); timing["gen_end"] = time.monotonic() - t0

        t = threading.Thread(target=generation); t.start(); time.sleep(0.2)
        t0 = time.monotonic(); during = eng.read("今天真好。", user="嗨"); timing["read"] = time.monotonic() - t0
        t.join(30)
        self.assertLess(timing["read"], timing["gen_end"] - 0.2)  # did not wait for the generation
        self.assertEqual(out["gen"]["text"], alone["text"])  # the generation is unaffected
        self.assertEqual(out["gen"]["readout"], alone["readout"])
        self.assertEqual(during["readout"], standalone_read["readout"])  # same answer as alone
        eng.H.remove()


class PrefixCache(unittest.TestCase):
    def test_reusing_a_cached_prompt_head_changes_nothing(self):
        tok = AutoTokenizer.from_pretrained(TOK_DIR)
        system = {"role": "system", "content": "你是小星。" + "你温柔、耐心，喜欢陪人聊天。" * 12}
        turn1 = [system, {"role": "user", "content": "你好"}]
        turn2 = turn1 + [{"role": "assistant", "content": "你好呀"}, {"role": "user", "content": "今天下雨了"}]
        v = np.random.RandomState(0).randn(4, 16).astype(np.float32) * 5
        pack = lambda: TonePack("anger", "tiny", 3, 4, 16, [[0, 1], [1, 0], [1, 3], [2, 2]], v, default_lambda=4.0, max_lambda=8.0,
                                readout={"a": 0.0, "b": 0.0}, expression="angry")
        for tone in ({}, {"anger": 1.0}):
            cached = ToneEngine(model=tiny(), tok=tok, device="cpu"); cached.add_pack(pack())
            fresh = ToneEngine(model=tiny(), tok=tok, device="cpu", prefix_cache=0); fresh.add_pack(pack())
            for msgs in (turn1, turn2, turn2):
                a = cached.generate(msgs, max_new_tokens=12, tone=tone)
                b = fresh.generate(msgs, max_new_tokens=12, tone=tone)
                self.assertEqual(a["text"], b["text"], f"tone={tone}")
                self.assertEqual(a["readout"], b["readout"])
            self.assertTrue(cached._prefixes); self.assertFalse(fresh._prefixes)
            ids, start = cached._render(turn2)
            reused, _ = cached._prefix_lookup(ids, start, steered=bool(tone))
            self.assertGreaterEqual(reused, ToneEngine.MIN_REUSE)
            if tone:  # steered positions are never reused
                self.assertLessEqual(reused, start)
            cached.H.remove(); fresh.H.remove()
