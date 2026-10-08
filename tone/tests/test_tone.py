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
        lams = [ev["lambda"].get("anger", 0) for ev in eng.stream(msgs, target={"anger": 0.8}, max_new_tokens=8) if not ev.get("done")]
        self.assertGreater(lams[-1], lams[0]); self.assertLessEqual(max(lams), 8.0)               # rises toward the target, capped
        ev = next(e for e in eng.stream(msgs, max_new_tokens=2) if not e.get("done"))
        self.assertIn("angry", ev["expression"]); self.assertIn("neutral", ev["expression"])
        with self.assertRaises(ValueError): next(eng.stream(msgs, tone={"joy": 1.0}))
        eng.H.remove()

    def test_build_end_to_end(self):
        d = tempfile.mkdtemp()
        p = B.build("tiny", "anger", d, k=3, n=3, gen=4, device="cpu", model=tiny(1), tok=self.tok, log=lambda *a: None)
        q = TonePack.load(d)
        self.assertLessEqual(len(q.heads), 3); self.assertEqual(q.vectors.shape[1], 16); self.assertIn("a", q.readout)
        self.assertEqual(set(q.meta["dose_curve"]), {"1.0", "2.0", "4.0", "8.0"}); self.assertEqual(q.expression, "angry")


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
