"""Readout probe: which tone the model's text carries, per token, specific across attributes.
  python -m nous_tone.readout MODEL_DIR OUT_DIR [--attrs anger joy sadness warmth] [--n 40] [--gen 48]
The steering heads are chosen for causal push, not for telling tones apart (in Qwen3-1.7B the four attributes share most
heads), so readout uses a separate multinomial linear probe on the residual stream, the approach validated in the research
(E01: an L19 probe tracked blind ratings at rho 0.54-0.71):
- texts: the model's own replies under the reference prompt and under each attribute prompt (same user messages)
- features: the running mean of the residual at one layer over the reply tokens, read with the REFERENCE prompt, so the
  probe sees the text's tone and not the system prompt that produced it
- layer chosen by message-grouped cross-validation over the middle layers; classes = reference + attributes
- sentence samples: avatars read one sentence at a time (ToneEngine.read), so every reply also contributes one sample per
  sentence, built exactly like read() builds it (the sentence's mean shrunk toward the reply mean with SHRINK pseudo-tokens)
- short variants (--short): every class is also generated with the system prompt asking for one or two sentences, so
  the probe also learns tone from short, spoken-style lines (companion speech), not only from 48-token replies
- temperature: the multinomial logits are rescaled by a temperature fitted on out-of-fold predictions (minimum NLL), so
  readouts are calibrated probabilities instead of saturating at 0/1 on text unlike the training replies"""
import argparse, json, os, re, time
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .build import pick_device, render
from .prompts import ATTRIBUTES, REFERENCE, USER_MESSAGES

SENTENCE_RE = re.compile(r"[^。！？!?]+[。！？!?]*")   # the split avatars use per TTS sentence (SoulForge gateway)
SHRINK = 8.0                                        # pseudo-tokens of reply mean in a sentence reading (= ToneEngine.read)
SHORT = "请只用一两句话简短回复，像面对面说话一样。"


def sentence_token_groups(tok, ids):
    """Token index lists, one per sentence of the decoded reply (tokens overlapping the sentence's characters)."""
    text, ends = "", []
    for i in range(len(ids)):
        text = tok.decode(ids[:i + 1], skip_special_tokens=True); ends.append(len(text))
    starts = [0] + ends[:-1]; groups = []
    for m in SENTENCE_RE.finditer(text):
        if not m.group().strip(): continue
        s, e = m.span(); idx = [i for i in range(len(ids)) if ends[i] > s and starts[i] < e]
        if idx: groups.append(idx)
    return groups


def fit_temperature(logits, y):
    """Scalar temperature minimising the NLL of softmax(logits / T) (grid search; logits: [n, classes])."""
    best = (np.inf, 1.0)
    for T in np.exp(np.linspace(np.log(0.25), np.log(50), 200)):
        z = logits / T; z = z - z.max(1, keepdims=True); lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        nll = -lp[np.arange(len(y)), y].mean()
        if nll < best[0]: best = (nll, float(T))
    return best[1], float(best[0])


class ReadoutProbe:
    def __init__(self, d):
        m = json.load(open(os.path.join(d, "readout.json"), encoding="utf-8")); z = np.load(os.path.join(d, "readout.npz"))
        self.layer, self.classes, self.meta = m["layer"], m["classes"], m
        self.mean, self.scale, self.coef, self.intercept = z["mean"], z["scale"], z["coef"], z["intercept"]
        self.temperature = float(m.get("temperature", 1.0))   # probes trained before calibration: 1.0 (uncalibrated)

    def probs(self, h):                                   # h: running-mean residual [hidden] (numpy)
        x = (h - self.mean) / self.scale; s = (self.coef @ x + self.intercept) / self.temperature
        s = np.exp(s - s.max()); p = s / s.sum()
        return {c: float(v) for c, v in zip(self.classes, p)}


def train(model_dir, out, attrs=("anger", "joy", "sadness", "warmth"), n=40, gen=48, device=None, model=None, tok=None, messages=None, log=print,
          short=False):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    t0 = time.time(); device = device or pick_device()
    if model is None:
        tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32 if device == "cpu" else torch.bfloat16).to(device).eval()
    NL = model.config.num_hidden_layers; layers = list(range(NL // 4, NL - NL // 6))
    msgs = (messages or USER_MESSAGES)[:n]; classes = ["reference"] + list(attrs)
    systems = {"reference": REFERENCE, **{a: ATTRIBUTES[a]["system"] for a in attrs}}
    X = {L: [] for L in layers}; y, g, kind = [], [], []                    # kind: 0 = whole reply, 1 = one sentence
    seen = {}                                                                # (message -> classes done once): 2nd pass = short variant
    with torch.no_grad():
        for i, u in enumerate(msgs):
            ref_ids = tok(render(tok, REFERENCE, u), add_special_tokens=False)["input_ids"]
            for ci, c in [(ci, c) for ci, c in enumerate(classes)] * (2 if short else 1):
                variant = short and (ci, c) in seen.setdefault(i, set()); seen[i].add((ci, c))
                ids = tok(render(tok, systems[c] + (SHORT if variant else ""), u), add_special_tokens=False)["input_ids"]
                r = model.generate(torch.tensor([ids], device=device), max_new_tokens=gen, do_sample=False, temperature=None, top_p=None, top_k=None,
                                   pad_token_id=tok.eos_token_id)[0, len(ids):].tolist()
                r = [t for t in r if t != tok.eos_token_id] or [tok.eos_token_id]
                hs = model(torch.tensor([ref_ids + r], device=device), output_hidden_states=True, use_cache=False).hidden_states
                sents = sentence_token_groups(tok, r)
                for L in layers:
                    H = hs[L + 1][0, len(ref_ids):].float().cpu().numpy(); whole = H.mean(0); X[L].append(whole)
                    for idx in sents: X[L].append((H[idx].sum(0) + SHRINK * whole) / (len(idx) + SHRINK))
                y += [ci] * (1 + len(sents)); g += [i] * (1 + len(sents)); kind += [0] + [1] * len(sents)
    y, g, kind = np.array(y), np.array(g), np.array(kind); folds = np.array_split(np.random.default_rng(0).permutation(len(msgs)), 5)
    cv, cv_whole, cv_sent, oof = {}, {}, {}, {}
    for L in layers:                                     # message-grouped CV; a layer is scored on whole replies AND sentences
        A = np.stack(X[L]); pred = np.zeros(len(y), int); logit = np.zeros((len(y), len(classes)))
        for f in folds:
            te = np.isin(g, f); sc = StandardScaler().fit(A[~te]); clf = LogisticRegression(max_iter=3000, C=0.05).fit(sc.transform(A[~te]), y[~te])
            logit[te] = clf.decision_function(sc.transform(A[te])); pred[te] = logit[te].argmax(1)
        ok = pred == y; cv_whole[L] = float(ok[kind == 0].mean()); cv_sent[L] = float(ok[kind == 1].mean()) if (kind == 1).any() else cv_whole[L]
        cv[L] = (cv_whole[L] + cv_sent[L]) / 2; oof[L] = logit
    L = max(cv, key=cv.get); A = np.stack(X[L]); sc = StandardScaler().fit(A); clf = LogisticRegression(max_iter=3000, C=0.05).fit(sc.transform(A), y)
    T, nll = fit_temperature(oof[L], y)                  # calibrate on out-of-fold logits, never on the training fit
    z = oof[L] - oof[L].max(1, keepdims=True); nll_raw = float(-(z - np.log(np.exp(z).sum(1, keepdims=True)))[np.arange(len(y)), y].mean())
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "readout.npz"), mean=sc.mean_, scale=sc.scale_, coef=clf.coef_, intercept=clf.intercept_)
    json.dump({"model": os.path.basename(str(model_dir).rstrip("/")), "layer": int(L), "classes": classes, "cv_accuracy_by_layer": {str(k): round(v, 4) for k, v in cv.items()},
               "cv_accuracy": round(cv_whole[L], 4), "cv_accuracy_sentence": round(cv_sent[L], 4), "chance": round(1 / len(classes), 3),
               "temperature": round(T, 4), "oof_nll": {"uncalibrated": round(nll_raw, 4), "calibrated": round(nll, 4)},
               "n_messages": len(msgs), "n_samples": {"replies": int((kind == 0).sum()), "sentences": int((kind == 1).sum())},
               "shrink": SHRINK, "short_variants": bool(short), "gen_tokens": gen, "seconds": round(time.time() - t0, 1)}, open(os.path.join(out, "readout.json"), "w"), ensure_ascii=False, indent=1)
    log(f"readout probe: layer {L}, {len(classes)}-way CV accuracy {cv_whole[L]:.3f} replies / {cv_sent[L]:.3f} sentences "
        f"(chance {1 / len(classes):.2f}), temperature {T:.2f} (OOF NLL {nll_raw:.3f} -> {nll:.3f}), {time.time() - t0:.0f}s")
    return out


def main():
    ap = argparse.ArgumentParser(prog="nous_tone.readout"); ap.add_argument("model"); ap.add_argument("out")
    ap.add_argument("--attrs", nargs="*", default=["anger", "joy", "sadness", "warmth"]); ap.add_argument("--n", type=int, default=40); ap.add_argument("--gen", type=int, default=48)
    ap.add_argument("--short", action="store_true", help="also train on one-or-two-sentence replies (spoken companion lines)")
    a = ap.parse_args(); train(a.model, a.out, a.attrs, a.n, a.gen, short=a.short)


if __name__ == "__main__":
    main()
