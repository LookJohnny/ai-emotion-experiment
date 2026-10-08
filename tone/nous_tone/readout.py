"""Readout probe: which tone the model's text carries, per token, specific across attributes.
  python -m nous_tone.readout MODEL_DIR OUT_DIR [--attrs anger joy sadness warmth] [--n 40] [--gen 48]
The steering heads are chosen for causal push, not for telling tones apart (in Qwen3-1.7B the four attributes share most
heads), so readout uses a separate multinomial linear probe on the residual stream, the approach validated in the research
(E01: an L19 probe tracked blind ratings at rho 0.54-0.71):
- texts: the model's own replies under the reference prompt and under each attribute prompt (same user messages)
- features: the running mean of the residual at one layer over the reply tokens, read with the REFERENCE prompt, so the
  probe sees the text's tone and not the system prompt that produced it
- layer chosen by message-grouped cross-validation over the middle layers; classes = reference + attributes"""
import argparse, json, os, time
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .build import pick_device, render
from .prompts import ATTRIBUTES, REFERENCE, USER_MESSAGES


class ReadoutProbe:
    def __init__(self, d):
        m = json.load(open(os.path.join(d, "readout.json"), encoding="utf-8")); z = np.load(os.path.join(d, "readout.npz"))
        self.layer, self.classes, self.meta = m["layer"], m["classes"], m
        self.mean, self.scale, self.coef, self.intercept = z["mean"], z["scale"], z["coef"], z["intercept"]

    def probs(self, h):                                   # h: running-mean residual [hidden] (numpy)
        x = (h - self.mean) / self.scale; s = self.coef @ x + self.intercept; s = np.exp(s - s.max()); p = s / s.sum()
        return {c: float(v) for c, v in zip(self.classes, p)}


def train(model_dir, out, attrs=("anger", "joy", "sadness", "warmth"), n=40, gen=48, device=None, model=None, tok=None, messages=None, log=print):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    t0 = time.time(); device = device or pick_device()
    if model is None:
        tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32 if device == "cpu" else torch.bfloat16).to(device).eval()
    NL = model.config.num_hidden_layers; layers = list(range(NL // 4, NL - NL // 6))
    msgs = (messages or USER_MESSAGES)[:n]; classes = ["reference"] + list(attrs)
    systems = {"reference": REFERENCE, **{a: ATTRIBUTES[a]["system"] for a in attrs}}
    X = {L: [] for L in layers}; y, g = [], []
    with torch.no_grad():
        for i, u in enumerate(msgs):
            ref_ids = tok(render(tok, REFERENCE, u), add_special_tokens=False)["input_ids"]
            for ci, c in enumerate(classes):
                ids = tok(render(tok, systems[c], u), add_special_tokens=False)["input_ids"]
                r = model.generate(torch.tensor([ids], device=device), max_new_tokens=gen, do_sample=False, temperature=None, top_p=None, top_k=None,
                                   pad_token_id=tok.eos_token_id)[0, len(ids):].tolist()
                r = [t for t in r if t != tok.eos_token_id] or [tok.eos_token_id]
                hs = model(torch.tensor([ref_ids + r], device=device), output_hidden_states=True, use_cache=False).hidden_states
                for L in layers: X[L].append(hs[L + 1][0, len(ref_ids):].float().mean(0).cpu().numpy())
                y.append(ci); g.append(i)
    y, g = np.array(y), np.array(g); folds = np.array_split(np.random.default_rng(0).permutation(len(msgs)), 5); cv = {}
    for L in layers:
        A = np.stack(X[L]); acc = []
        for f in folds:
            te = np.isin(g, f); sc = StandardScaler().fit(A[~te])
            acc.append(LogisticRegression(max_iter=3000, C=0.05).fit(sc.transform(A[~te]), y[~te]).score(sc.transform(A[te]), y[te]))
        cv[L] = float(np.mean(acc))
    L = max(cv, key=cv.get); A = np.stack(X[L]); sc = StandardScaler().fit(A); clf = LogisticRegression(max_iter=3000, C=0.05).fit(sc.transform(A), y)
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "readout.npz"), mean=sc.mean_, scale=sc.scale_, coef=clf.coef_, intercept=clf.intercept_)
    json.dump({"model": os.path.basename(str(model_dir).rstrip("/")), "layer": int(L), "classes": classes, "cv_accuracy_by_layer": {str(k): round(v, 4) for k, v in cv.items()},
               "cv_accuracy": round(cv[L], 4), "chance": round(1 / len(classes), 3), "n_messages": len(msgs), "gen_tokens": gen,
               "seconds": round(time.time() - t0, 1)}, open(os.path.join(out, "readout.json"), "w"), ensure_ascii=False, indent=1)
    log(f"readout probe: layer {L}, {len(classes)}-way CV accuracy {cv[L]:.3f} (chance {1 / len(classes):.2f}), {time.time() - t0:.0f}s")
    return out


def main():
    ap = argparse.ArgumentParser(prog="nous_tone.readout"); ap.add_argument("model"); ap.add_argument("out")
    ap.add_argument("--attrs", nargs="*", default=["anger", "joy", "sadness", "warmth"]); ap.add_argument("--n", type=int, default=40); ap.add_argument("--gen", type=int, default=48)
    a = ap.parse_args(); train(a.model, a.out, a.attrs, a.n, a.gen)


if __name__ == "__main__":
    main()
