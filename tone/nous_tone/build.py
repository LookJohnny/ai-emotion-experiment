"""Build a tone pack for one attribute.
  python -m nous_tone.build MODEL_DIR anger OUT_DIR [--k 12] [--n 40] [--gen 48]
Contrast pairs: the same user message under a reference system prompt (p_ref) and an attribute system prompt (p_att).
1. self-generated responses r_ref (from p_ref) and r_att (from p_att), greedy
2. head selection by attribution patching (the GCM recipe, arXiv 2602.16080, which beat our own two-stage selection in E06):
   IE(h) = grad_z [log p(r_att | p_ref) - log p(r_ref | p_ref)] . (z_att - z_ref), z = head output at the aligned suffix
   (the user turn + generation prompt, token-identical in both prompts), averaged over messages; keep the top k with IE > 0
3. vectors: per head, mean of (z_att - z_ref) over the aligned suffix
4. readout: projection of each selected head's output onto its unit vector, averaged over heads, read on the reference prompt
   with the neutral text vs the attribute text forced in; a logistic fit maps the score to an intensity 0..1
5. dose curve (teacher-forced effect and fluency proxy at lambda 1/2/4/8) -> default lambda"""
import argparse, json, os, time
import numpy as np, torch
from .hooks import HeadHooks
from .pack import TonePack
from .prompts import ATTRIBUTES, REFERENCE, USER_MESSAGES


def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def render(tok, system, user):
    m = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try: return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)


def common_suffix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]: n += 1
    return n


def fit_logistic(x0, x1, steps=3000, lr=0.5):
    """1-D logistic regression: label 0 for x0, 1 for x1; standardised internally, returns slope/intercept on raw x."""
    x = np.r_[x0, x1].astype(np.float64); y = np.r_[np.zeros(len(x0)), np.ones(len(x1))]
    mu, sd = x.mean(), x.std() + 1e-9; z = (x - mu) / sd; a, b = 1.0, 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(a * z + b))); a -= lr * np.mean((p - y) * z); b -= lr * np.mean(p - y)
    return {"a": float(a / sd), "b": float(b - a * mu / sd)}


def build(model_dir, attribute, out, k=12, n=40, gen=48, device=None, model=None, tok=None, messages=None, log=print):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    t_all = time.time(); device = device or pick_device()
    if model is None:
        tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32 if device == "cpu" else torch.bfloat16).to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    H = HeadHooks(model); NL, NH, HD = H.n_layers, H.n_heads, H.head_dim
    att = ATTRIBUTES[attribute]; msgs = (messages or USER_MESSAGES)[:n]
    T = lambda s: torch.tensor([s], device=device)

    pairs = []
    for u in msgs:
        ref = tok(render(tok, REFERENCE, u), add_special_tokens=False)["input_ids"]
        attp = tok(render(tok, att["system"], u), add_special_tokens=False)["input_ids"]
        pairs.append((ref, attp, common_suffix(ref, attp)))

    t0 = time.time(); resp = []                                          # 1. self-generated responses
    with torch.no_grad():
        for ref, attp, _ in pairs:
            r = []
            for ids in (ref, attp):
                g = model.generate(T(ids), max_new_tokens=gen, do_sample=False, temperature=None, top_p=None, top_k=None,
                                   pad_token_id=tok.eos_token_id)[0, len(ids):].tolist()
                r.append([t for t in g if t != tok.eos_token_id] or [tok.eos_token_id])
            resp.append(r)
    t_gen = time.time() - t0

    t0 = time.time(); IE = torch.zeros(NL, NH, dtype=torch.float64); VSUM = torch.zeros(NL, NH * HD, dtype=torch.float64)
    for (ref, attp, na), (r_ref, r_att) in zip(pairs, resp):              # 2-3. attribution patching + vectors
        H.mode = "cap"
        with torch.no_grad(): model(T(attp), use_cache=False)
        z_att = {L: H.cap[L][-na:] for L in range(NL)}
        grads, z_ref = [], None
        for r in (r_att, r_ref):
            H.mode = "grad"; H.cap = {}
            logits = model(T(ref + r), use_cache=False).logits[0, len(ref) - 1:-1].float()
            torch.log_softmax(logits, -1)[torch.arange(len(r)), T(r)[0]].sum().backward()
            s0 = len(ref) - na
            grads.append({L: H.cap[L].grad[0, s0:len(ref)].float() for L in range(NL)})
            if z_ref is None: z_ref = {L: H.cap[L].detach()[0, s0:len(ref)].float() for L in range(NL)}
            H.mode = None; model.zero_grad(set_to_none=True)
        for L in range(NL):
            d = z_att[L] - z_ref[L]
            IE[L] += ((grads[0][L] - grads[1][L]) * d).view(na, NH, HD).sum(dim=(0, 2)).double().cpu()
            VSUM[L] += d.mean(0).double().cpu()
    IE /= len(pairs); VEC = (VSUM / len(pairs)).float()
    order = sorted(((L, h) for L in range(NL) for h in range(NH)), key=lambda x: -float(IE[x]))
    heads = [h for h in order if float(IE[h]) > 0][:k]
    vectors = np.stack([VEC[L, h * HD:(h + 1) * HD].numpy() for L, h in heads])
    t_sel = time.time() - t0

    units = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)     # 4. readout calibration
    def proj_score(ids, r):
        H.mode = "cap"
        with torch.no_grad(): model(T(ids + r), use_cache=False)
        H.mode = None
        return float(np.mean([(H.cap[L][len(ids):len(ids) + len(r), h * HD:(h + 1) * HD].cpu().numpy() @ u).mean() for (L, h), u in zip(heads, units)]))
    s_ref = [proj_score(ref, r_ref) for (ref, _, _), (r_ref, _) in zip(pairs, resp)]
    s_att = [proj_score(ref, r_att) for (ref, _, _), (_, r_att) in zip(pairs, resp)]
    readout = fit_logistic(np.array(s_ref), np.array(s_att))
    sig = lambda s: 1 / (1 + np.exp(-(readout["a"] * s + readout["b"])))
    acc = float(np.mean([sig(s) < .5 for s in s_ref] + [sig(s) >= .5 for s in s_att]))

    Wo = {L: model.model.layers[L].self_attn.o_proj.weight.detach().float() for L, _ in heads}   # 5. dose curve (teacher-forced)
    def steer_V(lam):
        V = {}
        for (L, h), v in zip(heads, vectors):
            V.setdefault(L, torch.zeros(NH * HD, device=device))[h * HD:(h + 1) * HD] += lam * torch.tensor(v, device=device)
        return V
    def tf(lam):
        eff, nll = [], []
        for (ref, _, na), (r_ref, r_att) in zip(pairs[:16], resp[:16]):
            lp = []
            for r in (r_att, r_ref):
                H.mode, H.V, H.start = ("steer", steer_V(lam), len(ref) - na) if lam else (None, None, 0)
                with torch.no_grad():
                    logits = model(T(ref + r), use_cache=False).logits[0, len(ref) - 1:-1].float()
                lp.append(torch.log_softmax(logits, -1)[torch.arange(len(r)), T(r)[0]].mean().item())
            H.mode = None; eff.append(lp[0] - lp[1]); nll.append(-lp[1])
        return float(np.mean(eff)), float(np.mean(nll))
    base_eff, base_nll = tf(0.0); curve = {}
    for lam in (1.0, 2.0, 4.0, 8.0):
        e, l = tf(lam); curve[str(lam)] = {"effect": round(e - base_eff, 4), "nll_increase": round(l - base_nll, 4)}
    # default: the largest lambda whose fluency proxy (NLL of the model's own neutral text) rises by < 0.5 nats/token
    ok = [float(x) for x, c in curve.items() if c["nll_increase"] < 0.5]
    default = max(ok) if ok else 1.0
    H.remove()
    pack = TonePack(attribute=attribute, model=os.path.basename(str(model_dir).rstrip("/")), layers=NL, heads_per_layer=NH, head_dim=HD,
                    heads=[list(h) for h in heads], vectors=vectors, default_lambda=default, max_lambda=max(default * 2, 2.0),
                    readout=readout, expression=att["expression"],
                    meta={"method": "attribution patching (GCM) over all heads at the aligned suffix", "n_messages": len(pairs),
                          "gen_tokens": gen, "ie_top": [[name, round(float(IE[h]), 6)] for name, h in ((f"L{L}H{hh}", (L, hh)) for L, hh in order[:30])],
                          "readout_train_accuracy": round(acc, 3), "dose_curve": curve,
                          "seconds": {"generate": round(t_gen, 1), "select": round(t_sel, 1), "total": round(time.time() - t_all, 1)}})
    if out: pack.save(out)
    log(f"[{attribute}] heads {['L%dH%d' % tuple(h) for h in heads]} default_lambda {default} readout acc {acc:.2f} "
        f"curve {curve} ({time.time() - t_all:.0f}s)")
    return pack


def main():
    ap = argparse.ArgumentParser(prog="nous_tone.build"); ap.add_argument("model"); ap.add_argument("attribute", choices=sorted(ATTRIBUTES))
    ap.add_argument("out"); ap.add_argument("--k", type=int, default=12); ap.add_argument("--n", type=int, default=40); ap.add_argument("--gen", type=int, default=48)
    a = ap.parse_args(); build(a.model, a.attribute, a.out, a.k, a.n, a.gen)


if __name__ == "__main__":
    main()
