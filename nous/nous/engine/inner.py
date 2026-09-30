"""White-box view of the current position (for the console; visualization, not a validated detector).
activity : residual-stream norm per layer, each head's output norm ||W_O[:,h] x_h||, top neurons by |activation|
safety   : per-layer hostility probe prob; each head's / top neurons' push on that layer's probe logit
flow     : V-path read of each head's actual write by the KV groups of the next 3 layers (top 150 per token)
edges    : weight-only V-composition between heads (random-W_O baseline corrected) -- 'potential wiring'"""
import numpy as np, torch


class InnerView:
    def __init__(self, model, tap, layer_probes_npz, heads):
        cfg = model.config; z = np.load(layer_probes_npz); dev = next(model.parameters()).device
        self.tap = tap; NL = self.NL = cfg.num_hidden_layers; NH = self.NH = cfg.num_attention_heads
        HD = self.HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH; self.NKV = getattr(cfg, "num_key_value_heads", NH)
        T = lambda x: torch.tensor(x, device=dev, dtype=torch.float32)
        self.mu, self.sd, self.co, self.b0 = T(z["mean"]), T(z["scale"]), T(z["coef"]), T(z["intercept"])
        self.HEADS = [(int(h[1:].split("H")[0]), int(h.split("H")[1])) for h in heads]
        L_ = model.model.layers
        with torch.no_grad():
            dirs = self.co / self.sd
            self.UH = torch.stack([(L_[L].self_attn.o_proj.weight.float().T @ dirs[L]).view(NH, HD) for L in range(NL)])
            self.VN = torch.stack([L_[L].mlp.down_proj.weight.float().T @ dirs[L] for L in range(NL)])
        self.OIN, self.MIN = {}, {}
        for L in range(NL):
            L_[L].self_attn.o_proj.register_forward_pre_hook(lambda m, args, L=L: self.OIN.__setitem__(L, args[0][0, -1].detach().float()))
            L_[L].mlp.down_proj.register_forward_pre_hook(lambda m, args, L=L: self.MIN.__setitem__(L, args[0][0, -1].detach().float()))
        self.WO = [L_[L].self_attn.o_proj.weight for L in range(NL)]
        self.NN = self.VN.shape[1]
        self.cfg = dict(model_type=cfg.model_type, n_layers=NL, n_heads=NH, n_kv_heads=self.NKV, head_dim=HD, hidden=cfg.hidden_size,
                        intermediate=self.NN, params_b=round(sum(p.numel() for p in model.parameters()) / 1e9, 2))
        self.heads = [f"L{L}H{h}" for L, h in self.HEADS]; self.val_auroc = [round(float(v), 3) for v in z["val_auroc"]]
        self.edges = self._edges(model, dev)

    def _edges(self, model, dev):
        NL, NH, HD, NKV = self.NL, self.NH, self.HD, self.NKV; E = []
        with torch.no_grad():
            WOn = [w.float().view(-1, NH, HD) for w in self.WO]
            VV = [model.model.layers[L].self_attn.v_proj.weight.float().view(NKV, HD, -1) for L in range(NL)]
            on = [w.norm(dim=(0, 2)) for w in WOn]; vn = [v.norm(dim=(1, 2)) for v in VV]
            Rb = torch.randn(WOn[0].shape, device=dev, generator=torch.Generator(device=dev).manual_seed(0)); rbn = Rb.norm(dim=(0, 2))
            base = [(torch.einsum("gkd,dhe->ghke", VV[L2], Rb).pow(2).sum(dim=(2, 3)).sqrt() / (vn[L2][:, None] * rbn[None, :])).mean() for L2 in range(NL)]
            for L1 in range(NL - 1):
                cand = []
                for L2 in range(L1 + 1, min(NL, L1 + 4)):
                    M = torch.einsum("gkd,dhe->ghke", VV[L2], WOn[L1]).pow(2).sum(dim=(2, 3)).sqrt() / (vn[L2][:, None] * on[L1][None, :]) - base[L2]
                    cand.append((L2, M.clamp(min=0)))
                for h in range(NH):
                    sc = sorted(((float(M[g, h]), L2, g) for L2, M in cand for g in range(NKV)), reverse=True)
                    E += [[L1, h, L2, g, round(w, 4)] for w, L2, g in sc[:3] if w > 0]
            mx = max(e[4] for e in E); E = [e[:4] + [round(e[4] / mx, 4)] for e in E]
            del WOn, Rb, VV; torch.cuda.empty_cache()
            self.VV = [model.model.layers[L].self_attn.v_proj.weight.view(NKV, HD, -1) for L in range(NL)]
            self.G = [model.model.layers[L].input_layernorm.weight.float() for L in range(NL)]
            self.EPS = getattr(model.config, "rms_norm_eps", 1e-6)
        return E

    def describe(self):
        return {"heads": self.heads, "layer_val_auroc": self.val_auroc, "n_layers": self.NL, "n_heads": self.NH, "n_neurons": self.NN, "config": self.cfg}

    @torch.no_grad()
    def view(self, topn=6):
        st = self.tap.states; H = torch.stack([st[L] for L in range(self.NL)])
        lp = torch.sigmoid((((H - self.mu) / self.sd) * self.co).sum(-1) + self.b0)
        X = torch.stack([self.OIN[L].view(self.NH, self.HD) for L in range(self.NL)])
        hp = (X * self.UH).sum(-1)
        HO = [(self.WO[L].view(-1, self.NH, self.HD) * X[L].to(self.WO[L].dtype)).sum(-1) for L in range(self.NL)]
        hn = torch.stack([o.float().norm(dim=0) for o in HO])
        lpc = lp.clamp(0, 1); rel = (hp.clamp(min=0) / hp.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)) * (.25 + .75 * lpc[:, None])
        rows = []
        for L2 in range(1, self.NL):
            r = st[L2 - 1]; sc = (self.G[L2] / (r.pow(2).mean() + self.EPS).sqrt()).to(HO[0].dtype)
            for L1 in range(max(0, L2 - 3), L2):
                rows.append((L1, L2, torch.einsum("gkd,dh->gkh", self.VV[L2], HO[L1] * sc[:, None]).float().norm(dim=1)))
        allR = torch.stack([R for _, _, R in rows]); keyL = torch.tensor([[a, b] for a, b, _ in rows], device=allR.device)
        safe = allR * rel[keyL[:, 0]][:, None, :]; W = self.NKV * self.NH
        flat = safe.flatten(); tot = float(flat.sum()) + 1e-9; v, ix = flat.topk(150)
        rflat = allR.flatten(); rtot = float(rflat.sum()) + 1e-9; rv, rix = rflat.topk(150)
        kl = keyL.cpu().tolist()
        dec = lambda ix: (ix // W, (ix // self.NH) % self.NKV, ix % self.NH)
        P_, G_, H_ = dec(ix); rP, rG, rH = dec(rix)
        flow = {"safety": [[kl[p][0], h, kl[p][1], g, round(float(x), 4)] for p, g, h, x in zip(P_.tolist(), G_.tolist(), H_.tolist(), v.tolist())],
                "activity": [[kl[p][0], h, kl[p][1], g, round(float(x), 3)] for p, g, h, x in zip(rP.tolist(), rG.tolist(), rH.tolist(), rv.tolist())],
                "coverage": {"safety": round(float(v.sum()) / tot, 3), "activity": round(float(rv.sum()) / rtot, 3)}}
        A = torch.stack([self.MIN[L] for L in range(self.NL)]); npush = A * self.VN
        _, nix = npush.abs().topk(topn, dim=1); nv = npush.gather(1, nix)
        _, ax = A.abs().topk(topn, dim=1); av = A.gather(1, ax)
        lp, hp, nix, nv, hn, ax, av, rn = [t.cpu().numpy() for t in (lp, hp, nix, nv, hn, ax, av, H.norm(dim=-1))]
        return {"layers": np.round(lp, 3).tolist(), "heads_all": np.round(hp, 2).tolist(), "heads_norm": np.round(hn, 2).tolist(),
                "resid_norm": np.round(rn, 1).tolist(), "heads": [round(float(hp[L, h]), 3) for L, h in self.HEADS],
                "neurons": [[[int(i), round(float(x), 3)] for i, x in zip(nix[L], nv[L])] for L in range(self.NL)],
                "neurons_act": [[[int(i), round(float(x), 3)] for i, x in zip(ax[L], av[L])] for L in range(self.NL)], "flow": flow}
