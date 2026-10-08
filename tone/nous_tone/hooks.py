"""Hooks on every layer's attention output projection (o_proj input = concatenated head outputs).
modes: None (pass), "cap" (store the activation), "grad" (keep it in the graph and retain its grad),
       "steer" (add per-layer vectors from position `start` on; during cached decoding every new token is steered).
Readout taps capture the current step's head outputs for the layers a pack reads."""
import torch


def attn_layers(model):
    return model.model.layers


class HeadHooks:
    def __init__(self, model):
        self.model, self.mode, self.V, self.start, self.cap, self.read_layers, self.tap = model, None, None, 0, {}, set(), {}
        cfg = model.config
        self.n_layers, self.n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.handles = [L.self_attn.o_proj.register_forward_pre_hook(self._hook(i)) for i, L in enumerate(attn_layers(model))]
        self.handles.append(model.model.embed_tokens.register_forward_hook(
            lambda m, a, out: out.requires_grad_(True) if self.mode == "grad" else None))

    def _hook(self, L):
        def h(mod, args):
            x = args[0]
            if L in self.read_layers: self.tap[L] = x[0, -1].detach().float()
            if self.mode == "cap": self.cap[L] = x[0].detach().float(); return None
            if self.mode == "grad": x.retain_grad(); self.cap[L] = x; return None
            if self.mode == "steer" and self.V and L in self.V:
                x = x.clone()
                if x.shape[1] > 1: x[0, self.start:] += self.V[L].to(x.dtype)     # prefill: from the steering start on
                else: x[0, -1] += self.V[L].to(x.dtype)                           # cached decoding: the new token
                return (x,)
            return None
        return h

    def head_slice(self, h):
        return slice(h * self.head_dim, (h + 1) * self.head_dim)

    def remove(self):
        for hd in self.handles: hd.remove()
