"""Residual-stream capture on a HF decoder model: after any forward (prefill or one decode step),
states[L] holds the decoder-layer-L output of the LAST position; full[L] holds all positions of the last prefill."""
import torch


class ResidualTap:
    def __init__(self, model, layers):
        self.states, self.full, self.handles = {}, {}, []
        for L in sorted(set(layers)):
            self.handles.append(model.model.layers[L].register_forward_hook(self._mk(L)))

    def _mk(self, L):
        def h(mod, args, out):
            x = out[0] if isinstance(out, tuple) else out
            self.states[L] = x[0, -1].detach().float()
            if x.shape[1] > 1: self.full[L] = x[0].detach().float()
        return h

    def remove(self):
        for h in self.handles: h.remove()
