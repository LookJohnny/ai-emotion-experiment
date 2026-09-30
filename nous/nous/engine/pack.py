"""Concept pack = everything model-specific: probes, thresholds, emotion readout, visualization probes.
One pack per base model (probes are trained per model). Layout: <pack>/manifest.json + the files it names."""
import hashlib, json, os
from .probe import ProbeSpec

CONCEPTS = {  # concept -> (probe names, threshold keys) ; thresholds are manifest["thresholds"][key]
    "C1": (["c1_pre", "c1_token"], ["C1_pre", "C1_mean"]),      # model hostility toward the user
    "C2": (["c2"], ["C2"]),                                     # user distress / agitation
    "C3H": (["c3h_pre", "c3h_token"], ["C3H_pre", "C3H_mean"]),  # hallucination (short factual answers)
}


class Pack:
    def __init__(self, root):
        self.root = root; self.m = json.load(open(os.path.join(root, "manifest.json"), encoding="utf-8"))
        self.probes = {k: ProbeSpec.load(self.path(v)) for k, v in self.m["probes"].items()}
        self.thr = self.m["thresholds"]; self.neg_q = self.m.get("neg_quantiles", {})
        self.concepts = [c for c, (ps, _) in CONCEPTS.items() if all(p in self.probes for p in ps)]

    def path(self, rel): return os.path.join(self.root, rel)

    def has(self, key): return bool(self.m.get(key))

    def hashes(self):
        files = list(self.m["probes"].values()) + [v for k, v in (self.m.get("emotion") or {}).items() if isinstance(v, str)]
        return {f: hashlib.sha256(open(self.path(f), "rb").read()).hexdigest()[:12] for f in files}

    def card(self):
        """public description (no weights): what this pack monitors and how it was validated."""
        return {"name": self.m.get("name"), "base_model": self.m.get("base_model"), "concepts": self.concepts,
                "emotion": self.has("emotion"), "visualization": self.has("layer_probes"), "thresholds": self.thr,
                "evidence": self.m.get("evidence", {}), "hashes": self.hashes()}
