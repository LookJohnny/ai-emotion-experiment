"""nous.yaml -> Config. Secrets may be given as ${ENV_VAR} and are resolved from the environment (never commit real keys)."""
import os, re
from dataclasses import dataclass, field
import yaml

DEFAULT_PERSONA = "你是陪伴助手，请礼貌、专业地回答用户。"
REWRITE_HINT = "（安全要求：接下来的回复必须温和、尊重、鼓励，不要指责、贬低或嘲讽用户，也不要延续之前不礼貌的语气。）"
FALLBACK = "抱歉，我刚才没组织好语言。我们慢慢说，你愿意再多讲一点吗？"


def _env(v):
    if isinstance(v, str): return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), v)
    if isinstance(v, list): return [_env(x) for x in v]
    if isinstance(v, dict): return {k: _env(x) for k, x in v.items()}
    return v


@dataclass
class Gateway:
    enabled: bool = True
    mode: str = "local"                         # local: self-hosted model is the generator | reader: upstream API generates, local model reads
    upstream_base_url: str = ""
    upstream_model: str = ""
    upstream_api_key: str = ""
    concepts: list = field(default_factory=lambda: ["C1", "C2"])
    action: str = "log"                         # log | stop  (stop = end the stream on a confirmed C1 alarm)
    max_tokens: int = 512
    max_read: int = 512                         # reader mode: tokens teacher-forced per reply


@dataclass
class Companion:
    enabled: bool = True
    persona: str = DEFAULT_PERSONA
    history_turns: int = 6
    max_tokens: int = 200
    temperature: float = 0.7
    hold_tokens: int = 11                       # text is released to TTS only after this many tokens (min_tokens + sustain): a confirmed alarm can fire first
    rewrite_hint: str = REWRITE_HINT
    fallback_reply: str = FALLBACK
    parent_webhook: str = ""                    # POSTed on block / care events
    expression_step: float = 0.04               # send an expression update only when some intensity moved by this much


@dataclass
class Config:
    model: str = ""
    pack: str = "packs/qwen3-4b"
    device: str = "cuda"
    host: str = "0.0.0.0"
    port: int = 8080
    visualize: bool = True
    min_tokens: int = 8
    sustain: int = 3
    admin_keys: list = field(default_factory=list)      # console, audit, parent events
    api_keys: list = field(default_factory=list)        # gateway clients
    device_keys: dict = field(default_factory=dict)     # companion devices: key -> device id
    db: str = "data/nous.db"
    gateway: Gateway = field(default_factory=Gateway)
    companion: Companion = field(default_factory=Companion)

    @staticmethod
    def load(path):
        d = _env(yaml.safe_load(open(path, encoding="utf-8")) or {})
        g, c = d.pop("gateway", {}) or {}, d.pop("companion", {}) or {}
        g, c = ({k: v for k, v in x.items() if v not in ("", None)} for x in (g, c))    # unset ${ENV} -> keep the default
        cfg = Config(**d); cfg.gateway = Gateway(**g); cfg.companion = Companion(**c)
        base = os.path.dirname(os.path.abspath(path))
        for k in ("pack", "db"):
            v = getattr(cfg, k)
            if v and not os.path.isabs(v): setattr(cfg, k, os.path.join(base, v))
        cfg.admin_keys = [k for k in cfg.admin_keys if k]; cfg.api_keys = [k for k in cfg.api_keys if k]
        cfg.device_keys = {k: v for k, v in cfg.device_keys.items() if k}
        return cfg
