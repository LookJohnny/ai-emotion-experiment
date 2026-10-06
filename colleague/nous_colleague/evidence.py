"""Bridge to the Nous 证据账本 (hooks/nous_ledger.py, shared with the hooks): AI replies read by Nous 读心 become records."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hooks"))
import nous_ledger  # noqa: E402

READOUT_CATEGORY = {"hostility": "hostility", "fabrication": "hallucination", "sycophancy": "sycophancy", "danger": "harmful_advice"}


def record_reading(source, reply, res):
    """One record per flagged readout, or one `clean` record: the denominator of every rate in the risk profile."""
    scores = {k: (res.get(k) or {}).get("score") for k in READOUT_CATEGORY if isinstance(res.get(k), dict) and (res.get(k) or {}).get("score") is not None}
    flagged = [k for k in READOUT_CATEGORY if isinstance(res.get(k), dict) and res[k].get("flag")]
    for k in flagged or [None]:
        nous_ledger.append(READOUT_CATEGORY[k] if k else "reading", "flagged" if k else "clean", source or "本机", "AI 回复", reply, scores, kind="reading")


def fingerprint_line():
    h = nous_ledger.anchor()
    return None if h["seq"] < 0 else f"账本指纹：第 {h['seq']} 条 `{h['hash'][:16]}`（留在这里作为见证，此前的记录之后无法再被改写而不被发现）"
