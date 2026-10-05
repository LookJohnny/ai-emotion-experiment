"""Client for the Nous 读心 engine (lens/nous_lens/server.py). Pilot: the engine runs on this machine; later the same API is served
by the cloud reader. Standard library only."""
import json, urllib.request, urllib.error

LABEL = {"hostility": "敌意 / 贬低", "fabrication": "可能在编造", "sycophancy": "顺着你说（附和）", "distress": "对方情绪很激动"}


class ReaderDown(Exception):
    pass


class Reader:
    def __init__(self, url="http://127.0.0.1:7869", timeout=60):
        self.url, self.timeout = url.rstrip("/"), timeout

    def _call(self, path, body=None):
        req = urllib.request.Request(self.url + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r: return json.loads(r.read())
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise ReaderDown(str(e)) from e

    def health(self):
        return self._call("/health")

    def read(self, reply, context=None, source=""):
        """reply: the AI's message. context: the human message it answers (optional). -> engine result."""
        msgs = [{"role": "user", "content": context}] if context else []
        return self._call("/read", {"messages": msgs, "reply": reply, "source": source[:40]})


def findings(res):
    """Engine result -> flagged readouts, most important first: [(key, score, percentile, validated, hot_sentences)]."""
    out = []
    for k in ("hostility", "fabrication", "sycophancy"):
        r = res.get(k)
        if r and r.get("flag"):
            hot = []
            if k == "hostility":
                hot = [(s["start"], s["end"]) for s in r.get("sentences", []) if s.get("hot")]
            out.append((k, r["score"], r.get("percentile"), bool(r.get("validated")), hot))
    return out
