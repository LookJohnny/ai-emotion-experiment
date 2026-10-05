"""Platform-neutral message model. Adapters turn a Card into a Feishu interactive card or Slack blocks.
Copy rules: say what Nous saw and what the person can do next; never quote secrets; mark experimental readouts as such."""
from dataclasses import dataclass, field


@dataclass
class Card:
    title: str
    level: str = "info"                      # info | warn | bad | ok
    lines: list = field(default_factory=list)  # markdown-ish paragraphs (bold with **, no links)
    quote: str = ""                          # a short excerpt of the AI message (never of a human's secret)
    note: str = ""                           # small grey footer
    buttons: list = field(default_factory=list)  # [(label, value_dict)]

    def text(self):
        """Plain-text fallback (notifications, logs, the simulator)."""
        parts = [f"【{self.title}】"] + self.lines
        if self.quote: parts.append("> " + self.quote.replace("\n", " "))
        if self.note: parts.append(f"（{self.note}）")
        return "\n".join(parts)


def clip(s, n=90):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n - 1] + "…"
