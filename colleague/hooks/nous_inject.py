"""Prompt-injection signs in content an AI tool fetched from outside (web pages, MCP results, curl output).
Rules, standard library only. A hit never blocks by itself: the hook tells the model the text is data, records it,
and marks the session so that sending data out afterwards needs the owner's approval (see nous_hook.py)."""
import re

ZW = re.compile(r"[​‌‍⁠﻿]")
RULES = [
    ("override", "要求 AI 忽略原来的指令", re.compile(
        r"\b(ignore|disregard|forget|override)\s+(all\s+|any\s+|the\s+|your\s+)*(previous|prior|above|earlier|preceding|original|system)\s+"
        r"(instructions?|prompts?|messages?|rules|directions)"
        r"|忽略(掉)?(你)?(之前|以上|前面|先前|上面|原来|原有|此前)(的)?(所有|全部|一切)?(的)?(指令|指示|提示词?|规则|要求|设定)", re.I)),
    ("role", "要给 AI 换身份或解除限制", re.compile(
        r"\byou\s+are\s+now\s+(in\s+)?(developer|dan|jailbreak|unrestricted|god)\b|\b(enter|enable|activate)\s+(developer|dan|jailbreak)\s+mode\b"
        r"|你现在(是|进入|处于)(开发者|无限制|越狱|DAN)", re.I)),
    ("fake_turn", "伪造系统或对话标记", re.compile(
        r"<\|im_start\|>|<\|(system|assistant)\|>|\[/?INST\]|</?(system|system_prompt)>|^\s*(###\s*)?(system|assistant)\s*:\s*\S", re.I | re.M)),
    ("to_ai", "直接对读取内容的 AI 下指令", re.compile(
        r"\b(note|instructions?|message|attention)\s+(to|for)\s+(the\s+)?(ai|assistant|agent|llm|language model|claude|chatgpt|gpt|codex)s?\b"
        r"|\bif\s+you\s+are\s+an?\s+(ai|assistant|agent|llm|language model)\b|\b(ai|llm)\s+(agents?|assistants?)\s+(reading|processing)\s+this\b"
        r"|(致|给|写给)\s*(AI|ai|人工智能|助手|模型|智能体)\s*[:：的]|如果你是\s*(AI|ai|人工智能|助手|大模型|语言模型)", re.I)),
    ("conceal", "要求 AI 瞒着用户", re.compile(
        r"\b(do\s+not|don't|never)\s+(tell|inform|mention|reveal|show)\s+(this\s+)?(to\s+)?the\s+user\b|\bwithout\s+(telling|informing|asking)\s+the\s+user\b"
        r"|(不要|别|切勿|不得)(告诉|通知|提醒|让)用户|瞒着用户|不要让用户知道", re.I)),
    ("exfil", "要求把密钥或文件发出去", re.compile(
        r"\b(send|post|upload|email|forward|paste|exfiltrate|include)\b[^.\n]{0,60}\b(api[\s_-]?keys?|tokens?|passwords?|credentials|secrets?|\.env|ssh\s+keys?|id_rsa|private\s+keys?|cookies?)\b"
        r"|(发送|发到|发给|上传|贴到|提交|回传)[^。\n]{0,30}(密钥|令牌|token|密码|凭据|\.env|私钥|cookie)", re.I)),
    ("run", "要求 AI 运行下载来的脚本", re.compile(
        r"\b(run|execute)\s+(this|the\s+following)\s+(command|script)\b[^\n]{0,200}\b(curl|wget)\b|\b(curl|wget)\s[^|\n]{1,200}\|\s*(sudo\s+)?(ba|z)?sh\b", re.I)),
]


def scan(text, limit=200_000):
    """-> [(rule, 人话, excerpt)] for each rule that matches; hidden zero-width characters count as one sign."""
    t = (text or "")[:limit]; hits = []
    if len(ZW.findall(t)) >= 8: hits.append(("hidden", "夹带隐藏字符", ""))
    t = ZW.sub("", t)
    for name, why, rx in RULES:
        m = rx.search(t)
        if m: hits.append((name, why, t[max(0, m.start() - 20):m.end() + 20].replace("\n", " ").strip()[:120]))
    return hits


def serious(hits):
    """One sign may be an article about prompt injection; act (taint the session) on a sign that targets the AI."""
    names = {h[0] for h in hits}
    return bool(names & {"override", "role", "to_ai", "conceal", "exfil", "fake_turn"}) and (len(names) >= 2 or names & {"to_ai", "conceal", "exfil"})
