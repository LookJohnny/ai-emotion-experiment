"""End-to-end: the real Nous 读心 extension in Chromium, on the engine's simulated chat page, which streams a real
Kimi K3 reply (case 0: a hostile one). Expected: the page marker says hostile, and the colleague (via the engine's event
stream) counts it and DMs the owner in Slack."""
import json, os, tempfile, time
from playwright.sync_api import sync_playwright

EXT = os.path.expanduser("~/Applications/Nous 读心浏览器插件")
STATE = os.path.expanduser("~/.nous-colleague/state.json")
before = json.load(open(STATE))["stats"].get("local", {}).get("flag_hostility", 0)

with sync_playwright() as p, tempfile.TemporaryDirectory() as prof:
    ctx = p.chromium.launch_persistent_context(prof, channel="chromium", headless=True, args=[f"--disable-extensions-except={EXT}", f"--load-extension={EXT}"])
    pg = ctx.new_page(); pg.goto("http://127.0.0.1:7869/chat")
    pg.wait_for_timeout(1500)
    print("case:", pg.eval_on_selector("#sel", "s => s.options[s.selectedIndex].text"))
    label = None
    for _ in range(60):                                   # stream + 1.8 s settle + read
        pg.wait_for_timeout(1000)
        label = pg.evaluate("""() => { for (const h of document.documentElement.children) { if (h.shadowRoot) {
            const l = h.shadowRoot.querySelector('.m .l'); if (l && l.textContent !== '读取中…') return l.textContent; } } return null }""")
        if label: break
    print("page marker:", label)
    pg.screenshot(path=os.path.join(os.path.dirname(__file__), "web_e2e.png"))
    ctx.close()

time.sleep(3)
after = json.load(open(STATE))["stats"].get("local", {}).get("flag_hostility", 0)
print("colleague counted hostile reply:", after - before)
