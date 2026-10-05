"""Browser test of lens/extension/guard.js on a ChatGPT-like page (contenteditable #prompt-textarea, send button,
Enter-to-send handled by the page itself). chrome.runtime is stubbed to record notices."""
import os
from playwright.sync_api import sync_playwright

GUARD = open("/Users/lovelyjoy/Documents/AI-Emotion-Experiment/lens/extension/guard.js", encoding="utf-8").read().replace('mode: "closed"', 'mode: "open"')  # open only for the test
KEY = "sk-proj-" + "simTEST1234567890abcdEFGHijkl"
PAGE = """<!doctype html><meta charset=utf-8><body style="font:14px sans-serif">
<div id=log></div>
<form id=f onsubmit="return false"><div id="prompt-textarea" contenteditable="true" style="border:1px solid #999;min-height:40px"></div>
<button type=button data-testid="send-button" id=send>Send</button></form>
<script>
const ed=document.getElementById('prompt-textarea'), log=document.getElementById('log');
function send(){ const t=ed.innerText.trim(); if(!t) return; const p=document.createElement('p'); p.className='sent'; p.textContent=t; log.appendChild(p); ed.innerText=''; }
document.getElementById('send').addEventListener('click', send);
ed.addEventListener('keydown', e=>{ if(e.key==='Enter'&&!e.shiftKey){ e.preventDefault(); send(); } });
</script>"""
STUB = "window.__notices=[]; window.chrome={runtime:{sendMessage:(m)=>window.__notices.push(m.kind)}};"

with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page()
    pg.set_content(PAGE); pg.evaluate(STUB); pg.add_script_tag(content=GUARD)
    sent = lambda: pg.eval_on_selector_all(".sent", "els=>els.map(e=>e.textContent)")
    ed = pg.locator("#prompt-textarea")

    ed.click(); ed.type("帮我修一下登录页的 bug"); pg.keyboard.press("Enter")
    print("1 ordinary message sent:", sent() == ["帮我修一下登录页的 bug"])

    ed.click(); ed.type(f"key 是 {KEY} 帮我调接口"); pg.keyboard.press("Enter")
    bar = pg.evaluate("() => { const h=[...document.documentElement.children].find(x=>x.shadowRoot===null && x.style.position==='fixed'); return !!h }")
    print("2 secret via Enter held:", len(sent()) == 1, "| notices:", pg.evaluate("window.__notices"))

    pg.click("#send")
    print("3 secret via button held:", len(sent()) == 1)
    pg.screenshot(path=os.path.join(os.path.dirname(__file__), "guard_bar.png"))

    pg.locator("button[data-a=send]").click()
    print("4 send anyway works:", len(sent()) == 2, "| notices:", pg.evaluate("window.__notices"))
    pg.evaluate("document.querySelector('#prompt-textarea').focus()")
    pg.evaluate("""k => { const dt=new DataTransfer(); dt.setData('text/plain', 'token: '+k);
        document.querySelector('#prompt-textarea').dispatchEvent(new ClipboardEvent('paste',{clipboardData:dt,bubbles:true})) }""", KEY)
    print("5 paste warning:", pg.evaluate("window.__notices")[-1])
    b.close()
