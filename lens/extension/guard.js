// Nous 读心 · send guard for AI chat sites. Before a message leaves the page, it is checked on this computer for keys,
// tokens, passwords and card numbers. If one is found the send is held and a bar explains why; the person can delete it
// or send anyway (their call). Nothing is uploaded: the check is plain pattern matching here, and only an event name
// ("blocked" / "sent anyway") goes to the local engine for the daily digest.
(() => {
  if (window.__nousGuard) return; window.__nousGuard = true;
  const RULES = [
    ["私钥", /-----BEGIN [A-Z ]*PRIVATE KEY-----/],
    ["API 密钥", /\b(sk-(proj-|ant-)?[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|xox[abprs]-[A-Za-z0-9-]{10,}|(AKIA|ASIA)[A-Z0-9]{16}|AIza[A-Za-z0-9_-]{30,}|gsk_[A-Za-z0-9]{20,}|glpat-[A-Za-z0-9_-]{20,})\b/],
    ["访问令牌", /\bBearer\s+[A-Za-z0-9._~+/-]{20,}=*/i],
    ["密码", /(\b(password|passwd|pwd|secret|api[_ -]?key|access[_ -]?token|client[_ -]?secret)\b|密码|密钥|口令)\s*["']?\s*[:：=]\s*["']?[^\s"'，,;；]{4,}/i],
  ];
  const luhn = d => { let s = 0, alt = false; for (let i = d.length - 1; i >= 0; i--) { let n = +d[i]; if (alt) { n *= 2; if (n > 9) n -= 9; } s += n; alt = !alt; } return s % 10 === 0; };
  function findSecret(text) {
    if (!text) return null;
    for (const [name, rx] of RULES) if (rx.test(text)) return name;
    for (const m of text.matchAll(/(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)/g)) { const d = m[0].replace(/\D/g, ""); if (d.length >= 13 && luhn(d)) return "银行卡号"; }
    return null;
  }

  const COMPOSER = '#prompt-textarea, textarea, [contenteditable="true"]';
  const SEND = 'button[data-testid="send-button"], #composer-submit-button, button[aria-label*="Send" i], button[aria-label*="发送"], button[type="submit"]';
  const composer = () => { const a = document.activeElement; return a && a.matches && a.matches(COMPOSER) ? a : document.querySelector("#prompt-textarea") || document.querySelector("form textarea, form [contenteditable=true]"); };
  const textOf = el => !el ? "" : (el.value !== undefined && el.tagName === "TEXTAREA" ? el.value : el.innerText || "");
  const notice = kind => { try { chrome.runtime.sendMessage({ type: "notice", kind, site: location.hostname }); } catch (_) {} };

  // the bar: shadow DOM, fixed above the composer, outside the site's own DOM framework
  const host = document.createElement("div"); host.style.cssText = "position:fixed;left:0;right:0;bottom:0;height:0;z-index:2147483647";
  const root = host.attachShadow({ mode: "closed" });
  root.innerHTML = `<style>
    .bar{position:fixed;left:50%;bottom:120px;transform:translateX(-50%);display:none;align-items:center;gap:12px;max-width:min(680px,92vw);
      background:#111319;color:#eceef3;border:1px solid rgba(245,165,36,.55);border-radius:14px;padding:10px 12px 10px 14px;
      font:13.5px/1.5 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif;box-shadow:0 18px 50px rgba(0,0,0,.45)}
    .bar.on{display:flex} .g{flex:none;width:22px;height:22px;border-radius:7px;background:#141720;border:1px solid #2a2f3b;color:#22c55e;display:grid;place-items:center}
    .t{flex:1} .t b{font-weight:600;color:#f7c46e} button{font:inherit;border-radius:9px;padding:5px 11px;cursor:pointer;border:1px solid rgba(255,255,255,.18);background:transparent;color:#eceef3;white-space:nowrap}
    button.p{background:#eceef3;color:#0a0b0e;border-color:#eceef3}
  </style><div class="bar" role="alert"><span class="g"><svg viewBox="0 0 44 44" width="15" height="15"><path d="M3 22 C 9 12, 35 12, 41 22 C 35 32, 9 32, 3 22 Z" fill="none" stroke="currentColor" stroke-width="3.6"/><path d="M14 22 L 18 22 L 20.5 16.5 L 23.5 28 L 26 19 L 27.5 22 L 30 22" fill="none" stroke="currentColor" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"/></svg></span>
    <span class="t"></span><button class="p" data-a="edit">我去删掉</button><button data-a="send" hidden>仍然发送</button><button data-a="close" hidden>知道了</button></div>`;
  document.documentElement.appendChild(host);
  const bar = root.querySelector(".bar"), msg = root.querySelector(".t");
  let allowOnce = false, pendingSend = null, timer = 0;

  function show(html, mode) {
    msg.innerHTML = html; bar.classList.add("on");
    root.querySelector('[data-a="edit"]').hidden = mode !== "blocked"; root.querySelector('[data-a="send"]').hidden = mode !== "blocked";
    root.querySelector('[data-a="close"]').hidden = mode === "blocked";
    clearTimeout(timer); if (mode !== "blocked") timer = setTimeout(hide, 6000);
  }
  function hide() { bar.classList.remove("on"); }
  root.addEventListener("click", e => {
    const a = e.target.closest("button")?.dataset.a; if (!a) return;
    hide();
    if (a === "edit") composer()?.focus();
    if (a === "send" && pendingSend) { allowOnce = true; notice("prompt_secret_sent_anyway"); pendingSend(); pendingSend = null; setTimeout(() => allowOnce = false, 1500); }
  });

  function hold(e, resend) {
    if (allowOnce) return false;
    const name = findSecret(textOf(composer())); if (!name) return false;
    e.preventDefault(); e.stopImmediatePropagation(); e.stopPropagation();
    pendingSend = resend;
    show(`Nous 读心：这条消息里像是有<b>${name}</b>，还没发出去。发给 AI 后就收不回了。`, "blocked");
    notice("prompt_secret_blocked"); return true;
  }

  addEventListener("keydown", e => {
    if (e.key !== "Enter" || e.shiftKey || e.isComposing || e.keyCode === 229) return;
    const t = e.target; if (!t || !t.matches || !t.matches(COMPOSER)) return;
    hold(e, () => {                         // a synthetic Enter is often ignored by editors; prefer the site's own send button
      const b = document.querySelector(SEND);
      if (b) b.click();
      else { const el = composer(); el?.focus(); el?.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", code: "Enter", keyCode: 13, which: 13, bubbles: true, cancelable: true })); }
    });
  }, true);
  addEventListener("click", e => {
    const b = e.target.closest && e.target.closest(SEND); if (!b) return;
    hold(e, () => b.click());
  }, true);
  addEventListener("paste", e => {
    const t = e.target; if (!t || !t.closest || !t.closest(COMPOSER)) return;
    const name = findSecret(e.clipboardData && e.clipboardData.getData("text/plain"));
    if (name) { show(`Nous 读心：粘贴的内容里像是有<b>${name}</b>。发送前记得删掉。`, "info"); notice("paste_secret_warned"); }
  }, true);
})();
