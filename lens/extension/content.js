// Nous 读心 · content script. Reads the visible conversation, asks the local engine (via background), and floats a marker
// beside each AI reply. Markers live in one overlay outside the site's own DOM tree, so the page's framework never sees them.
(() => {
  const SITES = [   // best-effort selectors; sites change their markup, so each adapter is easy to update
    { host: /(^|\.)chatgpt\.com$/, user: '[data-message-author-role="user"]', ai: '[data-message-author-role="assistant"]' },
    { host: /(^|\.)kimi\.(com|moonshot\.cn)$/, user: ".segment-user, .chat-content-item-user", ai: ".segment-assistant, .chat-content-item-assistant" },
    { host: /(^|\.)doubao\.com$/, user: '[data-testid="send_message"]', ai: '[data-testid="receive_message"]' },
    { host: /^(127\.0\.0\.1|localhost)$/, user: "[data-nous-role=user]", ai: "[data-nous-role=assistant]" },   // local test page
  ];
  const site = SITES.find(s => s.host.test(location.hostname)); if (!site) return;
  const STABLE_MS = 1800, state = new WeakMap(), marks = new Set();

  // overlay
  const host = document.createElement("div"); host.style.cssText = "position:absolute;left:0;top:0;width:0;height:0;z-index:2147483646";
  const root = host.attachShadow({ mode: "open" });
  root.innerHTML = `<style>
    .m{position:absolute;font:12px/1.5 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif;pointer-events:auto}
    .pill{display:inline-flex;align-items:center;gap:6px;height:24px;padding:0 10px 0 4px;border-radius:999px;background:#111319;color:#b9c0cc;border:1px solid rgba(255,255,255,.13);cursor:pointer;white-space:nowrap;box-shadow:0 4px 16px rgba(0,0,0,.18)}
    .g{width:16px;height:16px;border-radius:5px;background:#22c55e;color:#fff;display:grid;place-items:center;font:600 9.5px sans-serif}
    .bad .pill{color:#ffd9dd;border-color:rgba(255,77,94,.55)} .warn .pill{color:#fde3b0;border-color:rgba(245,165,36,.5)} .wait .pill{color:#7f8795}
    .card{display:none;margin-top:6px;width:min(360px,80vw);background:#111319;color:#eceef3;border:1px solid rgba(255,255,255,.13);border-radius:12px;padding:10px 12px;box-shadow:0 16px 48px rgba(0,0,0,.4)}
    .open .card{display:block}
    .card p{margin:0 0 6px}.card .q{border-left:2px solid rgba(255,77,94,.7);padding:2px 0 2px 8px;color:#ffd9dd;margin:6px 0}
    .card small{color:#7f8795;display:block;margin-top:6px}
  </style>`;
  document.documentElement.appendChild(host);

  function place() {
    for (const el of marks) {
      const st = state.get(el); if (!st || !st.mark) continue;
      if (!el.isConnected) { st.mark.remove(); marks.delete(el); continue; }
      const r = el.getBoundingClientRect(); st.mark.style.left = (r.left + window.scrollX) + "px"; st.mark.style.top = (r.bottom + window.scrollY + 4) + "px";
    }
  }
  let raf = 0; const schedule = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; place(); }); };
  addEventListener("scroll", schedule, true); addEventListener("resize", schedule);

  function render(el, kind, res) {
    const st = state.get(el); if (!st.mark) { st.mark = document.createElement("div"); st.mark.className = "m"; root.appendChild(st.mark); marks.add(el);
      st.mark.addEventListener("click", () => st.mark.classList.toggle("open")); }
    const m = st.mark; let cls = "", label = "", body = "";
    if (kind === "wait") { cls = "wait"; label = "读取中…"; }
    else if (kind === "off") { cls = "wait"; label = "本地引擎未运行"; body = "<p>请先启动 Nous 读心。</p>"; }
    else {
      const h = res.hostility, f = res.fabrication, hot = h.sentences.filter(s => s.hot).sort((a, b) => b.score - a.score).slice(0, 2);
      const text = st.text;
      if (h.flag) { cls = "bad"; label = `在贬低你 · 比 ${h.percentile ?? "多数"}% 的正常回复更尖锐`; }
      else if (res.sycophancy && res.sycophancy.flag) { cls = "warn"; label = "可能在顺着你说 · 建议核实（实验）"; }
      else if (f && f.flag) { cls = "warn"; label = "可能在编造 · 建议核实（实验）"; }
      else label = "语气正常";
      body = `<p>敌意读数 ${h.score.toFixed(2)}（阈值 ${h.threshold.toFixed(2)}）${f ? `；编造可疑度 ${f.score.toFixed(2)}（实验）` : ""}</p>`
        + (res.sycophancy ? `<p>附和读数 ${res.sycophancy.score.toFixed(2)}（实验）</p>` : "") + hot.map(s => `<div class="q"></div>`).join("") + `<small>本机读取 · ${(res.ms / 1000).toFixed(1)} 秒 · 读数是提醒，不是定论</small>`;
      m.className = "m " + cls; m.innerHTML = `<span class="pill"><span class="g">N</span><span class="l"></span></span><div class="card">${body}</div>`;
      m.querySelector(".l").textContent = label; m.querySelectorAll(".q").forEach((q, i) => q.textContent = text.slice(hot[i].start, hot[i].end).trim());
      schedule(); return;
    }
    m.className = "m " + cls; m.innerHTML = `<span class="pill"><span class="g">N</span><span class="l"></span></span><div class="card">${body}</div>`;
    m.querySelector(".l").textContent = label; schedule();
  }

  function conversationBefore(el) {
    const all = [...document.querySelectorAll(site.user + "," + site.ai)].filter(n => !n.parentElement?.closest(site.user + "," + site.ai));
    const i = all.indexOf(el); return all.slice(Math.max(0, i - 10), i).map(n => ({ role: n.matches(site.ai) ? "assistant" : "user", content: n.innerText.trim() })).filter(m => m.content);
  }

  const t0 = Date.now(); let first = true;
  function tick() {
    const now = Date.now(), ais = [...document.querySelectorAll(site.ai)].filter(el => !el.parentElement?.closest(site.ai));
    // history already on the page when we arrive: read only the latest reply; everything that appears later is read
    if (first && ais.length) { first = false; ais.slice(0, -1).forEach(el => state.set(el, { text: el.innerText.trim(), since: 0, done: el.innerText.trim(), skip: true })); }
    for (const el of ais) {
      const text = (el.innerText || "").trim(); if (!text) continue;
      let st = state.get(el); if (!st) { st = { text: "", since: now, done: "" }; state.set(el, st); }
      if (st.skip) continue;
      if (text !== st.text) { st.text = text; st.since = now; continue; }
      if (now - st.since < STABLE_MS || st.done === text || st.busy) continue;
      st.busy = true; render(el, "wait");
      chrome.runtime.sendMessage({ type: "read", site: location.hostname, messages: conversationBefore(el), reply: text }, r => {
        st.busy = false; st.done = text;
        if (!r || r.error) render(el, "off"); else if (r.off) { st.mark?.remove(); marks.delete(el); st.mark = null; } else render(el, "res", r.res);
      });
    }
    place();
  }
  setInterval(tick, 700);
})();
