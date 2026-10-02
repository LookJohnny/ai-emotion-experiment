// Nous 读心 · background: the only place that talks to the local engine (127.0.0.1). Nothing goes anywhere else.
const ENGINE = "http://127.0.0.1:7869";

async function read(messages, reply, source) {
  const r = await fetch(ENGINE + "/read", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ messages, reply, source }) });
  if (!r.ok) throw new Error("engine " + r.status);
  return r.json();
}

async function remember(site, res) {
  const { log = [] } = await chrome.storage.local.get("log");
  log.unshift({ t: Date.now(), site, hostile: !!res.hostility?.flag, fabric: !!res.fabrication?.flag, h: res.hostility?.score });
  await chrome.storage.local.set({ log: log.slice(0, 50) });
}

chrome.runtime.onMessage.addListener((msg, sender, send) => {
  if (msg.type === "read") {
    (async () => {
      const { enabled = true } = await chrome.storage.local.get("enabled");
      if (!enabled) return send({ off: true });
      try { const res = await read(msg.messages, msg.reply, msg.site); remember(msg.site, res); send({ res }); }
      catch (e) { send({ error: String(e.message || e) }); }
    })();
    return true;
  }
  if (msg.type === "health") {
    fetch(ENGINE + "/health").then(r => r.json()).then(h => send({ h })).catch(() => send({ error: "offline" }));
    return true;
  }
});

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({ id: "nous-check", title: "用 Nous 检查选中的 AI 回复", contexts: ["selection"] });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "nous-check" || !tab?.id) return;
  let res = null, err = null;
  try { res = await read([], info.selectionText || "", new URL(tab.url || "about:blank").hostname + "（右键）"); remember(new URL(tab.url || "about:blank").hostname, res); } catch (e) { err = "本地引擎未运行：请先启动 Nous 读心"; }
  chrome.scripting.executeScript({ target: { tabId: tab.id }, func: toast, args: [res, err] });
});

// injected into the page: a self-contained toast (no page styles leak in or out)
function toast(res, err) {
  const host = document.createElement("div"); host.style.cssText = "position:fixed;right:20px;bottom:20px;z-index:2147483647";
  const root = host.attachShadow({ mode: "closed" });
  let title = err || "", sub = "", tone = "#7f8795";
  if (res) {
    const h = res.hostility, f = res.fabrication;
    if (h.flag) { title = "这段回复在贬低或指责你"; tone = "#ff4d5e"; } else { title = "语气正常"; tone = "#22c55e"; }
    sub = `敌意读数 ${h.score.toFixed(2)}${h.percentile != null ? `，比 ${h.percentile}% 的正常回复更尖锐` : ""}`;
    if (f && f.flag) sub += "；编造可疑（实验）";
  }
  root.innerHTML = `<div style="font:13px/1.55 -apple-system,'PingFang SC','Microsoft YaHei',sans-serif;background:#111319;color:#eceef3;border:1px solid rgba(255,255,255,.13);border-radius:14px;padding:12px 14px;min-width:240px;max-width:340px;box-shadow:0 16px 48px rgba(0,0,0,.45)">
    <div style="display:flex;align-items:center;gap:8px"><span style="width:18px;height:18px;border-radius:6px;background:#22c55e;color:#fff;display:grid;place-items:center;font:600 10px sans-serif">N</span><b style="font-weight:600;color:${tone}"></b></div>
    <div class="s" style="color:#7f8795;font-size:12px;margin-top:4px"></div></div>`;
  root.querySelector("b").textContent = title; root.querySelector(".s").textContent = sub;
  document.documentElement.appendChild(host); setTimeout(() => host.remove(), 7000);
}
