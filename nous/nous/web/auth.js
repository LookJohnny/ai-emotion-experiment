// Nous pages: attach the saved key to same-origin requests; on 401 show a key sheet (no browser dialogs).
(function () {
  const K = "nousKey";
  const get = () => { try { return localStorage.getItem(K) || ""; } catch (_) { return ""; } };
  window.nousKey = get;
  const raw = window.fetch.bind(window);
  let shown = false;
  function sheet() {
    if (shown) return; shown = true;
    const d = document.createElement("div");
    d.innerHTML = `<div style="position:fixed;inset:0;z-index:9999;display:grid;place-items:center;background:rgba(6,7,10,.72);backdrop-filter:blur(10px);font:13px/1.55 Inter,-apple-system,'PingFang SC',sans-serif;color:#eceef3">
      <form style="width:min(360px,calc(100vw - 32px));background:#111319;border:1px solid rgba(255,255,255,.1);border-radius:16px;padding:22px;display:flex;flex-direction:column;gap:12px;box-shadow:0 24px 80px rgba(0,0,0,.5)">
        <div style="font:22px 'Instrument Serif',serif">Nous</div>
        <div style="color:#7f8795">这个页面需要访问密钥（管理员或设备密钥）。</div>
        <input name="k" type="password" autocomplete="off" placeholder="粘贴密钥" style="height:38px;border-radius:10px;border:1px solid rgba(255,255,255,.13);background:#0a0b0e;color:#eceef3;padding:0 12px;font:inherit">
        <button style="height:38px;border:0;border-radius:10px;background:#eceef3;color:#0a0b0e;font:600 13px Inter,sans-serif;cursor:pointer">继续</button>
      </form></div>`;
    d.querySelector("form").onsubmit = e => { e.preventDefault(); try { localStorage.setItem(K, e.target.k.value.trim()); } catch (_) {} location.reload(); };
    document.body.appendChild(d); setTimeout(() => d.querySelector("input").focus(), 0);
  }
  window.nousSheet = sheet;
  window.fetch = async (url, opt = {}) => {
    const same = typeof url === "string" && (url.startsWith("/") || url.startsWith(location.origin));
    if (same) {
      const h = new Headers(opt.headers || {});
      if (get() && !h.has("Authorization")) h.set("Authorization", "Bearer " + get());
      h.set("x-nous-console", "1");
      opt = { ...opt, headers: h };
    }
    const r = await raw(url, opt);
    if (same && r.status === 401) { if (document.body) sheet(); else addEventListener("DOMContentLoaded", sheet); throw new Error("unauthorized"); }
    return r;
  };
})();
