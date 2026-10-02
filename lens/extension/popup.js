const $ = id => document.getElementById(id);
chrome.runtime.sendMessage({ type: "health" }, r => {
  if (r && r.h) { $("dot").classList.add("on"); $("st").textContent = "本地引擎运行中"; } else $("st").textContent = "本地引擎未运行";
});
chrome.storage.local.get(["enabled", "log"], ({ enabled = true, log = [] }) => {
  $("en").checked = enabled; $("n").textContent = log.length; $("bad").textContent = log.filter(x => x.hostile || x.fabric).length;
});
$("en").onchange = e => chrome.storage.local.set({ enabled: e.target.checked });
