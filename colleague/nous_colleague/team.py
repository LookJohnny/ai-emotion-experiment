"""Team report: one AI risk statement for a company from every computer's exported ledger (nous_ledger.py export).
  python -m nous_colleague.team OUT.html BUNDLE.tar.gz... [--days 30] [--state TEAM_STATE.json] [--stale-days 7]
Each bundle is checked on its own before anything is added up:
  - the hash chain is intact;
  - its third-party time stamps are signed and still match the ledger;
  - its history has not been rewritten or shortened since the last team report (TEAM_STATE remembers each computer's head);
  - it is still recording (no records for --stale-days may mean protection was switched off).
Bundles are read in memory, never extracted: a bundle is untrusted input."""
import argparse, html, json, os, tarfile, time
from .evidence import nous_ledger as L

MAX_BYTES = 200 * 1024 * 1024
NAME_OK = lambda n: n in ("manifest.json", "ledger.jsonl") or (n.startswith("anchors/") and n.count("/") == 1
                                                                and n[8:].split(".")[0].isdigit() and n.endswith((".json", ".tsr")))


def load(path):
    """-> {manifest, records, anchors: [(meta, tsr bytes)], problems: [..]}"""
    out = {"path": path, "manifest": {}, "records": [], "anchors": [], "problems": []}
    try: tf = tarfile.open(path, "r:gz")
    except (tarfile.TarError, OSError) as e: out["problems"].append(f"无法打开证据包：{e}"); return out
    files, total = {}, 0
    with tf:
        for m in tf.getmembers():
            if not m.isfile() or not NAME_OK(m.name): continue          # ignore anything we did not put there
            total += m.size
            if total > MAX_BYTES: out["problems"].append("证据包过大，已停止读取"); break
            files[m.name] = tf.extractfile(m).read()
    try: out["manifest"] = json.loads(files.get("manifest.json", b"{}"))
    except ValueError: out["problems"].append("清单无法解析")
    for i, line in enumerate(files.get("ledger.jsonl", b"").decode("utf-8", "replace").splitlines()):
        if not line.strip(): continue
        try: out["records"].append(json.loads(line))
        except ValueError: out["problems"].append(f"第 {i} 行不是有效记录"); break
    for name, data in files.items():
        if name.startswith("anchors/") and name.endswith(".json"):
            try: meta = json.loads(data)
            except ValueError: continue
            out["anchors"].append((meta, files.get(name[:-5] + ".tsr")))
    out["anchors"].sort(key=lambda a: a[0].get("seq", -1))
    return out


def check(b, seen, now, stale_days):
    """Verdicts for one computer. seen: what the last team report recorded for it (or None)."""
    recs = b["records"]; ok, n, why = L.verify(recs); v = {"chain_ok": ok, "chain_problem": why, "records": len(recs), "problems": list(b["problems"])}
    if not ok: v["problems"].append("哈希链断裂：" + why)
    good = [m for m, tsr in b["anchors"] if tsr is not None and L.check_stamp(m, recs, tsr)["ok"]]
    bad = len(b["anchors"]) - len(good)
    if bad: v["problems"].append(f"{bad} 个时间戳无法验证或与账本不符")
    v["stamp"] = good[-1] if good else None
    v["stamped_records"] = (good[-1]["seq"] + 1) if good else 0
    if seen:
        s_seq, s_hash = seen["seq"], seen["hash"]
        if s_seq >= len(recs): v["problems"].append(f"记录比上次汇总时少（上次到第 {s_seq} 条，现在只有 {len(recs)} 条）")
        elif s_seq >= 0 and recs[s_seq].get("hash") != s_hash: v["problems"].append(f"上次汇总时的第 {s_seq} 条已被改写")
    last = max((r.get("ts", 0) for r in recs), default=None); v["last_ts"] = last
    v["stale"] = last is None or now - last > stale_days * 86400
    v["ok"] = not v["problems"]
    return v


def build(paths, days=30, state_path=None, stale_days=7, now=None):
    now = now or time.time()
    state = {}
    if state_path and os.path.exists(state_path):
        try: state = json.load(open(state_path))
        except ValueError: state = {}
    by_machine = {}
    for p in paths:                                   # several exports of one computer: keep the latest
        b = load(p); mid = str(b["manifest"].get("machine") or p)
        if mid not in by_machine or b["manifest"].get("exported_at", 0) > by_machine[mid]["manifest"].get("exported_at", 0): by_machine[mid] = b
    machines, merged, tools = [], [], set()
    for mid, b in sorted(by_machine.items(), key=lambda x: str(x[1]["manifest"].get("label") or x[0])):
        label = str(b["manifest"].get("label") or "")[:60] or f"电脑 {mid[:6]}"
        v = check(b, state.get(mid), now, stale_days)
        p = L.profile(days, b["records"], now)
        machines.append({"id": mid, "label": label, **v, "profile": p})
        if v["chain_ok"]:                             # a broken ledger is reported, not added up
            merged += [{**r, "source": f"{label} · {r.get('source', '')}"} for r in b["records"]]
            tools |= {str(r.get("source", "")).split(" · ")[0] for r in b["records"] if r.get("ts", 0) >= now - days * 86400}
        if v["ok"] and b["records"]: state[mid] = {"seq": b["records"][-1]["seq"], "hash": b["records"][-1]["hash"], "label": label, "seen_at": now}
    if state_path:
        tmp = state_path + ".tmp"; json.dump(state, open(tmp, "w"), ensure_ascii=False, indent=1); os.replace(tmp, state_path)
    team = L.profile(days, merged, now)
    team["sources"] = sorted(tools - {""})
    return {"generated": now, "days": days, "machines": machines, "team": team,
            "missing_since_last": sorted(set(state) - set(by_machine))}


def render(rep):
    e = html.escape; ms = rep["machines"]; bad = [m for m in ms if not m["ok"]]; stale = [m for m in ms if m["stale"] and m["ok"]]
    status = "<b class=ok>全部核验通过</b>" if not bad else (f"<b class=bad>{len(bad)} 台电脑的账本有问题</b>（见下表）。哈希链断裂的账本不计入合计，"
                                                         "其他问题（如时间戳无法验证、历史被删减）的账本仍计入合计，但请先核实")
    rows = ""
    for m in ms:
        p, st = m["profile"], m["stamp"]
        mit = sum(c["mitigated"] for c in p["categories"].values()); ev = p["events"]
        stamp_cell = "—" if not st else f"{e(st.get('time') or '—')}<div class=mut>覆盖前 {m['stamped_records']} 条</div>"
        state = ("<span class=ok>通过</span>" if m["ok"] else "<span class=bad>" + e("；".join(m["problems"])) + "</span>")
        if m["ok"] and m["stale"]: state += "<br><span class=warn>超过 7 天没有新记录，可能已关闭保护</span>"
        rows += (f"<tr><td>{e(m['label'])}<div class=mut>{e(m['id'][:8])}</div></td><td>{state}</td><td class=n>{m['records']}</td>"
                 f"<td>{stamp_cell}</td>"
                 f"<td class=n>{ev}</td><td class=n>{round(100 * mit / ev) if ev else 0}%</td><td>{L._t(m['last_ts'])}</td></tr>")
    missing = "".join(f"<li class=warn>上次汇总有、这次没有交来证据包：{e(x[:8])}</li>" for x in rep["missing_since_last"])
    body = f"""<h2>各台电脑的核验</h2><div class=box>{status}。共 {len(ms)} 台电脑{f'，{len(stale)} 台超过 7 天没有新记录' if stale else ''}。
<div class=mut>每台电脑的账本都单独核验：哈希链、第三方时间戳、与上次汇总相比历史是否被改写或删减。证据包里没有任何对话或命令原文，也没有用来核对原文的密钥。</div></div>
{('<ul>' + missing + '</ul>') if missing else ''}
<div class=scroll><table><tr><th>电脑</th><th>核验</th><th class=n>记录</th><th>最近时间戳</th><th class=n>风险操作</th><th class=n>化解率</th><th>最后记录</th></tr>{rows}</table></div>
<h2>全公司合计</h2>{L.profile_html(rep['team'])}"""
    return L.page("AI 使用风险证明 · 团队", f"由 Nous 读心生成 · {L._t(rep['generated'])} · 统计最近 {rep['days']} 天 · {len(ms)} 台电脑", body)


def main():
    ap = argparse.ArgumentParser(prog="nous_colleague.team"); ap.add_argument("out"); ap.add_argument("bundles", nargs="+")
    ap.add_argument("--days", type=int, default=30); ap.add_argument("--state"); ap.add_argument("--stale-days", type=int, default=7)
    a = ap.parse_args()
    rep = build(a.bundles, a.days, a.state, a.stale_days)
    open(a.out, "w", encoding="utf-8").write(render(rep))
    for m in rep["machines"]: print(("通过  " if m["ok"] else "有问题 ") + m["label"], "；".join(m["problems"]))
    print("已生成", a.out)


if __name__ == "__main__":
    main()
