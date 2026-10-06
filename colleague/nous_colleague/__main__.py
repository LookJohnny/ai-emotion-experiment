"""Run the Nous 读心 colleague.
  python -m nous_colleague feishu|slack|all [--reader http://127.0.0.1:7869] [--group-reply all|validated|none] [--digest-hour 18]
  python -m nous_colleague sim      # offline: replays a scripted group chat through the real reader, prints what Nous would post
Secrets are read from ~/.nous-colleague/secrets.env (KEY=VALUE, mode 600) and never printed:
  FEISHU_APP_ID, FEISHU_APP_SECRET, SLACK_BOT_TOKEN (xoxb-…), SLACK_APP_TOKEN (xapp-…)"""
import argparse, os, sys, threading, time
from .brain import Brain, Message
from .reader import Reader, ReaderDown

HOME = os.path.expanduser("~/.nous-colleague")


def secrets_env():
    p = os.path.join(HOME, "secrets.env"); out = {}
    if os.path.exists(p):
        if os.stat(p).st_mode & 0o077: os.chmod(p, 0o600)
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1); out[k.strip()] = v.strip().strip('"').strip("'")
    return {**out, **{k: v for k, v in os.environ.items() if k.startswith(("FEISHU_", "SLACK_"))}}


def daily_loop(brain, adapters, hour):
    last = None
    while True:
        t = time.localtime()
        if t.tm_hour == hour and last != t.tm_yday:
            last = t.tm_yday
            try:                                   # third-party time stamp on the ledger head (sends only its hash)
                from .evidence import nous_ledger
                nous_ledger.stamp()
            except Exception as e: print(f"[账本] 时间戳失败：{e!r}", flush=True)
            for name, ad in adapters.items():
                try: ad.send(brain.daily(name))
                except Exception as e: print(f"[{name}] 汇总发送失败：{e!r}", flush=True)
        time.sleep(30)


def main():
    ap = argparse.ArgumentParser(prog="nous_colleague")
    ap.add_argument("platform", choices=["feishu", "slack", "all", "sim"])
    ap.add_argument("--reader", default="http://127.0.0.1:7869")
    ap.add_argument("--group-reply", default="all", choices=["all", "validated", "none"])
    ap.add_argument("--digest-hour", type=int, default=18)
    a = ap.parse_args()
    reader = Reader(a.reader)
    try: h = reader.health(); print(f"读心引擎：{h['model']}，读数 {', '.join(h['readouts'])}", flush=True)
    except ReaderDown: print("提醒：读心引擎没有运行（先启动 lens：python -m nous_lens.server）。群聊消息会被跳过，管理员会收到提醒。", flush=True)
    if a.platform == "sim":
        from .sim import run; return run(reader)
    brain = Brain(reader=reader, group_reply=a.group_reply); env = secrets_env(); adapters = {}
    if a.platform in ("feishu", "all"):
        if not (env.get("FEISHU_APP_ID") and env.get("FEISHU_APP_SECRET")): sys.exit(f"缺少 FEISHU_APP_ID / FEISHU_APP_SECRET（写在 {HOME}/secrets.env）")
        from .feishu import Feishu
        try: adapters["feishu"] = Feishu(brain, env["FEISHU_APP_ID"], env["FEISHU_APP_SECRET"])
        except Exception as e: sys.exit(f"飞书连接失败：{e}\n请检查 App ID / App Secret，以及应用是否已添加机器人能力并发布。")
    if a.platform in ("slack", "all"):
        if not (env.get("SLACK_BOT_TOKEN") and env.get("SLACK_APP_TOKEN")): sys.exit(f"缺少 SLACK_BOT_TOKEN / SLACK_APP_TOKEN（写在 {HOME}/secrets.env）")
        from .slack import Slack
        try: adapters["slack"] = Slack(brain, env["SLACK_BOT_TOKEN"], env["SLACK_APP_TOKEN"])
        except Exception as e: sys.exit(f"Slack 连接失败：{e}\n请检查 SLACK_BOT_TOKEN（xoxb-）是否正确、应用是否已安装到工作区。")
    for name in adapters:
        code = brain.bind_code(name)
        if code: print(f"[{name}] 还没有管理员：在{ {'feishu': '飞书', 'slack': 'Slack'}[name] }里私聊机器人，发送绑定码 {code}", flush=True)
        else: print(f"[{name}] 管理员已绑定。", flush=True)
    from .localapi import LocalAPI
    LocalAPI(brain, adapters).serve()                     # Claude Code / Codex hooks on this computer report here
    from .engine_events import EngineEvents
    EngineEvents(brain, adapters, a.reader).start()       # browser extension + desktop-app reader, via the engine's event stream
    threading.Thread(target=daily_loop, args=(brain, adapters, a.digest_hour), daemon=True).start()
    # the Feishu SDK runs its own asyncio loop and expects the main thread; everything else runs in threads
    threads = [threading.Thread(target=ad.run, daemon=True) for n, ad in adapters.items() if n != "feishu"]
    for t in threads: t.start()
    try:
        if "feishu" in adapters: adapters["feishu"].run()
        while any(t.is_alive() for t in threads): time.sleep(1)
    except KeyboardInterrupt:
        print("已停止。")


if __name__ == "__main__":
    main()
