"""Send this computer's ledger bundle to the Nous admin in a Slack DM, once a day.
  python -m nous_colleague.ship            # time-stamp the head (best effort), export, upload to the admin
Uses only Slack's Web API with SLACK_BOT_TOKEN, no Socket Mode connection: Slack hands each event to one random
connection of an app, so only the admin's computer runs the full colleague; every other computer just ships.
The admin is SLACK_ADMIN_USER (from ~/.nous-colleague/secrets.env) or, on the admin's own computer, the bound owner.
The Slack app needs the files:write scope. A bundle holds no conversation or command text and no HMAC key."""
import glob, json, os, sys, time
from .evidence import nous_ledger as L

OUTBOX = os.path.join(os.path.dirname(L.PATH), "outbox")
KEEP = 7


class ShipError(Exception):
    pass


def make_bundle():
    """Stamp (if the network allows), export to the outbox, keep the last KEEP bundles. -> (path, manifest, stamp or None)"""
    try: st = L.stamp()
    except Exception: st = None                       # offline: the bundle still ships; the next stamp will cover these records
    os.makedirs(OUTBOX, mode=0o700, exist_ok=True)
    m = L.machine(); path = os.path.join(OUTBOX, time.strftime("%Y%m%d-%H%M%S") + f"-{m['id'][:8]}.tar.gz")
    man = L.export(path); os.chmod(path, 0o600)
    for old in sorted(glob.glob(os.path.join(OUTBOX, "*.tar.gz")))[:-KEEP]: os.remove(old)
    return path, man, st


def comment(man, st):
    h = man["head"]; who = man["label"] or f"电脑 {man['machine'][:6]}"
    lines = [f"*{who}* 的 Nous 证据包：{man['records']} 条记录，账本指纹 `{h['hash'][:16]}`（第 {h['seq']} 条）。",
             f"第三方时间戳：{st['time']}，覆盖到第 {st['seq']} 条。" if st else "这次没能签上第三方时间戳（可能是网络问题），下次会补上。",
             "包里没有任何对话或命令原文。汇总多台电脑：`python -m nous_colleague.team 报告.html 包… --state team.json`"]
    if not man["chain_ok_at_export"]: lines.insert(1, ":red_circle: 导出时账本的哈希链已经断裂，请核实这台电脑。")
    return "\n".join(lines)


def upload(client, user_id, path, text):
    """Slack DM with the bundle attached. Raises ShipError with a plain explanation."""
    from slack_sdk.errors import SlackApiError
    try:
        ch = client.conversations_open(users=user_id)["channel"]["id"]
        client.files_upload_v2(channel=ch, file=path, filename=os.path.basename(path), title="Nous 证据包 " + os.path.basename(path)[:15],
                               initial_comment=text)
    except SlackApiError as e:
        err = e.response.get("error", "")
        if err == "missing_scope":
            raise ShipError("Slack 应用缺少 files:write 权限：在 api.slack.com/apps → OAuth & Permissions 里添加 files:write，然后重新安装到工作区。") from e
        if err in ("user_not_found", "channel_not_found", "cannot_dm_bot"):
            raise ShipError(f"找不到管理员（{err}）：检查 SLACK_ADMIN_USER 是否是管理员的 Slack 成员 ID（U 开头）。") from e
        raise ShipError(f"Slack 返回错误：{err}") from e


def ship(client, user_id):
    path, man, st = make_bundle()
    upload(client, user_id, path, comment(man, st))
    return path, man, st


def admin_user(env):
    if env.get("SLACK_ADMIN_USER"): return env["SLACK_ADMIN_USER"]
    try: return json.load(open(os.path.expanduser("~/.nous-colleague/state.json")))["owners"].get("slack")
    except Exception: return None


def main():
    from .__main__ import secrets_env
    env = secrets_env()
    if not env.get("SLACK_BOT_TOKEN"): sys.exit("缺少 SLACK_BOT_TOKEN（写在 ~/.nous-colleague/secrets.env）")
    uid = admin_user(env)
    if not uid: sys.exit("不知道管理员是谁：在 ~/.nous-colleague/secrets.env 里写 SLACK_ADMIN_USER=U…（管理员的 Slack 成员 ID）")
    from slack_sdk import WebClient
    try: path, man, st = ship(WebClient(token=env["SLACK_BOT_TOKEN"]), uid)
    except ShipError as e: sys.exit(str(e))
    print(f"已发给管理员：{os.path.basename(path)}，{man['records']} 条记录" + ("，已签时间戳" if st else "，时间戳未签上"))


if __name__ == "__main__":
    main()
