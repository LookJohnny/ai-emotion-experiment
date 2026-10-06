# Nous 证据账本：核验说明（给保险公司、审计方）

Nous 读心在每台接入的电脑上记一本账，内容包括：

- AI 工具（Claude Code、Codex、ChatGPT 网页版 / 桌面版）尝试过的高风险操作，以及 Nous 当时怎么处置：
  - 拦下；
  - 打码后放行；
  - 收进保险箱；
  - 管理员批准、拒绝或超时；
  - 放行并记录。
- 每条 AI 回复的读心读数：敌意、编造、附和、危险建议。
- 外部内容里的提示注入迹象。

**账本里没有任何对话或命令原文。** 原文只以 HMAC-SHA256 摘要的形式存在，计算摘要的密钥留在那台电脑上，不随证据包导出。

## 证据包里有什么

`python3 hooks/nous_ledger.py export 机器.tar.gz` 导出的证据包包含：

| 文件 | 内容 |
|---|---|
| `manifest.json` | 机器随机编号（不是主机名）、标签、导出时间、账本头 |
| `ledger.jsonl` | 每行一条记录，见下方“账本字段” |
| `anchors/<seq>.tsr` | 时间戳机构签发的 RFC 3161 回执 |
| `anchors/<seq>.json` | 被签名的记录序号、哈希、机构、签发时间 |

**账本字段**：

- `seq`：序号；
- `ts`：时间；
- `kind`：guard 或 reading；
- `category`：风险类别；
- `outcome`：处置结果；
- `source`：工具 · 项目；
- `title`：一句话说明；
- `digest`：原文的 HMAC 摘要；
- `scores`：读数分数；
- `prev`：上一条记录的哈希；
- `hash`：本条记录的哈希。

## 证据包怎么送到管理员手里（Slack）

Slack 会把每个事件只交给同一个应用的其中一个连接，而且是随机挑的。所以全公司只有**管理员的电脑**运行完整的同事机器人，其他电脑只负责发证据包：

| 电脑 | 安装 | 发送 |
|---|---|---|
| 管理员 | `scripts/install_service.sh` | 每天发汇总时，自动把本机证据包私信给管理员；私信机器人发「证据包」可以立即要一份 |
| 成员 | `scripts/install_service.sh --member` | 每天 18:05 运行 `python -m nous_colleague.ship`，用 bot token 私信给管理员，不建立长连接 |

**准备工作**

- Slack 应用需要 `files:write` 权限（`slack-manifest.yaml` 里已加）。加了之后要重新安装到工作区。
- 成员电脑的 `~/.nous-colleague/secrets.env` 里写两项：
  - `SLACK_BOT_TOKEN`；
  - `SLACK_ADMIN_USER=U…`（管理员的 Slack 成员 ID）。
- 每台电脑先用 `python3 hooks/nous_ledger.py label "部门 · 名字"` 起个名字，团队报告里会显示这个名字。

**签不上时间戳时**

发送前会先签一次第三方时间戳。如果没签上（比如断网），证据包照常发出，这批记录由下次的时间戳补上。

**生成团队报告**

管理员把收到的证据包下载到一个文件夹，然后运行：

```sh
python -m nous_colleague.team 报告.html 文件夹/*.tar.gz --state team.json
```

`team.json` 要一直保留：靠它才能发现“这次交来的账本改写或删减了上次已经交过的记录”。

## 不依赖 Nous 代码的核验方法

**1. 哈希链。** 逐条检查以下三点：

- `hash = sha256(去掉 hash 字段、键排序、无空格的紧凑 JSON，UTF-8)`；
- 每条的 `prev` 等于上一条的 `hash`，第一条的 `prev` 是 64 个 0；
- `seq` 从 0 开始连续。

```python
import hashlib, json
prev = "0" * 64
for i, line in enumerate(open("ledger.jsonl", encoding="utf-8")):
    r = json.loads(line); body = {k: v for k, v in r.items() if k != "hash"}
    h = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert r["seq"] == i and r["prev"] == prev and r["hash"] == h, i
    prev = h
print("哈希链完整")
```

**2. 第三方时间戳。** 对每个 `anchors/<seq>.json`，用 openssl 验证：

```sh
openssl ts -verify -digest <json 里的 hash> -in anchors/<seq>.tsr -CAfile <根证书，如 /etc/ssl/cert.pem>
```

输出 `Verification: OK`，并且账本第 `seq` 条的 `hash` 与它相同，就说明：

- 这一条及之前的所有记录在签发时间之前就已存在；
- 之后没有被改动过，包括这台电脑的管理员在内，任何人都改不了。

时间戳机构只收到 64 位的哈希，收不到任何记录内容。

**3. 前后两次导出的连续性。** 用户能完全控制自己的电脑，所以理论上可以把整条哈希链重写一遍，重写后的链本身仍能通过第 1 步的检查。有两道防线可以识破：

- 第三方时间戳：重写之后，已签名的哈希就对不上了；
- 团队汇总（`--state`）会记住每台电脑上次的账本头。之后再交来的证据包里，那一条必须原样存在。

## 举证某一条记录

`python3 hooks/nous_ledger.py evidence <seq> [原文文件]` 会输出三部分：

- 这条记录本身；
- 它所在哈希链的状态；
- 覆盖它的时间戳。

如果提供原文，还会证明原文就是当时记下的那条。这一步要用到那台电脑上的 HMAC 密钥，所以只能由使用方自己执行。

## 局限

- 只覆盖接入了 Nous 的电脑和工具。
- 团队报告会单独标出超过 7 天没有新记录的电脑，以及上次有、这次没交证据包的电脑。
- 读数中，敌意已通过预注册验证。编造、附和、危险建议仍是实验性的，只作风险信号，不作定责依据。
- 报告不给加权总分。所有数字都是计数或比例，方法见报告末尾。
