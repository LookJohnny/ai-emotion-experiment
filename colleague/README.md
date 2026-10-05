# Nous 读心 · AI 安全同事（飞书 / Slack 试点）

Nous 读心以机器人同事的身份加入飞书或 Slack 工作区，不需要员工安装任何东西。

- **在群里**：读 AI 机器人的每一条回复。发现它在贬低人、可能在编造、或者一味附和提问的人，就在那条消息下面标注一张卡片；严重的同时私聊管理员。没问题时不说话。
- **有人把密钥贴进群里**：只私聊发消息的人，提醒撤回并更换。提醒里不会重复密钥本身。
- **私聊它**：转发或粘贴任何 AI 的回复（包括 ChatGPT、Kimi 等 App 里的），它读一遍再回复你。
- **每天 18:00**：私聊管理员一份汇总，包括读了多少条、标注了什么、同事反馈的误报。

读取由 Nous 读心引擎完成（`../lens`，读模型内部状态，不是关键词）。密钥检测复用 `nous-safety-core` 的规则。

## 试点怎么跑

试点阶段，所有东西都跑在一台电脑上：飞书用长连接，Slack 用 Socket Mode，都不需要公网服务器。

```bash
# 1. 读心引擎（另开一个终端，保持运行）
cd lens && ~/.nous-lens/venv/bin/python -m nous_lens.server

# 2. 同事服务
cd colleague
~/.nous-colleague/venv/bin/python -m nous_colleague feishu     # 或 slack / all
```

第一次启动会在终端打印一个 6 位**绑定码**。在飞书或 Slack 里私聊机器人发送这个码，你就成为管理员。重要提醒和每日汇总会发给管理员，「暂停」「继续」也只有管理员能用。

不连任何平台的离线演练：`python -m nous_colleague sim`。它会用 Kimi K3 的真实回复模拟一个群聊，并打印 Nous 会发出的每一条消息。

## 密钥放在哪里

写在 `~/.nous-colleague/secrets.env`（权限 600），不进仓库，程序也不会打印：

```
FEISHU_APP_ID=cli_xxx
FEISHU_APP_SECRET=xxx
SLACK_BOT_TOKEN=xoxb-xxx
SLACK_APP_TOKEN=xapp-xxx
```

## 飞书：创建应用（约 10 分钟，需要企业管理员审核）

1. 打开 [飞书开放平台](https://open.feishu.cn/app)，选「创建企业自建应用」，名称填「Nous 读心」，图标用 `assets/avatar-1024.png`。
2. 「添加应用能力」里添加「机器人」。
3. 「权限管理」里开通以下权限：

   | 权限 | 用途 |
   |---|---|
   | `im:message.group_bot_msg:readonly` 获取群组中其他机器人发送的消息 | 读 AI 机器人的回复（核心） |
   | `im:message.group_msg:readonly` 获取群聊中所有的用户聊天消息 | AI 回复的上下文，以及密钥提醒 |
   | `im:message.p2p_msg:readonly` 读取用户发给机器人的单聊消息 | 私聊检查、管理员命令 |
   | `im:message:send_as_bot` 以应用的身份发消息 | 标注卡片和私聊 |
   | `im:chat:readonly` 获取群组信息 | 在提醒里显示群名 |

4. 「事件与回调」→「事件配置」：订阅方式选**使用长连接接收事件**，添加事件 `im.message.receive_v1`。
   「回调配置」：同样选长连接，添加 `card.action.trigger`，用于卡片上的「误报 / 有用」按钮。

   > 长连接要求程序先运行起来，后台才能保存。所以先把 App ID 和 Secret 写进 `secrets.env`，启动 `python -m nous_colleague feishu`，再回到后台保存。
5. 「版本管理与发布」：创建版本，可用范围选试点成员，然后提交发布，等企业管理员审核通过。
6. 把机器人拉进有 AI 机器人的测试群：群设置 →「群机器人」→ 添加。

## Slack：创建应用（约 5 分钟）

1. 打开 [api.slack.com/apps](https://api.slack.com/apps)，选「Create New App」→「From an app manifest」，选好工作区，粘贴 `slack-manifest.yaml` 的内容。
2. 「Basic Information」→「App-Level Tokens」：生成一个带 `connections:write` 的 token，这就是 `SLACK_APP_TOKEN`（`xapp-` 开头）。
3. 「Install App」：安装到工作区，拿到 `SLACK_BOT_TOKEN`（`xoxb-` 开头）。图标同样用 `assets/avatar-1024.png`。
4. 在要试点的频道里输入 `/invite @Nous 读心`。

## 能看到什么，看不到什么

- 只能读**它所在的群**里的消息。员工和 AI 的私聊、其他群、员工电脑上的其他 AI，它都看不到。
- 群里的标注属于**事后提醒**：AI 的回复已经发出去了，Nous 不能撤回，也不能阻止别人看到。
- 编造和附和两项读数是**实验性的**，会有误报。卡片上的「误报」按钮会计入每日汇总，用来衡量试点效果。只想让群里出现敌意标注的话，用 `--group-reply validated`；不想在群里出现任何标注、只私聊管理员，用 `--group-reply none`。
- 消息正文只在内存里保留最近 40 条，用来找上下文。状态文件 `~/.nous-colleague/state.json` 只存计数、管理员 ID 和开关。
- 试点阶段，消息会发到运行 Nous 的这台电脑上读取。正式版会换成云端读心服务；企业也可以选择私有部署。

## 测试

```bash
~/.nous-colleague/venv/bin/python -m unittest discover -s tests
```

19 项测试，覆盖：标注与私聊管理员、安静时不说话、私聊提醒限频、密钥提醒不复述密钥、手机号不算告警、私聊检查的问答解析、绑定码与管理员权限、引擎离线只提醒一次、状态文件不含正文、飞书和 Slack 的消息解析与卡片渲染，以及 Claude Code hook 的各项拦截决定。

## 保护本机的 Claude Code，开机自动运行

```bash
bash colleague/scripts/install_service.sh          # 安装：读心引擎 + 同事服务开机自启，Claude Code hooks 指向安装副本
bash colleague/scripts/install_service.sh --off    # 停止并移除后台服务
~/.nous-colleague/venv/bin/python colleague/nous_colleague/install.py claude --off   # 关闭 Claude Code 保护
```

- 在 Claude Code 里发的消息带密钥：发给模型之前拦下。
- AI 要把密钥发到外部：直接拒绝。
- AI 要读密钥文件、强制推送、大范围删除、下载即执行、上传数据、sudo、读钥匙串：私聊管理员审批，2 分钟没人确认就拒绝；Nous 连不上时退回 Claude Code 自己的本机确认。
- AI 回答完：读心在后台读一遍，有问题私聊管理员。

代码会复制到 `~/.nous-colleague/app` 再运行，因为后台进程默认没有权限读 `~/Documents`。改完代码要重新运行安装脚本。日志在 `~/.nous-colleague/logs`。读心引擎常驻时约占 5 GB 内存。
