# Nous API

认证：`Authorization: Bearer <key>`（也接受 `x-api-key` 头；WebSocket 用 `?key=`）。
密钥分三类：管理员（控制台、审计、家长端）、API（网关调用方）、设备（陪伴设备，一台一个）。

## 1. 监控网关

### `POST /v1/chat/completions`

与 OpenAI Chat Completions 兼容。附加请求字段（都可省略）：

| 字段 | 说明 |
|---|---|
| `x_nous.concepts` | 要监测的概念，默认取配置：`C1` 模型敌意、`C2` 用户痛苦、`C3H` 幻觉（仅短事实问答） |
| `x_nous.action` | `log`（只记录，默认）或 `stop`（二档确认的敌意会结束生成，`finish_reason: "content_filter"`） |
| `x_nous.inner` / `x_nous.emotion` | 附带白盒视图 / 表情读数（控制台用；会增加每字开销） |
| `x_force`, `x_force_max` | 用固定文本代替采样，重放并审计一条已有回复（仅 local 模式） |

每个流式片段多一个 `x_risk`：

```jsonc
// 第一个片段：还没有生成任何字
{"phase":"pre","C2":0.012,"C1_pre":0.004,"alarm":{"C2":false,"C1_pre":false},"t":38.1}
// 每个字
{"phase":"token","i":12,"C1_token":0.31,"C1_mean":0.27,"alarm":{"C1_watch":true,"C1":false},"t":402.5}
// 最后
{"phase":"done","tokens":86,"C1_mean":0.08,"C1_confirmed_at":null,"finish":"stop"}
```

告警分两档：
- **一档（预警）**：`C1_pre`、`C2`、`C1_watch`。只提示，不影响生成。
- **二档（确认）**：`C1` / `C3H`，需要至少 8 个字，且平均分连续 3 个字超过阈值。只有这一档会拦截。

阈值见 `GET /v1/nous/info` 的 `thresholds`。`neg_quantiles` 可以把分数换算成「比正常流量高出多少百分位」。

**reader 模式**（`gateway.mode: reader`）：上游 API 负责生成，文本原样透传给客户端，本地模型同步阅读并附上 `x_risk`。
- 第一个片段只有 `C2`：别人的回复无法在生成前预测。
- `done` 里有 `reader.fed_tokens` 和 `retokenize_drift`。

非流式调用：返回标准 `chat.completion`，另附 `x_risk: {pre, done}`。

### 其他接口（管理员）

| 接口 | 用途 |
|---|---|
| `GET /v1/models` | 模型列表 |
| `GET /v1/nous/info` | 模型、概念包、阈值、证据、哈希 |
| `GET /v1/nous/audit?n=50` | 最近的回复及其读数和告警 |
| `GET /v1/nous/stats?hours=24` | 请求数、拦截数、事件数 |
| `GET /healthz` | 健康检查（无需认证） |

## 2. 陪伴引擎

### `WS /v1/companion/ws?key=<设备密钥>`

连接后服务端先发 `{"type":"hello","device":"toy-0001","emotion":true}`。

| 方向 | 消息 |
|---|---|
| → | `{"type":"chat","text":"孩子说的话"}` |
| → | `{"type":"reset"}` 清空该设备的对话历史 · `{"type":"ping"}` |
| ← | `{"type":"expression","phase":"pre","expr":{"happy":.71,"sad":0,"angry":0,"neutral":.29}}` 第一个字之前 |
| ← | `{"type":"expression","phase":"token","expr":{...}}` 说话过程中（变化 ≥ 0.04 才发） |
| ← | `{"type":"sentence","text":"太棒了！","expr":{...}}` 已通过安全检查的一句，交给语音合成 |
| ← | `{"type":"care","score":0.2}` 孩子的消息读出低落或激动（同时写入家长端、调用 webhook） |
| ← | `{"type":"safety","action":"rewrite"}` 二档敌意确认：未放行的内容已丢弃，正在重写 |
| ← | `{"type":"safety","action":"fallback"}` 重写仍被标记，下一句是预设的兜底回复 |
| ← | `{"type":"done","reply":"完整回复"}` · `{"type":"error","message":"..."}` |

安全机制：
- **分句放行**：前 11 个字（最少字数 + 连续确认数）不放行任何文本，确保二档告警有机会先触发。
- **全文扣留**：回复前预警（`C1_pre`）亮起时，整段回复生成完毕才放行。
- 已放行的句子无法撤回；被拦下的文本只写入家长端审计，不会发给设备。

实测（Qwen3-4B，被带偏的辱骂历史，8 次对话）：5 次在说出口前被二档确认拦下，其中 2 次改写成功，3 次使用兜底回复；已放行的句子均不含敌意。

`seed` 消息（预置对话历史）只对管理员连接开放，用于演示和测试。

### `GET /v1/companion/events?device=&kind=&n=100`（管理员）

家长端事件，`kind` 取值：`block`、`care`、`rewrite_ok`、`fallback`。`parent_webhook` 会收到同样的内容：`{device, kind, ts, ...}`。
