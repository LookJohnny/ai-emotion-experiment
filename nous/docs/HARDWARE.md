# 陪伴硬件接入

设备端只负责录音、放音和执行表情，推理放在私有云。一张 4090 跑 Qwen3-4B，同一时间服务一路对话，每秒 20–30 字。多路并发需要多卡或批处理，这是下一版的工作。

```
麦克风 → ASR ──text──▶ Nous /v1/companion/ws ──expression──▶ 屏幕表情 / 面部舵机
                                            └──sentence────▶ TTS → 扬声器
```

## 表情映射

`expr` 是四个 0–1 的强度：`happy`、`sad`、`angry`、`neutral`（其中 `neutral = 1 − max(其他)`）。建议的映射：

| 设备 | 映射 |
|---|---|
| 屏幕 / VRM 形象 | 直接写入 blendshape：`happy→happy`、`sad→sad`、`angry→angry`，`neutral` 高时加一点 `relaxed` |
| 小智 ESP32（表情字） | 取最大项；强度 < 0.3 视为平静，发送 `{"type":"llm","emotion":"happy"}` 一类的表情帧 |
| 舵机脸（如 17 舵机脸板） | 每种情绪一组关键帧，按强度在「平静帧」和「情绪帧」之间插值；`phase:"pre"` 那一帧要在 TTS 开始前执行完 |

`phase:"pre"` 的表情在第一个字之前到达。实测从请求到首个表情约 0.4–0.5 秒（含 WebSocket、预填充和用户情绪读数），到首句约 1 秒。设备应该**先执行表情，再开始播放语音**，这正是「先有表情再说话」的效果。

## 小智（xiaozhi-esp32）适配思路

小智设备连的是自己的服务端（ASR → LLM → TTS）。接入方式是在这个服务端里把「调用 LLM」换成 Nous 的 WebSocket：

1. ASR 得到文本后发送 `{"type":"chat","text":...}`。
2. 收到 `expression`：转成小智的表情帧发给设备。
3. 收到 `sentence`：直接送进原有的分句 TTS 流程。Nous 已经分好句，并且做过安全检查。
4. 收到 `safety`：不需要处理，后面的 `sentence` 就是改写后的内容。
5. 收到 `care`：可以在设备上显示关怀表情，家长端由 Nous 通知。

> 这一节是接口设计，**尚未在真机上联调**。

## 家长端

- 事件会写入审计库，家长端页面地址是 `/parent`。
- 如果要推送到自己的 App，在配置里设置 `companion.parent_webhook`。每次拦截或关怀事件，服务端都会 POST `{device, kind, ts, ...}` 到这个地址。
