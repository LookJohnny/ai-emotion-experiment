# Nous

读取大模型在**开口之前**和**说话过程中**的内部状态：同一次前向计算同时给出表情、安全预警和可审计的内部读数，不需要额外调用模型。

一个引擎，两个产品：

| 产品 | 给谁 | 做什么 |
|---|---|---|
| **监控网关** | 自部署 Qwen / DeepSeek 的企业、集成商 | OpenAI 兼容代理：换一个地址即可接入，每个流式片段附带风险读数；两档告警（预警只提示，连续确认才拦截）；审计日志；白盒控制台 |
| **陪伴引擎** | AI 毛绒玩具、陪伴机器人厂商 | 一台设备一条 WebSocket：第一个字之前就给出表情；不当回复在说出口前拦下并改写；孩子情绪低落时通知家长 |

## 证据（预注册、封存测试集，Qwen3-4B-Instruct-2507）

| 能力 | 结果 |
|---|---|
| 表情预同步 | 54/54 在第一个字之前出现表情；起动 23 ms（整句后分析 274 ms）；中性任务误报 0/18 |
| 模型敌意 C1 | 逐字 AUROC 0.929（通过），回复前 0.919；**最终准确率不胜全文本分类器**（+0.009），优势在时间和成本 |
| 用户痛苦 C2 | 无关键词迁移 AUROC 0.876（通过），比同数据文本分类器高 +0.22 |
| 幻觉 C3-H | 答案表征 AUROC 0.811，回答前 0.722（不确定） |
| 闭源 API 读者代理 | 读 Kimi K3 的回复：敌意 AUROC 0.973（通过）；幻觉 0.737（不确定，受读者自身知识限制） |

完整实验报告见仓库 `2026-09-28/`。标签来自两名 LLM 盲评员，人工一致性抽检仍在进行。

## 快速开始

```bash
pip install -e .                                   # 需要 CUDA GPU（≥16 GB，4090 上 20–30 字/秒）
cp nous.example.yaml nous.yaml                     # 填模型路径
nous keygen --n 3                                  # 生成管理员 / API / 设备密钥，写进环境变量
nous check -c nous.yaml
nous serve -c nous.yaml                            # http://HOST:8080
```

Docker：

```bash
cp deploy/.env.example deploy/.env                 # 填密钥和模型目录
docker compose -f deploy/docker-compose.yml up -d
```

页面：`/` 服务首页 · `/console` 监控控制台 · `/companion` 陪伴演示 · `/parent` 家长端 · `/docs` API。

## 接入

**网关**（任何 OpenAI SDK 不改代码可用，`x_risk` 是附加字段）：

```python
from openai import OpenAI
client = OpenAI(base_url="http://HOST:8080/v1", api_key=NOUS_API_KEY)
for ch in client.chat.completions.create(model="nous", messages=[...], stream=True):
    print(ch.choices[0].delta.content or "", ch.model_extra.get("x_risk"))
```

**陪伴设备**：见 [docs/API.md](docs/API.md) 和 [docs/HARDWARE.md](docs/HARDWARE.md)；客户端 `clients/python/nous_client.py`、`clients/js/nous-client.js`。

## 边界（请如实告知客户）

- 完整能力（回复前预警、表情预同步、白盒可视化）只适用于**能拿到内部激活的自部署开源模型**。
- 闭源 API 走「读者代理」：本地模型读对方的输出，敌意监测可用，**没有回复前预警**；幻觉监测受本地模型知识限制。
- 探针按模型训练：每换一个基座模型需要一个新的概念包（`packs/`），约 1–2 天 GPU + 标注。
- 目前只在 Qwen3-4B 上验证；阈值按验证集 5% 误报率设定，上线前建议按客户场景校准一次。
- 表情和情绪读数是模型内部激活的探针估计，不代表模型有主观感受；「关怀」提醒不是心理诊断。

## 结构

```
nous/engine/     共用引擎：探针、两档告警、表情平滑、白盒视图、读者代理
nous/gateway.py  监控网关（OpenAI 兼容）
nous/companion.py 陪伴引擎（WebSocket）
nous/web/        控制台、陪伴演示、家长端（离线可用，无外部 CDN）
packs/qwen3-4b/  概念包：冻结的探针 + 阈值 + 证据
clients/         Python / JS 客户端
deploy/          Dockerfile、compose
tests/           单元测试（CPU）+ 端到端测试（GPU，复现研究分数）
```

测试：`pytest tests/test_units.py`；GPU：`NOUS_MODEL=/path/to/model NOUS_RESEARCH=../2026-09-28 pytest tests/test_gpu.py`。
