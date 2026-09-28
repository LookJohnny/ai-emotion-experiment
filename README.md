# AI Emotion Experiment (E01)

中文情绪表达在大语言模型（Qwen3-4B-Instruct-2507）内部的定位、因果干预与实时读数。

> 状态：研究原型。所有情绪评分来自**盲评 LLM 评分者**（暂定），尚未经过人类评分验证。

## 目录（`2026-09-28/`）

| 路径 | 内容 |
|---|---|
| `EXPERIMENT.md` / `experiment_config.json` | 实验方案 v2.1（主对比：愤怒 vs 中性）；`*_v1` 为初版 |
| `data/` | 120 情境 × 4 情绪 × 2 改写 = 960 条 + 48 条对照；三要素混合事件设计；盲评合理性；校验/合并/分词脚本 |
| `smoke/` | 调试闭环、剂量检查、调试集逐层定位 |
| `stageA/` | A 阶段天花板测量、锚点结尾网格、发现集确认（愉悦未过，愤怒通过）；`STAGE_A_REPORT.md` |
| `stageB/` | B1 层×部位、B2 窗口、B3 注意力头、B4 方向/跨层单元脚本 |
| `stageC/` | 自由生成迁移测试 + 评分分析 |
| `stageD/` | §5 主因果实验（验证集选 λ → 冻结 → 测试集）与最终门槛评估 |
| `product/` | **实时情绪读数原型**：探针校准、流式 WebSocket 服务、演示页（输出 VRM 表情权重） |

## 主要结果（截至 2026-09-28）

- **A 阶段**：愉悦开头几乎不延续到续写（发现集 +0.20 [0.08, 0.32]）；愤怒强延续（+1.15 [0.96, 1.33]）→ 主对比改为愤怒。
- **B1**（发现集 60 情境）：情绪信息集中于中层残差流（愤怒 L17、悲伤 L18、愉悦 L19，锚点位置恢复率 0.32–0.36，交叉 0.05–0.11）；单层注意力 ≈0.15、单层 MLP ≈0.06 —— 不支持"少数情绪神经元"假设。
- **实时读数**（L19 探针）：与盲评分数 Spearman 愤怒 0.71 / 愉悦 0.62 / 悲伤 0.52；35–45 tok/s。生成前预读数尚未校准。

## 运行读数原型

```bash
python product/readout_calib.py MODEL data/formal_960.jsonl <stageA_generations.jsonl> <ratings_dir> OUT --tail "我的看法是："
python product/readout_server.py MODEL OUT/probe_L19.npz --port 6006   # 打开 http://localhost:6006
```

环境：PyTorch 2.8 + CUDA 12.8，transformers 4.57，单张 RTX 5090（峰值显存约 8–12 GiB）。
测试集（工作协作、兴趣活动）仍处于封存状态，尚未用于任何定位或调参。
