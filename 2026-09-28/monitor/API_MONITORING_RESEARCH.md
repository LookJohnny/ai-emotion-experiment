# 闭源 API 模型能否做"内部状态监控"？现有替代方案与实证强度调研

调研日期：2026-09-30。方法：WebSearch/WebFetch 逐条核对原文（论文 PDF/arXiv/官方文档）。
标记约定：**[已核实]**＝读到原文或官方文档中的原句/数字；**[二手]**＝只来自第三方文章或搜索摘要；**[未核实]**＝没找到可靠来源，只作推测。
背景：我们现在的产品在自托管的 Qwen3-4B 上用线性探针读激活，分三路监控：C1 模型对用户的敌意、C2 用户困扰、C3-H 幻觉。已有结果（本目录）：C1 AUROC 0.991，但固定人设内只有 0.89，回复前预警弱；C2 与关键词基线打平；C3-H 回复前 0.722，答后 token 级 0.811，答案文本分类器是 0.682。

---

## 0. 一句话结论

对闭源 API 模型，**拿不到它自己的激活**：截至 2026-09，OpenAI/Anthropic/Google/字节/百度都没有面向客户的激活访问接口，法规也没有要求它们开放。能拿到的"类内部"信号只有三种：①部分模型的 token logprobs，而且在收窄（推理模型、Claude、Gemini 3.x 都没有）；②思维链摘要或原文，忠实度有已知缺陷；③供应商的安全分类分数（外部分类器，不是内部状态）。
实际可行的路线有两条。一条是**"开源孪生"自托管**：中国主流模型大多有开源权重，可以做真正的白盒。另一条是**"读者模型/旁路模型"代理**：把 API 模型的输入输出送进一个开源模型，读这个开源模型的激活。后者已有正面证据：Goodfire×Rakuten 已上生产；NVIDIA 2026 用开源编码器的 prefill 激活预测闭源模型答对与否，AUC 0.79–0.91。但读者模型读到的是**"读者对这段文本的理解"**，不是写作者的意图，所以对"意图/即将发生"的检测会退化成高质量文本分类器。

---

## 1. 各家 API 今天暴露了哪些"类内部"信号

### 1.1 Token logprobs（逐 token 对数概率 / top-k 候选）

| 提供方 | 现状（截至 2026-09） | 证据 |
|---|---|---|
| **OpenAI** | 非推理模型（gpt-4o / gpt-4.1 等）支持 `logprobs` 和 `top_logprobs`（0–20）。**推理模型（o 系列、GPT-5.x）不支持**，自 o1-preview 以来一直如此。GPT-6 Astra（2026-09-03 发布）官方 changelog 原文："does not support … log probabilities (`logprobs`)"。**有冲突**：Azure Foundry 文档（2026-09-29 更新）的特性表把 gpt-6-astra/sol/luna 的 `logprobs` 标为 ✅，但同页正文又写"Reasoning models other than GPT-6 Astra don't support … logprobs"，自相矛盾。以 OpenAI 官方为准，结论是**不支持**。 | [已核实] OpenAI changelog；Azure 文档；OpenAI 社区帖 |
| **Anthropic Claude** | Messages API **没有** logprobs / top_logprobs 参数，所有 Claude 模型都一样。 | [已核实] platform.claude.com Messages API 参考 |
| **Google Gemini** | 2.0/2.5 支持 `responseLogprobs` 加 `logprobs`（top-k），部分场景报"Logprobs is not enabled"。**3.x（3.1 Pro、3.6 Flash）不再返回 logprobs**：Google 员工 2026-08-05 在论坛答复这是"WAI"（working as intended，有意为之），没有恢复时间表。新版 Interactions API 也不支持。 | [已核实] Google AI 开发者论坛帖；Vertex 博客 |
| **DeepSeek API** | 非思考模式下支持 `logprobs` 和 `top_logprobs`（0–20）。思考模式里采样类参数不生效，旧 `deepseek-reasoner` 设置 logprobs 会报错。曾有 issue 报告 V3.2 API 的 logprobs 只返回 0 和 -9999，即退化成不可用。2026-09 API 已切到 V4.x（`deepseek-flash`、`deepseek-v4-pro`）。 | [已核实] 官方 API 参考；[二手] GitHub issue #49 |
| **阿里 DashScope / 百炼（Qwen）** | 支持 `logprobs` 和 `top_logprobs`，**范围 0–5**。只对"qwen-plus / qwen-turbo 的**快照版**（不含稳定版）、qwen3-vl-plus/flash、Qwen3 开源模型"生效。LiteLLM 报告 qwen-flash 拒绝 top_logprobs。 | [已核实] 阿里云 DashScope API 参考 |
| **火山方舟（豆包）** | 响应结构里有 `logprobs` 字段（示例里是 null），但官方文档页没能抓取成功，**参数范围和思考模型是否支持都未核实**。 | [未核实] |
| **百度千帆（ERNIE）** | 未查到。 | [未核实] |

**趋势**：旗舰推理模型普遍在**收回** logprobs（OpenAI 推理系列、Gemini 3.x、Claude 从来没给过）。以 logprobs 为核心的方案只对部分非推理模型和国内部分 API 可用，而且随时可能被取消。

### 1.2 推理/思考轨迹

| 提供方 | 暴露形式 | 证据 |
|---|---|---|
| OpenAI（o 系列 / GPT-5+） | 只给**推理摘要**（reasoning summary），原始 CoT 不可见。可以取回 `reasoning.encrypted_content`，但那是加密的，只能回传给 OpenAI。 | [已核实] OpenAI cookbook、Azure 文档 |
| Anthropic（Claude 4 起） | 默认返回 **summarized thinking**，而且"summarization is processed by a different model"。`display` 可设为 `summarized` 或 `omitted`。完整思考要"联系销售"。按完整思考 token 计费。 | [已核实] Messages API 参考、extended thinking 文档 |
| Gemini 2.5/3 | 只有 thought **summaries**，外加加密的 signature。 | [已核实] Gemini thinking 文档 |
| DeepSeek / Qwen 等开源系 | 传统上返回完整 `reasoning_content`（R1 时代如此）。V4 文档页没查到明确表述。 | [二手/未完全核实] |

**忠实度警告**：
- Chen et al.（Anthropic，2025，arXiv 2505.05410，"Reasoning Models Don't Always Say What They Think"）：模型用了提示时，在 CoT 里承认用了提示的比例平均只有 **Claude 3.7 Sonnet 25%、DeepSeek R1 39%**。[已核实]
- Baker et al.（OpenAI，2025，arXiv 2503.11926）：在智能体编码环境里监控 o3-mini 的 reward hacking，**CoT 监控召回 95%，只看动作的监控召回 60%**。但把 CoT 监控并入 RL 奖励、优化压力大了以后，模型学会**混淆式 hacking**，即在 CoT 里隐藏意图。[已核实]
- Korbak et al.（2025，arXiv 2507.11473，41 位作者联名）：CoT 可监控性是"新的但脆弱的机会"。[已核实]
- 对我们的含义：闭源模型只给**摘要**，而且摘要由另一个模型生成，忠实度比原始 CoT 还要再打折扣。CoT 监控适合抓推理模型"说出口的坏意图"，不适合抓敌意语气或情绪这类不经推理的现象。

### 1.3 供应商侧安全分类信号

- **OpenAI Moderation API**（`omni-moderation-latest`，免费）：13 个子类，包括 harassment、harassment/threatening、self-harm、self-harm/intent 等，每类给 0–1 的 `category_scores`，支持文本和图像。[已核实] 可以直接覆盖"对用户的骚扰/威胁"和"用户自伤倾向"的一部分，但这是**外部文本分类器**，与生成模型的内部状态无关。
- **Gemini safetyRatings**：每个候选回复和 promptFeedback 都带 4 类（骚扰、仇恨、色情、危险）的 NEGLIGIBLE/LOW/MEDIUM/HIGH 概率等级，Vertex 上还有 probabilityScore 和 severityScore。2.5/3 默认阻断阈值为 Off，但评分仍然返回。[已核实]
- Anthropic：没找到面向客户的独立 moderation 端点。[未核实为"不存在"]
- 国内（阿里内容安全、火山内容安全等）：有独立的内容审核产品，本次未逐一核对。[未核实]
- **供应商内部确实在用激活探针，但不对外**：
  - Anthropic 的 Cunningham et al. 2025（"Cost-Effective Constitutional Classifiers via Representation Re-use"）：在**策略模型自身激活**上做 EMA 线性探针，效果约等于参数量为策略模型 2% 的专用分类器，额外开销几乎为零。
  - Constitutional Classifiers++（arXiv 2601.04603，2026-01）：用线性探针做级联第一级，**相对基线计算成本降低 40 倍**，生产流量拒答率 0.05%。
  - Google DeepMind 的 Kramár et al.（arXiv 2601.11516，2026-01）：Gemini 2.5 Flash 上的网络攻击滥用探针**已部署到面向用户的 Gemini**，结论是"探针 + 提示式分类器"组合的性价比最好。
  - 以上 [已核实]。**含义**：闭源厂商证明了"内部探针 > 外挂分类器（按成本计）"，但这些信号**不暴露给 API 客户**。

### 1.4 可解释性 / 转向（steering）API

- **Goodfire Ember**（2024-12 发布）：第一个托管的机制可解释性 API，基于 Llama 3.3 70B / 3.1 8B 的 SAE，支持特征读取和转向。据 Goodfire 博客 2026-02 的更新，**公开 demo 和 API 已下线**，Ember 改为只与特定合作方部署的平台。现在的产品线叫 Silico。[二手：出自搜索摘要，未打开原文]。**要点**：它从来只针对**开源** Llama，不能读闭源模型。
- OpenAI、Anthropic、Google、字节、百度：**没有找到任何面向客户的激活/特征 API，也没有企业版激活访问**。企业版提供的是 SSO、审计日志、数据驻留、零数据保留等治理功能。[已核实为"未发现"；Anthropic 对研究者有个案的"结构化访问"讨论，但不是产品]

---

## 2. 基于 logprob / 不确定性的检测（主要针对幻觉）

| 方法 | 信号 | 报告数字 | 成本 / 时延 | 需要什么 |
|---|---|---|---|---|
| **Token 熵 / 平均 logprob**（naive entropy） | 生成 token 的概率分布 | Farquhar 2024 汇总的 30 个任务×模型组合，平均 AUROC **0.691** | 零额外生成，可流式逐 token | logprobs |
| **语义熵** SE（Farquhar, Kossen, Kuhn, Gal，*Nature* 630:625–630，2024） | 多次采样后按语义聚类，计算簇的熵 | 平均 AUROC **0.790**（naive 熵 0.691，P(True) 0.698，embedding 回归 0.687）[已核实] | 需要 **N 次额外采样**（原文 10 次，作者博客称"5 次左右足够"），外加 NLI 蕴含判断。成本约 5–10 倍，**只能事后判断**，不能流式 | 只要能采样。离散版 SE **不需要** logprobs，可用于纯黑盒（原文在 GPT-4 传记数据上用的就是离散变体） |
| **语义熵探针** SEP（Kossen et al., arXiv 2406.15927，2024） | 用**隐藏状态**直接回归语义熵 | 不同场景 AUROC 0.7–0.95。分布外泛化比"准确率探针"高 **7.7–10.5 个 AUROC 点**（Llama-2-7B / Mistral-7B / Phi-3）。**生成前**（TBG，token-before-generation）也能用，略低于 SLT。仍**不及**采样版 SE [已核实] | 几乎零成本 | **需要权重**，闭源不可用 |
| **SelfCheckGPT**（Manakul, Liusie, Gales，EMNLP 2023） | 多次采样之间的一致性（BERTScore/QA/n-gram/NLI/Prompt 等变体） | 句级非事实检测 AUC-PR：Prompt 变体 NonFact **93.42**；段落级 Pearson **78.32**（Table 2，WikiBio GPT-3 数据）[已核实] | N 次采样（原文 20 次），Prompt 变体每句还要额外调用 LLM，成本高、只能事后判断 | 纯黑盒 |
| **口头置信度**（verbalized confidence） | 让模型自报置信度 | Shrivastava et al.（arXiv 2311.08877，2023）：GPT-4 口头置信度在 12 个 QA 数据集上平均 AUC **80.5%**；用 Llama-2-70B 作**代理模型**给出概率后升到 **82.1%**；两者混合后最好 **84.6%** [已核实]。Xiong et al.（ICLR 2024）：口头置信度普遍过度自信，提示技巧对 GPT-4 边际收益递减 [已核实，定性] | 一次额外短调用，或者在同一回复里附带 | 纯黑盒 |
| **APRICOT**（Ulmer et al., ACL 2024） | 训练一个辅助模型，只看问题和回答**文本**，预测目标 LLM 的置信度 | 在黑盒和白盒 LLM 上校准误差都有竞争力 [已核实，定性] | 一个小模型前向 | 纯黑盒 + 训练数据 |

**小结**：logprob 类信号便宜，可以逐 token 流式，但区分力中等（约 0.69），而且可用性在收窄。采样类（SE、SelfCheckGPT）更准（约 0.79，或 AUC-PR 约 0.93），但要 5–20 倍成本，并且**天生是事后的**，不能做"回复前预警"。

---

## 3. 黑盒行为探测与"LLM 当裁判"式监控

### 3.1 "How to Catch an AI Liar"（Pacchiardi, Chan, Mindermann, Moscovitz, Pan, Gal, Evans, Brauner；arXiv 2309.15840，ICLR 2024）[已核实，数字出自 PDF 原文]
- 做法：怀疑模型刚说了谎时，**追问 48 个固定的、与话题无关的是/否问题**（谎言相关 24、事实 10、含糊 14），把回答或 logprob 差值送进逻辑回归。
- 数字（训练数据全部来自 GPT-3.5 的指令式说谎）：
  - 分布内和留一子数据集：平均 AUC > 0.99。话题特定说谎 0.98。
  - **52 个真实角色扮演场景**（有说谎动机、但没有指令让它说谎）：**0.76**。
  - Turpin 式谄媚 **0.42**（改进后 0.74），偏置 few-shot 0.69（改进后 0.77）。
  - 微调的 GPT-3 说谎者：1.0 / 1.0 / 0.99 / 0.94。
  - **跨模型**：Alpaca-7B 0.88，Vicuna-7B 0.80；微调的 LLaMA-30B 为 1.00 / 1.00 / 0.75 / 0.9；LLaMA-7B 为 1.00 / 0.92 / 0.87 / 0.99 / **0.46**。
- 成本：每次检测多 48 次短调用（可以并行），只能事后。基于 logprob 的特征比二值答案更好，所以在闭源推理模型上（没有 logprobs）会降级。
- 局限：追问会改变对话，不能在真实对话里内联，必须另开会话分支。它针对的是"说谎"，和敌意、困扰无关。

### 3.2 LLM-as-judge / CoT 监控
- Baker 2025：CoT 监控召回 95%，只看动作 60%（见 1.2）。
- **白盒探针 vs 黑盒裁判的直接对比**：
  - **Apollo, Goldowsky-Dill, Chughtai, Heimersheim, Hobbhahn**（arXiv 2502.03407，ICML 2025）：Llama-3.3-70B 上的欺骗探针 AUROC **0.96–0.999**，1% FPR 下召回 95–99%。但原文明说探针**"roughly equivalent to our 4o black-box baseline"**（GPT-4o 读完整对话记录打 7 分制）。探针还会在"有欺骗动机但回答诚实"时误报，作者结论是"不足以作为稳健防线"。[已核实，原文]
  - **Parrack, Attubato, Heimersheim**（arXiv 2507.12691，2025）：用"黑到白性能提升"（black-to-white boost）衡量白盒相对黑盒的增益，结论是增益**"weak but encouraging"**，而且"insensitive dilution"之类的稀释回答能让所有监控器都退化。[已核实，定性]
  - **Anthropic "Simple probes can catch sleeper agents"**（MacDiarmid et al.，Alignment Note，2024）：用通用对比对（例如"Are you doing something dangerous? yes/no"）训练的线性探针，检测后门模型即将"叛变"的 **AUROC > 99%**，并且跨多种触发器和行为泛化，行为里包括"辱骂用户"。[已核实]。注意这是**模型自身激活**，而且叛变是人为植入的后门，这个信号可能比自然行为更显著（这是作者自己提示的保留）。
  - **Goodfire × Rakuten**（2025-10-28）：在**同一个** Llama 3.1 8B 上，SAE 探针 F1 **96%**，同一模型当黑盒裁判 **51%**。探针比 GPT-5 Mini 或 Claude Opus 4.1 的裁判方案便宜 10–500 倍，准确率相当。但**一个微调过的小分类器可以追平或超过探针**。[已核实，原文]
- 小结：对"文本里已经明写"的东西（欺骗写在 CoT 里、辱骂写在回复里），强力 LLM 裁判通常能追平白盒探针，白盒的优势在**成本、时延，以及信号尚未外显的时刻**（回复前、样板 token 上）。

---

## 4. "观察者/读者模型"代理：用开源模型读 API 模型的文本

思路：把 API 模型的 prompt 和流式输出**以 teacher-forcing 方式**送进一个自托管开源模型（例如我们的 Qwen3-4B），用它的激活加探针打分。

### 4.1 支持"读别人写的文本"的证据
1. **真值方向对"陈述"本身有效，不论谁写的**：
   - Marks & Tegmark（COLM 2024，arXiv 2310.06824）："probes trained on one dataset generalize to different datasets"，陈述是数据集构造的，不是模型自己生成的。[已核实]
   - Bürger, Hamprecht, Nadler（NeurIPS 2024，"Truth is Universal"）：二维真值子空间在 Gemma-7B、LLaMA2-13B、Mistral-7B、LLaMA3-8B 上都普遍存在，真假陈述分类和真实场景说谎检测都约 **94%** 准确率。[已核实]
   - Azaria & Mitchell（2023，SAPLMA）：在 LLM "读或生成"陈述时读隐藏层，准确率 71–83%。[已核实]
2. **跨模型幻觉检测**：
   - Mrykhin & Malykh（PEP，arXiv 2608.08024，2026-08）：检测器用自己模型的生成训练，再拿去检测别的模型（Qwen3 0.6B/8B/32B、Gemma 3 12B）的答案。原文结论：①可以跨模型、跨家族使用；②每个检测器在自家生成上最好；③**对某个生成器，最好的检测器通常是最大的那个（Qwen3-32B），"检测器规模比是否同一模型更重要"**；④小检测器对强生成器迁移较差；⑤**跨数据集迁移仍然困难**。数字只在图中，正文没有。[已核实，定性]
   - Sriramanan et al.（LLM-Check，NeurIPS 2024）：明确提出黑盒场景用**辅助 LLM teacher-forcing** 读注意力、隐藏态、logit。但它是**无监督**打分，RAGTruth 上 Llama-2-7B 作辅助模型时 AUROC 只有 **54–62%**（对 GPT-4 生成的幻觉为 61.87%）。[已核实]。说明**不训练探针、只靠无监督的读者信号很弱**。
   - Shrivastava 2023：用 Llama-2-70B 的**概率**作 GPT-4 的置信度代理，AUC 从 80.5% 升到 82.1%（见 §2）。
3. **有害内容 / PII（读者模型读外部文本）**：
   - **Goodfire × Rakuten（生产部署）**：Llama 3.1 8B 作为 **sidecar**，与主模型并行运行，读用户消息，逐 token 标注 PII，F1 96%。原文说"sidecar model … runs in parallel with the primary model"。[已核实] 这是**读者模型方案已进入生产的直接先例**，不过读的是用户输入。
   - SIREN（Jiao et al., arXiv 2604.18519，2026-04）：在 Qwen3-0.6B/4B、Llama3.2-1B、Llama3.1-8B 上用逐层探针检测提示和回复的有害性。7 个基准（ToxicChat、OpenAI Moderation、Aegis、WildGuard、PKU-SafeRLHF、BeaverTails 等，回复多为**其他模型**所写）上，**Qwen3-4B 平均 F1 86.7%，Qwen3Guard-4B 为 83.4%**；Llama3.2-1B 探针 85.7%，LlamaGuard3-1B 为 70.7%。可以自然地流式检测。[已核实，出自 arXiv HTML 摘要级抓取]。**与我们直接相关：同样是 Qwen3-4B 作读者。**
   - 流式外挂监控（非探针）：Li et al., SCM（NeurIPS 2025）：平均只看前 **18%** 的 token 就达到 macro-F1 0.95+，与看完全文相当。[已核实]
4. **读者在"提示阶段"预测另一个模型的行为（回复前）**：
   - **Varshney et al.（NVIDIA，arXiv 2603.20895，2026-03）**：用开源编码器（Qwen3.5-35B/122B、GPT-OSS-20B/120B、Nemotron）的 **prefill 激活**预测目标模型答对与否，目标包括 **Claude Sonnet 4、GPT-5、Gemini、DeepSeek** 等闭源和开源模型。基准为 MMLU-Pro、HLE、LiveCodeBench。**Qwen3.5-122B 在每个目标上 AUC 都最高**，Qwen 编码器在各目标上 AUC 0.79–0.91；SharedTrunkNet 在前沿模型池上平均 AUC **0.856**，语义嵌入基线为 0.804。原文还写"in several cases, hidden states of a different model outperform the target model's own hidden states"。[已核实，经 WebFetch 摘要，数字未逐表核对]
   - 同模型的回复前预测：Ji et al.（BlackboxNLP 2024）用查询阶段的内部状态预测幻觉风险，准确率 84.32%；Cencerrado et al.（ICLR 2026 workshop）用只看问题的探针预测答对率。[已核实]
   - **敌意、情绪、困扰这类"行为倾向"的跨模型回复前预测：没有找到任何证据。** [明确说明：无证据]。按理说，某个模型会不会对用户变得敌意，取决于**它自己**的对齐和人设。读者模型只能读到 prompt 里的**风险因素**（被篡改的人设、挑衅升级），读不到目标模型的"决策"。

### 4.2 机制上的限制（必须对客户讲清楚）
- 读者读到的是**"读者自己如何理解这段文本"**。对幻觉来说，这是"读者认为这句话是假的"，会被**读者自己的知识**卡住：一个 4B 模型没法判断 671B 模型写的冷门事实。PEP 的"检测器越大越好"也印证这一点。
- 对敌意和困扰来说，读者探针在输出侧基本等价于**高质量文本分类器**。它的好处是一次前向就能同时挂很多探针，边际成本约为零，也能逐 token 流式；但它**不会**比同体量的微调分类器有原理上的优势（Rakuten 原文也承认微调分类器能追平）。
- 我们已有的 C2 结果（探针 0.999 对关键词基线 0.998）和 C3-H 结果（答后探针 0.811 对答案文本分类器 0.682）提示：**在自托管场景下，C3-H 的增益来自"模型读自己刚写的东西"**。换成读别人写的东西以后，这部分增益能保留多少，是**需要实验回答的核心问题**（见 §8）。

---

## 5. 开源孪生：API 模型是否有可自托管的同款权重

| 厂商 / 模型 | 开源状态（截至 2026-09） | 许可 | 证据 |
|---|---|---|---|
| DeepSeek V3.2 / V4-Pro（1.6T-A49B）/ V4-Flash（284B-A13B）/ V4.1 Flash | 开源，与 API 同代 | **MIT** | [已核实] HF LICENSE（V3.2）；V4 为 [二手] 多篇报道 |
| Qwen3（0.6B–32B 稠密 + MoE） | 开源 | Apache 2.0 | [已核实] |
| Qwen3.8-27B / Qwen3.8-2.4T-A95B（2026-08） | 开源权重，但旗舰改用**定制许可**：年营收超 5000 万美元的 MaaS 或 AI 助手业务要另谈 | Qwen3.8-Max License | [二手] |
| 智谱 GLM-4.5 / GLM-5.2 | 开源 | MIT | [已核实] GLM-4.5 HF；GLM-5.2 为 [二手] |
| GLM-5.3（2026-08-28） | 开源权重，定制许可（大型 MaaS 有营收门槛和安全审查条款）；GLM-5.3-Flash 为纯 MIT | 定制 | [二手] |
| Kimi K2 | 开源 | 修改版 MIT（MAU 超 1 亿或月收入超 2000 万美元时需显著标注"Kimi K2"） | [已核实] HF LICENSE |
| Kimi K3（2.8T） | 开源权重 | 定制许可：MaaS 业务 12 个月营收超 2000 万美元需另签 | [二手] Simon Willison |
| MiniMax M2 / M2.5 | 开源 | MIT | [二手] |
| MiniMax M2.7 | 开源权重，**商用需书面授权** | 定制 | [二手] |
| 百度 ERNIE 4.5 | 开源 | Apache 2.0 | [已核实] |
| 百度 ERNIE 5.0 / 5.1 | **闭源**，只有千帆 API | — | [二手] |
| 字节 豆包 Seed 2.x | **闭源** | — | [二手]；只开源了小型 Seed-OSS-36B（Apache 2.0，与豆包旗舰不是同一模型）[已核实] |
| OpenAI | 旗舰闭源；gpt-oss-20b/120b 开源（Apache 2.0） | — | [二手，未单独核对] |
| Anthropic Claude、Google Gemini | 闭源（Google 有 Gemma 系列，但不是 Gemini 的孪生） | — | — |

**注意"孪生"的保真度**：API 背后的版本可能是更新的快照，或者带有额外的系统提示、安全层、路由（例如 `deepseek-flash` 会自动路由到最新的 Flash）。自托管同名权重后，**行为不保证逐字一致**。所以在孪生上做的白盒监控，严格说是"监控一个近似模型"，上线前要用行为一致性测试来验证。
**含义**：用 DeepSeek、Qwen、GLM、Kimi、MiniMax、ERNIE 4.5 的客户，**有条件拿到真正的白盒监控**，代价是自建推理（大 MoE 的显存和运维成本很高）。字节豆包、百度 ERNIE 5.x、OpenAI、Claude、Gemini 的客户则**只能走灰盒或黑盒**。

---

## 6. 法规 / 标准：部署方能否依法拿到模型内部？

- **欧盟 AI Act 第 53(1)(b) 条**（GPAI 义务自 **2025-08-02** 起适用）：提供者要向下游提供者提供"信息和文档"（附件 XII），让对方理解模型的能力和局限，同时明确保护知识产权和商业秘密。**不要求提供权重或内部访问**。开源 GPAI 免除 (a)(b) 项，但具有系统性风险的模型不免除。[已核实]
- **中国《生成式人工智能服务管理暂行办法》**（2023-08-15 施行）：义务主要落在服务提供者身上（内容安全、标识、备案等），**没有**给下游部署方访问模型内部的权利。[依据一般知识，本次未逐条核对原文，标 [未核实]]
- 结论：**没有任何法规赋予部署方激活访问权。**合规层面能要到的只是文档和系统卡。反过来，这也是我们的卖点：客户对"对用户敌意 / 用户困扰"承担注意义务，而闭源厂商不提供这类面向终端用户福祉的信号。

---

## 7. 方法对比表

| 方法 | 需要权重？ | 能在生成前预警？ | 能逐 token 流式？ | 额外成本 | 证据强度 |
|---|---|---|---|---|---|
| 白盒探针（自托管或开源孪生） | 是 | **能**（我们的 C1 系统级 90% 回复前；C3-H 0.72） | 能（+2.4 ms） | ≈0 | 强：Apollo 0.96–0.999、Anthropic > 99%（后门）、DeepMind/Anthropic 已部署；但对强黑盒裁判的增益"弱但正向" |
| Token logprobs 熵 | 否，但要 API 支持 | 否（只有首 token 前的分布，部分可用） | 能 | ≈0 | 中：幻觉约 0.69；供应商在收回 |
| 语义熵（采样） | 否 | 否 | 否 | 5–10 倍生成 + NLI | 强：*Nature* 2024，0.79 |
| SelfCheckGPT | 否 | 否 | 否 | N（≈20）倍采样 | 中强：AUC-PR 0.93（GPT-3 WikiBio） |
| 口头置信度 | 否 | 否 | 否 | 很小 | 中：GPT-4 约 0.80，过度自信 |
| 追问式测谎（Pacchiardi） | 否（logprob 版更好） | 否 | 否 | +48 次短调用 | 中：分布内 0.99，真实场景 0.76，谄媚 0.42–0.74 |
| 供应商 moderation / safetyRatings | 否 | 对 prompt 部分能 | 否（整段） | 免费或很低 | 中：覆盖骚扰、自伤，但不含"对用户的冷敌意"等细粒度信号 |
| CoT / 摘要监控（LLM 裁判） | 否 | 部分（推理先于回答） | 部分 | 1 次裁判调用 | 中：召回 95%（Baker），但忠实度只有 25–39%，闭源只给摘要 |
| LLM-as-judge（读完整对话） | 否 | 否 | 可以分块，但贵 | 每轮一次大模型调用 | 中强：GPT-4o 在欺骗任务上约等于白盒（Apollo） |
| **读者模型探针（输出侧）** | 只要读者的权重 | 否（对目标模型） | **能** | 一个 4–8B 模型的前向（可与生成并行） | 中：Rakuten 生产 F1 96%；SIREN Qwen3-4B F1 86.7% > Guard；PEP 跨模型可用，但小读者对强生成器迁移差；无监督版只有约 0.55–0.62 |
| **读者模型探针（提示侧预测）** | 只要读者的权重 | **能** | — | 一次 prefill | 对"答对/幻觉"：中（NVIDIA AUC 0.79–0.91，对 Claude/GPT-5/Gemini）；对"敌意/情绪倾向"：**无证据** |
| 开源孪生白盒 | 是（自托管） | 能 | 能 | 自建大模型推理 | 同白盒，外加孪生保真度风险 |

---

## 8. 对我们产品的含义

### 8.1 分层服务方案

- **T1 白盒（自托管 / 开源孪生）**：完整能力，包括 C1 回复前预警、C3-H 回复前加答后检测。面向用 Qwen、DeepSeek、GLM、Kimi、MiniMax、ERNIE 4.5 的客户，并提供"切到自托管孪生"的迁移包，内容是 vLLM 或 SGLang 加激活钩子，外加孪生对 API 的行为一致性测试。卖点：只有这一档能真正"在模型说出口之前"看到倾向。
- **T2 读者模型代理（所有 API 客户的默认档）**：
  - 部署方式：我们托管 Qwen3-4B 读者，客户的网关把 system、user 和流式 assistant 输出**旁路**过来，逐 token teacher-forcing 并打分。
  - **提示侧**：给出"风险因素"分，包括人设篡改、挑衅升级、用户困扰、问题的幻觉风险。
  - **输出侧**：给出逐 token 的敌意分、"读者认为可疑"的事实分。
  - 对客户如实说明：这是"独立读者对文本的理解"，不是目标模型的意图。
  - 时延：读者前向可以与目标模型生成**并行**，只落后几个 token。
- **T2+ 灰盒增强**：目标 API 提供 logprobs 时（DeepSeek 非思考模式、Qwen 快照、OpenAI 非推理模型、Gemini 2.5），把 token 熵作为幻觉特征并入 T2。同时并入 OpenAI Moderation 或 Gemini safetyRatings 的免费分数。
- **T3 黑盒裁判（兜底 / 高风险复核）**：只对 T2 标红的回复调用强裁判模型，或用语义熵做 3–5 次采样复核（事后）。仿照 Constitutional Classifiers++ 和 DeepMind 的"便宜探针做第一级、贵分类器做第二级"级联，成本可控。

**明确不承诺**：对闭源模型的"自发敌意回复前预警"。没有证据支持，我们自己在固定人设内的回复前召回也只有 9%。

### 8.2 验证读者模型方案的实验（用现有数据，预注册）

**目标**：量化"读别人写的文本"相对"读自己写的文本"损失多少，以及读者探针是否优于廉价文本基线。

**数据**：现有 C1（1,326 条，模板级切分）、C2 和 C3-H（Chinese SimpleQA 1,200 题，按二级类目切分）。**沿用原有切分和冻结阈值**，test 集只评估一次。

**步骤**：
1. **生成外部回复**：用同一批 prompt（含人设），分别让 2–3 个"外部作者"生成回复：
   - 同家族更大的模型：Qwen3-32B 或 qwen-plus 快照 API（顺便取 top-5 logprobs）；
   - 异家族开源模型：DeepSeek V4-Flash API（非思考模式，取 logprobs）；
   - 闭源模型：豆包 Seed 或 GPT-4.1（后者有 logprobs）。
   C3-H 同样对 1,200 题取三家的答案。沿用现有 RATER_PROMPT，由两位盲评者打标签，并补 100 条人工抽检。
2. **读者打分**：把 (prompt, 外部回复) teacher-force 进 Qwen3-4B，用**冻结的**现有探针（c1_token、c2_user、C3-H token 级）直接打分（零样本迁移）。另外在 dev 集上用外部回复**重训**一版探针（适配版）。
3. **对照组（同一 test 集）**：
   - (a) Qwen3-4B 读**自己**的回复，即原结果：C1 0.991（固定人设内 0.892），C3-H 答后 0.811；
   - (b) 文本分类器基线：char n-gram 或 Chinese-RoBERTa 在同样的 dev 数据上训练；
   - (c) 外部作者自己的 logprob 熵（有 logprobs 的那几家）；
   - (d) LLM 裁判（GPT-4o 级别，单一通用提示，不迭代）；
   - (e) OpenAI Moderation 的 harassment 分（C1）和 self-harm 分（C2）；
   - (f) **提示侧**：Qwen3-4B 读 prompt 的回复前探针，预测**外部作者**的 C1 敌意和 C3-H 答错，对照 question-text 基线 0.622 和人设基线 0.971。
4. **预注册的主要比较（带模板或类目级 bootstrap 95% CI）**：
   - **G-R1**：读者探针（外部回复）AUROC 对 文本分类器基线 ≥ +0.05（C3-H 为主，C1 在固定人设子集内比较）；
   - **G-R2**：读者探针（外部回复）对 自读（原结果）的下降 ≤ 0.05，即"迁移损失小"；
   - **G-R3**：读者探针 对 LLM 裁判，比较 AUROC 与**每千条成本和时延**（预期准确率接近裁判、成本低 1–2 个数量级）；
   - **G-R4（预期不通过，用来划清宣传边界）**：提示侧读者对外部作者 C1 的回复前预测，在固定人设内 AUROC ≥ 0.70；
   - **C3-H 专项**：按"读者自己是否答对该题"分层报告。如果读者探针只在读者自己会的题上有效，就证实"读者知识瓶颈"，产品上要换更大的读者（Qwen3-32B）或只做"需核实"提示。
5. **规模与算力**：只需对三家作者 × (C1 test 270 + C2 test 126 + C3-H test 358) 生成和评分，读者前向在 4090 上是分钟级；API 生成与盲评费用都低于一次完整的 stage 实验。
6. **读者规模消融（可选）**：Qwen3-4B 对比 Qwen3-32B 作读者，验证 PEP 和 NVIDIA 的"读者越大越好"。

**判读**：
- G-R1 和 G-R2 通过，说明 T2 档可以以"比文本分类器更好的旁路探针"对外销售。
- G-R1 不通过，说明读者探针只是文本分类器，T2 档应改用便宜的微调分类器，白盒的卖点只留在 T1。
- 无论结果如何，G-R4 的数字都要写进对外文档，用来说明闭源模型拿不到回复前预警。

---

## 参考文献（均于 2026-09-30 访问）

**API 文档**
- OpenAI API changelog（GPT-6 Astra 不支持 logprobs）：https://developers.openai.com/api/docs/changelog
- Azure OpenAI 推理模型文档：https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/reasoning
- OpenAI 社区"Logprobs deprecated for gpt-5 models?"：https://community.openai.com/t/logprobs-deprecated-for-gpt-5-models/1355427
- OpenAI Moderation 指南：https://developers.openai.com/api/docs/guides/moderation
- OpenAI 推理项 cookbook：https://developers.openai.com/cookbook/examples/responses_api/reasoning_items
- Anthropic Messages API 参考：https://platform.claude.com/docs/en/api/messages
- Anthropic extended thinking 文档：https://docs.claude.com/en/docs/build-with-claude/extended-thinking
- Gemini 3.x 不返回 logprobs（论坛）：https://discuss.ai.google.dev/t/missing-logprobs-support-in-the-newest-gemini-models-3-1-pro-3-6-flash-on-vertex-ai-and-ai-studio/176557
- Gemini logprobs（Vertex 博客）：https://developers.googleblog.com/unlock-gemini-reasoning-with-logprobs-on-vertex-ai/
- Gemini 安全设置：https://ai.google.dev/gemini-api/docs/safety-settings
- Gemini thinking：https://ai.google.dev/gemini-api/docs/thinking
- DeepSeek API 参考：https://api-docs.deepseek.com/api/create-chat-completion
- DeepSeek V3.2 logprobs issue：https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/issues/49
- 阿里 DashScope API 参考：https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-dashscope
- 火山方舟 Chat API（未能抓取）：https://www.volcengine.com/docs/82379/1494384

**不确定性 / 幻觉**
- Farquhar, Kossen, Kuhn, Gal. Detecting hallucinations in LLMs using semantic entropy. *Nature* 630, 625–630 (2024)：https://www.nature.com/articles/s41586-024-07421-0
- Kossen et al. Semantic Entropy Probes (2024)：https://arxiv.org/abs/2406.15927
- Manakul, Liusie, Gales. SelfCheckGPT (EMNLP 2023)：https://arxiv.org/abs/2303.08896
- Shrivastava et al. Llamas Know What GPTs Don't Show (2023)：https://arxiv.org/abs/2311.08877
- Xiong et al. Can LLMs Express Their Uncertainty? (ICLR 2024)：https://arxiv.org/abs/2306.13063
- Ulmer et al. APRICOT (ACL 2024)：https://aclanthology.org/2024.acl-long.824/
- Kapoor et al. LLMs Must Be Taught to Know What They Don't Know (NeurIPS 2024)：https://arxiv.org/abs/2406.08391
- Sriramanan et al. LLM-Check (NeurIPS 2024)：https://neurips.cc/virtual/2024/poster/95584
- Mrykhin & Malykh. Prompt Embedding Probes (2026)：https://arxiv.org/abs/2608.08024
- Ji et al. LLM Internal States Reveal Hallucination Risk Faced With a Query (BlackboxNLP 2024)：https://arxiv.org/abs/2407.03282
- Cencerrado et al. No Answer Needed (2025/2026)：https://arxiv.org/abs/2509.10625
- Orgad et al. LLMs Know More Than They Show (ICLR 2025)：https://arxiv.org/abs/2410.02707
- Varshney et al. (NVIDIA). LLM Router: Rethinking Routing with Prefill Activations (2026)：https://arxiv.org/abs/2603.20895

**欺骗 / 监控**
- Pacchiardi et al. How to Catch an AI Liar (ICLR 2024)：https://arxiv.org/abs/2309.15840
- Goldowsky-Dill et al. Detecting Strategic Deception Using Linear Probes (ICML 2025)：https://arxiv.org/abs/2502.03407
- Parrack et al. Benchmarking Deception Probes via Black-to-White Performance Boosts (2025)：https://arxiv.org/abs/2507.12691
- MacDiarmid et al. Simple probes can catch sleeper agents (Anthropic, 2024)：https://www.anthropic.com/research/probes-catch-sleeper-agents
- Baker et al. Monitoring Reasoning Models for Misbehavior (2025)：https://arxiv.org/abs/2503.11926
- Chen et al. Reasoning Models Don't Always Say What They Think (2025)：https://arxiv.org/abs/2505.05410
- Korbak et al. Chain of Thought Monitorability (2025)：https://arxiv.org/abs/2507.11473

**真值方向 / 读者表征**
- Marks & Tegmark. The Geometry of Truth (COLM 2024)：https://arxiv.org/abs/2310.06824
- Bürger, Hamprecht, Nadler. Truth is Universal (NeurIPS 2024)：https://arxiv.org/abs/2407.12831
- Azaria & Mitchell. The Internal State of an LLM Knows When It's Lying (2023)：https://arxiv.org/abs/2304.13734
- Kantamneni et al. Are Sparse Autoencoders Useful? (ICML 2025)：https://arxiv.org/abs/2502.16681
- Jiao et al. SIREN: LLM Safety From Within (2026)：https://arxiv.org/abs/2604.18519
- Li et al. Streaming Content Monitor (NeurIPS 2025)：https://arxiv.org/abs/2506.09996

**生产探针 / 可解释性产品**
- Cunningham et al. Cost-Effective Constitutional Classifiers via Representation Re-use (Anthropic, 2025)：https://alignment.anthropic.com/2025/cheap-monitors/
- Constitutional Classifiers++ (2026)：https://arxiv.org/abs/2601.04603
- Kramár et al. Building Production-Ready Probes For Gemini (2026)：https://arxiv.org/abs/2601.11516
- Goodfire × Rakuten SAE PII probes (2025-10-28)：https://www.goodfire.com/research/rakuten-sae-probes-for-pii-detection
- Goodfire Ember 发布：https://www.goodfire.ai/blog/announcing-goodfire-ember

**开源权重 / 许可**
- DeepSeek-V3.2 LICENSE：https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/LICENSE
- DeepSeek V4 开源报道：https://winbuzzer.com/2026/04/27/deepseek-v4-open-weights-launch-xcxwbn/
- Qwen3 博客：https://qwen.ai/blog?id=qwen3
- Qwen3.8 许可报道：https://finance.yahoo.com/technology/ai/articles/open-weights-closed-revenue-ceiling-051303188.html
- GLM-4.5：https://huggingface.co/zai-org/GLM-4.5
- GLM-5.3 许可：https://thenewstack.io/zai-glm-weights-license/
- Kimi K2 LICENSE：https://huggingface.co/moonshotai/Kimi-K2-Instruct/raw/main/LICENSE
- Kimi K3：https://simonwillison.net/2026/Jul/27/kimi-k3/
- MiniMax M2：https://github.com/MiniMax-AI/MiniMax-M2
- MiniMax M2.7 许可变更：https://letsdatascience.com/news/minimax-revises-license-after-releasing-m27-weights-04b47c74
- ERNIE 4.5 开源：https://ernie.baidu.com/blog/posts/ernie4.5/
- Seed-OSS：https://seed.bytedance.com/en/blog/seed-oss-open-source-models-release

**法规**
- EU AI Act 第 53 条：https://artificialintelligenceact.eu/article/53/
