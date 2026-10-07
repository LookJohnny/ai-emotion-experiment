# 文献综述：LLM 情绪表征的跨位置携带、注意力头定位与转向、跨模型迁移、读者模型

> 内部文件，保密（专利待申请）。撰写日期：2026-10-07。
> 核实方式：除特别标注外，每条引用都已通过 arXiv API / arXiv 摘要页核对标题、第一作者和编号（2026-10-07）。会议/期刊信息只在核实过或把握很高时才写，否则只写 arXiv 年份。〔未核实〕= 未能核对原文。
> 检索过程只用了通用关键词，没有向任何外部表单提交我们的方法细节。

---

## 1. LLM 中的情绪/情感表征

- **Tigges, Hollinsworth, Geiger, Nanda 等. "Linear Representations of Sentiment in Large Language Models." arXiv 2023, 2310.15154.**
  发现情感（正/负）主要由单一线性方向表示，有少量注意力头和神经元参与。另外提出 **summarization motif**：情感会被"汇总"到本身不带情感的中间位置（逗号、人名），消融逗号处的情感方向就会损失大量分类能力。
  *与我们的关系*：和 E01/E02 的"锚点携带"最接近。他们证明了情绪会被搬到中性位置，但用的是自然文本，没有做 token 完全相同的锚点对照；对象是二元情感，不是多类情绪；也没有拿这个现象去选头、做转向。

- **Tak, Banayeeanzade, Bolourani, Kian, Jia, Gratch. "Mechanistic Interpretability of Emotion Inference in Large Language Models." Findings of ACL 2025, arXiv 2502.05489.**
  覆盖 10 个模型（主模型 Llama-3.2-1B）。情绪信号在中层就基本定型（1B 模型在第 10/16 层达到峰值）；activation patching 显示**最后一个 token 上的中层 MHSA** 起决定作用；沿评价（appraisal）概念方向转向，结果符合心理学预期。
  *关系*："情绪在中层成形、注意力起关键作用"这一点和 E02 的峰值深度 0.43–0.56 一致，属于已知结论。他们研究的是情绪推断（分类决策，看最后一个 token），我们研究的是开头情绪被携带到生成文本里，并且做 head 级的定位和转向。

- **Wang, Zhang, Yu 等. "Do LLMs 'Feel'? Emotion Circuits Discovery and Control." arXiv 2510.11328（2025-10；ICML 2026 官网有 poster 页）.**
  构建 SEV 数据集：同一场景、同一参与者，只换结局，得到不同情绪的变体（内容受控）。**在每个场景组内减去跨情绪均值**，提取与上下文无关的情绪方向。神经元按解析式对齐分数排序 β=W_dᵀv_e；注意力头用**单头置零后情绪投影下降的幅度**排序（这是带符号的因果筛选）。之后做消融/增强验证，组装成全局回路。回路调制在测试集上的情绪表达准确率达 99.65%。主模型 LLaMA-3.2-3B-Instruct，附录验证了 Qwen2.5-7B，覆盖 6 类基本情绪。转向注入在 user 输入的最后一个 token。
  *关系*：**和我们最接近的情绪工作**。(i) 场景内去均值，和我们"场景内情绪方差占比"分数的思路同源；(ii) 先用解析式给组件打分，再做因果消融/增强验证，形式上已经是"相关性/解析排序 + 因果验证"；(iii) 用单头消融对头排序时天然带符号。差异：他们的头排序是对**所有头**做消融，没有先用廉价分数预筛再做小规模筛查；读出位置是 prompt 末尾，不是 token 完全相同的锚点段；没有 teacher forcing 下的等量推送筛查；没有跨尺寸按索引迁移。

- **Zhang, Zhong. "Decoding Emotion in the Deep: A Systematic Study of How LLMs Represent, Retain, and Express Emotion." arXiv 2510.04064（2025）.**
  用约 40 万条 Reddit 语料，对 Qwen3 和 LLaMA 逐层做探针。情绪信号很早出现、在网络中段达到峰值；情绪状态"可塑"（受系统提示影响），也"持久"：**开头的情绪基调在后续几百个 token 中仍可解码**。
  *关系*：直接覆盖了"情绪会被携带"这一定性结论。我们的增量在于用 token 完全相同的锚点做受控设计（排除内容混杂），以及量化"方差占比低但 AUROC 高"。

- **Sofroniew, Kauvar, Saunders 等. "Emotion Concepts and their Function in a Large Language Model." arXiv 2604.07729（2026, Anthropic, Claude Sonnet 4.5）.**
  发现情绪概念表征能跨情境泛化，并跟踪每个 token 位置上"当前起作用的情绪"。中前层编码当前内容的情绪色彩，中后层编码与预测后续文本相关的情绪。这些表征**因果地影响偏好以及奖励投机、勒索、谄媚等失准行为**。
  *关系*：位置追踪和"为预测后续而携带"与我们 E02 高度相关。更重要的是，他们证明情绪表征会因果地影响谄媚，这和我们"安全探针（敌意/编造/谄媚）与模型自身情绪方向正交"的结论**存在张力**。两者不一定矛盾（正交的是读出方向，不排除因果影响），但论文里必须正面讨论。

- **Hollowell. "Replicating the Geometry of Emotion Representations in a Base Open-Weights Model." arXiv 2609.22208（2026）.** 在 gemma-2-27b 上复现了 Sofroniew 等的情绪几何：PC1 为效价、呈环形结构；在第 22–26 层出现明显"接缝"；至少 52% 的向量峰值落在非概念 token 上。
  *关系*：给"效价分裂"提供了旁证，也提醒 max-activation 存在混杂。

- **van der Ben, Baur, Metz 等. "Where Do Models Find Happiness? Emotion Vectors in Open-Source LLMs." arXiv 2606.26987（2026）.** 在 Apertus-8B 和 Gemma-4-E4B 上复现效价几何，但效价随深度的走势在两个模型间相反。
  *关系*：说明"效价分裂从约 1/4 深度出现、三模型一致"不是普适规律，我们只能声称在所测的三个模型上成立。

- **Reichman, Avsian, Heck. "Emotions Where Art Thou: Understanding and Characterizing the Emotional Latent Space of Large Language Models." arXiv 2510.22042（PDF 页眉标注 ICLR 2026）.** 发现低维情绪流形，结构跨层稳定，并能泛化到 5 种语言、8 个数据集；用一个学习得到的干预模块转向情绪感知。
  *关系*：跨语言、分布式编码与我们一致；但他们的方法是学习式干预，不是挑选头。

- **Sun, Yan, Lu 等. "Valence-Arousal Subspace in LLMs: Circular Emotion Geometry and Multi-Behavioral Control." arXiv 2604.03147（2026）.** 在 Llama-3.1-8B、**Qwen3-8B、Qwen3-14B** 上找到效价–唤醒（VA）二维环形子空间；沿 VA 转向可以**双向控制拒答和谄媚**。
  *关系*：用了和我们 E03/E04 相同的模型；并且"情绪方向可以控制谄媚"同样与我们的正交结论有张力，需要讨论。

- **Venkatesh. "Negative Before Positive: Asymmetric Valence Processing in Large Language Models." arXiv 2605.05653（2026）.** 用 patching 和转向发现，负效价在浅层定位，正效价在中后层达到峰值；固定话题只翻转效价时，得到符号相反的响应。
  *关系*：效价不对称的直接先例。我们的"悲伤离愤怒比离喜悦更近"属于同类发现，只能算佐证，不是新发现。

- **Wang, Dang, Wang 等. "Read-Best Is Not Steer-Best: A Probing–Steering Layer Dissociation in Omni-Modal Large Language Models." arXiv 2609.22135（2026）.** 以情绪为测试平台，发现**探针最佳层 ≠ 转向最佳层**；转向有效层集中在归一化深度的中后段；与随机方向对照相比有约 26 倍的因果差距。
  *关系*：在**层**粒度上预示了我们"相关性分数找得到相关头，却定不了符号/效用"（E03）的结论；我们的贡献在**头**粒度和符号问题上。

- **Liu 等（2502.04075；v1 标题为 "Controllable Emotion Generation with Emotion Vectors"，v2 改为 "From Rational Answers to Emotional Resonance: The Role of Controllable Emotion Generation in Language Models"）arXiv 2025.**〔第一作者未在 API 输出中核对〕 用中性回答与情绪化回答之间的激活差得到逐层情绪向量，推理时注入隐藏状态。
  *关系*：情绪的残差流级转向，不涉及选头。

- **Konen, Jentzsch, Diallo 等. "Style Vectors for Steering Generative Large Language Models." arXiv 2402.01618（2024）.** 用激活差得到情绪/风格向量来转向生成。另有人类评估跟进工作 **"The Effectiveness of Style Vectors for Steering Large Language Models: A Human Evaluation." arXiv 2601.21505（2026）**：中等强度能可靠增强目标情绪。
  *关系*：情绪转向的早期基线，可以作为对照组。

- **"Extracting and Steering Emotion Representations in Small Language Models: A Methodological Comparison." arXiv 2604.04064（2026）.**〔作者未核对〕 覆盖 9 个小模型，发现情绪表征集中在约 50% 深度，呈 U 形且跨架构不变；转向效果分为"精确 / 重复崩溃 / 爆炸"三种状态。
  *关系*：峰值深度结论与我们 E02 的 0.43–0.56 几乎重合，属于已知结论。

- **跨语言 / 中文**："AI shares emotion with humans across languages and cultures" arXiv 2506.13978（2025），从多语言 LLM 中提取覆盖中英文的情绪概念特征，用作转向向量；"Multilingual Emotion Neurons in Large Audio-Language Models" arXiv 2608.08772（2026），发现按单语言分别找出的情绪神经元重叠极少。
  *关系*：**我们没有找到以中文为主要语料、在 head 级做情绪因果定位和转向的论文**。中文设定本身是小的差异点，但学术上不算强新颖性。

## 2. 转向方法

- **Li, Patel, Viégas, Pfister, Wattenberg. "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model." NeurIPS 2023, arXiv 2306.03341.**
  在每个头的输出上训练线性探针，**按探针验证准确率选 top-K 头**（纯相关性），沿均值差 / 探针方向偏移，偏移作用在**生成的每一步**。
  *关系*：head 级、每个 token 持续注入的直接先例（专利点 c）。它的选头方式正是我们所说的 map-only 基线；我们 E03 的失败可以看作 ITI 式选头失效的一个案例。

- **Hoscilowicz 等. "Non-Linear Inference Time Intervention." arXiv 2403.18680（2024）；Chen 等. "Truth Forest." arXiv 2312.17484（2023）.** 都是 ITI 的变体，选头仍基于探针。

- **Panickssery（Rimsky）, Gabrieli, Schulz 等. "Steering Llama 2 via Contrastive Activation Addition." ACL 2024, arXiv 2312.06681.** 用残差流的均值差向量，加在用户 prompt 之后**所有 token 位置**上。
  *关系*：持续注入 + 均值差的先例。

- **Zou, Phan, Chen 等. "Representation Engineering: A Top-Down Approach to AI Transparency." arXiv 2310.01405（2023）.** 提出 RepE 的读出（探针/PCA）与控制框架，包含情绪等概念。
  *关系*：同时覆盖"读出 + 控制"的总体框架。

- **Turner, Thiergart, Leech 等. "Steering Language Models With Activation Engineering"（ActAdd）. arXiv 2308.10248（2023）.** 用单对 prompt 的差值，在前几个位置注入。
  *关系*：属于"只在前段注入"一类。我们发现"只在锚点注入一次无效"，与 ActAdd 式前缀注入的局限相呼应。

- **Todd, Li, Sen Sharma 等. "Function Vectors in Large Language Models." ICLR 2024, arXiv 2310.15213.** 对每个头做 causal mediation，按**平均间接效应（AIE）**选 top 头（约 10 个），把这些头的平均输出相加得到 FV。
  *关系*：**用因果效应选头的经典先例**。和我们 E04 第二阶段（单头因果筛查）的思想同源。区别：他们对所有头做因果打分，没有相关性预筛；任务是 ICL，不是生成转向；也没有"等量推送、只保留显著为正"的符号筛查。

- **Hendel, Geva, Globerson. "In-Context Learning Creates Task Vectors." arXiv 2310.15916（2023）；Liu 等. "In-context Vectors." arXiv 2311.06668（2023）.** 都是任务/上下文向量类工作。

- **Darm, Riccardi. "Head-Specific Intervention Can Induce Misaligned AI Coordination in Large Language Models." arXiv 2502.05945（TMLR 2025）.**
  **逐个头单独施加干预，在二选一任务上测转向效果**，得到每个头的"准确率图"，再挑出 1–4 个头。没有预筛。
  *关系*：**和我们第二阶段（单头转向筛查）高度相似**。区别：他们是全量扫描，没有相关性预筛；指标是二选一准确率，不是 teacher forcing 下的 logit/得分，也没有等量推送和显著性检验。

- **Sankaranarayanan, Zur, Geiger, Hadfield-Menell. "Activation Steering via Generative Causal Mediation"（GCM）. arXiv 2602.16080（2026）.**
  从对比的长文本回答出发，用 activation patching（也评估了 attribution patching 近似）给每个头算间接效应，**选 top-k% 头做均值差转向**。在拒答、谄媚、风格任务上**优于 ITI 式探针选头**，模型包括 Qwen-14B 等。
  *关系*：**"因果选头 > 相关性选头"已被这篇论文公开证明**，包括在 14B 级模型上。我们 E04 中"two-stage 优于 map-only"的定性结论已不新。我们剩下的差异点：相关性预筛（top 60）+ 小规模单头符号筛查的组合与成本控制；锚点打分。

- **Yin, Ye, Durrett. "LoFiT: Localized Fine-tuning on LLM Representations." NeurIPS 2024, arXiv 2406.01563.** 先学习头的缩放因子来选头，再在选中头上学习偏置。
  **Lai, Fraser, Titov. "Joint Localization and Activation Editing for Low-Resource Fine-Tuning"（JoLA）. arXiv 2502.01179（2025）.** 用门控同时学习选哪些头和怎么编辑。
  *关系*：学习式选头，属于另一路线。

- **Nam, Conklin, Yang 等. "Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers." arXiv 2505.13737（2025）.** 用可学习的门把头分为促进 / 干扰 / 无关三类。
  *关系*：**"头有相反因果符号"的直接先例**，可用来解释 E03 的抵消现象。

- **"Steering at the Source: Style Modulation Heads for Robust Persona Control." arXiv 2603.13249（2026；ICLR/ICML 2026 页面）.**〔作者未核对〕 结合逐层余弦相似度和逐头贡献分（几何/相关性）找到 3 个"风格调制头"，只干预这些头，连贯性优于残差流转向。
  *关系*：相关性几何选头 + head 级转向，同属一类，但没有因果符号筛查。

- **"LocUS: Head Selection and Subspace Projection for Targeted Activation Steering." arXiv 2609.31122（2026）.** 按头与输出词表子空间的对齐程度选头，任务含情感重定向。
  **"REAL: Reading Out Transformer Activations for Precise Localization in Language Model Steering." arXiv 2506.08359（2025）.** 按 VQ-AE 编码的判别度选头。
  **Wang, Yang, Peng. "Semantics-Adaptive Activation Intervention for LLMs via Dynamic Steering Vectors"（SADI）. arXiv 2410.12299（2024）.**
  **"Sycophancy Hides Linearly in the Attention Heads." arXiv 2601.16644（2026）.** 谄媚信号在中层少数头上线性可分，这些头最适合转向。
  *关系*：都是相关性/判别度选头的变体，说明"相关性打分选头"已是拥挤的领域。

- **副作用与可靠性**：Tan, Chanin, Lynch 等. "Analyzing the Generalization and Reliability of Steering Vectors." arXiv 2407.12404（2024）；Wu, Arora, Geiger 等. "AxBench." arXiv 2501.17148（2025）；Im, Li. "A Unified Understanding and Evaluation of Steering Methods." arXiv 2502.02716（2025）；Jorgensen 等. "Mean-Centring." arXiv 2312.03813（2023）；Chalnev, Siu, Conmy. "SAE-TS." arXiv 2411.02193（2024）；"A Coin Flip Per Token: Bernoulli Sparse Steering." arXiv 2607.05615（2026），每个 token 以概率 p 决定是否转向，约 50% 的 token 就能恢复大部分效果，说明效果取决于累积"剂量"。
  *关系*：最后一篇直接讨论了"每个 token 都注入 vs 稀疏注入"，和我们"只在锚点注入一次无效、持续注入有效"同属剂量问题，削弱了专利点 (c)。

- **Chen, Arditi, Sleight 等. "Persona Vectors: Monitoring and Controlling Character Traits in Language Models." arXiv 2507.21509（2025）.** 用自动流程提取谄媚、幻觉、恶意等特质的方向，同时用于**监测和转向**。
  **Arditi 等. "Refusal in Language Models Is Mediated by a Single Direction." arXiv 2406.11717（2024）.**
  *关系*：安全相关方向的提取和监测；与 Nous 的探针相关。

## 3. 因果定位的廉价近似；探针与因果的差异

- **Nanda. "Attribution Patching: Activation Patching At Industrial Scale."（博客, 2023）**〔非论文，未核实具体 URL〕。用一阶泰勒展开近似 patching。
- **Kramár, Lieberum, Shah, Nanda. "AtP*: An efficient and scalable method for localizing LLM behaviour to components." arXiv 2403.00745（2024, DeepMind）.**
  原文明确写道：先用 AtP 作为**预筛步骤（prefiltering step）**，再按 |AtP 估计| 从大到小对节点做真正的 activation patching **验证（verification）**，过滤假阳性；用 "cost of verified recall" 衡量。
  *关系*：**这是专利点 (a)"廉价预筛 + 小规模因果筛查"最接近的现有技术**，已经公开了"廉价排序 → 只对头部候选做因果验证"的两阶段框架。差异：AtP 是基于梯度的因果近似，不是前向的相关性/方差分数；验证的目的是电路发现（效应幅度），不是为转向选择**带正号**的头；没有等量推送和显著性筛查；没有锚点。
- **Syed, Rager, Conmy. "Attribution Patching Outperforms Automated Circuit Discovery"（EAP）. arXiv 2310.10348（2023）；Hanna, Pezzelle, Belinkov. "Have Faith in Faithfulness"（EAP-IG）. arXiv 2403.17806（2024）；Conmy 等. ACDC. arXiv 2304.14997（NeurIPS 2023）；Marks 等. "Sparse Feature Circuits." arXiv 2403.19647（2024）.**
  *关系*：都是用廉价近似做大规模组件打分。
- **Michel, Levy, Neubig. "Are Sixteen Heads Really Better than One?" NeurIPS 2019, arXiv 1905.10650**（基于梯度的头重要性）；**Voita 等. "Analyzing Multi-Head Self-Attention..." ACL 2019, arXiv 1905.09418.**
- **Zhang, Nanda. "Towards Best Practices of Activation Patching..." arXiv 2309.16042（ICLR 2024）；Heimersheim, Nanda. "How to use and interpret activation patching." arXiv 2404.15255（2024）；Goldowsky-Dill 等. "Localizing Model Behavior with Path Patching." arXiv 2304.05969（2023）.**
- **探针与因果的差异**：Hewitt, Liang. "Designing and Interpreting Probes with Control Tasks." EMNLP 2019, arXiv 1909.03368；Ravichander, Belinkov, Hovy. "Probing the Probing Paradigm..." EACL 2021, arXiv 2005.00719；Elazar 等. "Amnesic Probing." arXiv 2006.00995（TACL 2021）；Belinkov. "Probing Classifiers: Promises, Shortcomings, and Advances." arXiv 2102.12452（Computational Linguistics）；以及 §1 的 "Read-Best Is Not Steer-Best"（2609.22135）。
  *关系*：经典文献已经指出"探针能读出的信息不一定被模型使用"。我们 E02 的"AUROC ≥0.99 但方差占比只有 3–5%"和 E03 的"分数能找到相关头，却给不出符号"都是这一观点的具体实例。
- **头的符号相反**：Wang 等. IOI 回路 arXiv 2211.00593（ICLR 2023）中的 negative name mover heads；McDougall 等. "Copy Suppression." arXiv 2310.04625（2023）；以及 Causal Head Gating（2505.13737）。
  *关系*：E03 中"正负头相互抵消"的机制早有先例，因此 E03 的现象不新。新的是我们把它作为两阶段方法的动机，并做了实证。

**关于"相关性预筛 + 小规模因果筛查"的检索结论（如实汇报）**：
1. **AtP\***：预筛 + 因果验证，框架层面最接近。
2. **Wang 等 2025 情绪回路**：解析/相关排序加因果消融验证，并用单头消融带符号排序。领域（情绪）最接近。
3. **Darm & Riccardi（HSI）**：单头转向逐一扫描。第二阶段的形式最接近。
4. **GCM**：用因果间接效应选头做转向，优于探针选头。
5. **Function Vectors**：用 AIE 选头。

我们**没有找到**完全相同的组合，即"前向相关性分数（场景内方差占比）预筛 top-N → teacher forcing 下对每个候选头做等量推送的单头转向符号筛查 → 只保留显著为正的头 → 联合持续转向"。但上面 1+3（或 2+3）的组合会构成很强的**显而易见性**论据。

## 4. 内容受控 / 最小对设计；跨位置的信息携带

- **Vig, Gehrmann, Belinkov 等. "Causal Mediation Analysis for Interpreting Neural NLP: The Case of Gender Bias." NeurIPS 2020, arXiv 2004.12265.** 最早用反事实最小对对神经元和头做中介分析。
- **Meng, Bau, Andonian, Belinkov. "Locating and Editing Factual Associations in GPT"（ROME / causal tracing）. NeurIPS 2022, arXiv 2202.05262.**
- **Geiger 等. "Causal Abstractions of Neural Networks." arXiv 2106.02997（NeurIPS 2021）；Geiger 等. "Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations"（DAS）. arXiv 2303.02536.** 互换干预（interchange intervention）的理论基础。
- **Kaushik, Hovy, Lipton. "Learning the Difference that Makes a Difference with Counterfactually-Augmented Data." ICLR 2020, arXiv 1909.12434.** 构造最小改动的反事实情感样本。
- **Ravfogel 等. INLP. arXiv 2004.07667（ACL 2020）；Belrose 等. LEACE. arXiv 2306.03819（NeurIPS 2023）.** 线性概念擦除，可以作为"情绪方向是否被使用"的检验工具。
- **信息跨位置搬运**：Geva, Bastings, Filippova, Globerson. "Dissecting Recall of Factual Associations in Auto-Regressive Language Models." arXiv 2304.14767（EMNLP 2023），发现注意力把属性信息搬到最后位置；Wang 等. "Label Words are Anchors: An Information Flow Perspective for Understanding In-Context Learning." arXiv 2305.14160（EMNLP 2023），发现 ICL 的信息聚合在标签词这个"锚点"上；Tigges 2023 的 summarization motif；Pal 等. "Future Lens." arXiv 2311.04897（2023）；Wu 等. "Do language models plan ahead for future tokens?" arXiv 2404.00859（2024）。
  *关系*："锚点"一词和"信息汇聚到中性位置"已有先例（Label Words are Anchors、summarization motif）。我们的差异是**刻意让锚点段的 token 完全相同**，使锚点上的情绪差异只能来自前文经注意力的携带。这是一种设计上的受控，用来分离"携带"和"局部内容"。我们没有找到把"token 完全相同的续写段"作为标准探测/打分位点的先例，但它与 IOI 式的模板化最小对、SEV 的同场景多结局非常接近，单独作为方法新颖性偏弱。

## 5. 跨模型迁移（转向向量 / 头 / 探针）

- **Oozeer, Nathawani, Prakash 等. "Activation Space Interventions Can Be Transferred Between Large Language Models." arXiv 2503.04429（2025）.** 学习激活空间之间的映射（自编码器），把拒答、后门移除等转向向量在 Llama/Qwen/Gemma 之间迁移，可以用小模型对齐大模型。
- **"Linear Representation Transferability Hypothesis: Leveraging Small Models to Steer Large Models." arXiv 2506.00653（2025）.**〔作者未核对〕 用小→大模型的仿射映射迁移转向向量。
- **"Transferring Linear Features Across Language Models With Model Stitching." arXiv 2506.06609（2025）.** 用仿射 stitching 在同族不同尺寸模型间迁移 SAE、探针和转向向量。
- **"Cross-model Transferability among Large Language Models on the Platonic Representations of Concepts." arXiv 2501.02009（2025）.** 用线性变换对齐转向向量，有弱→强迁移。
- **Lee, Weber, Viégas 等. "Shared Global and Local Geometry of Language Model Embeddings." arXiv 2503.21073（2025）.** 提出 EMB2EMB，线性迁移转向向量。
- **"The Master Key Hypothesis..." arXiv 2604.06377（2026）.** 免训练、免标签，用低秩线性对齐迁移能力方向，例如 Qwen3-4B-Base → Qwen3-14B-Base。
- **"Cross-Architecture Steering Transfer in Language Models: A Systematic Empirical Study." arXiv 2608.05164（2026）.** 约 1.7B 是迁移能否成功的阈值。
- **"Signed-Permutation Coordinate Transport for RMSNorm Transformers." arXiv 2606.31963（2026）.** 指出对 RMSNorm 模型，迁移坐标索引对象（top-k 神经元集合、转向向量）必须考虑带符号置换；仅用置换对齐时，拒答转向会反号。
  *关系*：与 E05"按索引迁移头"在概念上相邻，但它研究的是同一 base 的微调检查点之间、残差通道坐标的迁移。
- **普适性**：Olsson 等. "In-context Learning and Induction Heads." arXiv 2209.11895（2022）；Gurnee 等. "Universal Neurons in GPT2 Language Models." arXiv 2401.12181（2024）；Chughtai, Chan, Nanda. "A Toy Model of Universality." arXiv 2302.03025（ICML 2023）；Tigges, Hanna 等. "LLM Circuit Analyses Are Consistent Across Training and Scale." arXiv 2407.10827（2024）；Wu 等. "Retrieval Head Mechanistically Explains Long-Context Factuality." arXiv 2404.15574（2024），发现继续预训练后**仍是同一组检索头**；Prakash 等. "Fine-Tuning Enhances Existing Mechanisms." arXiv 2402.14811（2024），发现微调前后是同一回路；Huh 等. "The Platonic Representation Hypothesis." arXiv 2405.07987（ICML 2024）；Lan 等. "Quantifying Feature Space Universality Across LLMs via SAEs." arXiv 2410.06981；Bansal, Nakkiran, Barak. "Revisiting Model Stitching." arXiv 2106.07682（NeurIPS 2021）；Kornblith 等. CKA. arXiv 1905.00414（ICML 2019）。
- **背景**：Qwen3 Technical Report, arXiv 2505.09388（2025）称小尺寸模型利用旗舰模型知识（蒸馏）构建。这可能是 Qwen3-4B 与 8B 头索引能对应的原因，属于**机制性解释，不是现有技术**。
  *关系*：迁移转向向量 / 探针 / SAE 的工作（多数通过学习映射）已很多。**我们没有找到"在小模型里按 (layer, head) 选出的头，直接按索引用到同族大模型，并与大模型原生选出的头做非劣效比较"的先例**；相关工作大多强调"索引不可靠，需要做相似度匹配"。所以 E05 有一定新意，但在同族、层数和头数相同的条件下，按索引迁移是本领域技术人员很容易想到的尝试。

## 6. 用代理/读者模型读取另一模型的输出；谄媚 / 幻觉探针

- **Sriramanan, Bharti, Sadasivan, Saha, Kattakinda, Feizi. "LLM-Check: Investigating Detection of Hallucinations in Large Language Models." NeurIPS 2024**（NeurIPS proceedings 页已核实）.
  在**黑盒设定下，用一个辅助 LLM 的隐藏状态、注意力图和输出概率**来检测单条回答中的幻觉。
  *关系*：**Nous"本地开源模型在 teacher forcing 下读闭源模型输出"的最直接先例**（幻觉方向）。我们的差异：用多个线性探针同时读出敌意、编造、谄媚三项；以及无标签的跨模型探针迁移。
- **"Prompt Embedding Probes (PEP): Hallucination Detection in LLMs from Hidden States." arXiv 2608.08024（2026）.** 包含**跨模型设定**：大的检测器模型读小的生成器模型的错误，跨族也有效。
- **Azaria, Mitchell. "The Internal State of an LLM Knows When It's Lying"（SAPLMA）. arXiv 2304.13734（Findings of EMNLP 2023）**，对喂入的陈述（不一定是本模型生成的）用探针判断真假；**Marks, Tegmark. "The Geometry of Truth." arXiv 2310.06824**；**Bürger 等. "Truth is Universal." arXiv 2407.12831**；**Burns 等. CCS. arXiv 2212.03827（ICLR 2023）**。
  *关系*：用模型读取"外部给定文本"的真伪，本质上就是读者模型范式。
- **CH-Wang, Van Durme, Eisner, Kedzie. "Do Androids Know They're Only Dreaming of Electric Sheep?" arXiv 2312.17249（Findings of ACL 2024）**，幻觉探针，跨任务泛化差；**Chen 等. INSIDE. arXiv 2402.03744（ICLR 2024）**；**Kossen 等. "Semantic Entropy Probes." arXiv 2406.15927**；**Chuang 等. "Lookback Lens." arXiv 2407.07071**（基于注意力图，可跨模型迁移）；**FactCheckmate. arXiv 2410.02899**；**DRIFT. arXiv 2601.14210**。
- **谄媚**：Sharma 等. "Towards Understanding Sycophancy in Language Models." arXiv 2310.13548（ICLR 2024）；Papadatos, Freedman. "Linear Probe Penalties Reduce LLM Sycophancy." arXiv 2412.00967；"Sycophancy Hides Linearly in the Attention Heads." arXiv 2601.16644。
- **探针监测**：Goldowsky-Dill 等. "Detecting Strategic Deception Using Linear Probes." arXiv 2502.03407（2025）；"Detecting High-Stakes Interactions with Activation Probes." arXiv 2506.10805（2025），探针作为廉价初筛加分级监测；"Constitutional Classifiers++." arXiv 2601.04603（2026, Anthropic），线性探针与分类器级联；"Unread or Unenforced?..." arXiv 2609.26178（2026），读取 guard 模型自身残差流、把探针迁移到编码后的条件。
  *关系*：这些工作大多读**被监测模型自身**的激活；读者是**另一个**模型时，先例主要是 LLM-Check 和 PEP。机器生成文本检测（如 Hans 等. "Binoculars." arXiv 2401.12070）也是"用代理模型的统计量读他人文本"，但用的是 logits，不是探针。

---

## 与我们工作的区别（论文定位）

**很可能已知 / 只能算复现或佐证的：**
1. 情绪/情感线性可解码、在中层达到峰值（Tak 2025；Zhang & Zhong 2025；2604.04064 约 50% 深度；Tigges 2023）。E02 的峰值深度 0.43–0.56 与此一致，**不新**。
2. 情绪会被携带到后续 token 和中性位置（Tigges 的 summarization motif；Zhang & Zhong "持续数百 token"；Sofroniew 2026 的位置追踪）。定性结论**不新**。
3. 情绪是分布式编码，消融少数神经元无效（Reichman 2025 的分布式流形；"probing ≠ use" 系列）。不过 Wang 2025 报告少量组件就能主导情绪表达，所以我们的"神经元消融无效"与他们部分相反，值得讨论，但不算新现象。
4. 效价不对称 / 效价结构（Venkatesh 2026；VA 环形几何 2604.03147；Sofroniew 2026）。"悲伤更靠近愤怒"属于佐证。
5. 因果选头优于相关性选头（GCM 2026；Function Vectors；Read-Best≠Steer-Best 的层级版本）。E04 的"two-stage 优于 map-only"定性结论已被预示。
6. 头的因果符号相反、会相互抵消（negative mover heads、copy suppression、Causal Head Gating）。E03 的机制解释不新。
7. 持续（每个 token）注入优于一次性注入（ITI/CAA 默认每 token 注入；Bernoulli 稀疏转向的"剂量"结论）。
8. 读者模型读外部文本做幻觉检测（LLM-Check；PEP；SAPLMA）。
9. 探针 / 转向向量用线性映射跨模型迁移（Oozeer；LRT；stitching；Master Key）。

**相对较新、可以作为贡献的（需措辞谨慎）：**
1. **token 完全相同的锚点设计**：作为"携带情绪"的受控度量位点，并在 3 个模型上给出"方差占比 3–5% 但 AUROC ≥0.99"的量化（信息在、但占比小）。
2. **闭式的逐头锚点打分**（场景内情绪方差占比）：在 4B 上召回 patching 找到的因果头（top 10% 中命中 8/12），并且有反例：8B 上按分数直接转向会失败。"便宜分数能找到相关头、但定不了符号"作为一个**完整的正负结果对**，在情绪领域没有看到先例。
3. **两阶段选头**（相关性预筛 + teacher forcing 下的等量推送单头符号筛查）在 14B 上通过预注册门槛，并做了消融（优于只用分数、只用筛查）。**预注册加封存测试集的严谨性**本身是方法学亮点。
4. **同族按 (layer, head) 索引迁移选头，并与目标模型原生选头做非劣效比较**（E05）：没有找到直接先例。
5. **安全探针方向与模型自身情绪方向正交**：有一定新意，但 Sofroniew 2026 和 2604.03147 显示情绪方向能**因果影响**谄媚。必须把"读出方向正交"和"因果影响"区分开讨论，否则审稿人会认为与已有结论冲突。
6. **Nous 的无标签跨模型探针迁移**（用约 300 条锚点文本做 ridge 对齐，迁移敌意/编造/谄媚探针，读者模型读闭源模型输出）：读者模型和线性对齐分别有先例，但**组合起来用于多维安全读出**的应用较新，学术新颖性中等。

## 专利新颖性风险（需代理人复核）

> 以下是研究者视角的初步判断，不构成法律意见。

**(a) 两阶段头选择（相关性预筛 → 单头因果符号筛查 → 只保留显著为正的头 → 联合转向）——风险：高（宽权利要求）/ 中（窄权利要求）**
- 最接近的现有技术：
  - **AtP\***（2403.00745）明确公开了"廉价估计预筛 → 按排序对候选做真实因果验证"的两阶段流程。
  - **Darm & Riccardi HSI**（2502.05945）公开了"逐个头单独转向、按效果选头"。
  - **Wang 等 2025 情绪回路**（2510.11328）公开了"解析/相关排序 + 因果消融验证"，并用带符号的单头消融给情绪头排序，而且领域就是情绪。
  - **GCM**（2602.16080）公开了"按因果中介效应选头做均值差转向，优于探针选头"。
  - **Function Vectors**（AIE 选头）、**ITI**（探针选头）。
- 判断：如果权利要求写成"先用低成本指标给头排序，再对候选做因果检验，选出用于转向的头"，会被 AtP\* + HSI/GCM 的组合直接预示，或被认定为显而易见。可能还站得住的窄化特征：①预筛指标是**锚点处、场景内的情绪方差占比**这种前向闭式量（不用梯度）；②筛查是在 **teacher forcing 下对每个候选施加等幅推送**，用**显著性检验只保留正向**头，明确把剔除反号头作为步骤；③预筛规模与筛查规模的具体组合；④有"仅用分数会失败（反号抵消）"的实验支撑，可以论证技术效果。即便如此，仍可能被认为是"已知要素的常规组合"。

**(b) 基于锚点的携带分数（token 完全相同的锚点段 + 逐头场景内情绪方差占比）——风险：中**
- 最接近的现有技术：Wang 2025 的"场景组内去均值"情绪方向提取和 SEV 同场景多结局设计；Tigges 2023 的 summarization motif（情绪汇聚到中性位置）；"Label Words are Anchors"（锚点信息流）；ITI / REAL / Style Modulation Heads 等各种逐头相关/判别分数；方差分解 / ANOVA 类指标属于通用统计方法。
- 判断：我们没有找到"在前文情绪不同、后续 token 完全相同的锚点段上，计算每个头输出中情绪解释的场景内方差占比"作为选头分数的先例。这一具体组合可能具备新颖性；但"同场景控制 + 逐头相关性分数"的各个要素都已公开，**创造性**会受到挑战。建议在权利要求中把"token 完全相同的锚点段"和"在锚点读出"写成必要技术特征。

**(c) 在锚点和每个生成 token 上持续注入——风险：高**
- 最接近的现有技术：ITI（在选中头上，生成的每一步都偏移）；CAA（加在 prompt 之后的所有位置）；"所有输出 token"已是标准注入模式（见 Bernoulli 稀疏转向 2607.05615 中对比的 dense 基线）；GCM 在生成序列的相应位置施加干预。
- 判断：**基本被预示**。"一次性注入无效"是有价值的实验发现，但不能成为可授权的方法特征。不建议作为独立权利要求，最多作为从属权利要求中的实施方式。

**(d) 同族按 (layer, head) 索引迁移选出的头——风险：中低（新颖性）/ 中（创造性）**
- 最接近的现有技术：转向向量 / 探针 / SAE 的跨模型迁移（Oozeer 2503.04429；LRT 2506.00653；stitching 2506.06609；Master Key 2604.06377，其中就有 Qwen3-4B→14B）；微调或继续预训练前后头保持一致（检索头 2404.15574、Prakash 2402.14811、Tigges 2407.10827）；坐标索引对象迁移的规范性问题（2606.31963）。
- 判断：我们没有找到"小模型选出的头集合按索引直接用于同族更大模型、并与原生选头非劣效"的公开先例，**新颖性风险较低**。但 Qwen3-4B 与 8B 层数和每层头数相同，"按索引照搬"是很自然的尝试，审查员可能以"显而易见的尝试"驳回。可以用以下几点支撑：非劣效的实证结果、成本节省（只在小模型上做筛查），以及"现有文献普遍认为索引不可靠、需要相似度匹配"这一反向教导（需代理人查找书面证据）。

**其他提醒**
- Nous 的读者模型 + 多探针安全读出：LLM-Check（NeurIPS 2024）和 PEP（2608.08024）构成读者模型范式的先例，Persona Vectors 和 deception probes 构成探针监测的先例。"用约 300 条锚点文本做 ridge 对齐、无标签迁移探针"与 Oozeer / LRT / stitching 的线性映射迁移相近。**风险：中高**，只能靠具体实现细节窄化。
- 请代理人另行检索**专利数据库**（USPTO/CNIPA/EPO）。本综述只覆盖学术文献，2026 年下半年的新预印本更新很快，申请前需要再检索一次。

---

## 参考文献

1. Tigges et al. Linear Representations of Sentiment in LLMs. arXiv:2310.15154 (2023)
2. Tak et al. Mechanistic Interpretability of Emotion Inference in LLMs. Findings of ACL 2025, arXiv:2502.05489
3. Wang et al. Do LLMs "Feel"? Emotion Circuits Discovery and Control. arXiv:2510.11328 (2025)
4. Zhang, Zhong. Decoding Emotion in the Deep. arXiv:2510.04064 (2025)
5. Sofroniew et al. Emotion Concepts and their Function in a Large Language Model. arXiv:2604.07729 (2026)
6. Hollowell. Replicating the Geometry of Emotion Representations in a Base Open-Weights Model. arXiv:2609.22208 (2026)
7. van der Ben et al. Where Do Models Find Happiness? arXiv:2606.26987 (2026)
8. Reichman et al. Emotions Where Art Thou. arXiv:2510.22042 (ICLR 2026)
9. Sun et al. Valence-Arousal Subspace in LLMs. arXiv:2604.03147 (2026)
10. Venkatesh. Negative Before Positive. arXiv:2605.05653 (2026)
11. Wang et al. Read-Best Is Not Steer-Best. arXiv:2609.22135 (2026)
12. From Rational Answers to Emotional Resonance (v1: Controllable Emotion Generation with Emotion Vectors). arXiv:2502.04075 (2025)
13. Konen et al. Style Vectors for Steering Generative LLMs. arXiv:2402.01618 (2024)
14. The Effectiveness of Style Vectors for Steering LLMs: A Human Evaluation. arXiv:2601.21505 (2026)
15. Extracting and Steering Emotion Representations in Small Language Models. arXiv:2604.04064 (2026)
16. AI shares emotion with humans across languages and cultures. arXiv:2506.13978 (2025)
17. Multilingual Emotion Neurons in Large Audio-Language Models. arXiv:2608.08772 (2026)
18. Li et al. Inference-Time Intervention. NeurIPS 2023, arXiv:2306.03341
19. Hoscilowicz et al. Non-Linear ITI. arXiv:2403.18680
20. Chen et al. Truth Forest. arXiv:2312.17484
21. Panickssery (Rimsky) et al. Steering Llama 2 via CAA. ACL 2024, arXiv:2312.06681
22. Zou et al. Representation Engineering. arXiv:2310.01405
23. Turner et al. Steering Language Models With Activation Engineering. arXiv:2308.10248
24. Todd et al. Function Vectors in LLMs. ICLR 2024, arXiv:2310.15213
25. Hendel et al. In-Context Learning Creates Task Vectors. arXiv:2310.15916
26. Liu et al. In-context Vectors. arXiv:2311.06668
27. Darm, Riccardi. Head-Specific Intervention... arXiv:2502.05945 (TMLR 2025)
28. Sankaranarayanan et al. Activation Steering via Generative Causal Mediation. arXiv:2602.16080 (2026)
29. Yin et al. LoFiT. arXiv:2406.01563 (NeurIPS 2024)
30. Lai et al. JoLA. arXiv:2502.01179
31. Nam et al. Causal Head Gating. arXiv:2505.13737
32. Steering at the Source: Style Modulation Heads. arXiv:2603.13249 (2026)
33. LocUS. arXiv:2609.31122 (2026)
34. REAL. arXiv:2506.08359 (2025)
35. Wang et al. SADI. arXiv:2410.12299
36. Sycophancy Hides Linearly in the Attention Heads. arXiv:2601.16644 (2026)
37. Tan et al. Generalization and Reliability of Steering Vectors. arXiv:2407.12404
38. Wu et al. AxBench. arXiv:2501.17148
39. Im, Li. Unified Understanding and Evaluation of Steering Methods. arXiv:2502.02716
40. Jorgensen et al. Mean-Centring. arXiv:2312.03813
41. Chalnev et al. SAE-TS. arXiv:2411.02193
42. A Coin Flip Per Token: Bernoulli Sparse Steering. arXiv:2607.05615 (2026)
43. Chen et al. Persona Vectors. arXiv:2507.21509
44. Arditi et al. Refusal Is Mediated by a Single Direction. arXiv:2406.11717
45. Nanda. Attribution Patching (blog, 2023) 〔未核实〕
46. Kramár et al. AtP*. arXiv:2403.00745 (2024)
47. Syed et al. Attribution Patching Outperforms ACDC. arXiv:2310.10348
48. Hanna et al. Have Faith in Faithfulness. arXiv:2403.17806
49. Conmy et al. ACDC. arXiv:2304.14997
50. Marks et al. Sparse Feature Circuits. arXiv:2403.19647
51. Michel et al. Are Sixteen Heads Really Better than One? arXiv:1905.10650
52. Voita et al. Analyzing Multi-Head Self-Attention. arXiv:1905.09418
53. Zhang, Nanda. Best Practices of Activation Patching. arXiv:2309.16042
54. Heimersheim, Nanda. How to use and interpret activation patching. arXiv:2404.15255
55. Goldowsky-Dill et al. Path Patching. arXiv:2304.05969
56. Hewitt, Liang. Control Tasks. arXiv:1909.03368
57. Ravichander et al. Probing the Probing Paradigm. arXiv:2005.00719
58. Elazar et al. Amnesic Probing. arXiv:2006.00995
59. Belinkov. Probing Classifiers. arXiv:2102.12452
60. Wang et al. IOI circuit. arXiv:2211.00593
61. McDougall et al. Copy Suppression. arXiv:2310.04625
62. Vig et al. Causal Mediation Analysis. arXiv:2004.12265
63. Meng et al. ROME. arXiv:2202.05262
64. Geiger et al. Causal Abstractions of Neural Networks. arXiv:2106.02997
65. Geiger et al. DAS. arXiv:2303.02536
66. Kaushik et al. Counterfactually-Augmented Data. arXiv:1909.12434
67. Ravfogel et al. INLP. arXiv:2004.07667
68. Belrose et al. LEACE. arXiv:2306.03819
69. Geva et al. Dissecting Recall of Factual Associations. arXiv:2304.14767
70. Wang et al. Label Words are Anchors. arXiv:2305.14160
71. Pal et al. Future Lens. arXiv:2311.04897
72. Wu et al. Do language models plan ahead? arXiv:2404.00859
73. Oozeer et al. Activation Space Interventions Can Be Transferred. arXiv:2503.04429
74. Linear Representation Transferability Hypothesis. arXiv:2506.00653
75. Transferring Linear Features Across LMs With Model Stitching. arXiv:2506.06609
76. Cross-model Transferability ... Platonic Representations of Concepts. arXiv:2501.02009
77. Lee et al. Shared Global and Local Geometry of LM Embeddings. arXiv:2503.21073
78. The Master Key Hypothesis. arXiv:2604.06377 (2026)
79. Cross-Architecture Steering Transfer in LMs. arXiv:2608.05164 (2026)
80. Signed-Permutation Coordinate Transport for RMSNorm Transformers. arXiv:2606.31963 (2026)
81. Olsson et al. In-context Learning and Induction Heads. arXiv:2209.11895
82. Gurnee et al. Universal Neurons in GPT2. arXiv:2401.12181
83. Chughtai et al. A Toy Model of Universality. arXiv:2302.03025
84. Tigges et al. LLM Circuit Analyses Are Consistent Across Training and Scale. arXiv:2407.10827
85. Wu et al. Retrieval Head. arXiv:2404.15574
86. Prakash et al. Fine-Tuning Enhances Existing Mechanisms. arXiv:2402.14811
87. Huh et al. Platonic Representation Hypothesis. arXiv:2405.07987
88. Lan et al. Feature Space Universality via SAEs. arXiv:2410.06981
89. Bansal et al. Revisiting Model Stitching. arXiv:2106.07682
90. Kornblith et al. CKA. arXiv:1905.00414
91. Qwen Team. Qwen3 Technical Report. arXiv:2505.09388
92. Sriramanan et al. LLM-Check. NeurIPS 2024
93. Prompt Embedding Probes (PEP). arXiv:2608.08024 (2026)
94. Azaria, Mitchell. The Internal State of an LLM Knows When It's Lying. arXiv:2304.13734
95. Marks, Tegmark. The Geometry of Truth. arXiv:2310.06824
96. Bürger et al. Truth is Universal. arXiv:2407.12831
97. Burns et al. CCS. arXiv:2212.03827
98. CH-Wang et al. Do Androids Know They're Only Dreaming...? arXiv:2312.17249
99. Chen et al. INSIDE. arXiv:2402.03744
100. Kossen et al. Semantic Entropy Probes. arXiv:2406.15927
101. Chuang et al. Lookback Lens. arXiv:2407.07071
102. FactCheckmate. arXiv:2410.02899
103. DRIFT. arXiv:2601.14210
104. Sharma et al. Towards Understanding Sycophancy. arXiv:2310.13548
105. Papadatos, Freedman. Linear Probe Penalties Reduce Sycophancy. arXiv:2412.00967
106. Goldowsky-Dill et al. Detecting Strategic Deception Using Linear Probes. arXiv:2502.03407
107. Detecting High-Stakes Interactions with Activation Probes. arXiv:2506.10805
108. Constitutional Classifiers++. arXiv:2601.04603 (2026)
109. Unread or Unenforced? arXiv:2609.26178 (2026)
110. Hans et al. Binoculars. arXiv:2401.12070
