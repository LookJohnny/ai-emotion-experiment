# LLM 内部可视化与激活监测：文献调研（面向 Qwen3-4B「LLM 内部监视器」）

> 调研日期：2026-09-30
> 范围：学术界和工业研究机构对「可视化、理解 LLM 内部」的工作，重点收录报告了**结果**（用户研究、定量评估、案例研究、可测量发现）的论文，纯 demo 类只简述。
> 核实方式：每条都用 WebSearch/WebFetch 打开原文（arXiv 页面、PDF 全文、transformer-circuits.pub 原文或官方博客）核对过。下文数字都摘自原文，核对不到的数字一律标「未核实」或不写。
> 我们的监视器（本文简称「本系统」）：Qwen3-4B，36 层 × 32 个 Q 头（8 个 KV 头，GQA），hidden 2560。3D 视图中每层是一圈注意力头，围绕残差流轴排列；头与头之间的连线是**由权重算出的 V-composition 边**；线性探针读内部激活，在生成前和生成中触发安全告警（敌意、用户痛苦、幻觉）。

---

## 0. 结论速览

1. **原始注意力不能当解释。** Jain & Wallace (2019) 发现注意力与梯度重要性相关性很低，而且可以构造出预测完全不变、分布却截然不同的注意力。Abnar & Zuidema (2020) 在 6 层模型上测得，高层原始注意力与消融重要性的 Spearman 相关只有 −0.11～0.29，rollout/flow 能提到约 0.7；但在 DistilBERT/SST-2 上三种方法都只有约 0.1。
2. **只用权重算的 composition 分数是「潜在连线」，不代表「本次计算走了哪条路」。** Mathematical Framework (2021) 原文就要求减去随机矩阵基线，并发现「大多数头不参与实质性的组合」。Circuit Tracing (2025) 更明确警告：只看权重的「虚拟权重」存在 interference，「从不共同激活的特征之间也可能有很大的虚拟权重」，需要按共激活统计加权（TWERA）。
3. **最有说服力的内部可视化，是「按 prompt 计算 + 剪枝 + 标出剪掉了多少」的归因图。** Circuit Tracing 的剪枝把节点数减少约 10 倍、保留约 80% 的影响力。它在 18 层模型上 replacement score 为 0.61，completeness 为 0.80，扰动验证的 Spearman 为 0.72。作者自己承认只在约 1/4 的 prompt 上得到「令人满意的洞见」，QK（注意力为什么这样分配）完全没有被解释。**Neuronpedia 已经提供 Qwen3-4B 的 circuit-tracer transcoder 与在线归因图。**
4. **线性探针做监测，在分布内非常强，跨分布就明显变弱。** Apollo 的欺骗探针 AUROC 为 0.96–0.999，在 1% 误报率下召回 95–99%，但换两层探针，召回就几乎降到 0；探针还会对「涉及欺骗话题」的诚实回复误报。Orgad et al. (ICLR 2025) 表明幻觉探针跨数据集几乎不优于看 logits。Bailey et al. (2024) 用对抗手段能把有害性探针的召回从 100% 压到 0%。
5. **生成前预警是可行的，但信号更弱。**
   - SEP 用「生成前最后一个 token」就能预测语义熵。
   - Orgad 的「问题末尾」探针 AUC 为 0.72–0.82，而在「答案精确 token」上可达 0.83–0.95。
   - Persona vectors 在 prompt 末 token 上的投影与后续人格表现相关 r=0.75–0.83，但相关主要来自不同 prompt 类型之间的差异。
6. **「可视化真的有帮助」需要用任务型用户研究来证明。**
   - Transformer Explainer（CHI'26 版）：90 人被试间实验，客观测验正确率 73.3%，显著高于博客和视频（p=0.021）。视频组「自评懂了、测验却没做对」，即「理解错觉」。
   - Kaur et al. (CHI 2020) 发现数据科学家会过度信任、误用可解释性工具。
   - Hase & Bansal (ACL 2020) 发现主观评分与解释的实际效用不相关。

---

## 1. 电路 / 归因图可视化

### 1.1 A Mathematical Framework for Transformer Circuits
- **引用**：Elhage, Nanda, Olsson, Henighan, Joseph, Mann 等（Anthropic），2021，Transformer Circuits Thread。https://transformer-circuits.pub/2021/framework/index.html
- **可视化对象**：注意力头的 QK 电路（决定注意力模式）和 OV 电路（决定被注意 token 如何影响输出）；头与头之间的 Q/K/V-composition；「虚拟注意力头」（路径展开）。
- **编码方式**：用热力图矩阵显示第 1 层头 × 第 2 层头的 Q/K/V-composition 强度，颜色按头的功能（前一 token 头、归纳头）着色。
- **定量定义**（原文）：V-composition 的计算式是 `‖W_OV^{h2} W_OV^{h1}‖_F / (‖W_OV^{h2}‖_F ‖W_OV^{h1}‖_F)`，Q/K 同理。原文写明：「默认我们减去同形状随机矩阵的经验期望值（多数注意力头的组合远小于随机矩阵）」。在他们分析的 2 层模型中，「只有显著的 K-composition，而且只和一个第 0 层的头有关」，「大多数注意力头不参与任何实质性的组合」。
- **结果**：对只有注意力层的 1–2 层小模型给出了归纳头机制的完整解释。
- **局限（作者自述）**：
  - 只研究「两层或更少、只有注意力块」的 transformer，没有 MLP，作者称这是「我们工作的一大弱点」。
  - 忽略 LayerNorm，理由是它「除一个可变缩放外」可以并入相邻权重。
  - 原文有一处勘误：composition 图曾因一个低秩线代库的 bug 画错，后来给出了修正图。这说明这类权重分数对实现细节很敏感。
- **与本系统的关系**：本系统的 V-composition 边正是这一定义，但 Qwen3-4B 有 36 层，带 MLP、RMSNorm 和 QK-Norm，并且 4 个 Q 头共享一组 W_V（GQA），远超该框架被验证过的范围（见 §8）。

### 1.2 Circuit Tracing: Revealing Computational Graphs in Language Models（方法篇）
- **引用**：Ameisen, Lindsey, Pearce 等（Anthropic），2025-03-27，Transformer Circuits Thread。https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- **可视化对象**：针对**单个 prompt** 的归因图（attribution graph）。节点分四类：输入 embedding、跨层转码器（CLT）特征、误差节点（error node）、输出 logit。边是在冻结注意力模式和归一化分母之后，节点之间的**线性直接效应**。
- **编码方式**：
  - 纵轴是层，横轴是 token 位置。
  - 正向边为绿色，负向边为紫色。
  - 功能相近的特征可以人工合并成 supernode。
  - 悬停时显示特征的最大激活样本和上/下调的 token。
  - 误差节点单独显示，代表未被解释的 MLP 输出。
- **剪枝**：
  - 先把邻接矩阵 A 按「每个节点的输入边之和为 1」归一化，再计算间接影响 `B = (I − A)^{-1} − I`，即所有路径强度之和。
  - 节点按对输出 logit 的累积影响剪枝，默认保留约 80% 的影响，节点数约减少 10 倍，completeness 约下降 20%。
  - 边再按阈值剪枝，文中展示了 0.95/0.9/0.8/0.7 等不同阈值下的取舍。
- **定量结果**（18 层模型）：
  - replacement score 0.61：不经误差节点能解释的比例。
  - completeness 0.80。
  - 归一化重建误差约 11.5%。
  - 扰动验证：预测的特征→特征效应与实际效应的 Spearman 为 0.72。
- **局限（作者自述）**：
  1. 注意力模式被冻结，**QK 电路没有被解释**，因此在依赖注意力的行为上（归纳、多选题推理）会失败；作者承认这在注意力现象上比之前基于神经元的电路工作「倒退」了。
  2. 较多计算留在误差节点里。
  3. 抑制型机制（未激活特征）捕捉得差。
  4. 扰动的保真度随层数累积变差。
  5. **全局权重的 interference**：「从不在分布上共同激活的特征之间，仍可能有（可能很大的）虚拟权重」。作者因此提出 TWERA，按共激活统计 `E[a_j a_i]/E[a_j]` 对虚拟权重加权。
- **与本系统的关系**：第 5 条几乎就是在描述「只用权重算 V-composition 边」的问题。

### 1.3 On the Biology of a Large Language Model（案例篇）
- **引用**：Lindsey et al.（Anthropic），2025-03-27。https://transformer-circuits.pub/2025/attribution-graphs/biology.html
- **案例（Claude 3.5 Haiku）**：
  - 多跳推理：Dallas → Texas → Austin，中间步骤可见。
  - 押韵诗的提前规划。
  - **幻觉与实体识别**：模型默认有一条「无法回答」回路；遇到已知实体时，「known answer」特征会抑制它；当这类特征对不熟悉的实体「误触发」时，就产生幻觉。
  - 拒答：各类具体危害特征在微调中汇聚成通用的「有害请求」检测。
  - 越狱：模型逐字母拼接出有害词，在完成整句之前没有「意识到」内容有害。
- **局限**：作者明确说方法只在「约四分之一的 prompt 上」得到令人满意的洞见。这些案例是「存在性证明」，不代表系统性覆盖。
- **与本系统的关系**：幻觉告警可以参考「已知实体 vs 无法回答」这条机制，但机制描述来自 Claude 3.5 Haiku，Qwen3-4B 上是否同构需要自己验证。

### 1.4 circuit-tracer 库 + Neuronpedia 在线归因图
- **引用**：Hanna, Piotrowski, Lindsey, Ameisen，「circuit-tracer: A New Library for Finding Feature Circuits」，BlackboxNLP 2025。https://aclanthology.org/2025.blackboxnlp-1.14/
- **关键事实**：
  - 支持 Gemma-2-2B、Llama 系 1B 模型（具体版本号未核实），**以及 Qwen3-4B**。Neuronpedia 页面说明 Qwen3-4B 用的是 Hanna & Piotrowski 于 2025-08 发布的 transcoder。https://www.neuronpedia.org/qwen3-4b
  - 支持按需生成归因图、剪枝阈值滑块、特征干预。
  - Neuronpedia 的界面分三栏：全图、选中节点的上下游、特征最大激活样本与 logit 效应。
- **局限**：继承 §1.2 的全部局限；使用的是逐层转码器（PLT）还是 CLT，取决于所加载的 transcoder（未核实 Qwen3-4B 版具体是哪一种）。

---

## 2. 特征 / 神经元仪表盘

### 2.1 Towards Monosemanticity
- **引用**：Bricken, Templeton, Batson 等（Anthropic），2023-10。https://transformer-circuits.pub/2023/monosemantic-features
- **可视化对象**：在 1 层 transformer 的 MLP 激活上训练的稀疏自编码器（SAE）特征。
- **编码方式**：每个特征一张「特征卡」，包括激活密度直方图、按激活分位区间抽样的样本（不只看 top 样本）、logit 效应（上调/下调哪些 token），以及特征分裂的浏览。
- **结果**：提取出 DNA、base64、阿拉伯文、希伯来文、HTTP 请求、法律文本等清晰的特征。从三方面论证特征是「真实」的：特异性、对输出 logit 的因果效应（如 DNA 特征会上调 A/C/G/T）、自动可解释性评分。核心结论是：**比单个神经元更好的分析单元是存在的**。
- **局限**：只在小模型上验证；存在未被解释的「暗物质」重建误差（具体数字本文未核实）。

### 2.2 Scaling Monosemanticity
- **引用**：Templeton, Conerly, Marcus, Batson 等（Anthropic），2024-05。https://transformer-circuits.pub/2024/scaling-monosemanticity/
- **对象**：Claude 3 Sonnet 中间层残差流上训练的 1M / 4M / 34M 特征 SAE。
- **结果**：特征抽象、多语言、多模态；找到与安全相关的特征，包括欺骗、谄媚、偏见、代码后门、权力寻求、危险内容、自我表征，且论文称这些特征对行为有因果影响（「Golden Gate」式的 steering）。
- **局限**：特征激活≠该行为一定会发生；特征集合不完备。原文在这方面的具体定量指标本文未逐项核实。

### 2.3 OpenAI「Language models can explain neurons in language models」
- **引用**：Bills, Cammarata, Mossing 等（OpenAI），2023-05-09。https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html
- **方法**：解释 → 模拟 → 打分。GPT-4 先为 GPT-2 XL 的神经元写解释，再由模拟器按解释预测激活；打分标准是模拟激活与真实激活的相关系数。
- **结果**：GPT-2 XL 共 307,200 个神经元中，只有**1000 多个**神经元的解释得分 ≥0.8（作者认为 ≥0.8 意味着「解释了大部分 top 激活行为」）。换言之，绝大多数神经元无法用一句话解释。
- **批评**：Huang et al.（BlackboxNLP 2023，「Rigorously Assessing Natural Language Explanations of Neurons」，https://arxiv.org/abs/2309.10312）用更严格的评估发现，即使是置信度最高的解释，错误率也很高，而且几乎没有因果效力。Bolukbasi et al. (2021)「An Interpretability Illusion for BERT」（https://arxiv.org/abs/2104.07143）指出，单个神经元（及其线性组合）可能**虚假地**看起来编码了某个简单概念，原因在于语料只覆盖了窄的一片。

### 2.4 Neuronpedia
- **引用**：Johnny Lin 于 2023 年创建的开源可解释性平台（没有同行评审的主论文）。https://www.neuronpedia.org/ ，源码 https://github.com/hijohnnylin/neuronpedia
- **内容**：SAE/transcoder 特征仪表盘（最大激活样本、logit 效应、自动解释）、steering、归因图（集成 circuit-tracer）、API。
- **证据类型**：属于基础设施，本身没有报告用户研究。

### 2.5 相关的「实际效用」证据
- **Marks et al. 2025「Auditing Language Models for Hidden Objectives」**（https://arxiv.org/abs/2503.10965）：盲审游戏中 4 支团队审计一个被植入隐藏目标（RM 谄媚）的模型，3 支成功。其中一队用 SAE 在约 1.5 研究员小时内找到了相关训练文档；只有 API 访问的一队失败。这是少数「可解释性工具在真实任务中有用」的对照式证据。
- **GDM 机制可解释性团队 2025「Negative Results for SAEs on Downstream Tasks」**（https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9）：在「检测有害意图」这类分布外探测任务上，**SAE 特征不如直接训练的线性探针**，线性探针本身的效果也出人意料地好。团队因此降低了 SAE 研究的优先级。
  → 对本系统的含义：**告警用线性探针，解释用特征。** 两者分工不同。

---

## 3. 逐层读出（Layer-wise readouts）

### 3.1 Logit Lens
- **引用**：nostalgebraist，「interpreting GPT: the logit lens」，LessWrong，2020。https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
- **对象与编码**：把各层残差流直接乘上 unembedding，读出「如果在这一层停下会预测什么」。画成层 × token 位置的网格，格子里写 top-1 token，颜色表示概率或排名。
- **发现**：在 GPT-2 上，各层预测大体单调地收敛到最终答案。
- **局限**：在其他模型族上不可靠（见 3.2）。

### 3.2 Tuned Lens
- **引用**：Belrose, Ostrovsky, McKinney, Furman, Smith, Halawi, Biderman, Steinhardt，「Eliciting Latent Predictions from Transformers with the Tuned Lens」，arXiv 2303.08112，2023。https://arxiv.org/abs/2303.08112
- **方法**：给每一层训练一个仿射「translator」，再接 unembedding。
- **编码**：层（纵）× token 位置（横）的热力图，格子里是 top-1 token，颜色表示熵或与最终分布的 KL。
- **结果**：
  - Logit lens 在 BLOOM、GPT-Neo、OPT-125M 上表现差：在 BLOOM 和 OPT-125M 上，「超过一半的层里 logit lens 的 top-1 往往是输入 token 本身」。在 GPT-Neo-2.7B 上，大多数层的偏差约 4–5 bits。原因是各层表示的协方差会「漂移」，并且存在 rogue dimensions。
  - Tuned lens 更具预测力、偏差更小，因果实验显示它用的是与模型自身相似的特征。
  - **异常检测**：把各层预测轨迹展平后，用 iForest/LOF 检测 prompt 注入。在 Pythia-12B 上，BoolQ 的 AUROC 为 0.99–1.00，MNLI 为 0.98–1.00，QQP 为 1.00；但 SciQ 和 ARC-Easy 只有 0.62–0.74。
- **局限**：每层都要训练一个 translator，论文中约需 8×A40 一小时；logit lens 则开箱即用。
- **与本系统的关系**：Qwen3-4B 的 logit lens 是否可靠未知，应先实测它与最终分布的 KL，再决定是否训练 tuned lens。

### 3.3 Patchscopes
- **引用**：Ghandeharioun, Caciularu, Pearce, Dixon, Geva，ICML 2024。https://arxiv.org/abs/2401.06102
- **方法**：把某层的隐藏表示「patch」进另一个（可以是同一个）模型的特定 prompt 中，让模型用自然语言说出这个表示编码了什么。作者把 logit lens、tuned lens 等统一为它的特例。
- **结果**：作者称它克服了早期层检查失效、表达力有限等问题，还可以用更强的模型来解释弱模型，并纠正多跳推理错误（具体数字本文未逐项核实）。
- **局限**：每次读出都要额外推理一次，实时代价较高；生成的自然语言解释本身也可能出错。

### 3.4 FFN 在词表空间中构建预测 + LM-Debugger
- **引用**：
  - Geva, Caciularu, Wang, Goldberg，「Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space」，EMNLP 2022。https://arxiv.org/abs/2203.14680
  - Geva et al.，「LM-Debugger」，EMNLP 2022 System Demos。https://aclanthology.org/2022.emnlp-demos.2/
- **对象与编码**：把每层 FFN 的更新分解为单个参数向量的「子更新」，投影到词表，列出每个子更新推高/压低的 token。LM-Debugger 展示 FFN 前后预测分布的变化，以及最主要的子更新。
- **结果**：子更新常常是人类可解释的概念；只干预少数向量就能**让 GPT-2 的毒性下降约 50%**；基于这一分析做早停可平均节省约 20% 计算。
- **局限**：只在 GPT-2 规模上验证；词表投影继承了 logit lens 的偏差。

### 3.5 LM Transparency Tool
- **引用**：Tufanov, Hambardzumyan, Ferrando, Voita，arXiv 2404.07004，2024。https://arxiv.org/abs/2404.07004
- **对象**：按 prompt 计算的「信息流路径」（information flow routes）。先给注意力头和 FFN 的贡献打重要性分，**只显示重要部分**，再把各组件的更新投影到词表。
- **与本系统的关系**：它是「按激活计算、按重要性过滤」的头级别连线的直接参照，比按权重算的边更贴近本次计算。

---

## 4. 注意力可视化及其批评

### 4.1 BertViz
- **引用**：Jesse Vig，「A Multiscale Visualization of Attention in the Transformer Model」，ACL 2019 System Demos。https://aclanthology.org/P19-3007/ ，https://github.com/jessevig/bertviz
- **编码**：三种视图。head view 是 token 之间的连线，线宽表示注意力权重；model view 是层 × 头的缩略图网格；neuron view 展示 q·k 的逐元素乘积。
- **证据**：三个用例（检测偏见、定位相关头、把神经元与行为关联），**没有用户研究或定量评估**。

### 4.2 AttentionViz
- **引用**：Yeh, Chen, Wu, Chen, Viégas, Wattenberg，IEEE VIS 2023 / IEEE TVCG 30(1), 2024。https://arxiv.org/abs/2305.03210 ，demo：http://attentionviz.com
- **对象与编码**：把 query 向量和 key 向量投到同一个嵌入空间（t-SNE/UMAP/PCA 散点）。Matrix View 把所有层 × 头的散点缩略图并排，用于全局比较。
- **证据**：第一轮访谈 5 位专家（E1–E5）收集需求；第二轮访谈 4 位（E2、E3，加上新增的 E6、E7），另有若干应用场景。
  - 优点：多位专家认为 Matrix View 的「全局视角」是最新颖、最有价值的部分。
  - 担忧 1，信息过载：E6 说「展示所有特征和头可能让人不知所措……能不能做汇总？或者聚焦于特定任务？」
  - 担忧 2，投影失真：E3 问「我怎么知道能不能相信我看到的东西？」
  - 专家的需求：量化两个头之间的相似度、「追踪注意力流以做假设检验」、看两个头的模式如何连接（例如归纳头配对）。
- **作者自述局限**：没有纳入 value 向量（作者把它列为未来方向）；预计算和内存开销较大。

### 4.3 「Attention is not Explanation」与「Attention is not not Explanation」
- **引用**：
  - Jain & Wallace，NAACL 2019。https://arxiv.org/abs/1902.10186
  - Wiegreffe & Pinter，EMNLP 2019。https://arxiv.org/abs/1908.04626
- **Jain & Wallace 的结果**：注意力权重与梯度类特征重要性的相关性很低；可以构造出截然不同、预测却相同的注意力分布。
- **Wiegreffe & Pinter 的反驳**：「解释」取决于定义，提出了均匀注意力基线、多随机种子方差校准、冻结权重诊断、端到端对抗训练四种检验，指出前者的结论被夸大了。**但他们并没有证明注意力是「忠实」的解释。**
- **注意**：两篇的实验对象都是 RNN + 注意力的分类模型，不是现代 decoder LLM。它们对本系统的意义是方法论层面的：不能默认「注意力 = 重要性」，必须用消融/梯度来对照。

### 4.4 Attention Rollout / Attention Flow
- **引用**：Abnar & Zuidema，「Quantifying Attention Flow in Transformers」，ACL 2020。https://arxiv.org/abs/2005.00928
- **方法**：rollout 把各层注意力矩阵（加上残差恒等项）连乘；flow 把注意力当作容量，求最大流。
- **结果**：Spearman 相关，原文 Table 1、2，6 层主谓一致模型，2000 个样本：

  | 方法 \ 层 | L1 | L3 | L6 |
  |---|---|---|---|
  | 原始注意力 vs 消融 | 0.69 | −0.11 | 0.29 |
  | Rollout vs 消融 | 0.32 | 0.51 | 0.71 |
  | Flow vs 消融 | 0.32 | 0.70 | 0.70 |
  | 原始注意力 vs 输入梯度 | 0.53 | −0.06 | 0.46 |
  | Flow vs 输入梯度 | 0.22 | 0.54 | 0.61 |

  在 DistilBERT/SST-2（原文 Table 3）上，**三种方法都只有约 0.08–0.14**，作者也承认「都非常低」。
- **局限**：rollout/flow 做了很多简化假设，而且只给出 token 级别的归因，不涉及头与头之间。

---

## 5. 教育型 / 结构型（含 3D）可视化

### 5.1 Transformer Explainer
- **引用**：Cho, Kim, Karpekov, Helbling, Wang, Lee, Hoover, Kahng（Georgia Tech），最初为 IEEE VIS 2024 短文/海报；扩展版含用户研究，发表于 **CHI '26**（arXiv 2408.04619v2）。https://arxiv.org/abs/2408.04619
- **对象与编码**：在浏览器里实时运行 GPT-2。整体用 Sankey 式的流向图（2D）表现 embedding → 注意力 → MLP → 概率的流动。可以逐级展开到矩阵运算（用动画过渡保持上下文），可以调 temperature / top-k / top-p 并实时看到效果，另有逐步引导卡片。
- **结果**：
  - 初步可用性评估中，145 人里有 74% 更偏好该工具而非博客。
  - 总结性研究为被试间设计，**90 名**非专家（从 128 人中剔除 38 名不认真作答者），三组分别使用工具、博客（Illustrated GPT-2）、视频（3Blue1Brown），各自学习最多 45 分钟后做 7 题闭卷测验。
  - 工具组测验正确率为 **73.3%**，显著高于博客组（p=0.021）和视频组（p=0.021，GLMM，控制先验知识）。
  - 工具组的学习体验评分也更高（对博客 p=0.026，对视频 p=0.033），心理负荷低于博客组（p=0.044）。
  - 视频组自评理解程度与工具组相当，但测验成绩与博客组一样低，作者称之为「理解错觉」。
  - 清晰度评分三组都不高，工具组最高也只有 3.07/5。
- **局限**：被试是非专家、学习的是教学概念，不涉及分析真实模型行为；上线后有 49 万以上用户，但这一数字不是效果证据。

### 5.2 Brendan Bycroft「LLM Visualization」（3D）
- **引用**：Bycroft，2023，https://bbycroft.net/llm
- **编码**：用 3D 方块展示 nano-gpt（约 85k 参数，任务是给 6 个字母排序）的每一次乘加。绿色方块是激活，蓝色方块是权重，配合分步讲解和高亮。
- **证据**：**没有**用户研究或定量评估，属于纯教学 demo。

### 5.3 LIT（Language Interpretability Tool）
- **引用**：Tenney et al.（Google PAIR），EMNLP 2020 System Demos。https://aclanthology.org/2020.emnlp-demos.15/ ，https://github.com/pair-code/lit
- **内容**：在浏览器中联动局部解释（显著性图、注意力）、聚合分析（嵌入投影、指标切片）和反事实生成。
- **证据**：以案例研究为主，没有对照用户研究。
- 注：InterpretML（Nori et al., 2019）是 Microsoft 的 GAM/EBM 可解释建模工具，面向表格数据，与 LLM 内部无关。它唯一相关的证据是 Kaur et al. (CHI 2020) 对它和 SHAP 的使用研究（见 §7）。

---

## 6. 用于安全的激活监测（含 AUROC）

| 工作 | 模型 | 监测对象 | 关键数字 | 主要局限 |
|---|---|---|---|---|
| RepE（Zou et al. 2023） | LLaMA-2 等 | 诚实、情绪、权力寻求等的「reading vector」 | TruthfulQA MC1 比零样本提高 18.1 个百分点；在「被指示诚实/撒谎」的留出样本上，诚实方向分类准确率 >90% | 作者自己区分了「相关 / 操纵 / 终止」三类实验，读出本身**只提供相关证据** |
| Sleeper-agent 探针（MacDiarmid et al., Anthropic 2024） | 人为植入后门的模型 | 叛变（defection） | AUROC >99%；单个通用对比对即可达 99.3%；中间层残差流最好 | 后门是人为插入的，显著性可能是插入过程的产物；自然产生的欺骗对齐是否同样可检测「仍是开放问题」 |
| Apollo 欺骗探针（Goldowsky-Dill et al. 2025） | Llama-3.3-70B | 策略性欺骗 | 诚实 vs 欺骗 AUROC 0.96–0.999（Roleplaying 0.964、Insider Trading Report 0.999、Confirmation 0.974、Sandbagging 0.983）；阈值定在对照聊天数据 1% FPR 时召回 95–99% | 见下 |
| Circuit Breakers（Zou et al. 2024） | Llama-3-8B、Mistral-7B | 有害表示；另测了有害性探针（HP） | 表示重定向（RR）使对有害请求的平均顺从率降低 87%；HP（线性和 MLP 探针，逐 token 检测，命中即中止生成）也显著降低了攻击成功率，但不如 RR | HP 在「攻击者不知道有探针」的较弱威胁模型下测试 |
| Obfuscated Activations（Bailey et al. 2024） | 多个 | 攻击 SAE、探针、OOD 检测等潜空间防御 | 能把有害性探针的召回**从 100% 降到 0%**，同时保持 90% 的越狱率 | 在复杂任务（写 SQL）上，混淆会降低模型能力 |
| SEP（Kossen et al. 2024） | Llama-2/3、Mistral、Phi-3 | 语义熵（幻觉风险） | 后层 AUROC 为 0.7–0.95（视场景而定）；**生成前最后一个 token（TBG）**上也能预测，略低于 SLT；跨任务泛化优于直接预测正确率的探针 | BioASQ 上的高 AUROC 部分来自「是不是是/否题」这一捷径；长文本生成中峰值出现在中间层 |
| Orgad et al.（ICLR 2025） | Mistral-7B-Instruct、Llama-3-8B-Instruct | 答案正误 | 「答案精确 token」上的探针 AUC：TriviaQA 0.85/0.83，Winobias 0.92/0.93，Math 0.92/0.95；「问题末尾」（生成前）AUC 为 0.72–0.82 | **跨数据集泛化扣除 logit 基线后基本不存在**，真实性编码是「技能相关的」，不是通用的；模型可能内部编码了正确答案却持续输出错误答案 |
| CH-Wang et al.（Findings ACL 2024） | 有依据的生成任务 | 幻觉 span | 在领域内，第 4 层就达到峰值性能的 95%；在响应级 F1 上超过一名专家标注员 | 探针对训练领域敏感，跨任务、从合成幻觉到自然幻觉的泛化都差（论文用 F1，不报 AUROC） |
| Persona Vectors（Chen, Arditi, Sleight, Evans, Lindsey 2025） | Qwen2.5-7B-Instruct、Llama-3.1-8B-Instruct | 邪恶、谄媚、幻觉倾向 | prompt 末 token 投影与后续特质表达的相关 **r=0.75–0.83**；微调引起的偏移与特质变化 r=0.76–0.97 | 作者说明 prompt 层面的相关「主要来自区分不同 prompt 类型」，控制 prompt 类型后相关更温和 |
| TalkTuner（Chen et al. 2024，Harvard Insight+Interaction Lab） | LLaMA2-Chat-13B | 用户画像（年龄/性别/教育/社会经济地位）探针，**实时仪表盘** | 真实对话 6 轮后用户画像平均准确率 78%；19 人被试内研究：用户更愿意使用仪表盘（p<0.05），信任随画像准确度而校准；近一半被试借此发现了偏见回复；7 人担心隐私 | 小样本；合成数据训练的探针 |

**Apollo 论文里对本系统特别有用的细节**：
- **逐 token 着色**：红色表示读为欺骗，蓝色表示读为诚实，这是他们主要的诊断界面。探针常在显式欺骗文字出现**之前**就已经激活。
- **三类失败**：
  - 伪相关：探针同时在追踪类似「道德」的东西。
  - 聚合失败：对 token 分数取平均后，长回复里被稀释。
  - 原因不明的失败。
- **探针在「涉及欺骗的场景」下对诚实回复也显著更高**，它部分检测的是「话题是否与欺骗相关」。
- **召回@1%FPR 对方法细节极其敏感**：探针位置往后移两层，Insider Trading 数据集上的召回就几乎降到 0，而 AUROC 仍然尚可。
- 分数用「对照分布的百分位」来表示。

RepE 的界面同样值得参考：它的「LAT scan」是层 × token 的小方块热力图，每格显示该层该位置的「不诚实神经活动」强度（原文 Figure 8/9）。RepE 还有情绪（快乐/悲伤/愤怒等 6 类）的读出与操纵章节。

---

## 7. 可视分析综述与「可视化是否真的有帮助」的用户研究

- **Braşoveanu, Scharl, Nixon, Andonie，「Visualizing Large Language Models: A Brief Survey」，IEEE IV 2024**。https://eprints.weblyzard.com/123/1/ieee_iv_2024_visualizing_llms_a_brief_survey.pdf 。把 LLM 可视化分为提示工程、因果/可解释性、多模态、LLM 评测、RAG/代码等类别，只收录「提出可视化系统」的论文。
- **Wang, Sun, Fu, Liang，「Human-Computer Interaction and Visualization in Natural Language Generation Models」**，Frontiers of Computer Science（已录用）。https://arxiv.org/abs/2410.08723 。按三大领域、六类任务梳理 NLG/LLM 的 HCI 与可视化工作，并讨论在 LLM 上的挑战。
- **ConceptViz（Li et al., IEEE TVCG 32(1), 2026）**。https://arxiv.org/abs/2509.20376 。围绕 SAE 特征做「识别 → 解释 → 验证」的可视分析，有两个使用场景加一次用户研究（样本量等细节本文未核实）。它的要点是把「用行为测试验证特征」直接放进工作流。
- **Kaur, Nori, Jenkins, Caruana, Wallach, Wortman Vaughan，「Interpreting Interpretability」，CHI 2020**。https://dl.acm.org/doi/10.1145/3313831.3376219 。11 人情境访谈加 197 人问卷，对象是 InterpretML（GAM）和 SHAP。结论：**数据科学家过度信任、误用可解释性工具，很少有人能准确描述这些可视化到底输出了什么。**
- **Hase & Bansal，「Evaluating Explainable AI: Which Algorithmic Explanations Help Users Predict Model Behavior?」，ACL 2020**。https://arxiv.org/abs/2005.01831 。用「可模拟性」（用户能否预测模型在新输入上的行为）来评估解释方法。只有少数方法在部分设定下有效（如 LIME 在表格数据上）；**主观评分与实际效用不相关**。
- **AttentionViz（§4.2）和 TalkTuner（§6）** 分别提供了专家访谈证据和最终用户证据。
- **干预类方法的「可解释性错觉」**：Makelov, Lange, Nanda，「Is This the Subspace You Are Looking For?」，arXiv 2023（NeurIPS 2023 会议站点有收录页；最终正式会议版本未核实）。https://arxiv.org/abs/2311.17030 。子空间 patching 可以让输出表现得「好像特征被改了」，实际却是激活了一条与输出因果无关的休眠通路。

**小结**：面向 LLM 内部的可视分析系统大多只做专家访谈或案例研究。真正**有对照、有客观指标**的用户研究很少，现有的几项（Transformer Explainer、TalkTuner、Kaur、Hase & Bansal）一致表明：**主观喜好不等于理解，理解不等于正确使用。**

---

## 8. 对我们监视器的启示（Implications for our monitor）

### 8.1 文献认为本系统当前设计中可能误导的地方

1. **把按权重算的 V-composition 边画成「信息流」。**
   - 这些分数与输入无关，不考虑本次的注意力模式，也不考虑激活大小。
   - Circuit Tracing 明确指出虚拟权重存在 interference：从不共同激活的单元之间也可能有很大的权重连接。
   - Framework 原文要求减去随机矩阵基线，并发现大多数头不参与实质组合。
   - 在 Qwen3-4B 上还有额外问题：
     - 头之间隔着 MLP 和 RMSNorm，RMSNorm 的缩放逐 token 变化，不能精确并入权重。
     - Qwen3 的 QK-Norm 让 Q/K 路径更加非线性。
     - GQA 下 4 个 Q 头共享同一个 W_V，同组头的 V-composition 分数会高度相关，这是结构造成的「假结构」。
   - 36 层 × 32 头 = 1152 个头，层间头对约 66 万条。不做基线校正就剪枝的话，看到的主要是「范数大的头」。
2. **把单个头或神经元当作有名字的「概念单元」。** OpenAI 的结果是 30.7 万个神经元中只有 1000 多个解释分 ≥0.8；Huang et al. 与 Bolukbasi et al. 证明单单元解释常常是错觉。环上给头贴「愤怒头」这类标签，会制造虚假的确定感。
3. **原始注意力高亮 = 模型「在看」/「因为」。** Jain & Wallace 和 Abnar & Zuidema 的数据表明，高层原始注意力与重要性的相关可能接近 0 甚至为负。
4. **把分布内 AUROC 当作上线可靠性。** Apollo 显示召回@1%FPR 对层选择极为敏感；Orgad 和 CH-Wang 显示幻觉探针跨领域泛化差；Bailey 显示探针可以被对抗性绕过。
5. **环形布局暗示了并不存在的结构。** 同一层内头的编号可以任意置换，因此「相邻头」「环上的角度」没有语义。如果观众从空间邻近读出了关系，这就是误导。

### 8.2 具体设计建议（12 条）

**R1｜把「连线」分成两层并明确标注：静态布线 vs 本次计算。**
- 默认显示**按激活计算、针对本 prompt 的**头间边。可选做法：
  - 上游头输出在下游头 value 输入方向上的投影 × 下游注意力权重，即冻结注意力后的线性直接效应，做法同 Circuit Tracing / LM Transparency Tool。
  - 或者用 attribution patching 计算。
- 原先的权重 V-composition 降级为可开关的「潜在布线」图层，图例写明「与输入无关，非本次计算路径」，并且：
  1. 减去同形状随机矩阵的期望（Framework 的默认做法）；
  2. 按 GQA 分组显示或合并同一 KV 组的头；
  3. 可选 TWERA 式修正：用真实语料上的共激活统计给权重加权。

**R2｜剪枝必须「可见、可量化、可调」。**
- 参照归因图：先用 `B=(I−A)^{-1}−I` 计算对告警目标（探针分数或 logit）的累积影响，再按节点累积影响（默认 0.8）和边阈值剪枝。
- 界面常驻显示「本视图保留了 X% 影响 / 隐去 N 个头」，提供阈值滑块（Neuronpedia 的做法）。
- **增加一个「未解释 / 残差」节点**，对应归因图中的 error node，避免让用户以为图就是全部。

**R3｜正式评估前，先接入 Neuronpedia/circuit-tracer 的 Qwen3-4B transcoder 作为「深挖」模式。**
- 告警触发后，一键为该 token 生成归因图，用特征（带最大激活样本和 logit 效应）替代「头」作为解释单元。
- 3D 头环只承担「导航和总览」。
- 需要在 UI 上注明两点：归因图不解释 QK（注意力为何如此分配），只对约 1/4 的 prompt 给出清晰机制。

**R4｜不要单独展示原始注意力。**
- 如果要显示「这个头在看哪个 token」，至少改用下面之一：
  - 注意力 × 被注意位置 value 输出的范数，即「实际搬运的量」；
  - rollout/flow 作为 token 级别的汇总。
- 旁边附上「注意力≠重要性」的提示。
- 需要重要性时，用消融或梯度 × 激活对照，并在界面上并列显示两者，让分歧可见。

**R5｜逐层读出用「层 × token 热力图」这种 2D 视图，并先校准 lens。**
- 在 Qwen3-4B 上先测 logit lens 各层与最终分布的 KL。如果早中层偏差大（Tuned Lens 在 BLOOM/OPT 上看到过这种情况），就训练 tuned lens，论文中约需 8×A40 一小时。
- 热力图格子里写 top-1 token，颜色表示熵或到最终分布的 KL。
- 同一网格上叠加**探针分数的层 × token 扫描**（即 RepE 的 LAT scan），比在 3D 中给头着色更易读，也更可比较。
- 可以把「预测轨迹异常」作为一个通用告警：Tuned Lens 用它检测 prompt 注入，多数任务 AUROC 约为 1.0，但在 SciQ/ARC 上只有 0.62–0.74，需要自测。

**R6｜告警分数用「对照分布的百分位」表示，阈值按固定误报率设定。**
- 参照 Apollo：以大量中性对话为对照集，报告**召回@1%FPR**（以及 0.1%FPR），不只报 AUROC。
- UI 上显示「高于 99.3% 的正常对话」，而不是裸的 logit。
- 告警阈值和所用的层要冻结在 prereg 中（本目录已有 `FREEZE_monitor.json`，建议补上 FPR 口径）；每次改层都要重新评估，因为「换两层召回归零」是有记录的。

**R7｜分成「生成前预警」和「生成中确认」两个档位，并诚实标注置信差异。**
- 生成前：读最后一个 prompt token（SEP 的 TBG、Persona Vectors、Orgad 的「问题末尾」），文献中 AUC 约 0.72–0.82，且相关主要来自 prompt 类型差异。标为「风险预测」，用黄色。
- 生成中：幻觉在「答案精确 token」上信号最强（AUC 0.83–0.95）；敌意/痛苦类用逐 token 分数的**滑窗最大值和均值两者**显示，避免 Apollo 所说的「平均稀释」。
- 只有确认档可以触发「中止/替换」这类动作，做法同 Circuit Breakers 的 HP：逐 token，命中即中止。

**R8｜逐 token 着色的文本视图是一等公民，不是附属品。**
- Apollo 和 RepE 的主要诊断都依赖逐 token 着色。建议把输出文本按探针分数着色，放在 3D 视图旁边联动：点 token → 3D 中高亮该位置的层 × 头贡献。
- 这也是用户研究（R11）中最可能帮助用户判断「告警对不对」的视图。

**R9｜针对「话题相关 ≠ 行为本身」设计困难负例，并在 UI 中提示。**
- Apollo 的探针会在「涉及欺骗的场景」里对诚实回复误报。同理，敌意探针可能在「用户讲述别人很凶」时触发，痛苦探针可能在「讨论心理健康知识」时触发。
- 训练和评估要加入这类困难负例，并在告警卡片上显示最相似的训练正例和负例，供人工判断。

**R10｜显式的分布外/泛化警示与对抗性声明。**
- 幻觉探针要标注训练领域（Orgad 和 CH-Wang 都表明跨任务泛化弱）。当输入远离训练分布时（例如用激活的马氏距离或 kNN 距离判断），显示「探针可靠性未知」。
- 文档和 UI 都声明：激活探针可以被对抗性优化绕过（Bailey 2024：召回 100%→0%），它只是多层防御中的一层，不是保证。

**R11｜用任务型、有对照、预注册的用户研究来证明「可视化有用」，不要用满意度证明。**
- 可以参照 Transformer Explainer 的设计：被试间、客观测验、GLMM 控制先验知识、剔除不认真作答者。
- 为监视器设计的任务：给定一段对话和告警，判断告警是真阳性还是假阳性，并指出触发 token。
- 设 3 组对照：(a) 只有分数和文本着色；(b) 加 2D 层 × token 热力图；(c) 加 3D 头环 + 连线。
- 指标：判断准确率、用时、**信任校准**（信心与正确性的相关，回应 Kaur 的过度信任问题）。不以主观评分作为主要结论（Hase & Bansal）。
- 另可做 Marks et al. 式的「植入故障审计游戏」：预先植入已知的触发器或偏见，看用户能否借助本系统找到它。

**R12｜每个「解释」都带证据等级徽章，并提供一键因果验证。**
- 按 RepE 自己的「相关 → 操纵 → 终止」三级：
  - 默认显示「相关」（探针读数）；
  - 用户可以一键做 steering 或消融（沿探针方向加减，或 ablate 选中的头），查看输出和探针分数的变化，结果升级为「已因果验证」。
- 同时提醒 Makelov 的「子空间 patching 错觉」：干预有效≠找到了模型原本使用的通路。

**附加（布局层面）**：
- 头在环上的排列应改为有意义的顺序，例如按 GQA 组分块，或按该头对当前告警探针方向的平均贡献排序。也可以明确声明「角度无意义」。
- 应对信息过载（AttentionViz 专家的担忧）：默认只显示剪枝后的前 k 个头和边，其余淡化（focus + context）。

---

## 9. 未能核实 / 需注意

- Scaling Monosemanticity 的具体重建与可解释性定量指标、Towards Monosemanticity 的自动可解释性具体分数：本文没有逐项核对原文数字，因此未引用。
- Patchscopes 各实验的具体数字未逐项核对。
- ConceptViz 用户研究的样本量与结果未核对。
- Neuronpedia 上 Qwen3-4B 所用的是 PLT 还是 CLT、覆盖哪些层：搜索结果只说明「Hanna & Piotrowski 2025-08 发布的 Circuit Tracer Transcoders」，接入前需要到页面上确认。
- 「Qwen3 带 QK-Norm」来自对 Qwen3 架构的一般了解，本次未能访问 HF config（网络证书问题）确认；本系统 dashboard 代码中的配置 (36/32/8/2560) 与本文一致。
- 部分检索结果中出现了 2026 年的 arXiv 编号（例如 Scaling Monosemanticity 的 arXiv 版本），本文一律以 transformer-circuits.pub 原始发布为准。

---

## 10. 参考文献（URL）

**电路 / 归因图**
1. Elhage et al. 2021, A Mathematical Framework for Transformer Circuits — https://transformer-circuits.pub/2021/framework/index.html
2. Ameisen et al. 2025, Circuit Tracing: Revealing Computational Graphs in Language Models — https://transformer-circuits.pub/2025/attribution-graphs/methods.html
3. Lindsey et al. 2025, On the Biology of a Large Language Model — https://transformer-circuits.pub/2025/attribution-graphs/biology.html
4. Hanna, Piotrowski, Lindsey, Ameisen 2025, circuit-tracer (BlackboxNLP) — https://aclanthology.org/2025.blackboxnlp-1.14/
5. Neuronpedia Qwen3-4B — https://www.neuronpedia.org/qwen3-4b

**特征 / 神经元**
6. Bricken et al. 2023, Towards Monosemanticity — https://transformer-circuits.pub/2023/monosemantic-features
7. Templeton et al. 2024, Scaling Monosemanticity — https://transformer-circuits.pub/2024/scaling-monosemanticity/
8. Bills et al. 2023, Language models can explain neurons in language models — https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html
9. Huang et al. 2023, Rigorously Assessing Natural Language Explanations of Neurons — https://arxiv.org/abs/2309.10312
10. Bolukbasi et al. 2021, An Interpretability Illusion for BERT — https://arxiv.org/abs/2104.07143
11. Neuronpedia — https://www.neuronpedia.org/ ; https://github.com/hijohnnylin/neuronpedia
12. Marks et al. 2025, Auditing Language Models for Hidden Objectives — https://arxiv.org/abs/2503.10965
13. GDM Mech Interp Team 2025, Negative Results for SAEs on Downstream Tasks — https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9

**逐层读出**
14. nostalgebraist 2020, interpreting GPT: the logit lens — https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
15. Belrose et al. 2023, Tuned Lens — https://arxiv.org/abs/2303.08112
16. Ghandeharioun et al. 2024, Patchscopes (ICML) — https://arxiv.org/abs/2401.06102
17. Geva et al. 2022, FFN Layers Build Predictions (EMNLP) — https://arxiv.org/abs/2203.14680
18. Geva et al. 2022, LM-Debugger (EMNLP Demos) — https://aclanthology.org/2022.emnlp-demos.2/
19. Tufanov et al. 2024, LM Transparency Tool — https://arxiv.org/abs/2404.07004

**注意力**
20. Vig 2019, BertViz (ACL Demos) — https://aclanthology.org/P19-3007/
21. Yeh et al. 2023, AttentionViz (IEEE VIS / TVCG) — https://arxiv.org/abs/2305.03210
22. Jain & Wallace 2019, Attention is not Explanation (NAACL) — https://arxiv.org/abs/1902.10186
23. Wiegreffe & Pinter 2019, Attention is not not Explanation (EMNLP) — https://arxiv.org/abs/1908.04626
24. Abnar & Zuidema 2020, Quantifying Attention Flow in Transformers (ACL) — https://arxiv.org/abs/2005.00928

**教育 / 结构可视化**
25. Cho et al., Transformer Explainer (IEEE VIS 2024 → CHI '26) — https://arxiv.org/abs/2408.04619
26. Bycroft 2023, LLM Visualization — https://bbycroft.net/llm
27. Tenney et al. 2020, LIT (EMNLP Demos) — https://aclanthology.org/2020.emnlp-demos.15/

**激活监测**
28. Zou et al. 2023, Representation Engineering — https://arxiv.org/abs/2310.01405
29. MacDiarmid et al. 2024, Simple probes can catch sleeper agents — https://www.anthropic.com/research/probes-catch-sleeper-agents
30. Goldowsky-Dill et al. 2025, Detecting Strategic Deception Using Linear Probes — https://arxiv.org/abs/2502.03407
31. Zou et al. 2024, Improving Alignment and Robustness with Circuit Breakers — https://arxiv.org/abs/2406.04313
32. Bailey et al. 2024, Obfuscated Activations Bypass LLM Latent-Space Defenses — https://arxiv.org/abs/2412.09565
33. Kossen et al. 2024, Semantic Entropy Probes — https://arxiv.org/abs/2406.15927
34. Orgad et al. 2025, LLMs Know More Than They Show (ICLR) — https://arxiv.org/abs/2410.02707
35. CH-Wang et al. 2024, Do Androids Know They're Only Dreaming of Electric Sheep? (Findings ACL) — https://arxiv.org/abs/2312.17249
36. Chen, Arditi, Sleight, Evans, Lindsey 2025, Persona Vectors — https://arxiv.org/abs/2507.21509
37. Chen et al. 2024, Designing a Dashboard for Transparency and Control of Conversational AI (TalkTuner) — https://arxiv.org/abs/2406.07882

**综述 / 用户研究 / 方法论批评**
38. Braşoveanu et al. 2024, Visualizing Large Language Models: A Brief Survey (IEEE IV) — https://eprints.weblyzard.com/123/1/ieee_iv_2024_visualizing_llms_a_brief_survey.pdf
39. Wang et al., HCI and Visualization in NLG Models (Frontiers of CS) — https://arxiv.org/abs/2410.08723
40. Li et al., ConceptViz (IEEE TVCG 2026) — https://arxiv.org/abs/2509.20376
41. Kaur et al. 2020, Interpreting Interpretability (CHI) — https://dl.acm.org/doi/10.1145/3313831.3376219
42. Hase & Bansal 2020, Evaluating Explainable AI (ACL) — https://arxiv.org/abs/2005.01831
43. Makelov, Lange, Nanda 2023, An Interpretability Illusion for Subspace Activation Patching — https://arxiv.org/abs/2311.17030
