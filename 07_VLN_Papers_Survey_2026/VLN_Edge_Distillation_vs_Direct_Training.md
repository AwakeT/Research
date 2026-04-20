# 端侧VLN为何大多直接训练小模型，而非大模型蒸馏？

> 本文结合9篇VLN论文对比分析文档（VLN_Papers_vs_Kinbot_Comparison.md）中的实践案例与额外文献调研，系统分析当前端侧部署VLN方案中"直接训练小模型"占主流而非"大模型蒸馏"的深层原因，并评估Kinbot蒸馏路线的合理性与风险。

---

## 一、现象观察：直接训练是主流

从9篇论文+近期文献来看，绝大多数端侧VLN方案选择了**在小模型上直接训练**，而非从大模型蒸馏：

| 方案 | 模型规模 | 训练方式 | 是否蒸馏 |
|------|---------|---------|---------|
| **ABot-N0** | Qwen3-4B | 认知预热→SFT→SAFE-GRPO | **否**，直接4B训练+部署 |
| **VLingNav** | LLaVA-Video-7B | 预训练→SFT→在线RL | **否**，单模型全流程 |
| **NavGRPO** | ScaleVLN | SFT→GRPO | **否**，直接RL强化 |
| **VLN-R1** | LVLM | SFT→GRPO-based RFT | **否**，10K样本RFT即超越完整SFT |
| **Efficient-VLN** | 直接训练 | 优化训练流程 | **否**，282 GPU-hours达到SOTA |
| **EmergeNav** | Qwen3-VL-8B/32B | 零样本 | **否**，无任何训练 |
| **SFCo-Nav** | GPT-4o+检测器 | 零样本 | **否**，无训练 |
| **MetaNav** | GPT-4o+检测器 | 零样本 | **否**，无训练 |
| **MiniVLN** | 蒸馏Student | 两阶段蒸馏 | **是**，唯一蒸馏方案 |
| **Kinbot** | 27B Teacher→4B Student | 多阶段训练+蒸馏 | **是**，蒸馏认知判断（非动作） |

蒸馏路线仅有MiniVLN和Kinbot两家，且MiniVLN面向的是**离散图导航**（节点选择），非连续动作空间。

---

## 二、核心原因分析

### 2.1 动作空间不匹配——大模型"没有"导航该蒸的知识

**这是最根本的原因。**

大尺寸VLM（如Qwen-VL-72B、GPT-4o等）的预训练分布是**视觉-语言对齐**（图文理解、VQA、推理），而VLN需要的是**连续3D空间中的精细动作控制**（x, y, θ路点）。大模型根本不具备细粒度动作级知识。

正如多项研究指出：

> "Large teacher models lack fine-grained action-level control knowledge needed for embodied navigation. The queries used to request robotic activities are usually separate from the distribution that VLMs are pre-trained to handle."
>
> —— VLN-R1 (arXiv: 2506.17221) & VLM Edge Survey (arXiv: 2502.07855)

**9篇论文中的直接印证：**

- **ABot-N0**：直接在Qwen3-4B上三阶段训练（认知预热→SFT→SAFE-GRPO），输出5个连续路点。16.9M专家轨迹+5.0M推理样本提供了导航专用知识——**这些知识不存在于任何通用大VLM中**
- **VLingNav**：在LLaVA-Video-7B上直接预训练+SFT+RL，输出连续(x,y,θ)。训练数据中的动作轨迹标注是从仿真器里采集的，不是从大模型蒸馏来的
- **NavGRPO**：在ScaleVLN上直接SFT+GRPO，RL奖励信号来自导航环境反馈，不来自Teacher模型

**本质问题**：蒸馏的前提是"Teacher有而Student没有的知识"。但对于导航动作预测，27B和4B模型的预训练知识都同样缺乏——它们的差距不在于"谁更懂导航"，而在于"谁更懂语言和视觉"。所以直接在4B上用导航专用数据训练，比绕一圈从27B蒸馏更高效。

### 2.2 RL/GRPO直接后训练在小模型上效果惊人

2025-2026年的一个关键发现：**基于强化学习的后训练**（尤其是GRPO系列）在小模型上效果出奇地好，大幅降低了对蒸馏的需求。

| 方案 | RL方法 | 关键发现 |
|------|--------|---------|
| **VLN-R1** | GRPO-based RFT | 仅10K样本RFT就**超过完整数据集SFT** |
| **ABot-N0** | SAFE-GRPO | 4B直接RL训练，7大基准全部SOTA |
| **NavGRPO** | DeGRPO | 去偏差变体消除超参敏感性，+14.89%鲁棒性 |

核心逻辑：**环境反馈信号（奖励函数）比Teacher模型的知识更适合教会小模型"怎么导航"**。仿真器可以提供无限的在线反馈，而蒸馏只能提供离线的Teacher输出分布。

VLN-R1的结论尤其值得关注：
> "Surprisingly, the LVLM fine-tuned with merely 10K samples via RFT outperforms its counterpart trained on the complete dataset. VLN-R1 proves LVLMs can drive embodied navigation and enhance task-specific reasoning through data-efficient, reward-driven post-training."

这意味着**奖励驱动的后训练**可能比传统SFT（蒸馏的典型范式）更高效。

### 2.3 跨模态对齐在蒸馏中极易被破坏

多模态蒸馏面临一个独特困难：**视觉-语言-动作三模态的对齐关系非常脆弱**。

VLA模型研究发现：
> "VLA models suffer from spurious forgetting — the alignment between robot actions and visual-text data appears fragile and susceptible to being overwritten during fine-tuning. Task interference arises where the conflicting parameter spaces of control and understanding tasks cause mutual performance degradation."
>
> —— ChatVLA (arXiv: 2502.14420) & Efficient VLA Survey (arXiv: 2510.17111)

具体困难包括：

1. **蒸馏损失设计困难**：VLN的输出是多维度的（位置、方向、置信度），不像NLP蒸馏只需对齐token分布。Kinbot设计的蒸馏损失（`0.4×region_ranking + 0.3×location_cls + 0.2×next_action + 0.1×confidence`）就体现了这种复杂性——4个子任务的损失加权比例如何确定？各子任务的蒸馏难度不同，权重是否需要动态调整？

2. **容量瓶颈**：27B→4B压缩比约7:1，对于需要同时保持空间推理、语义理解、动作决策的多模态任务，4B的参数容量可能不足以忠实复现Teacher的完整行为

3. **VLM蒸馏 ≠ LLM蒸馏**：纯文本LLM蒸馏已有成熟方法论（NVIDIA Minitron等），但多模态VLM的蒸馏还需要处理：
   - 视觉编码器对齐（Teacher/Student用不同视觉骨干怎么办）
   - 跨模态注意力保持（蒸馏后是否丢失视觉-语言关联）
   - 多语言能力保持（VLM蒸馏中多语言能力损失显著）

### 2.4 导航专用数据规模比模型规模更重要

2025-2026的实践一致表明：**导航专用数据的规模和质量**是性能的决定性因素，而非模型大小。

| 方案 | 模型规模 | 训练数据量 | 核心性能 |
|------|---------|-----------|---------|
| ABot-N0 | 4B直接训练 | 16.9M轨迹+5.0M推理 | R2R SR **66.4**, VLN-CE SOTA |
| VLingNav | 7B直接训练 | 1.6M预训练+4.5M SFT | 连续环境高SR |
| MiniVLN | 蒸馏（12%参数） | 继承Teacher数据 | R2R SR 77.59（离散图导航） |
| Efficient-VLN | 直接训练 | 优化训练流程 | R2R SR 64.2（仅282 GPU-hours）|
| VLN-R1 | 直接RFT | 仅10K样本 | 超过完整数据集SFT |

**关键洞察**：ABot-N0用4B直接训练在7大基准取得SOTA，核心壁垒是16.9M轨迹的数据飞轮，不是模型大小。**与其花精力设计蒸馏流程，不如把资源投入构建大规模导航专用数据。**

ABot-N0的数据来源组合提供了可复制的方法论：
- 互联网视频伪轨迹：2.0M（低成本快速扩充）
- 3D场景合成：1.7M（仿真器生成）
- 真机演示：340K（高质量但高成本）
- 认知推理标注：5.0M（VLM自动生成+人工校验）

### 2.5 专用架构设计优于通用压缩

从头设计适配端侧的高效架构，比压缩通用大模型更有效。

Small VLM（SVLM）领域的研究指出：
> "Unlike post-hoc compression (e.g., distillation), many SVLMs are designed from scratch for efficiency, incorporating shallow transformer layers, factorized embeddings, and optimized fusion blocks. This architecture-first approach helps mitigate knowledge loss and suboptimal adaptation, common challenges in compressed large models."
>
> —— Small Vision-Language Models Survey (ScienceDirect, 2025)

9篇论文中的具体案例：
- **ABot-N0**：推理头+动作头双头分离，避免推理污染动作——这种架构决策是蒸馏无法传递的，因为它改变了模型的输出结构
- **SFCo-Nav**：慢脑（LLM规划）+快脑（Grounding-DINOv2检测）的双速架构——架构创新比模型大小更重要
- **EmergeNav**：PST三阶段结构化推理框架——将规划、执行、阶段转换解耦为独立模块

**Tiny-VLA**的研究进一步证明：
> "Expensive full-parameter adaptation of large pre-trained models is unnecessary. LoRA等PEFT方法+小规模领域数据就能实现端侧部署。"

### 2.6 蒸馏流程的工程复杂度高

蒸馏路线引入了显著的工程复杂度：

1. **双模型维护成本**：需要同时维护Teacher（27B）和Student（4B）两套训练流程
2. **蒸馏超参调优**：损失加权、温度参数、蒸馏阶段划分等超参数需要大量实验
3. **Teacher训练本身的成本**：27B模型的多阶段训练需要大量算力，且Teacher性能直接决定蒸馏天花板
4. **迭代周期长**：每次改进都要先训Teacher再蒸Student，迭代效率低于直接训练小模型

相比之下，直接在4B模型上训练：
- 单模型流程，迭代快
- RL/GRPO可以端到端优化
- 数据工程投入直接转化为性能提升

---

## 三、MiniVLN——蒸馏路线的唯一成功案例及其局限

[MiniVLN](https://arxiv.org/abs/2409.18800)（ICRA 2024）是目前VLN领域蒸馏路线的唯一系统性成功案例，值得详细分析：

### 3.1 MiniVLN的方法

- **两阶段渐进蒸馏**：
  - 预训练阶段：Embedding蒸馏 + Attention-based蒸馏 + Hidden States蒸馏（Transformer逐层对齐）
  - 微调阶段：导航特定蒸馏（动作预测分布对齐）
- **效果**：仅用Teacher 12%参数，比非蒸馏基线高约4% SR
- **R2R test unseen**：SR 77.59, SPL 68.05

### 3.2 MiniVLN成功的前提条件

MiniVLN的蒸馏成功依赖于特殊条件：
1. **离散图导航**（节点选择），非连续动作预测——Teacher/Student的输出分布形态相似
2. **Teacher和Student架构高度一致**——同为VLN Transformer，仅层数/维度不同
3. **预训练蒸馏+微调蒸馏两阶段**——不仅蒸馏任务知识，还蒸馏预训练表征

### 3.3 MiniVLN暴露的蒸馏局限

1. **仅4%的提升**：相比非蒸馏基线仅提升约4% SR，性价比存疑
2. **不适用于连续动作空间**：离散图导航（选节点）→蒸馏可行；连续动作（输出x,y,θ）→Teacher缺乏动作级知识，蒸馏困难
3. **未与RL路线对比**：MiniVLN未比较"蒸馏 vs 直接RL后训练"，而VLN-R1表明RL后训练可能获得更大提升
4. **缺乏边缘部署验证**：MiniVLN仅在仿真中评估，未验证端侧推理效率

---

## 四、对Kinbot蒸馏路线的启示

Kinbot的蒸馏路线与上述分析有一个**关键区别**：**Kinbot不蒸馏动作输出，而是蒸馏认知判断**（区域排序、位置分类、搜索建议的结构化JSON）。

### 4.1 Kinbot蒸馏路线的合理性

Kinbot的设计实际上**巧妙地避开了"动作空间不匹配"这个最大陷阱**：

1. **蒸馏目标是语义推理而非动作预测**：27B Teacher确实比4B Student有更强的空间推理和语义理解能力——这正是通用VLM的核心优势。蒸馏"哪个房间最可能有目标物"（region_ranking）、"当前位于什么类型的空间"（location_cls）这类认知任务，Teacher确实拥有Student所不具备的知识
2. **结构化JSON输出比连续轨迹更适合蒸馏**：
   - `region_ranking`是一个排序任务 → 可用rank loss蒸馏
   - `location_cls`是分类任务 → 可用KL散度蒸馏
   - `confidence`是标量回归 → 可用MSE蒸馏
   - 每个子任务都有明确的蒸馏损失设计，不存在"输出分布形态不匹配"的问题
3. **"不输出动作"的设计决策**让蒸馏聚焦于VLM的核心优势区域（视觉理解+语义推理），而非其天然弱项（动作控制）

### 4.2 需要警惕的风险

1. **数据瓶颈**：ABot-N0用16.9M轨迹达到4B SOTA。Kinbot的Teacher训练至少需要同量级的认知标注数据（场景图+区域排序+位置分类标注），数据建设成本不可忽视

2. **蒸馏增益是否值得**：如果直接在4B上用大规模认知标注数据SFT+RL，可能接近甚至超过蒸馏效果。VLN-R1的结论（10K样本GRPO > 完整数据集SFT）暗示：奖励驱动训练可能是更高效的路径

3. **27B Teacher的训练成本**：Teacher本身需要多阶段训练（T1→T2→T3→T4），每个阶段都需要大量数据和算力。如果Teacher性能不够好，蒸馏出的Student也无法超越直接训练的4B

4. **蒸馏损失调优成本**：4个子任务的加权比例（0.4:0.3:0.2:0.1）是静态的，但不同训练阶段、不同场景下最优权重可能不同

### 4.3 建议实验

在P0阶段进行**对照实验**以验证蒸馏路线的必要性：

| 实验组 | 方法 | 目标 |
|-------|------|------|
| **A组** | 4B直接SFT（大规模认知标注数据） | 基线 |
| **B组** | 4B直接SFT + GRPO（奖励：搜索准确度+效率） | 验证RL增益 |
| **C组** | 27B Teacher训练 → 蒸馏到4B | 验证蒸馏增益 |
| **D组** | 27B Teacher训练 → 蒸馏到4B → GRPO后训练 | 蒸馏+RL组合 |

重点对比：
- C组 vs A组：蒸馏增益有多大？
- B组 vs C组：直接RL和蒸馏哪个更高效？
- D组 vs B组：蒸馏是否为RL提供了更好的初始化？

### 4.4 折中建议：蒸馏作为初始化，RL作为精调

结合当前趋势，Kinbot可以考虑**混合路线**：

1. **Phase 1**：27B Teacher生成大规模认知标注数据（利用Teacher的推理能力做数据标注，而非做在线蒸馏）
2. **Phase 2**：4B Student在Teacher标注的数据上SFT（本质上是离线蒸馏/数据蒸馏）
3. **Phase 3**：4B Student做GRPO后训练（环境反馈+认知质量奖励）

这条路线：
- 避免了在线蒸馏的工程复杂度
- 利用了Teacher的数据生成能力（27B标注数据比人工标注更快更便宜）
- 保留了RL后训练的优势（VLN-R1证明了其有效性）
- 本质上类似ABot-N0的做法（它的5.0M认知推理样本也是VLM自动生成的）

---

## 五、总结

### 5.1 "直接训练 vs 蒸馏"的适用场景

| 场景 | 推荐路线 | 原因 |
|------|---------|------|
| 端到端动作预测（输出x,y,θ） | **直接训练 + RL** | 大模型缺乏动作知识，RL奖励更直接 |
| 离散图导航（节点选择） | 蒸馏可行 | Teacher/Student输出分布一致（MiniVLN已验证） |
| **认知判断输出（排序/分类/JSON）** | **蒸馏合理，但需验证增益** | Teacher有语义推理优势，但直接SFT+RL可能也够 |
| Token极度受限的端侧 | 直接训练小模型 | 专用架构设计比压缩通用模型更高效 |

### 5.2 Kinbot的独特定位

Kinbot的"蒸馏认知判断而非动作"的路线在当前VLN领域是**独特且合理的**——它避开了蒸馏最大的陷阱（动作空间不匹配），聚焦于大模型确实擅长的能力（语义推理）。但需要通过对照实验验证蒸馏增益是否能覆盖其额外成本，并考虑"Teacher做数据标注 + Student直接SFT+RL"的混合路线作为备选。

### 5.3 一句话总结

**当前VLN领域不用蒸馏的核心原因是：导航动作知识不在大模型里，导航专用数据+RL比蒸馏更直接有效。但Kinbot蒸馏的是认知判断而非动作，路线本身合理——关键风险在于数据规模和蒸馏增益的性价比，建议通过对照实验验证。**

---

## 参考文献

1. [Vision-Language Models for Edge Networks: A Comprehensive Survey (arXiv: 2502.07855)](https://arxiv.org/abs/2502.07855)
2. [MiniVLN: Efficient Vision-and-Language Navigation by Progressive Knowledge Distillation (ICRA 2024, arXiv: 2409.18800)](https://arxiv.org/abs/2409.18800)
3. [VLN-R1: Vision-Language Navigation via Reinforcement Fine-Tuning (arXiv: 2506.17221)](https://arxiv.org/abs/2506.17221)
4. [ABot-N0: Technical Report on the VLA Foundation Model for Versatile Embodied Navigation (arXiv: 2602.11598)](https://arxiv.org/abs/2602.11598)
5. [Efficient Vision-Language-Action Models for Embodied Manipulation: A Systematic Survey (arXiv: 2510.17111)](https://arxiv.org/abs/2510.17111)
6. [Pure Vision Language Action Models: A Comprehensive Survey (arXiv: 2509.19012)](https://arxiv.org/abs/2509.19012)
7. [Large-Scale Model-Enhanced Vision-Language Navigation (Sensors, 2026)](https://www.mdpi.com/1424-8220/26/7/2022)
8. [Scaling down, Powering up: A Survey on Small Vision-Language Models (ScienceDirect, 2025)](https://www.sciencedirect.com/science/article/abs/pii/S156625352500867X)
9. [Efficient-VLN: A Training-Efficient Vision-Language Navigation Model (arXiv: 2512.10310)](https://arxiv.org/pdf/2512.10310)
10. [ChatVLA: Unified Multimodal Understanding and Robot Control (arXiv: 2502.14420)](https://arxiv.org/abs/2502.14420)
11. [A Survey on Efficient Vision-Language-Action Models (arXiv: 2510.24795)](https://arxiv.org/abs/2510.24795)
12. [Knowledge Distillation and Dataset Distillation of Large Language Models (arXiv: 2504.14772)](https://arxiv.org/abs/2504.14772)
