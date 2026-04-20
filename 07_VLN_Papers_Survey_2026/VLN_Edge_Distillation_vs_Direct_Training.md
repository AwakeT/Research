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
| **Kinbot** | 27B Teacher→4B Student | 多阶段训练+蒸馏 | **是**，蒸馏房间理解+目标检测等空间认知能力（**不蒸馏动作**，动作输出由后续阶段/其他模块负责） |

蒸馏路线仅有MiniVLN和Kinbot两家，且MiniVLN面向的是**离散图导航**（节点选择），非连续动作空间。值得注意的是，Kinbot的蒸馏目标与MiniVLN及其他VLN方案有本质区别——Kinbot蒸馏的是**空间认知能力**（房间理解、目标检测），而非导航动作。

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

### 4.0 Kinbot蒸馏路线的本质——蒸馏空间认知能力，而非动作

**必须首先澄清一个关键事实：Kinbot的蒸馏目标与上述VLN方案的端到端动作输出是完全不同的范畴。**

Kinbot的阶段性蒸馏目标是将Teacher（27B Qwen VL）训练或微调后获得的**房间理解能力**和**目标检测能力**继承到Student（4B）上。具体而言：
- **蒸馏的是**：空间语义认知（房间类型识别、区域划分理解、家具/物品识别与定位、空间关系推理）
- **不蒸馏的是**：导航动作（x, y, θ轨迹点、底盘控制指令）
- **动作输出的规划**：等认知能力蒸馏完成、整流程跑通后，再单独构建动作输出模块，或由SLAM+局部规划+安全链等其他模块负责

这意味着Kinbot的蒸馏**完全落在VLM的核心能力圈内**——大模型确实比小模型更擅长视觉场景理解、目标检测、空间关系推理。第二节分析的"动作空间不匹配"问题**对Kinbot不适用**，因为Kinbot根本不在蒸馏阶段涉及动作空间。

### 4.1 Kinbot蒸馏路线的深层合理性

理解了Kinbot蒸馏的真正目标后，其合理性远比"一般认知蒸馏"更强：

**（1）房间理解和目标检测恰好是大模型相对小模型的核心优势区域**

27B VLM相比4B VLM的能力差距，主要体现在：
- 复杂场景的语义理解（区分"客厅的休闲区"和"客厅的用餐区"）
- 细粒度目标检测与属性识别（"红色的遥控器"而非仅"遥控器"）
- 空间关系推理（"沙发左边的茶几上"的多跳推理）
- 多义性消解（"那个柜子"在多个柜子的场景中指代谁）

这些能力**完全在VLM预训练分布内**，是蒸馏最有效的知识类型。这与其他VLN方案试图蒸馏"导航动作"（不在预训练分布内）的困境形成鲜明对比。

**（2）认知能力蒸馏的损失设计天然明确**

Kinbot的结构化认知输出（JSON格式）为蒸馏提供了清晰的监督信号：
- `region_ranking`（区域排序）→ rank loss / listwise loss蒸馏
- `location_cls`（位置分类）→ KL散度蒸馏
- `confidence`（置信度）→ MSE蒸馏
- 搜索建议 → 序列级蒸馏

每个子任务都有成熟的蒸馏损失设计，**不存在"输出分布形态不匹配"的问题**——这是端到端动作蒸馏的最大难题，但Kinbot完全避开了。

**（3）分阶段蒸馏策略与能力分层高度耦合**

Kinbot的P0→P1→P2能力分层为蒸馏提供了天然的课程学习框架：
- P0阶段蒸馏：空间理解（房间识别、区域划分）+ 基础目标检测 → 最基础的视觉认知
- P1阶段蒸馏：多视角融合 + 几何约束理解 + 精细空间推理 → 增强认知
- 动作输出：**不在蒸馏流程内**，由独立模块或后续阶段构建

这种"先蒸馏认知，后构建动作"的分离策略，实际上比端到端VLA的"认知+动作一起训练"更工程友好——每个阶段的目标单一明确，易于评估和调试。

**（4）与其他方案的本质区别：蒸馏的是"看懂"而非"怎么走"**

| 方案 | 蒸馏/训练目标 | 是否在VLM能力圈内 | 蒸馏可行性 |
|------|-------------|----------------|----------|
| 端到端VLA（ABot-N0等） | 视觉→连续动作轨迹 | **否**，动作控制非VLM预训练内容 | 低（故选直接训练+RL） |
| MiniVLN | 视觉+指令→离散节点选择 | 部分（节点选择≈分类） | 中（已验证+4% SR） |
| **Kinbot** | **视觉→房间理解+目标检测+空间关系** | **是**，完全在VLM核心能力内 | **高**（蒸馏目标与VLM优势完全对齐） |

### 4.2 需要关注的风险

尽管Kinbot的蒸馏路线在原理上高度合理，仍有以下风险需要管理：

1. **认知标注数据的规模与质量**：Teacher的房间理解和目标检测能力来自其训练数据。ABot-N0用5.0M认知推理样本训练4B模型直接达到高水平——Kinbot的Teacher训练数据至少需要同等量级的场景认知标注（房间类型、物品位置、空间关系等），数据建设是核心瓶颈

2. **27B→4B的认知压缩损失**：7:1的压缩比下，4B模型能否保留Teacher在复杂场景下的细粒度认知能力（如区分"衣帽间的外套区"和"衣帽间的鞋区"），需要实验验证。简单场景可能无损，复杂场景可能出现显著退化

3. **Teacher训练本身的投入**：27B Teacher需要多阶段训练（T1→T2→T3→T4），如果Teacher在房间理解/目标检测上没有显著超越4B直接训练的效果，蒸馏增益将不明显

4. **蒸馏损失的多任务平衡**：`0.4×region_ranking + 0.3×location_cls + 0.2×next_action + 0.1×confidence`的静态权重是否最优？不同能力的蒸馏难度不同（空间关系推理可能比简单目标检测更难蒸馏），可能需要动态加权

### 4.3 建议实验

在P0阶段进行**对照实验**，验证"蒸馏认知能力"相对于"直接训练认知能力"的增益：

| 实验组 | 方法 | 验证目标 |
|-------|------|---------|
| **A组** | 4B直接SFT（大规模房间理解+目标检测标注数据） | 基线：4B模型直接训练能达到什么水平 |
| **B组** | 4B直接SFT + GRPO（奖励：房间识别准确率+目标检测recall） | 验证：RL后训练对认知能力的增益 |
| **C组** | 27B Teacher训练 → 蒸馏到4B（多轮蒸馏） | 验证：蒸馏路线的认知能力 |
| **D组** | 27B Teacher训练 → 蒸馏到4B → GRPO后训练 | 验证：蒸馏+RL组合是否最优 |

重点对比：
- **C组 vs A组**：蒸馏增益有多大？如果C组在房间理解和目标检测上显著优于A组，蒸馏路线成立
- **B组 vs C组**：对于认知能力（非动作），蒸馏和RL哪个增益更大？
- **D组 vs B组**：蒸馏是否为RL提供了更好的认知初始化？

**预期**：由于Kinbot蒸馏的是VLM核心能力（视觉场景理解），C组大概率优于A组，尤其在复杂场景下。但需要量化增益是否值得27B Teacher的训练成本。

### 4.4 补充建议：Teacher作为数据标注引擎

除了传统的在线蒸馏（Teacher/Student同时前向传播+对齐loss），Kinbot还可以充分利用27B Teacher作为**高质量认知标注的数据引擎**：

1. **Phase 1 — Teacher标注**：27B Teacher对大量室内场景图像生成结构化认知标注（房间类型、区域划分、物品清单、空间关系描述），构建大规模认知数据集
2. **Phase 2 — Student SFT**：4B Student在Teacher标注的数据上SFT（本质上是离线蒸馏/数据蒸馏，将Teacher的认知能力转化为训练数据）
3. **Phase 3 — Student RL后训练**：在仿真环境中用GRPO进一步强化Student的认知判断质量
4. **后续阶段 — 动作输出构建**：认知能力稳定后，独立构建动作输出模块或接入SLAM+局部规划系统

这条路线的优势：
- 利用了Teacher的数据生成能力（27B标注数据比人工标注更快更便宜），本质上类似ABot-N0的做法（其5.0M认知推理样本也是VLM自动生成的）
- 在线蒸馏和离线数据蒸馏可以**并行使用**：在线蒸馏对齐中间表征，离线数据蒸馏扩充训练样本
- 认知能力与动作输出的**解耦开发**降低了整体工程风险——认知蒸馏失败不影响动作模块开发，反之亦然

---

## 五、总结

### 5.1 "直接训练 vs 蒸馏"的适用场景

| 蒸馏目标 | 推荐路线 | 原因 |
|---------|---------|------|
| 端到端动作预测（输出x,y,θ） | **直接训练 + RL** | 大模型缺乏动作知识，RL环境奖励更直接 |
| 离散图导航（节点选择） | 蒸馏可行 | Teacher/Student输出分布一致（MiniVLN已验证） |
| **视觉场景认知（房间理解、目标检测、空间推理）** | **蒸馏高度合理** | **完全在VLM核心能力圈内**，大模型的认知优势可有效传递 |
| Token极度受限的端侧 | 直接训练小模型 | 专用架构设计比压缩通用模型更高效 |

### 5.2 Kinbot的独特定位

当前VLN领域不用蒸馏的核心原因是**蒸馏目标与大模型能力不匹配**——端到端VLA方案需要的动作控制知识不在大VLM中，所以直接训练+RL更高效。

**但Kinbot走的是完全不同的路线。** Kinbot的蒸馏阶段目标是将Teacher训练/微调后的**房间理解能力**和**目标检测能力**继承到Student上，动作输出留给后续阶段或其他模块。这使得Kinbot的蒸馏目标**精准落在VLM的核心优势区域**——视觉场景理解、语义推理、目标识别——27B模型在这些任务上确实显著优于4B模型，蒸馏有明确的知识增量。

这一设计选择使Kinbot**同时享有两个路线的优势**：
- 认知能力通过蒸馏高效继承（利用大模型在视觉理解上的绝对优势）
- 动作输出通过独立模块构建（避开动作空间不匹配的蒸馏陷阱）

### 5.3 一句话总结

**当前VLN领域不用蒸馏的核心原因是端到端动作知识不在大模型里。但Kinbot蒸馏的是房间理解和目标检测等空间认知能力——这恰恰是大模型的核心优势区域，蒸馏路线不仅合理且有充分的理论依据。动作输出留给后续阶段独立构建，认知与动作的解耦是Kinbot方案的关键架构优势。**

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
