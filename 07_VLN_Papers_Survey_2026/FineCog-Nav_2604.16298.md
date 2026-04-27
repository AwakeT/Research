---
title: "FineCog-Nav: Integrating Fine-grained Cognitive Modules for Zero-shot Multimodal UAV Navigation"
method_name: "FineCog-Nav"
authors: [Dian Shao, Zhengzheng Xu, Peiyang Wang, Like Liu, Yule Wang, Jieqi Shi, Jing Huo]
year: 2026
venue: arXiv
tags: [uav-navigation, vision-language-navigation, zero-shot-navigation, cognitive-architecture, hierarchical-memory, llm-agent]
zotero_collection: ""
image_source: online
arxiv_html: https://arxiv.org/html/2604.16298v1
created: 2026-04-27
---

# 论文笔记：FineCog-Nav: Integrating Fine-grained Cognitive Modules for Zero-shot Multimodal UAV Navigation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Northwestern Polytechnical University, Nanjing University |
| 日期 | April 2026 |
| 项目主页 | [Project Page](https://smartdianlab.github.io/projects-FineCogNav) |
| 对比基线 | [[NavGPT]], [[DiscussNav]] |
| 链接 | [arXiv](https://arxiv.org/abs/2604.16298) / [Project](https://smartdianlab.github.io/projects-FineCogNav) |

---

## 一句话总结

> FineCog-Nav 以人类认知功能为蓝本，将零样本无人机 VLN 分解为 8 个细粒度认知模块（解析、感知、注意、记忆、想象、判断、决策），每个使用中等规模基础模型配以角色特化提示。

---

## 核心贡献

1. **认知启发式零样本框架**: 首个显式建模核心认知功能间相互依赖关系的无人机 VLN 框架，模块按认知功能而非 agent 身份组织
2. **AerialVLN-Fine 基准**: 从 AerialVLN 精选 300 条高质量轨迹，提供句子级指令-轨迹对齐和精炼指令
3. **广泛验证**: 在多种 LLM（8B-72B）上一致超越基线，人类研究进一步确认优势

---

## 问题背景

### 要解决的问题
无人机 VLN 要求从第一人称视角跟随多步模糊指令在复杂 3D 环境中导航——比地面 VLN 更具挑战性（长距离、连续飞行、3D 空间推理）。

### 现有方法的局限
- 严重依赖大型 LLM/VLM：当 GPT-4V 替换为 LLaVA-7B 时性能从 28.3 暴跌至 1.7（CityNavAgent）
- 使用通用 prompt 和松散的模块协调，缺乏层次化规划、动态子目标提取和记忆机制
- 现有框架要么单模块（BaseModel），要么多 agent 讨论（DiscussNav）但效率低

### 本文的动机
人类导航涉及多种协同认知功能：语言理解、场景感知、注意力聚焦、记忆管理、想象预测、进度判断、决策执行。将这些功能显式建模为独立但相互依赖的模块，可以用中等规模模型实现高质量导航。

---

## 方法详解

### 模型架构

FineCog-Nav 采用 **自顶向下认知模块化** 架构：
- **输入**: 自然语言指令 $I$ + 第一人称 RGB-D 视图 $V_t$
- **8 个认知模块**: Instruction Parser → Subgoal Extractor → Perception → Attention → Imagination → Memory → Subgoal Judger → Decision-Making
- **VLM**: [[Qwen2.5-VL]]-32B（固定）
- **LLM**: 可变（8B-72B 多种模型）
- **动作空间**: TaskFinish, MoveForward($d$), TurnLeft/Right($\theta$), Ascend/Descend($h$), MoveLeft/Right($d$)，$\theta=15°, h=2, d=5$

### 核心模块

#### 模块1: Instruction Parsing & Subgoal Extraction

**设计动机**: 将长程多步指令分解为可执行的子目标序列，实现层次化规划。

**具体实现**:
- **Instruction Parser** $\mathcal{S}$ 将指令分割为句子+地标对序列
- **Subgoal Extractor** $\mathcal{E}$ 从每条句子生成有序可执行子目标
- 子目标优先按执行顺序排列而非句法结构

#### 模块2: Perception guided by Attention

**设计动机**: 利用[[注意力机制]]引导感知模块聚焦于与当前子目标相关的视觉线索。

**具体实现**:
- **Attention Module** 识别当前和下一句子的关键地标 $\{L_i, L_{i+1}\}$，生成目标查询
- **Perception Module** $\mathcal{P}$ 基于查询描述当前 FPV 场景
- 轻量规则安全模块从深度图 $V_t^{(D)}$ 估计碰撞风险，输出安全警告 $W_t$

#### 模块3: Subgoal Judgement with Imagination

**设计动机**: 通过"想象"子目标完成时的预期场景，提供判断当前子目标是否达成的参考依据。

**具体实现**:
- **Imagination Module** 生成地标导向的预期场景描述 $R^{[g_i^{(k)}]}$
- **Subgoal Judger** $\mathcal{J}$ 综合当前观测 $O_t$、子目标记忆 $M^{[g_i^{(k)}]}$ 和想象参考 $R^{[g_i^{(k)}]}$ 判断是否完成

#### 模块4: Multi-level Memory Management

**设计动机**: 在不同粒度上管理历史信息，为长程导航提供充分的上下文。

**具体实现**:
- **Step Memory** $M^{[t]}$: 每步的观测和动作（"I see..., I do..."）
- **Subgoal Memory** $M^{[g_i^{(k)}]}$: 子目标期间累积的步记忆，完成后压缩为 $M^{*[g_i^{(k)}]}$
- **Instruction Memory** $M^{[I_i]}$: 聚合已完成子目标的压缩摘要
- 层次结构防止上下文窗口溢出，保留关键信息

#### 模块5: Decision-Making

**设计动机**: 综合所有认知模块的输出，选择最优动作并生成可解释的推理过程。

**具体实现**:
- 输出动作偏好得分 $p$、选定动作 $a_t$ 和可解释理由 $\tau$

---

## 关键公式

### 公式1: [[Instruction Parsing|指令解析]]

$$
\mathcal{S}(I) = \{(I_1, L_1), (I_2, L_2), \ldots, (I_N, L_N)\}
$$

**含义**: 将自然语言指令分割为 $N$ 个句子-地标对。

**符号说明**:
- $I$: 完整导航指令
- $I_i$: 第 $i$ 条指令句子
- $L_i$: 第 $i$ 句中的关键地标

### 公式2: [[Subgoal Planning|子目标提取]]

$$
\{g_i^{(k)}\}_{k=1}^{K} = \mathcal{E}(I_i, O_t)
$$

**含义**: 从指令句子和当前观测中提取有序子目标序列。

**符号说明**:
- $g_i^{(k)}$: 第 $i$ 句的第 $k$ 个子目标
- $\mathcal{E}$: 子目标提取器
- $O_t$: 当前时间步观测

### 公式3: [[Attention Mechanism|注意力引导查询]]

$$
\{Q_i\} = \text{Attention}(\{L_i, L_{i+1}\})
$$

**含义**: 注意力模块基于当前和下一句地标生成感知查询。

### 公式4: [[Visual Perception|引导感知]]

$$
O_t = \mathcal{P}(V_t^{(\text{RGB})}, \{Q_i\})
$$

**含义**: 感知模块基于注意力查询描述当前 FPV 场景。

### 公式5: [[Subgoal Planning|子目标判断]]

$$
\mathcal{J}(O_t, g_i^{(k)}, M^{[g_i^{(k)}]}, R^{[g_i^{(k)}]}) \to \{\text{True}, \text{False}\}
$$

**含义**: 综合观测、记忆和想象参考判断当前子目标是否完成。

### 公式6: [[Hierarchical Memory|指令级记忆聚合]]

$$
M^{[I_i]} = \{M^{*[g_i^{(1)}]}, M^{*[g_i^{(2)}]}, \ldots, M^{*[g_i^{(k)}]}\}
$$

**含义**: 指令级记忆由所有已完成子目标的压缩摘要组成。

### 公式7: [[Decision Making|决策输出]]

$$
\{p, a_t, \tau\} = \mathcal{D}(g_i^{(k)}, I_i, O_t, W_t, M^{[I_i]})
$$

**含义**: 决策模块综合子目标、指令、观测、安全警告和记忆，输出动作偏好、选定动作和推理理由。

**符号说明**:
- $p$: 动作偏好得分
- $a_t$: 选定动作
- $\tau$: 可解释推理过程
- $W_t$: 碰撞安全警告

---

## 关键图表

### Figure 1: Framework Overview / 框架概览

![Figure 1](https://arxiv.org/html/2604.16298v1/x1.png)

**说明**: FineCog-Nav 模拟人类多种认知功能：指令解析、子目标提取、感知、注意、想象、记忆、子目标判断、决策。每个模块由中等规模基础模型驱动。

### Figure 2: Detailed Architecture / 详细架构

![Figure 2](https://arxiv.org/html/2604.16298v1/x2.png)

**说明**: FineCog-Nav 详细架构，展示认知模块间的相互依赖关系。LLM/VLM 模块通过结构化 I/O 协议协作，显式建模认知功能间的依赖。

### Figure 3: AerialVLN-Fine Dataset / 数据集概览

![Figure 3](https://arxiv.org/html/2604.16298v1/x3.png)

**说明**: AerialVLN-Fine 数据集。左：细粒度标注示例（句子级指令-轨迹对齐）；右：场景、指令和轨迹长度分布可视化。

### Figure 4: Qualitative Example / 定性示例

![Figure 4](https://arxiv.org/html/2604.16298v1/x4.png)

**说明**: FineCog-Nav 定性示例。左：逐步推理过程与子目标；右：鸟瞰轨迹视图。

### Figure 5: Human Study Results / 人类研究结果

![Figure 5](https://arxiv.org/html/2604.16298v1/x5.png)

**说明**: 约 70 名参与者对匿名 FPV 导航视频评分（1-5 分），FineCog-Nav 一致优于基线方法，即使在失败案例中也被更青睐。

### Figure 6: Real-World Deployment / 真实世界部署

![Figure 6](https://arxiv.org/html/2604.16298v1/x6.png)

**说明**: 在 [[RoboMaster TT]] 无人机上的初步真实部署。给定指令"帮我找到椅子，然后降落在桌上"，agent 在 17 步后到达目标区域。

### Figure 7: AerialVLN Dataset Issues / AerialVLN 数据集问题

![Figure 7](https://arxiv.org/html/2604.16298v1/x7.png)

**说明**: 对 200 个随机采样的指令-轨迹对的人工分析揭示四类问题：轨迹-指令错位 (47%)、碰撞异常 (27%)、模糊指令 (17%)、不可见地标 (15%)。

### Figure 8: AerialVLN-Fine Construction Process / 数据集构建流程

![Figure 8](https://arxiv.org/html/2604.16298v1/x8.png)

**说明**: AerialVLN-Fine 构建过程，包括轨迹筛选、指令分割、轨迹分割和双盲质量控制。

### Figure 9: Scene Distribution / 场景分布

*(图片外链不可达，为补充材料中的场景分布图)*

**说明**: AerialVLN-Fine 15 个场景的分布和示例，覆盖日间/夜间、城市/乡村、工业区等多种环境。包括城市（Abandoned City, Nighttime City, Rail Station, Hong Kong City）、乡村/郊区（Abandoned Rural Area, Southern Village, Industrial Zone, Warehouse Area）等。

### Table 1: Comparison with BaseModel (AerialVLN-Fine)

| Model | Method | SR2D↑ | SR3D↑ | OSR↑ | NE↓ | nDTW↑ |
|-------|--------|-------|-------|------|-----|-------|
| Gemini 2.5 Flash-Lite | BaseModel | 1.33 | 1.00 | 3.67 | 240.04 | 8.84 |
| Gemini 2.5 Flash-Lite | FineCog-Nav | 4.00 | 2.67 | 6.33 | 120.70 | 15.66 |
| GPT-4o-mini | BaseModel | 0.33 | 0.33 | 2.00 | 325.98 | 8.74 |
| GPT-4o-mini | FineCog-Nav | 4.00 | 2.33 | 3.67 | 100.37 | 20.45 |
| InternLM3-8B | BaseModel | 0.67 | 0.33 | 3.33 | 128.13 | 14.01 |
| InternLM3-8B | FineCog-Nav | 2.67 | 2.33 | 6.67 | 120.72 | 14.91 |
| ChatGLM-4-9B | BaseModel | 1.00 | 0.33 | 1.33 | 124.03 | 13.27 |
| ChatGLM-4-9B | FineCog-Nav | 3.00 | 2.67 | 2.67 | 97.05 | 19.26 |
| InternLM2.5-20B | BaseModel | 0.33 | 0.33 | 1.67 | 152.34 | 9.94 |
| InternLM2.5-20B | FineCog-Nav | 2.00 | 2.00 | 4.00 | 103.94 | 19.35 |
| ChatGLM-4-32B | BaseModel | 2.33 | 2.00 | 5.00 | 180.66 | 10.59 |
| ChatGLM-4-32B | FineCog-Nav | 3.33 | 2.33 | 5.33 | 94.18 | 21.25 |
| Qwen3-32B | BaseModel | 2.67 | 3.00 | 6.33 | 142.72 | 17.07 |
| Qwen3-32B | FineCog-Nav | 5.00 | 4.00 | 7.00 | 95.31 | 20.31 |
| Llama3.3-70B | BaseModel | 3.00 | 2.67 | 5.67 | 263.10 | 9.67 |
| Llama3.3-70B | FineCog-Nav | 6.67 | 6.00 | 9.67 | 98.84 | 20.17 |
| Qwen2.5-72B | BaseModel | 2.67 | 2.67 | 6.33 | 270.10 | 10.69 |
| **Qwen2.5-72B** | **FineCog-Nav** | **8.00** | **6.00** | **9.00** | **91.43** | **22.48** |

**说明**: FineCog-Nav 在所有 LLM 上一致超越 BaseModel，且大模型显著放大 FineCog-Nav 的优势。

### Table 2: Comparison with Framework Baselines (Qwen2.5-72B + Qwen2.5-VL-32B)

| Dataset | Method | SR2D↑ | SR3D↑ | OSR↑ | nDTW↑ | NE↓ |
|---------|--------|-------|-------|------|-------|-----|
| AerialVLN-Fine | NavGPT | 0.33 | 0.00 | 0.67 | 15.90 | 110.94 |
| AerialVLN-Fine | DiscussNav | 2.67 | 2.67 | 3.33 | 19.63 | 98.36 |
| AerialVLN-Fine | **FineCog-Nav** | **8.00** | **6.00** | **9.00** | **22.48** | **91.43** |
| AerialVLN-S | NavGPT | 0.12 | 0.12 | 0.46 | 8.29 | 135.65 |
| AerialVLN-S | DiscussNav | 0.46 | 0.46 | 0.93 | 8.33 | 158.46 |
| AerialVLN-S | **FineCog-Nav** | **1.97** | **1.50** | **1.85** | **11.47** | **130.32** |

**说明**: NavGPT 几乎不探索；DiscussNav 依赖多轮对话效率低；FineCog-Nav 显著领先。

### Table 3: Module Ablation (AerialVLN-Fine, Qwen2.5-72B)

| Attn. | Imag. | Subgoal | Mem. | SR↑ | OSR↑ | NE↓ | nDTW↑ |
|-------|-------|---------|------|-----|------|-----|-------|
| ✘ | ✓ | ✓ | H | 3.00 | 7.00 | 104.38 | 19.55 |
| ✓ | ✘ | ✓ | H | 3.67 | 5.00 | 101.27 | 19.67 |
| ✓ | ✓ | ✘ | H | 2.00 | 4.33 | 102.32 | 19.13 |
| ✓ | ✓ | ✓ | P (plain) | 0.67 | 2.67 | 97.76 | 19.69 |
| ✓ | ✓ | ✓ | **H (full)** | **6.00** | **9.00** | **91.43** | **22.48** |

**关键发现**: 层次化记忆替换为普通历史（plain）导致最大性能下降（SR 6.00→0.67），证明层次化记忆是核心组件。模块间紧耦合——单独移除某模块的降幅可能小于联合移除。

### Table 4: Sentence-Level Analysis by LLM

| Base LLM | S-SR↑ | S-nDTW↑ | S-NE↓ |
|----------|-------|---------|-------|
| Gemini2.5-Flash-Lite | 16.40 | 26.33 | 79.00 |
| GPT-4o-mini | 19.56 | 28.58 | 65.96 |
| InternLM3-8B | 15.62 | 28.19 | 72.01 |
| ChatGLM-4-9B | 19.92 | 31.42 | 62.18 |
| InternLM2.5-20B | 19.00 | 34.11 | 67.67 |
| ChatGLM-4-32B | 15.95 | 23.20 | 62.67 |
| Qwen3-32B | 18.76 | 30.04 | 62.42 |
| Llama3.3-70B | 21.32 | 33.89 | 64.82 |
| **Qwen2.5-72B** | **22.03** | **35.37** | **59.02** |

**说明**: 句子级性能随 LLM 规模提升；小的句子级差异在长程轨迹上累积放大。

### Table 5: Per-Scene Statistics (AerialVLN-Fine)

15 个场景的详细统计，轨迹数 2-35，路径长度 96-425m，动作数 47-143，句子数 3.5-7.5。

### Table 6: Capability Dimensions

| Capability | Proportion |
|-----------|-----------|
| Spatial Relations | 99.36% |
| Multi-Target Path Planning | 52.72% |
| Trajectory Constraints | 40.40% |
| Temporal Relations | 29.58% |
| Ordinal/Cardinal Recognition | 14.83% |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[AerialVLN-Fine]] | 300 trajectories, 15 scenes | 句子级指令-轨迹对齐，精炼指令，平均 189m/76 步/4.6 句 | 主要评估 |
| [[AerialVLN]]-S-Val | 864 trajectories | 原始 AerialVLN 验证集 | 泛化验证 |

### 实现细节

- **VLM**: [[Qwen2.5-VL]]-32B（固定）
- **LLM**: 可变，最优为 [[Qwen2.5]]-72B
- **评估指标**: NE (↓), SR2D/SR3D (↑), OSR (↑), nDTW (↑), PL, Steps
- **成功阈值**: 距目标 20m（2D/3D）
- **硬件**: 未明确

### 可视化结果

Figure 4 展示了 FineCog-Nav 的逐步推理过程：agent 依次完成子目标，遇到碰撞风险时安全模块发出警告并调整路径。真实无人机部署在 RoboMaster TT 上验证了框架的实际可行性（Figure 6）。

---

## 批判性思考

### 优点
1. **认知科学理论基础**: 以认知心理学为指导设计模块，比 ad-hoc 多 agent 方法更有理论支撑
2. **对小模型友好**: 8B 模型即可获得基线以上的性能，不强依赖闭源大模型
3. **AerialVLN-Fine 数据质量**: 精细标注解决了原 AerialVLN 数据集 47% 的轨迹-指令错位问题

### 局限性
1. **绝对性能偏低**: 最优配置 SR2D 仅 8%，说明零样本无人机 VLN 仍极具挑战性
2. **模块间紧耦合**: 消融实验显示单独移除某模块可能不降反升，暗示模块间依赖关系复杂
3. **仅在仿真中评估**: 真实无人机部署仅为初步演示（17 步简单任务）

### 潜在改进方向
1. 引入视觉基础模型进行更精确的场景理解（如 3D 重建辅助）
2. 利用强化学习微调决策模块
3. 在更复杂的真实无人机场景中系统验证

### 可复现性评估
- [ ] 代码开源（有项目页但未确认代码发布）
- [ ] 预训练模型
- [x] 训练细节完整（零样本无需训练）
- [ ] 数据集可获取（AerialVLN-Fine 待发布确认）

---

## 关联笔记

### 基于
- [[AerialVLN]]: 里程碑式无人机 VLN 基准（ICCV 2023）
- [[NavGPT]]: 基于 LLM 的显式推理导航

### 对比
- [[DiscussNav]]: 多专家讨论 VLN
- [[CityNavAgent]]: 层次化语义规划

### 方法相关
- [[Hierarchical Memory]]: 多层次记忆管理
- [[Subgoal Planning]]: 子目标规划
- [[Qwen2.5-VL]]: 视觉语言模型
- [[Instruction Parsing]]: 指令解析

### 硬件/数据相关
- [[RoboMaster TT]]: DJI 教育无人机平台
- [[AerialVLN-Fine]]: 精细标注无人机 VLN 基准

---

## 速查卡片

> [!summary] FineCog-Nav: Fine-grained Cognitive Modules for Zero-shot UAV VLN
> - **核心**: 认知启发的 8 模块零样本无人机导航框架
> - **方法**: 解析→子目标→感知→注意→想象→记忆→判断→决策
> - **结果**: Qwen2.5-72B 上 SR2D 8.0%, nDTW 22.48, NE 91.43m (AerialVLN-Fine)
> - **代码**: [Project Page](https://smartdianlab.github.io/projects-FineCogNav)

---

*笔记创建时间: 2026-04-27*
