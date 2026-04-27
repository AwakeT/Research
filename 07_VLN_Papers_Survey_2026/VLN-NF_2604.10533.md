---
title: "VLN-NF: Feasibility-Aware Vision-and-Language Navigation with False-Premise Instructions"
method_name: "VLN-NF"
authors: [Hung-Ting Su, Ting-Jun Wang, Jia-Fong Yeh, Min Sun, Winston H. Hsu]
year: 2026
venue: arXiv
tags: [vision-language-navigation, false-premise, infeasibility-detection, abstention, benchmark, embodied-navigation]
zotero_collection: ""
image_source: online
arxiv_html: https://arxiv.org/html/2604.10533v1
created: 2026-04-27
---

# 论文笔记：VLN-NF: Feasibility-Aware Vision-and-Language Navigation with False-Premise Instructions

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | National Taiwan University, National Tsing Hua University, DRIC |
| 日期 | April 2026 |
| 项目主页 | [Project Page](https://vln-nf.github.io/) |
| 对比基线 | [[DUET]], [[NavGPT]], [[MapGPT]] |
| 链接 | [arXiv](https://arxiv.org/abs/2604.10533) / [Project](https://vln-nf.github.io/) |

---

## 一句话总结

> VLN-NF 首次提出针对虚假前提指令的 VLN 基准和评估指标 REV-SPL，以及两阶段混合方法 ROAM（房间级导航+LLM/VLM 室内探索），使 agent 能在目标不存在时基于证据输出 NOT-FOUND。

---

## 核心贡献

1. **VLN-NF 基准**: 首个评估 agent 处理虚假前提（目标不存在）指令能力的 VLN 数据集，通过 LLM 重写+VLM 缺失验证的可扩展流水线构建（<2% 人类评审错误率）
2. **REV-SPL 评估指标**: 联合评估到达房间、探索覆盖度和决策正确性的综合指标，定义参考探索协议归一化效率
3. **ROAM 方法**: 两阶段混合框架——监督式房间级导航 + LLM/VLM 驱动的室内探索与验证，配合 FREE 自由空间先验

---

## 问题背景

### 要解决的问题
现有 VLN 基准假设所有指令都是可执行的（目标存在），但现实中人类常犯错误——认知科学研究表明人类每七次物品-位置回忆中约有一次错误。Agent 需要能够判断目标不存在并输出 NOT-FOUND。

### 现有方法的局限
- 所有主流 VLN 基准（[[R2R]], [[REVERIE]], [[CVDN]]）都假设指令可行
- 简单地在动作空间添加 NOT-FOUND 会导致 agent 因协变量偏移和复合误差而过早放弃（premature abstention）
- 标准 VLN 指标（SR, SPL）无法区分"到达正确房间但未充分探索"和"在错误房间停止"

### 本文的动机
在部分可观察的 3D 环境中，agent 需要到达正确房间、充分探索收集证据、然后做出 FOUND 或 NOT-FOUND 的判断——这是一个证据驱动的弃权问题。

---

## 方法详解

### 模型架构

ROAM 采用 **两阶段混合** 架构：
- **Stage 1 — Room-Level Navigator**: 监督式学习（[[DUET]] backbone），以房间标签为监督信号，导航至目标房间
- **Stage 2 — In-room Explorer**: [[LLM]]/[[VLM]] 驱动的室内探索，结合 FREE 自由空间先验
- **检测**: [[Grounding-DINO]] 开放词汇物体检测
- **分割**: [[Grounded-SAM]] 地面分割（FREE 模块）
- **LLM**: [[GPT-3.5]]-turbo / [[GPT-4o]]

### 核心模块

#### 模块1: Room-Level Navigator

**设计动机**: 利用监督学习的稳定性解决从起点到目标房间的宏观导航，避免 LLM 在长程导航中的不可靠性。

**具体实现**:
- 环境建模为导航图 $G = (V, E)$，定义目标房间子图 $R = \{v \in V \mid r(v) = r^*\}$
- 入口视点 $v_{\text{room}} = \arg\min_{v \in R} d_G(v_{\text{start}}, v)$
- 使用 [[DUET]] 作为 backbone，仅用房间标签训练
- 运行直到 agent 进入目标房间或步数预算耗尽

#### 模块2: In-room Explorer

**设计动机**: 在目标房间内需要灵活的语义推理来探索、定位目标或收集证据证明目标不存在。

**具体实现**:
- 动作空间: $\mathcal{A}_t = \{\text{move}(v) \mid v \in \mathcal{N}(v_t)\} \cup \{\text{FOUND}, \text{NOT-FOUND}\}$
- VLM 提供语义上下文（场景描述、检测物体）
- 开放词汇检测器 $D(v, o') \in [0,1]$ 提供目标置信度
- 若 $D(v_t, o') \geq \tau$ 则输出 FOUND
- 否则持续探索，基于证据停止或 frontier 耗尽时输出 NOT-FOUND

#### 模块3: FREE (Free-space Raycasting Estimation Engine)

**设计动机**: LLM 擅长常识推理但在几何/空间推理上不可靠，FREE 提供显式的自由空间距离信号辅助导航决策。

**具体实现**:
- 对每个候选朝向 $\theta_k$，估计自由空间距离 $d_{\text{free}}(\theta_k)$
- 使用 [[Grounded-SAM]] 分割可导航地面区域
- 深度图反投影到 3D 空间进行光线投射
- 将 $d_{\text{free}}$ 作为额外信号注入 LLM 的导航提示

---

## 关键公式

### 公式1: [[SPL|Reach SPL]]

$$
\text{Reach SPL} = \frac{1}{N}\sum_{i=1}^{N} S_{\text{reach},i} \cdot \frac{l_{\text{reach},i}}{\max(p_{\text{reach},i},\; l_{\text{reach},i})}
$$

**含义**: 评估 agent 到达目标房间的效率，考虑成功率和路径长度效率。

**符号说明**:
- $S_{\text{reach},i}$: episode $i$ 是否成功到达目标房间
- $l_{\text{reach},i}$: 参考到达路径长度
- $p_{\text{reach},i}$: 实际到达路径长度

### 公式2: [[SPL|REV-SPL]] (核心指标)

$$
\text{REV-SPL} = \frac{1}{N}\sum_{i} S_i^r S_i^d C_i \frac{\ell_i}{\max(p_i, \ell_i)} \cdot \min\!\left(1, \frac{p_i}{\ell_i}\right)
$$

其中物体覆盖度:

$$
C_i = \frac{|\bigcup_{v \in P_i} \text{OBJECTSVISIBLE}(v)|}{|O_i|}
$$

**含义**: 联合评估到达目标房间 ($S_i^r$)、决策正确性 ($S_i^d$)、探索覆盖度 ($C_i$) 和路径效率的综合指标。

**符号说明**:
- $S_i^r$: 是否到达目标房间
- $S_i^d$: FOUND/NOT-FOUND 决策是否正确
- $C_i$: 物体覆盖度（已观察到的房间物体比例）
- $\ell_i$: 参考探索路径长度
- $p_i$: 实际路径长度
- $\min(1, p_i/\ell_i)$: 惩罚过早终止（探索不足）

### 公式3: [[Navigation Graph|入口视点]]

$$
v_{\text{room}} = \arg\min_{v \in R} d_G(v_{\text{start}}, v)
$$

**含义**: 选择目标房间中距起点最近的视点作为 Stage 1 的导航目标。

### 公式4: [[Object Navigation|室内探索动作空间]]

$$
\mathcal{A}_t = \{\text{move}(v) \mid v \in \mathcal{N}(v_t)\} \cup \{\text{FOUND}, \text{NOT-FOUND}\}
$$

**含义**: agent 可选择移动到邻居视点、声明找到目标或声明目标不存在。

### 公式5: [[Object Navigation|综合验证评分]]

$$
S = \omega \cdot \phi(\{V_k\}) + (1 - \omega) \cdot C_i
$$

**含义**: 结合 360 度全景 LLM 评估和检测器置信度判断是否停止。（注：此公式来自 OVAL 论文，VLN-NF 的 ROAM 使用检测阈值 $D(v, o') \geq \tau$ 进行 FOUND 判断。）

---

## 关键图表

### Figure 1: Failure Modes under Unreliable Instructions / 不可靠指令下的失败模式

![Figure 1](https://arxiv.org/html/2604.10533v1/x1.png)

**说明**: 三种场景对比：(a) 标准 VLN 无 NOT-FOUND 选项时 agent 无限搜索；(b) 简单添加 NOT-FOUND 导致过早放弃；(c) ROAM 的证据收集方法——先到达房间、充分探索、再做判断。

### Figure 2: Dataset Construction Pipeline / 数据集构建流水线

![Figure 2](https://arxiv.org/html/2604.10533v1/x2.png)

**说明**: VLN-NF 的可扩展构建流水线。给定指令和目标物体，LLM Rewriter 提出合理替代物体并重写指令，VLM Verifier 通过全景开放词汇检测确认替代物体在目标房间中不存在。

### Figure 3: ROAM Overview / ROAM 方法概览

![Figure 3](https://arxiv.org/html/2604.10533v1/x3.png)

**说明**: ROAM 两阶段框架。左：Room-Level Navigator（DUET backbone，房间标签监督），到达目标房间；右：In-room Explorer（LLM/VLM 驱动+FREE 自由空间先验），探索并判断 FOUND/NOT-FOUND。

### Figure 4: FREE Mechanism / FREE 机制

![Figure 4](https://arxiv.org/html/2604.10533v1/x4.png)

**说明**: FREE 分割当前视图中的可导航地面区域，使用深度光线投射估计每个候选方向的自由空间距离 $d_{\text{free}}$，为 LLM 提供几何感知信号。

### Table 1: VLN-NF Dataset Statistics

| Split | #Scans | #Pairs | #Instructions | #FOUND | #NF |
|-------|--------|--------|---------------|--------|-----|
| Train | 55 | 2,362 | 4,724 | 2,362 | 2,362 |
| Val-seen | 38 | 234 | 468 | 234 | 234 |
| Val-unseen | 10 | 718 | 1,436 | 718 | 718 |

**说明**: 每对包含一条原始 REVERIE 指令（FOUND）和一条 VLN-NF 生成指令（NOT-FOUND），共享相同的场景、起点和目标房间。

### Table 2: VLN-NF val_unseen Results

| Setting | Method | Coverage | Reach SR | R&D SR | Reach SPL | REV-SPL |
|---------|--------|----------|----------|--------|-----------|---------|
| supervised | DUET | 69.5% | 53.8% | 33.8% | 37.0% | 4.2% |
| unsupervised | NavGPT | 59.1% | 8.2% | 5.4% | 7.0% | 1.0% |
| unsupervised | MapGPT | 63.3% | 30.0% | 14.0% | 18.2% | 3.2% |
| unsupervised | SoM-Gemini-2.0-Flash | 80.9% | 39.4% | 22.0% | 28.9% | 1.5% |
| hybrid | **ROAM-GPT-3.5** | 82.1% | 58.6% | 37.6% | 44.1% | **6.1%** |
| hybrid | ROAM-GPT-4o | 82.8% | 62.6% | 41.4% | 45.4% | 5.6% |

**说明**: ROAM-GPT-3.5 以 6.1% REV-SPL 超越 DUET 45%。注意 ROAM-GPT-4o R&D SR 更高但 REV-SPL 更低，因为 GPT-4o 的更长探索降低了效率分。

### Table 3: Ablation Study

| 2-Stage | FREE | Coverage | Reach SR | R&D SR | Reach SPL | REV-SPL |
|---------|------|----------|----------|--------|-----------|---------|
| ✗ | ✗ | 75.7% | 7.8% | 4.4% | 6.7% | 0.8% |
| ✓ | ✗ | 79.2% | 59.4% | 37.2% | 44.2% | 5.6% |
| ✓ | ✓ | **82.1%** | **58.6%** | **37.6%** | **44.1%** | **6.1%** |

**关键发现**: 两阶段设计是核心贡献（Reach SR 7.8%→59.4%，REV-SPL 0.8%→5.6%），FREE 进一步提升覆盖度（79.2%→82.1%）和 REV-SPL（5.6%→6.1%）。

### Table 4: Transfer to Standard REVERIE val_unseen

| 2-Stage | FREE | Reach SR | Reach SPL | SR | SPL |
|---------|------|----------|-----------|------|------|
| ✗ | ✗ | 8.1% | 6.6% | 9.2% | 6.8% |
| ✓ | ✗ | 62.0% | 48.1% | 42.0% | 23.7% |
| ✓ | ✓ | **64.0%** | **48.4%** | **45.2%** | **25.1%** |

**说明**: ROAM 框架在标准可行 VLN 任务上也有效，FREE 提升 SR 3.2% 和 SPL 1.4%。

### Table 5: False NOT-FOUND Error Analysis

| Error Source | Share |
|-------------|-------|
| Room-reaching failure | 55.7% |
| Perception/grounding uncertainty | 31.0% |
| Exploration-control failure | 13.3% |

**说明**: 大多数错误 NOT-FOUND 源于未能到达正确房间（55.7%），其次是感知/接地不确定性（31.0%）。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[VLN-NF]] | 3,314 pairs (train+val) | 基于 REVERIE 扩展，含 FOUND/NF 配对 | 训练/测试 |
| [[REVERIE]] | 原始基准 | 目标驱动导航和远程物体接地 | 对比迁移 |
| [[Matterport3D]] | 3D 室内场景 | 基础环境 | 仿真平台 |

### 实现细节

- **Room Navigator**: [[DUET]] fine-tuned on VLN-NF reach paths
- **Room Explorer**: [[GPT-3.5]]-turbo 或 [[GPT-4o]]（基于 NavGPT 改造）
- **物体检测**: [[Grounding-DINO]]-base（阈值 0.75）
- **地标提取**: [[Gemini 1.5 Flash]]
- **指令重写**: gpt-3.5-turbo
- **缺失验证**: GLIP SWIN-Large（阈值 0.7）
- **FREE 分割**: [[Grounded-SAM]]
- **硬件**: 未明确

### 可视化结果

Figure 1 直观展示了三种场景：无 NOT-FOUND 时的无限搜索、简单添加后的过早放弃、以及 ROAM 的证据驱动弃权。

---

## 批判性思考

### 优点
1. **问题定义的新颖性**: 首次系统研究 VLN 中的虚假前提问题，填补重要空白
2. **REV-SPL 指标设计精巧**: 联合评估到达、探索和决策，参考探索协议归一化效率
3. **实用的两阶段解决方案**: 结合监督学习的稳定性和 LLM 的灵活性

### 局限性
1. **数据覆盖有限**: 仅保留 REVERIE 45% 的 episode，可能存在分布偏差
2. **仅处理目标级虚假前提**: 不涵盖多目标、模糊描述等更广泛的不可靠指令类型
3. **REV-SPL 绝对值偏低**: 最好的方法仅 6.1%，说明问题极具挑战性

### 潜在改进方向
1. 扩展到更多类型的不可靠指令（步骤错误、描述歧义）
2. 添加 NOT-FOUND 后的恢复策略（如建议替代目标）
3. 探索端到端方法替代两阶段流水线

### 可复现性评估
- [ ] 代码开源（有项目页，待确认代码发布）
- [ ] 预训练模型
- [x] 训练细节完整
- [ ] 数据集可获取（有项目页，待确认数据发布）

---

## 关联笔记

### 基于
- [[REVERIE]]: 目标驱动 VLN 基准
- [[DUET]]: 双尺度图 Transformer VLN agent
- [[NavGPT]]: LLM 驱动的显式推理导航

### 对比
- [[MapGPT]]: 地图引导 GPT 导航
- [[Gemini]]: 多模态大模型

### 方法相关
- [[Grounding-DINO]]: 开放词汇物体检测
- [[Grounded-SAM]]: 接地分割
- [[SPL]]: Success weighted by Path Length 评估指标
- [[FREE]]: 自由空间光线投射估计引擎

### 硬件/数据相关
- [[Matterport3D]]: 大规模 3D 室内扫描数据集
- [[VLN-NF]]: 虚假前提 VLN 基准

---

## 速查卡片

> [!summary] VLN-NF: Feasibility-Aware VLN with False-Premise Instructions
> - **核心**: 首个虚假前提 VLN 基准 + REV-SPL 评估指标
> - **方法**: ROAM 两阶段混合（监督式房间导航 + LLM 室内探索 + FREE 自由空间先验）
> - **结果**: 6.1% REV-SPL, 超越监督基线 DUET 45%
> - **代码**: [Project Page](https://vln-nf.github.io/)

---

*笔记创建时间: 2026-04-27*
