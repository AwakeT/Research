---
title: "XEmbodied: A Foundation Model with Enhanced Geometric and Physical Cues for Large-Scale Embodied Environments"
method_name: "XEmbodied"
authors: [Kangan Qian, ChuChu Xie, Yang Zhong, Jingrui Pang, Siwen Jiao, Sicong Jiang, Zilin Huang, Yunlong Wang, Kun Jiang, Hao Ye, Guanghao Zhang, Hangjun Ye, Guang Chen, Long Chen, Diange Yang]
year: 2026
venue: arXiv
tags: [foundation-model, embodied-ai, 3d-adapter, physical-cues, autonomous-driving, vlm, geometric-reasoning, curriculum-learning, mamba]
zotero_collection: ""
image_source: online
arxiv_html: ""
created: 2026-05-06
---

# 论文笔记：XEmbodied: A Foundation Model with Enhanced Geometric and Physical Cues for Large-Scale Embodied Environments

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tsinghua University, Xiaomi Automotive and Robotics, NUS, McGill, UW-Madison |
| 日期 | April 2026 |
| 项目主页 | N/A |
| 对比基线 | GPT-4o, Gemini-1.5, Qwen2.5-VL (7B/32B), Qwen3-VL (A3B-30B/32B), UniVG-R1-7B, PR1 系列, DriveMM, Cosmos-R1, Mimo-Embodied |
| 链接 | [arXiv](https://arxiv.org/abs/2604.18484) |

---

## 一句话总结

> 提出 XEmbodied，云端具身基础模型，通过 3D Adapter (3DA) 将 3D 几何先验内生注入 VLM 语义 token，配合 Efficient Image-Embodied Adapter (EIEA) 将物理线索（检测/占据/地图分割）蒸馏为紧凑潜在 token，结合渐进式领域课程训练，在 18 个公开 benchmark 上实现 SOTA（空间推理、交通语义、具身 affordance）。

---

## 核心贡献

1. **3D Adapter (3DA)**: 双流架构——2D 语义编码器 + 3D 几何编码器 (VGGT)，通过 cross-attention 将 3D 几何先验注入 2D 语义 token，实现内生 3D 空间意识
2. **Efficient Image-Embodied Adapter (EIEA)**: 基于 Mamba 的多模态解释器，将检测/BEV 占据/地图分割等异构工具输出蒸馏为紧凑物理线索 token，即插即用注入 VLM 上下文
3. **渐进式领域课程**: 四阶段数据分类（空间熵评分 + 四层认知分级 T1-T4）+ 四阶段训练流水线（领域语义→3D 几何→端到端认知→EIEA 物理线索），避免灾难性遗忘
4. **18 Benchmark SOTA**: 覆盖 3D 理解（Ego3DBench, SURDS, VLADBench, STRIDE-QA）、语义推理（DriveBench, DriveLMM-o1, MapLMv2, LingoQA, OmniDrive）、具身 affordance（RoboAfford, Cosmos-R1, Embodied-R1, RoboRefIt, VABench, Where2place）

---

## 问题背景

### 要解决的问题
VLA 模型训练依赖云端数据闭环：车载模型采集 → 云端标注/场景挖掘 → 迭代训练。但现有云端 VLM 因 2D 图文预训练缺乏 3D 几何语义和物理线索理解能力，无法有效处理车道拓扑、度量距离、占据关系等具身概念。

### 现有方法的三大局限
1. **2D 预训练的几何盲区**: VLM 的互联网规模 2D 图文预训练无法捕获车道拓扑、度量距离、占据关系等 3D 概念
2. **领域知识稀疏**: 交通规则、驾驶术语等领域特定知识在通用预训练数据中覆盖不足；朴素微调导致灾难性遗忘和 OOD 退化
3. **物理线索集成困难**: 点云、3D 框、占据网格、轨迹等原生具身模态难以融入标准 VLM 上下文；显式工具调用效率低且工具输出与语言推理对齐差

### 本文的动机
类比人类空间认知——不依赖精确深度值，而是通过多源结构化线索（相对距离、运动轨迹）灵活推理。XEmbodied 构建具有**内生几何能力**的基础模型，并与外部物理线索**动态交互**。

---

## 方法详解

### 任务形式化

给定视觉观测 $X_v$（RGB 图像/视频）、文本指令 $X_T$、辅助驾驶模态 $X_m$（BEV 占据、检测、地图分割），学习：

$$Y = f_{\text{XEmbodied}}(X_v, X_T, X_m) \tag{1}$$

### 模型架构

三个模块化组件：

#### 模块1: 3D Adapter (3DA)

受人类认知中语义（"什么"）和几何（"哪里"）双通道分离处理启发：

**语义流**: 2D 视觉编码器（Qwen3-VL）提取语义 token：

$$\mathbf{e}_{2\text{D}} \in \mathbb{R}^{N \times d_{2\text{D}}}$$

捕获高层物体类别和场景属性。

**几何流**: 3D 视觉几何编码器（基于 VGGT）生成 3D 几何 token：

$$\mathbf{e}_{3\text{D}} \in \mathbb{R}^{M \times d_{3\text{D}}}$$

利用预训练编码器和跨帧融合解码器嵌入丰富的 3D 空间先验。

**Token 对齐与融合**:

1. 确定性插值将 $\mathbf{e}_{3\text{D}}$ 映射到 2D token 网格：$\bar{\mathbf{e}}_{3\text{D}} = \text{Interpolation}(\mathbf{e}_{3\text{D}}) \in \mathbb{R}^{N \times d_{3\text{D}}}$
2. 投影 $\bar{\mathbf{e}}_{3\text{D}}$ 到维度 $d_{2\text{D}}$，得到 $\hat{\mathbf{e}}_{3\text{D}}$
3. Cross-attention 注入几何线索——2D 语义 token 为 Query，3D 几何 token 为 Key/Value：

$$\mathcal{E}_{2\text{D}3\text{D}} = \text{CrossAttn}(\mathbf{Q} = \mathbf{e}_{2\text{D}}, \mathbf{K} = \hat{\mathbf{e}}_{3\text{D}}, \mathbf{V} = \hat{\mathbf{e}}_{3\text{D}}) \tag{2}$$

4. 残差连接保留基础 2D 语义特征：

$$\mathbf{e}_{\text{fusion}} = \mathbf{e}_{2\text{D}} + \mathcal{E}_{2\text{D}3\text{D}} \tag{3}$$

$\mathbf{e}_{\text{fusion}}$ 替代原始 2D token 输入语言模型。

#### 模块2: Efficient Image-Embodied Adapter (EIEA)

将异构具身模态（检测框、BEV 占据、地图分割）转化为紧凑物理线索 token，即插即用注入 VLM 上下文。

**三阶段流水线**:

**Step 1 - 模态特定特征提取**: 每个辅助模态 $X_m$ 通过模态编码器 $\mathcal{E}_m$ 和轻量投影器 $\phi_m$ 映射到共享维度 $d$：

$$\mathbf{H}_m = \mathcal{E}_m(X_m) \in \mathbb{R}^{L_m \times d_m}, \quad \mathbf{Z}_m = \phi_m(\mathbf{H}_m) \in \mathbb{R}^{L_m \times d}$$

$\mathcal{E}_m$ 可用 ResNet-50（相机视觉线索）或 SigLIP（语义线索），$\phi_m$ 为两层 MLP + GELU。

**Step 2 - Mamba 多模态解释器**: 拼接文本 query token $\mathbf{X}_T$ 和所有投影模态 token，送入轻量 Mamba 解释器：

$$\mathbf{H}_{\text{text}} = g_{\text{Mamba}}\left([\mathbf{X}_T; \mathbf{Z}_{m_1}; \ldots; \mathbf{Z}_{m_{|\mathcal{M}|}}]\right)$$

再通过潜在投影模块 $\phi_R$ 映射到 VLM 词嵌入空间：

$$\mathbf{Z}_{\text{phy}} = \phi_R(\mathbf{H}_{\text{text}})$$

**Step 3 - 蒸馏压缩**: 将物理线索 token $\mathbf{Z}_{\text{phy}}$ 作为紧凑中间表示（latent rationale），避免直接将冗长工具输出传入 VLM 上下文。

**最终推理**:

$$Y = f_{\text{VLM}}(X_v, X_T, \mathbf{Z}_{\text{phy}}) \tag{4}$$

#### 模块3: 渐进式领域课程

**数据分类与过滤** (4 阶段流水线):

1. **空间熵评分**: 基于规则的轻量评分 $\mathcal{H}(x_i)$，从深度方差和 3D 物体分布统计量化场景几何复杂度
2. **四层数据分级 (T1-T4)**: 融合空间熵和语义标签，按认知需求递增：
   - T1: 场景 grounding + 常识推理
   - T2: 空间定位
   - T3: 多视角/时间空间推理
   - T4: 时空理解
3. **数据质量评估**: 基于 Qwen3-VL-235B 的质量评估 agent，评估 correctness / completeness / clarity / relevance 四维
4. **质量过滤**: 基于评估分数淘汰低质量样本

### 训练流水线

四阶段渐进训练，所有阶段（除 Stage 4）使用相同的 next-token 预测目标：

$$\mathcal{L}_{\text{CE}}(\theta) = -\mathbb{E}_{(X_v, X_T, Y) \sim \mathcal{D}} \left[\frac{1}{|Y|} \sum_{t=1}^{|Y|} \log P_\theta(Y_t \mid X_v, X_T, Y_{<t})\right] \tag{5}$$

| 阶段 | 目标 | 数据 | 可训练模块 | 冻结模块 |
|------|------|------|---------|---------|
| **S1: Domain Semantic Alignment** | 适配 VLM 到驾驶/机器人领域语言 | 领域指令 QA (交通规则、道路拓扑、场景描述) | LLM 权重 | 图像编码器, 3D 模块 |
| **S2: 3D Geometry Alignment** | 训练 3DA 将 3D 几何注入语义流 | T2 主导数据（空间关系、车道连接、遮挡） | 3DA | VLM 骨干 |
| **S3: End-to-End Geometric Cognition** | 联合训练 3DA + LLM，实现双向耦合 | T2 起步，渐进混入 T3-T4 (课程调度) | 3DA + LLM LoRA | 图像编码器 |
| **S4: EIEA Training** | 训练 EIEA 编码物理线索 | 带工具输出的 VQA 数据 | EIEA + VLM LoRA | 3DA |

**Stage 4 两步训练**:
- **Step 1 - Alignment**: 对齐潜在物理线索 token 与 LLM response embedding
- **Step 2 - RL fine-tuning**: 使用 GRPO 优化，奖励函数关注格式和答案准确性：

$$r = \lambda_{\text{format}} r_{\text{format}} + \lambda_{\text{ans}} r_{\text{ans}}$$

$$\mathcal{L}_{\text{GRPO}} = \mathbb{E}\left[\min\left(\rho_t \hat{A}_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \tag{7}$$

其中 $\rho_t$ 为策略比率，$\hat{A}_t$ 为组归一化优势。

---

## 关键图表

### Figure 1: 三种具身场景理解范式对比

**说明**: (a) 传统规则流水线: 高透明度但可扩展性差。(b) 原始 VLM: 强通用推理但几何意识弱、易幻觉。(c) XEmbodied: 通过 3DA 注入几何先验 + EIEA 注入物理线索，兼具知识库 + 良好可扩展性 + 隐式几何意识 + 高效物理线索注入。

### Figure 2: 18 Benchmark 雷达图

**说明**: 三维评估——空间&3D 理解 / 语义&推理 / 具身&Affordance。XEmbodied (SFT) 在几乎所有维度上包络其他模型。

### Figure 3: 整体架构

**说明**: 输入图像/视频 → Semantic Encoder (2D token) + Geometric Encoder (3D token) → 3DA cross-attention 融合 → geometric-aware semantic tokens → LLM。EIEA 分支: Embodied Tools (Detection/Occupancy/Map Seg) → Embodied Encoder + Semantic Encoder → Mamba interpreter → Latent Projector → Physical cue tokens (plug-and-play) → 注入 LLM 上下文。

### Figure 4: 数据策划流水线

**说明**: Stage 1 空间熵评分 ($\mathcal{H}(x)$, 基于深度+3D 物体统计) → Stage 2 四层分级 (T1-T4) → Stage 3 数据质量评估 (Qwen3-VL-235B, 四维评分) → Stage 4 质量过滤。

### Figure 5: EIEA 两阶段训练

**说明**: (a) Alignment: 潜在物理线索 token 与 LLM response embedding 对齐。(b) RL fine-tuning: GRPO 优化，最终答案奖励驱动。

---

## 实验

### 实现细节

| 项目 | 配置 |
|------|------|
| 骨干 VLM | Qwen3-VL-30B-A3B-Instruct |
| 3D 编码器 | VGGT (预训练) |
| 训练框架 | ms-swift |
| 训练硬件 | 128× NVIDIA H20 GPU |
| 优化器 | AdamW + gradient accumulation |
| EIEA 模态编码器 | ResNet-50 / SigLIP |
| EIEA 解释器 | 轻量 Mamba |

### 训练数据

多源 VQA 聚合: OmniDrive, SURDS, DriveLMM-o1, DrivingVQA, LingoQA, MapLMv2, BDD, RoboVQA, RoboRefIt, RefCOCO, VQAv2, GQA, Flickr 等。

**关键**: Cosmos-R1, Ego3DBench, DriveBench 等 OOD benchmark 完全排除在训练集外。

### 核心结果

#### TABLE 1: 空间&3D 理解

| Model | Ego3DBench ACC | Ego3DBench RMSE | SURDS | VLADBench | STRIDE-QA |
|-------|---------------|----------------|-------|-----------|-----------|
| GPT-4o | 52.70 | 19.20 | 13.31 | 56.00 | 14.70 |
| Gemini-1.5 | 53.50 | 19.62 | 32.77 | 54.23 | - |
| Qwen2.5-VL-7B | 41.10 | 30.36 | 12.61 | 50.75 | 1.41 |
| Qwen2.5-VL-32B | 57.30 | 15.87 | 38.82 | 61.33 | 6.58 |
| Qwen3-VL-A3B-30B | 53.15 | 13.84 | 38.69 | 59.81 | 8.69 |
| Qwen3-VL-32B | **59.93** | 15.70 | 41.53 | 61.64 | 6.00 |
| UniVG-R1-7B | 46.82 | 53.85 | 1.96 | 46.51 | 4.61 |
| DriveMM | 49.73 | 12.68 | 8.73 | 47.45 | 1.12 |
| Cosmos-R1 | 45.62 | 23.41 | 10.05 | 55.93 | 1.03 |
| Mimo-Embodied | 53.62 | 16.57 | 24.04 | 55.82 | 7.90 |
| **XEmbodied (Best)** | 55.28 | **9.25** | **83.83** | **68.61** | **27.76** |

**亮点**: SURDS 83.83（vs 次优 41.53，+102%），STRIDE-QA 27.76（vs 次优 14.70，+89%），Ego3DBench RMSE 9.25（SOTA）。

#### TABLE 2: 语义&推理

| Model | DriveBench | DriveLMM-o1 | MapLMv2 | LingoQA | OmniDrive |
|-------|-----------|-------------|---------|---------|-----------|
| GPT-4o | 47.97 | 57.84 | 57.81 | 53.58 | 4.54 |
| Qwen3-VL-32B | 51.70 | 55.29 | 46.77 | 54.30 | 0.00 |
| DriveMM | 44.50 | **65.91** | 52.18 | 37.70 | 1.10 |
| Mimo-Embodied | 52.95 | 40.31 | **65.23** | **68.60** | 4.90 |
| **XEmbodied (Best)** | **53.18** | 77.01 | **78.55** | 65.70 | **25.43** |

**亮点**: DriveLMM-o1 77.01（vs DriveMM 65.91，+17%），OmniDrive 25.43（所有其他模型几乎为零）。

#### TABLE 3: 具身&Affordance

| Model | Affordance-2K | RoboAfford | Cosmos-R1 | Embodied-R1 | RoboRefIt | VABench | Where2place |
|-------|-------------|------------|-----------|-------------|-----------|---------|-------------|
| GPT-4o | 2.70 | 3.80 | 67.40 | 0.35 | 13.40 | 0.40 | 0.44 |
| Qwen3-VL-32B | 6.90 | 4.00 | 75.50 | 0.05 | 83.15 | 1.84 | 1.50 |
| Mimo-Embodied | 8.85 | 3.25 | **81.00** | 0.00 | 78.35 | 1.85 | 1.25 |
| **XEmbodied (Best)** | **78.50** | **4.35** | 76.00 | **3.80** | **87.15** | **3.50** | **2.30** |

**亮点**: Affordance-2K 78.50（vs 次优 8.85，+787%），Embodied-R1 3.80（唯一显著非零），RoboRefIt 87.15。

### 零样本泛化

Cosmos-R1, DriveBench, Ego3DBench 均为 OOD 数据集（完全排除在训练集外），XEmbodied 仍保持强性能，验证泛化能力。

### 消融实验

#### TABLE 5: 模型架构与数据策略消融

| Group | Data Strategy | 3DA 融合方式 | Spatial&3D | Semantic&Reasoning | Embodied&Affordance | Avg |
|-------|--------------|-----------|-----------|-------------------|-------------------|-----|
| (a) | ✗ (无数据策略) | ✗ | 73.39 | 82.17 | 60.94 | 71.48 |
| (b) | ✓ | MLP | 75.45 | **89.85** | 63.70 | 75.68 |
| (c) | ✓ | MLP | 74.88 | 75.49 | 59.42 | 69.07 |
| (d) | ✓ | QFormer | 78.75 | 86.02 | 56.50 | 72.52 |
| **(e)** | **✓** | **Cross-Attention** | **77.02** | 89.48 | **70.01** | **78.45** |

**关键发现**:
- MLP 直接投影 (c) 性能低于 baseline (b)：静态线性投影无法对齐 3D 与 2D token，引入噪声
- QFormer (d) 在语义推理上强但具身 affordance 下降 (56.50)：高级融合缺乏动态对齐
- Cross-Attention (e) 最优：2D token 可选择性聚合相关 3D 信息，具身 affordance 70.01（+13.51 vs QFormer）

#### TABLE 6: 课程学习消融

| Group | Mixture Training | CL | S1 | S2 | S3 | Spatial&3D | Semantic&Reasoning | Embodied&Affordance | Avg |
|-------|-----------------|----|----|----|----|-----------|-------------------|-------------------|-----|
| (a) | ✗ | ✗ | ✗ | ✗ | ✗ | 59.56 | 63.41 | 60.59 | 61.24 |
| (b) | ✓ | ✗ | ✗ | ✗ | ✗ | 69.14 | 73.75 | 64.76 | 68.97 |
| (c) | ✗ | ✓ | ✓ | ✗ | ✗ | 65.99 | 75.21 | 67.53 | 69.66 |
| (d) | ✗ | ✓ | ✓ | ✓ | ✗ | 69.63 | 81.75 | 88.05 | 80.83 |
| **(e)** | **✗** | **✓** | **✓** | **✓** | **✓** | **90.41** | **82.71** | **96.29** | **90.13** |

**关键发现**: 渐进课程 S1→S2→S3 (e) 总分 90.13，远超朴素混合训练 (b) 的 68.97。每阶段贡献递增且不破坏前阶段能力。

#### TABLE 4: EIEA 性能与效率

| Model | Size (B) | EIEA S1 | EIEA S2 | DriveLMM-o1 Score | DriveLMM-o1 Time (s) | DriveBench Score | DriveBench Time (s) | Ego3D ACC | Ego3D RMSE | Ego3D Time (s) |
|-------|---------|---------|---------|-------------------|---------------------|-----------------|-------------------|-----------|-----------|---------------|
| XEmbodied (SFT) | 30 | ✗ | ✗ | 64.32 | 1.52 | 53.18 | 1.64 | 55.28 | 9.44 | 1.25 |
| AgentThink | 7 | ✗ | ✗ | 71.35 | 3.17 | 55.41 | 3.36 | - | - | - |
| XEmbodied w/EIEA | 30 | ✓ | ✗ | 75.28 | - | 56.72 | - | 55.45 | 6.85 | - |
| **XEmbodied w/EIEA** | **30** | **✓** | **✓** | **77.01** | **0.46** | **57.33** | **0.52** | **59.53** | **6.83** | **0.40** |

**关键发现**:
- EIEA alignment (S1) 带来显著提升：DriveLMM-o1 64.32→75.28 (+17%)
- GRPO (S2) 进一步提升到 77.01，推理时间 0.46s（vs AgentThink 3.17s，**6.89x 加速**）
- Ego3D RMSE 从 9.44 降到 6.83

---

## 批判性思考

### 优点
1. 3DA 的 cross-attention 设计优雅：2D token 为 Query 使模型能选择性聚合 3D 信息，残差连接保留原始语义，消融实验充分验证优于 MLP 和 QFormer
2. EIEA 的 Mamba 解释器 + 蒸馏压缩避免了直接传递冗长工具输出的效率问题，推理速度 6.89x 优于显式工具调用
3. 四阶段课程训练设计严谨，每阶段冻结/解冻策略明确防止灾难性遗忘，消融实验从 61.24→90.13 验证效果
4. 18 个 benchmark 全面覆盖 3D/语义/affordance 三维度，含多个 OOD 零样本测试
5. 128x H20 GPU 大规模训练展示了工业级实力（清华 + 小米汽车）
6. Affordance-2K 78.50 vs 次优 8.85，说明 3DA+EIEA 对物理 affordance 理解有质的突破

### 局限性
1. 30B 参数（A3B 激活），仅适用于云端部署，无法端侧运行
2. 依赖预训练 VGGT 3D 编码器，VGGT 本身的失败模式（如纹理稀疏场景）会传播
3. EIEA 需要具身工具（检测器/占据预测器/地图分割器）的输出作为输入，增加了系统复杂度
4. 训练数据以自动驾驶为主，机器人操作 benchmark（RoboAfford 4.35, Where2place 2.30）提升有限
5. 未与端到端 VLA 方法（如 OpenVLA, π₀）在闭环控制任务上对比
6. 课程训练需要 128x H20 GPU，可复现性门槛极高

### 潜在改进方向
1. 研究 3DA 在室内导航场景（如 VLN-CE, R2R）的迁移效果
2. 将 EIEA 的物理线索注入机制应用于 VLN 中的语义地图/拓扑图信息融合
3. 探索轻量化版本（如 7B backbone）用于端侧部署
4. 在闭环仿真（如 CARLA, nuPlan）中评估对下游 VLA 性能的影响

### 可复现性评估
- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练流水线描述完整（四阶段 + 数据策略）
- [x] Benchmark 标准化（18 个公开 benchmark）
- [ ] 训练数据完整列表

---

## 关联笔记

### 基于
- [[Qwen3-VL]]: 骨干 VLM (Qwen3-VL-30B-A3B-Instruct)
- [[VGGT]]: 3D 视觉几何编码器
- [[Mamba]]: 状态空间模型（EIEA 多模态解释器）
- [[GRPO]]: Group Relative Policy Optimization (Stage 4 RL)
- [[LoRA]]: 低秩适配 (Stage 3-4 训练)

### 对比
- [[DriveMM]]: 多模态驾驶模型
- [[Cosmos-R1]]: 物理常识具身推理
- [[Mimo-Embodied]]: X-embodied 基础模型
- [[AgentThink]]: 工具增强 CoT 推理（显式物理线索注入）

### 方法相关
- [[3D Perception]]: 3D 空间理解
- [[Physical Cue Injection]]: 物理线索注入
- [[Curriculum Learning]]: 课程学习
- [[Data Closed-Loop]]: 数据闭环系统
- [[Embodied VQA]]: 具身视觉问答
- [[Cross-Attention Fusion]]: 跨注意力融合

---

## 速查卡片

> [!summary] XEmbodied
> - **核心**: 云端具身基础模型，3DA 内生注入 3D 几何 + EIEA 蒸馏物理线索 → VLM
> - **3DA**: Semantic Encoder (2D) + Geometric Encoder (VGGT, 3D) → Cross-Attention 融合 + 残差
> - **EIEA**: 异构工具输出 → 模态编码器 → Mamba 解释器 → 紧凑 physical cue tokens (plug-and-play)
> - **训练**: 四阶段渐进课程 (Domain→3D Geo→End-to-End→EIEA+GRPO)，128× H20 GPU
> - **结果**: 18 benchmark SOTA，SURDS 83.83 (+102%), Affordance-2K 78.50 (+787%), EIEA 6.89x 加速
> - **骨干**: Qwen3-VL-30B-A3B + VGGT + Mamba

---

*笔记创建时间: 2026-05-06*
