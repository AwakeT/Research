---
title: "PROSPECT: Unified Streaming Vision-Language Navigation via Semantic-Spatial Fusion and Latent Predictive Representation"
method_name: "PROSPECT"
authors: [Zehua Fan, Wenqi Lyu, Wenxuan Song, Linge Zhao, Yifei Yang, Xi Wang, Junjie He, Lida Huang, Haiyan Liu, Bingchuan Sun, Guangjun Bao, Xuanyao Mao, Liang Xu, Yan Wang, Feng Gao]
year: 2026
venue: arXiv
tags: [vision-language-navigation, streaming-vla, 3d-spatial-encoding, world-model, latent-prediction, real-robot]
zotero_collection: ""
image_source: online
arxiv_html: https://arxiv.org/html/2603.03739v1
created: 2026-04-27
---

# 论文笔记：PROSPECT: Unified Streaming Vision-Language Navigation via Semantic-Spatial Fusion and Latent Predictive Representation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Shanghai Jiao Tong University, Tsinghua University, University of Adelaide, Wuhan University, HKUST(GZ), Beijing Jiaotong University, Lenovo |
| 日期 | March 2026 |
| 项目主页 | N/A |
| 对比基线 | [[StreamVLN]], [[NaVILA]], [[NaVid]], [[Uni-NaVid]] |
| 链接 | [arXiv](https://arxiv.org/abs/2603.03739) / Code (to be released) |

---

## 一句话总结

> 将 [[CUT3R]] 流式3D空间编码与 [[JEPA]] 风格的潜在预测学习统一到流式VLA中，提升长程VLN鲁棒性且预测分支不增加推理开销。

---

## 核心贡献

1. **统一流式VLN框架**: 将流式VLA与潜在预测表征学习集成，在VLN-CE上达到第一梯队性能
2. **CUT3R流式3D感知**: 利用 [[CUT3R]] 提供绝对尺度空间特征，高效支持长上下文导航
3. **Stream Query Tokens + 流式因果注意力掩码**: 实现2D/3D潜在特征预测，同时解耦两个模态目标
4. **真实机器人部署**: 在室内外多种光照条件下实现高频控制（~4 Hz）

---

## 问题背景

### 要解决的问题
强大的导航不仅需要语义理解，还需要对环境动态和空间结构的**预测建模**能力。现有VLN方法主要关注动作生成和语言 grounding，缺乏空间理解和未来预测。

### 现有方法的局限
- 低维状态空间模型的预测表达力有限
- 在显式像素/深度空间做监督可能过拟合到与任务无关的细节（纹理、光照）
- 2D语义编码器（如 [[SigLIP]]）缺乏空间智能
- [[VGGT]] 对长序列内存开销大，需截断历史，且只提供相对尺度表示

### 本文的动机
- [[JEPA]] 启发：在紧凑的潜在空间预测未来特征，而非预测像素/深度，避免建模任务无关噪声
- [[CUT3R]] 天然支持流式处理，提供绝对尺度空间表示，适合长上下文流式导航

---

## 方法详解

### 模型架构

PROSPECT 采用**统一流式VLA + 潜在预测**架构：
- **输入**: 语言指令 $I$ + 流式RGB观测 $o_t$（无地图、无里程计）
- **2D编码器**: 冻结的 [[SigLIP]] 提取语义特征
- **3D编码器**: 冻结的 [[CUT3R]] 提取绝对尺度空间特征
- **融合**: [[Cross-Attention]] 将3D特征融入2D语义特征
- **LLM骨干**: LLaVA-NeXT-Video-7B (Qwen1.5-7B)
- **输出**: 每步 $n_a=4$ 个离散动作
- **预测分支**: 训练时活跃，推理时移除（零额外开销）

### 核心模块

#### 模块1: 流式VLA问题建模

**设计动机**: 将VLN形式化为流式VLA问题，支持长上下文处理

**具体实现**:
- 流式上下文定义包含 [[KV Cache]] 滑动窗口和长期记忆 token
- 短期窗口 $N=8$，8个长期关键帧
- 动作空间 $\mathcal{A} = \{\uparrow, \leftarrow, \rightarrow, \text{STOP}\}$

#### 模块2: 2D-3D感知融合

**设计动机**: 结合语义和空间信息，弥补纯2D编码器的空间理解不足

**具体实现**:
- [[SigLIP]] 编码2D语义特征 $F_t^{2D}$
- [[CUT3R]] 通过ViT编码器 + 流式解码器提取3D空间特征 $F_t^{3D}$
- [[Cross-Attention]] 融合：以2D特征为Query，3D特征为Key/Value
- 历史关键帧通过相同管道编码后压缩为单个长期记忆token

#### 模块3: Stream Query Tokens 潜在预测

**设计动机**: 利用 [[JEPA]] 思想，在潜在空间预测未来特征，塑造更具预测性的内部表示

**具体实现**:
- 可学习的查询token $\langle q_t^{2D} \rangle$ 和 $\langle q_t^{3D} \rangle$ 追加到LLM输入
- 通过LLM提取流式上下文信息
- 两个轻量 [[Transformer]] 解码器（各2层）重建下一步潜在特征
- 使用冻结的 [[SigLIP]] 和 [[CUT3R]] 教师作为监督目标（无梯度）
- 2D用余弦距离，3D用MSE

#### 模块4: 流式注意力掩码

**设计动机**: 防止未来信息泄露，隔离不同步骤和模态的查询token

**约束**:
- 因果性：查询token只能关注当前和之前的轮次
- 隔离性：不同轮次的查询token相互隔离
- 模态解耦：2D和3D查询token相互屏蔽

---

## 关键公式

### 公式1: [[KV Cache|流式上下文]]

$$
\text{Stream}_{0:t} := \{ \text{KV}(\mathcal{W}_t),\; o_t,\; M \}
$$

**含义**: 定义流式上下文，包含滑动窗口KV缓存、当前观测和长期记忆

**符号说明**:
- $\text{KV}(\mathcal{W}_t)$: 短期滑动窗口的Key-Value缓存
- $o_t$: 当前RGB观测
- $M$: 均匀采样历史关键帧的长期记忆token

### 公式2: 动作空间

$$
a_t^{(i)} \in \mathcal{A} := \{\uparrow, \leftarrow, \rightarrow, \text{STOP}\}
$$

**含义**: 每步输出4个离散导航动作

**符号说明**:
- $\uparrow$: 前进25cm
- $\leftarrow, \rightarrow$: 左/右转15度

### 公式3: [[Vision-Language-Action|VLA策略]]

$$
a_t = \text{VLA}(I, \text{Stream}_{0:t})
$$

**含义**: VLA策略根据指令和流式上下文生成动作

### 公式4: 统一模型输出

$$
a_t,\; \hat{F}_{t+1}^{2D},\; \hat{F}_{t+1}^{3D} = \text{UM}(I, \text{Stream}_{0:t})
$$

**含义**: 统一模型同时输出导航动作和下一步2D/3D潜在预测

### 公式5: [[SigLIP|2D语义编码]]

$$
F_t^{2D} = \text{SigLIP}(o_t)
$$

**含义**: SigLIP编码器提取2D语义特征

### 公式6: [[CUT3R|3D编码器前处理]]

$$
F_t^{3D,pre} = \text{Encoder}(o_t)
$$

**含义**: CUT3R的ViT编码器对当前帧进行初始编码

### 公式7: [[CUT3R|3D流式解码]]

$$
[p'_t, F_t^{3D}],\; s_t = \text{Decoders}([p_t, F_t^{3D,pre}],\; s_{t-1})
$$

**含义**: CUT3R解码器利用上一步状态token进行流式3D特征解码

**符号说明**:
- $p_t$: 可学习的位姿token
- $s_{t-1}$: 上一步的状态token
- $p'_t$: 更新后的位姿token

### 公式8: [[Cross-Attention|2D-3D融合]]

$$
F_t^{fuse} = \text{softmax}\!\Bigg(\frac{(F_t^{2D} W_Q)(F_t^{3D} W_K)^\top}{\sqrt{d_k}}\Bigg)(F_t^{3D} W_V)
$$

**含义**: 以2D语义特征为Query、3D空间特征为Key和Value的交叉注意力融合

**符号说明**:
- $W_Q, W_K, W_V$: 可学习投影矩阵
- $d_k$: Key维度

### 公式9-10: Stream Query Token 嵌入提取

$$
e_{t+1}^{2D} = \text{LLM}(I, \text{Stream}_{0:t} \mid \langle q_t^{2D} \rangle)
$$

$$
e_{t+1}^{3D} = \text{LLM}(I, \text{Stream}_{0:t} \mid \langle q_t^{3D} \rangle)
$$

**含义**: 2D和3D查询token通过LLM提取流式上下文中的预测性嵌入

### 公式11-12: 潜在特征解码

$$
\hat{F}_{t+1}^{2D} = \text{Decoder}_{2D}(e_{t+1}^{2D} \mid \langle m_t^{2D} \rangle)
$$

$$
\hat{F}_{t+1}^{3D} = \text{Decoder}_{3D}(e_{t+1}^{3D} \mid \langle m_t^{3D} \rangle)
$$

**含义**: 轻量Transformer解码器将嵌入还原为token级潜在特征

**符号说明**:
- $\langle m_t^{2D} \rangle, \langle m_t^{3D} \rangle$: 可学习的掩码token，重复到目标长度
- 每个解码器2层

### 公式13: [[Cosine Similarity|2D预测损失]]

$$
\mathcal{L}_{2D} = 1 - \cos(\hat{F}_{t+1}^{2D},\; F_{t+1}^{2D})
$$

**含义**: 2D潜在预测使用余弦距离损失（与SigLIP的pairwise sigmoid loss几何对齐）

### 公式14: [[Mean Squared Error|3D预测损失]]

$$
\mathcal{L}_{3D} = \text{MSE}(\hat{F}_{t+1}^{3D},\; F_{t+1}^{3D})
$$

**含义**: 3D潜在预测使用MSE损失（CUT3R特征上MSE稳定）

### 公式15: 总损失

$$
\mathcal{L}_{all} = \mathcal{L}_{nav} + \gamma(\alpha \mathcal{L}_{2D} + \beta \mathcal{L}_{3D})
$$

**含义**: 总损失由导航交叉熵损失和加权预测损失组成

**符号说明**:
- $\mathcal{L}_{nav}$: 动作交叉熵损失
- $\gamma = 0.01$: 预测分支总权重
- $\alpha = 0.25, \beta = 0.75$: 2D和3D预测的相对权重

---

## 关键图表

### Figure 1: Overview / 系统概览

![Figure 1](https://arxiv.org/html/2603.03739v1/x1.png)

**说明**: PROSPECT总览。(a) 流式设置：流式注意力掩码确保时间因果性并隔离2D/3D查询token。SigLIP和CUT3R分别提供2D语义和绝对尺度3D空间特征流，通过交叉注意力融合。(b) 统一模型：训练时stream query tokens在冻结SigLIP/CUT3R监督下预测下一步潜在特征（推理无开销）。推理时仅VLA策略运行。(c) 在VLN-CE上取得第一梯队性能，RxR上改进更大。

### Figure 2: Architecture / 模型架构

![Figure 2](https://arxiv.org/html/2603.03739v1/x2.png)

**说明**: PROSPECT架构。指令和观测（历史关键帧+当前帧）共享管道：冻结SigLIP和CUT3R通过交叉注意力融合；关键帧压缩为长期记忆M。模型使用KV缓存并自回归输出导航动作。训练时2D/3D查询token反向查询流式上下文，轻量解码器在余弦（2D）和MSE（3D）损失下预测下一步潜在特征。预测分支在推理时移除。

### Figure 3: Streaming Attention Mask / 流式注意力掩码

![Figure 3](https://arxiv.org/html/2603.03739v1/x3.png)

**说明**: PROSPECT使用的流式注意力掩码。上方（灰色）：导航上下文和动作的因果掩码。中间（红色）：2D查询token只关注当前和之前轮次的上下文/动作，隔离其他查询。下方（蓝色）：3D查询token同理。

### Figure 4: Real Robot Views / 真实机器人视角

![Figure 4](https://arxiv.org/html/2603.03739v1/x4.png)

**说明**: ARX-Lift2机器人在不同室内外光照条件下的第一人称视角，包括办公室、仓库、走廊、午后、黄昏、夜间街道。

### Table 1: VLN-CE R2R & RxR Val-Unseen 主结果

| Method | Pano. | Odo. | Depth | S.RGB | R2R NE↓ | R2R OSR↑ | R2R SR↑ | R2R SPL↑ | RxR NE↓ | RxR SR↑ | RxR SPL↑ | RxR nDTW↑ |
|--------|-------|------|-------|-------|---------|----------|---------|----------|---------|---------|----------|-----------|
| HPN+DN | Yes | Yes | Yes | - | 6.31 | 40.0 | 36.0 | 34.0 | - | - | - | - |
| CMA | Yes | Yes | Yes | - | 6.20 | 52.0 | 41.0 | 36.0 | 8.76 | 26.5 | 22.1 | 47.0 |
| VLN-BERT | Yes | Yes | Yes | - | 5.74 | 53.0 | 44.0 | 39.0 | 8.98 | 27.0 | 22.6 | 46.7 |
| GridMM | Yes | Yes | Yes | - | 5.11 | 61.0 | 49.0 | 41.0 | - | - | - | - |
| LAW | - | Yes | - | Yes | 6.83 | 44.0 | 35.0 | 31.0 | 10.90 | 8.0 | 8.0 | 38.0 |
| NavMorph | - | - | Yes | Yes | 5.75 | 56.9 | 47.9 | 33.2 | 8.85 | 30.8 | 22.8 | 44.2 |
| NaVid | - | - | - | Yes | 5.47 | 49.1 | 37.4 | 35.9 | - | - | - | - |
| Uni-NaVid | - | - | - | Yes | 5.58 | 53.3 | 47.0 | 42.7 | 6.24 | 48.7 | 40.9 | - |
| NaVILA* | - | - | - | Yes | 5.37 | 57.6 | 49.7 | 45.5 | - | - | - | - |
| StreamVLN* | - | - | - | Yes | 5.47 | 57.8 | 50.8 | 45.7 | 6.72 | 48.6 | 42.5 | 60.2 |
| **PROSPECT*** | - | - | - | Yes | **5.31** | **60.3** | **52.0** | **46.2** | **5.93** | **52.7** | **42.8** | **60.6** |
| NaVILA+ | - | - | - | Yes | 5.22 | 62.5 | 54.0 | 49.0 | 6.77 | 49.3 | 44.0 | 58.8 |
| StreamVLN+ | - | - | - | Yes | 5.10 | 64.0 | 55.7 | 50.9 | 6.22 | 52.9 | 46.0 | 61.9 |
| **PROSPECT+** | - | - | - | Yes | **4.92** | **65.2** | **58.9** | **54.0** | **5.70** | **54.6** | **46.2** | **62.1** |

**说明**: *=仅MP3D+VideoQA训练, +=加入ScaleVLN等额外数据。PROSPECT在两种设置下均达到最优。RxR上的改进更大（更长轨迹+更长指令），表明对长程导航特别有益。

### Table 2: 模块消融 (R2R val-unseen)

| Setting | NE↓ | OSR↑ | SR↑ | SPL↑ |
|---------|-----|------|-----|------|
| Baseline (SigLIP only) | 6.05 | 53.8 | 45.5 | 41.6 |
| + CUT3R | 5.91 | 55.0 | 46.7 | 41.8 |
| + WM-2D only | 5.89 | 56.0 | 47.0 | 42.0 |
| + WM-3D only | 5.90 | 55.4 | 47.2 | 41.9 |
| **+ WM-2D + WM-3D** | **5.82** | **57.6** | **48.7** | **42.9** |

**关键发现**: 2D和3D潜在预测提供互补信号，组合后效果最佳

### Table 3: 空间编码器消融 (R2R val-unseen)

| Encoder | Time (s) | SR↑ | SPL↑ | OSR↑ | NE↓ |
|---------|----------|-----|------|------|-----|
| VGGT | OOM | OOM | OOM | OOM | OOM |
| InfiniteVGGT | 0.284 | 43.2 | 38.0 | 54.4 | 6.61 |
| **CUT3R (Ours)** | **0.245** | **48.7** | **42.9** | **57.6** | **5.82** |

**关键发现**: VGGT在长序列上OOM；CUT3R在精度和延迟上均优于InfiniteVGGT，归因于绝对尺度vs相对尺度表示

### Table 4: 任务复杂度分析 (R2R val-unseen)

| Horizon | Model | SR↑ | SPL↑ | OSR↑ | NE↓ |
|---------|-------|-----|------|------|-----|
| Short (1-50步) | Baseline | 51.20 | 48.18 | 55.34 | 5.08 |
| Short (1-50步) | PROSPECT | 51.23 | 48.84 | 54.53 | 4.86 |
| Medium (50-100步) | Baseline | 49.61 | 43.79 | 61.27 | 5.64 |
| Medium (50-100步) | PROSPECT | **54.29** | **48.04** | **63.71** | 5.46 |
| Long (>=100步) | Baseline | 20.18 | 10.61 | 34.21 | 9.11 |
| Long (>=100步) | PROSPECT | **24.32** | **14.25** | **40.75** | **8.74** |

**关键发现**: 短任务上持平，中/长任务上显著提升，表明在长流式上下文下泛化更强

### Table 5: 注意力掩码消融 (R2R val-unseen)

| Mask Design | NE↓ | OSR↑ | SR↑ | SPL↑ |
|-------------|-----|------|-----|------|
| Leaky | 6.81 | 51.3 | 40.2 | 35.7 |
| w/o Isolation | 6.98 | 51.1 | 39.9 | 35.3 |
| **Ours** | **5.82** | **57.6** | **48.7** | **42.9** |

**关键发现**: 隔离性和因果严格性对稳健导航至关重要

### Table 6: 真实机器人成功率

| Scene | Lighting | NaVid | StreamVLN | PROSPECT |
|-------|----------|-------|-----------|----------|
| Indoor - Office | Bright | 7/30 | 12/30 | **20/30** |
| Indoor - Warehouse | Bright | 6/30 | 12/30 | **18/30** |
| Indoor - Corridor | Moderate | 11/30 | 16/30 | **22/30** |
| Outdoor - Afternoon | Bright | 6/30 | 10/30 | **18/30** |
| Outdoor - Dusk | Moderate | 4/30 | 6/30 | **11/30** |
| Outdoor - Night Street | Low | 2/30 | 6/30 | **9/30** |

**关键发现**: PROSPECT在所有场景和光照条件下均大幅超越基线，尤其在困难光照条件（黄昏、夜间）下优势更明显

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| R2R-CE | ~5% | 标准VLN指令 | Stage 1训练 + 评估 |
| RxR-CE | ~14% | 长指令（~120词） | Stage 1训练 + 评估 |
| R2R-EnvDrop | ~80% | 数据增强 | Stage 1训练 |
| DAgger | ~260K | 在线纠错样本 | Stage 2训练 |
| ScaleVLN | ~314K | 大规模增强 | Stage 2训练 |
| LLaVA-Video-178K | 178K | 视频QA | Stage 2混合 |
| ScanQA | - | 3D场景QA | Stage 2混合 |

### 实现细节

- **Backbone**: LLaVA-NeXT-Video-7B (Qwen1.5-7B LLM)
- **2D编码器**: SigLIP（冻结）
- **3D编码器**: CUT3R（冻结）
- **学习率**: SigLIP 5e-6, 其他 2e-5 (peak)
- **训练轮数**: Stage 1: 1 epoch, Stage 2: 1 epoch
- **硬件**: 8x A800 GPU
- **训练成本**: Stage 1: 560 A800 GPU-hours, Stage 2: ~1900 A800 GPU-hours
- **推理速度**: ~4 Hz（远程推理）
- **预测分支**: 196个掩码token + 9个查询token/模态

### 可视化结果

真实机器人部署在ARX-Lift2上，搭载RealSense 405头部相机。室内通过Wi-Fi/LAN远程推理（双RTX-4090，~0.25s/步），室外通过公网（双A800，~0.27s/步）。单RTX 4070车载推理可行但成功率较低。

---

## 批判性思考

### 优点
1. 预测分支仅在训练时活跃，不增加推理开销——非常优雅的设计
2. CUT3R的流式特性完美契合长程导航场景，避免了VGGT的OOM问题
3. 在潜在空间而非像素/深度空间做预测，更关注动态感知而非纹理细节
4. 真实机器人部署验证了方法的实用性

### 局限性
1. 仍依赖离散动作空间（前进25cm/转15度），限制了在连续控制场景的应用
2. 需要远程GPU推理（双4090或双A800），车载部署有精度损失
3. 训练成本较高（~2460 A800 GPU-hours）

### 潜在改进方向
1. 扩展到连续动作空间
2. 模型蒸馏以支持更高效的车载推理
3. 探索在线适应机制

### 可复现性评估
- [ ] 代码开源（计划中）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整
- [x] 数据集可获取

---

## 关联笔记

### 基于
- [[StreamVLN]]: 流式VLA基线，本文在其基础上扩展
- [[CUT3R]]: 流式3D基础模型，提供绝对尺度空间特征
- [[SigLIP]]: 2D语义编码器
- [[JEPA]]: 潜在预测学习的思想来源

### 对比
- [[NaVILA]]: 端到端VLN方法
- [[NaVid]]: 视频导航方法
- [[Uni-NaVid]]: 统一导航方法
- [[NavMorph]]: 使用深度+RGB的方法

### 方法相关
- [[Cross-Attention]]: 2D-3D特征融合
- [[KV Cache]]: 流式上下文管理
- [[Vision-Language-Action]]: VLA范式
- [[World Model]]: 预测表征学习

### 硬件/数据相关
- [[ARX-Lift2]]: 部署机器人平台
- [[Matterport3D]]: VLN-CE评估环境

---

## 速查卡片

> [!summary] PROSPECT
> - **核心**: 统一流式VLA + 潜在预测表征学习，CUT3R 3D编码
> - **方法**: Stream Query Tokens在训练时预测下一步2D/3D潜在特征（推理时移除）
> - **结果**: VLN-CE R2R SR 58.9%, SPL 54.0%; 真实机器人多光照部署
> - **代码**: 计划开源

---

*笔记创建时间: 2026-04-27*
