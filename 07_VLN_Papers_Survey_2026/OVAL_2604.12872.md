---
title: "OVAL: Open-Vocabulary Augmented Memory Model for Lifelong Object Goal Navigation"
method_name: "OVAL"
authors: [Jiahua Pei, Yi Liu, Guoping Pan, Yuanhao Jiang, Houde Liu, Xueqian Wang]
year: 2026
venue: arXiv
tags: [object-navigation, open-vocabulary, lifelong-navigation, memory-model, frontier-exploration, semantic-map]
zotero_collection: ""
image_source: online
arxiv_html: https://arxiv.org/html/2604.12872v1
created: 2026-04-27
---

# 论文笔记：OVAL: Open-Vocabulary Augmented Memory Model for Lifelong Object Goal Navigation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Shenzhen International Graduate School, Tsinghua University |
| 日期 | April 2026 |
| 项目主页 | N/A |
| 对比基线 | [[VLFM]], [[GOAT]] |
| 链接 | [arXiv](https://arxiv.org/abs/2604.12872) |

---

## 一句话总结

> OVAL 通过开放语义记忆模型（基于颜色直方图+SuperGlue 特征匹配的实例识别）和概率地图探索策略（距离+语义+足迹三因素），实现终身开放词汇物体导航中 68.1% SR。

---

## 核心贡献

1. **开放语义记忆模型**: 使用记忆描述符（标签、图像缓冲、3D 坐标、HSV 直方图、置信度）和两阶段实例匹配（直方图+SuperGlue），实现跨 episode 的结构化记忆管理
2. **概率地图探索策略**: 融合距离因素、语义共现因素和足迹因素的多值 frontier 评分，提升终身探索效率
3. **终身 ObjectNav 评估**: 在 HM3D 和 MP3D 上建立终身开放词汇 ObjectNav 评估协议

---

## 问题背景

### 要解决的问题
[[Object Navigation|物体目标导航]]（ObjectNav）中，现有方法处理孤立的单物体导航任务，但真实机器人需要连续完成多个开放词汇目标（如酒店场景中"找杯子→找遥控器→..."），需要终身记忆支持。

### 现有方法的局限
- 不可预测的开放词汇自动标注（重复存储、同义词、误识别）给结构化记忆带来障碍
- [[GOAT]] 等方法使用 Mask R-CNN 需要预定义词汇表，无法处理真正的开放词汇
- 场景信息与 frontier 选择之间存在鸿沟，持续探索中无法有效利用累积价值图

### 本文的动机
记忆应捕获物体的区分性特征而非原始标签，通过特征匹配进行实例识别；探索应整合语义信息同时防止重复访问。

---

## 方法详解

### 模型架构

OVAL 采用 **三模块管道** 架构：
- **Frontier Exploration**: 深度+位姿构建 2D 俯视图网格，[[DBSCAN]] 聚类 frontier，概率地图选择
- **Open-Semantic Memory Model**: [[Grounded-SAM2]] 零样本检测 → 关键词过滤 → 记忆描述符 → 实例匹配（HSV+[[SuperGlue]]）
- **Navigation**: KMP 子串搜索跨粒度查询 → 路径规划 → 360° 全景 + LLM 验证 → STOP

### 核心模块

#### 模块1: Frontier Exploration with Probability Map

**设计动机**: 在终身导航中，单纯的距离优先或语义优先都不够——需要同时考虑距离、语义关联和历史足迹。

**具体实现**:
- 深度+位姿构建 2D 俯视图网格地图
- [[DBSCAN]] 聚类 frontier 点
- 概率地图综合三个因素选择 frontier
- 记录已尝试 frontier 避免重复

#### 模块2: Open-Semantic Memory Model

**设计动机**: 开放词汇检测的输出不稳定（同一物体可能被标记为不同名称），需要基于视觉特征而非标签进行实例识别。

**具体实现**:
- **预处理**: [[Grounded-SAM2]] Autolabel 模式零样本检测，关键词匹配过滤噪声标签（"wall"等）
- **记忆描述符**: 五元组 $C = \{S_i, I_i, X_i, H_i, C_i\}$（标签、图像缓冲、3D 坐标、HSV 直方图、置信度）
- **实例管理**: 新标签 → 创建新实例；已有标签 → 计算相似度 $S_m$ → 阈值判断 → 模糊区间使用 [[SuperGlue]] 特征匹配

#### 模块3: Navigation with Verification

**设计动机**: 开放词汇场景中目标名可能不完全匹配（如"desk"对应"green desk"），需要模糊匹配和置信度验证。

**具体实现**:
- KMP 子串搜索解决跨粒度查询
- 贪心路径规划器导航至目标 $X_i$
- 到达后 360° 全景捕获，LLM 估计目标存在概率 $\phi(\cdot)$
- 综合验证评分 $S = \omega \cdot \phi(\{V_k\}) + (1-\omega) \cdot C_i$ 超过阈值则 STOP

---

## 关键公式

### 公式1: [[Frontier Exploration|Frontier 选择概率]]

$$
P(F) \propto o_d(F) + o_s(F) + o_f(F)
$$

**含义**: Frontier 被选择的概率正比于距离因素、语义因素和足迹因素之和。

**符号说明**:
- $o_d(F)$: 距离因素（高斯分布，优先选择近处 frontier）
- $o_s(F)$: 语义因素（基于物体共现关联的语义相关性）
- $o_f(F)$: 足迹因素（负高斯和，惩罚已访问区域）

### 公式2: [[Frontier Exploration|三因素详细定义]]

**距离因素**:
$o_d(F) = A_d \cdot \exp(-\|F - p_{\text{agent}}\|^2 / (2\sigma_d^2))$，距离超过 $d_{\text{th}}$ 截断

**语义因素**:
$o_s(F) = A_s \cdot \sum_k P(G_t, S_k) \cdot \exp(-\|F - X_k\|^2 / (2\sigma_s^2))$

**足迹因素**:
$o_f(F) = -A_f \cdot \sum_j \exp(-\|F - p_j\|^2 / (2\sigma_f^2))$

**符号说明**:
- $A_d, A_s, A_f$: 各因素振幅权重
- $\sigma_d, \sigma_s, \sigma_f$: 空间衰减率
- $P(G_t, S_k)$: 目标 $G_t$ 与物体类别 $S_k$ 的语义共现概率
- $p_j$: 历史 agent 位置

### 公式3: [[Feature Matching|记忆置信度]]

$$
C_i = \exp\!\Big(-\sigma \cdot \|\bar{p}_c - p_c\| \cdot \frac{D_{t,\bar{p}_c}}{A_b}\Big)
$$

**含义**: 基于物体在图像中的中心偏移、深度距离和边界框面积计算观测置信度。

**符号说明**:
- $\bar{p}_c$: 边界框中心
- $p_c$: 图像中心
- $D_{t,\bar{p}_c}$: 物体深度距离
- $A_b$: 边界框面积
- $\sigma$: 超参数

### 公式4: [[Feature Matching|实例相似度]]

$$
S_m = \lambda_H \cdot \text{Sim}(H_i, H_j) - \lambda_X \cdot \text{Sigmoid}(k \cdot \|X_i - X_j\|)
$$

**含义**: 综合颜色直方图相似度和空间距离判断两个观测是否为同一实例。

**符号说明**:
- $H_i, H_j$: HSV 颜色直方图
- $X_i, X_j$: 3D 坐标
- $\lambda_H, \lambda_X$: 权重系数
- $\text{Sim}(\cdot,\cdot)$: HSV 通道逐元素最小值

### 公式5: [[SuperGlue|特征匹配判定]]

$$
M_{\text{sg}} = \{(p_m, q_n) \mid \text{score}(p_m, q_n) > \tau_{\text{sg}}\}
$$

**含义**: SuperGlue 特征匹配成功要求匹配对数 $|M_{\text{sg}}| \geq \tau_M$。

### 公式6: [[Object Navigation|综合验证评分]]

$$
S = \omega \cdot \phi(\{V_k\}) + (1 - \omega) \cdot C_i
$$

**含义**: 结合 360° 全景 LLM 评估概率 $\phi$ 和记忆置信度 $C_i$ 判断是否到达目标。

**符号说明**:
- $\omega$: 权重（默认 0.5）
- $\phi(\{V_k\})$: LLM 对全景图的目标存在概率估计
- $C_i$: 记忆中的物体置信度

---

## 关键图表

### Figure 1: Motivation / 动机示例

![Figure 1](https://arxiv.org/html/2604.12872v1/x1.png)

**说明**: 酒店场景中的终身开放词汇物体导航：机器人需在未知房间中依次寻找 cup、remote control 等物体，需要跨 episode 的记忆管理。

### Figure 2: Pipeline Overview / 系统流水线

![Figure 2](https://arxiv.org/html/2604.12872v1/x2.png)

**说明**: OVAL 流水线。Frontier exploration 构建网格地图并进行概率选择；Open-semantic memory model 通过自动标注、关键词过滤和相似度计算管理记忆；Navigation 模块使用 KMP 查询和验证进行目标确认。

### Figure 3: Memory Model Management / 记忆模型管理

![Figure 3](https://arxiv.org/html/2604.12872v1/x3.png)

**说明**: 实例匹配器工作流。新观测通过标签检查、HSV 直方图+空间距离相似度、模糊区间 SuperGlue 匹配三阶段判断是新实例还是已有实例。

### Figure 4: Lifelong Performance Comparison / 终身性能对比

![Figure 4](https://arxiv.org/html/2604.12872v1/x4.png)

**说明**: 在 HM3D 上与 GOAT 的终身 ObjectNav 性能对比（柱状图/折线图显示 SR 和 SPL）。

### Figure 5: Ablation on Lifelong Targets / 终身目标数消融

![Figure 5](https://arxiv.org/html/2604.12872v1/x5.png)

**说明**: 不同终身目标数量下的性能消融。性能在 4+ 个目标后趋于稳定。

### Table I: Lifelong ObjectNav (1000 episodes)

| Method | Open-Vocab | Lifelong Memory | HM3D SR | HM3D SPL | MP3D SR | MP3D SPL |
|--------|-----------|----------------|---------|----------|---------|----------|
| VLFM | ✓ | ✗ | 53.7 | 30.6 | 35.1 | 16.7 |
| GOAT | ✗ | ✓ | 59.2 | 31.2 | N/A | N/A |
| **OVAL** | **✓** | **✓** | **68.1** | **33.8** | **44.1** | **18.6** |

**说明**: OVAL 同时具备开放词汇和终身记忆能力，在 HM3D 上 SR 68.1% 显著超越 VLFM（53.7%）和 GOAT（59.2%）。GOAT 在 MP3D 上因复杂多样的物体类别需要开放词汇能力而失败。

### Table II: Ablation on HM3D Lifelong ObjectNav

| Verify STOP | Memory | Prob Map | SR | SPL |
|-------------|--------|----------|------|------|
| ✗ | ✓ | ✓ | 61.3 | 31.2 |
| ✓ | ✗ | ✓ | 56.1 | 26.0 |
| ✓ | ✓ | ✗ | 66.8 | 32.3 |
| ✓ | ✓ | ✓ | **68.1** | **33.8** |

**关键发现**: 记忆系统是核心组件（去除后 SR 降 12%）；验证停止和概率地图各贡献约 1-7% SR 提升。

### Table III: Standard ObjectNav (1000 episodes)

| Method | Open-Vocab | Training-Free | Lifelong | HM3D SR | HM3D SPL | MP3D SR | MP3D SPL |
|--------|-----------|--------------|----------|---------|----------|---------|----------|
| PONI | ✗ | ✗ | ✗ | - | - | 31.8 | 12.1 |
| SemEXP | ✗ | ✗ | ✗ | - | - | 36.0 | 14.4 |
| L3MVN | ✗ | ✓ | ✗ | 50.4 | 23.1 | - | - |
| Habitat-Web | ✓ | ✗ | ✗ | 41.5 | 16.0 | 31.6 | 8.5 |
| ZSON | ✓ | ✗ | ✗ | 25.5 | 12.6 | 15.3 | 4.8 |
| OVRL | ✓ | ✗ | ✗ | - | - | 28.6 | 7.4 |
| OVRL-V2 | ✓ | ✗ | ✗ | 64.7 | 28.1 | - | - |
| PixNav | ✓ | ✗ | ✗ | 37.9 | 20.5 | - | - |
| ESC | ✓ | ✓ | ✗ | 39.2 | 22.3 | 28.7 | 14.2 |
| VLFM | ✓ | ✓ | ✗ | 52.5 | 30.4 | 36.4 | 17.5 |
| VoroNav | ✓ | ✓ | ✗ | 42.0 | 26.0 | - | - |
| OpenFMNav | ✓ | ✓ | ✗ | 54.9 | 24.4 | - | - |
| TopV-Nav | ✓ | ✓ | ✗ | 45.9 | 28.0 | - | - |
| Instruct-Nav | ✓ | ✓ | ✗ | 58.0 | 20.9 | - | - |
| GOAT | ✗ | ✓ | ✓ | 50.6 | 24.1 | - | - |
| **OVAL** | **✓** | **✓** | **✓** | **58.2** | 24.5 | **41.1** | 15.3 |

**说明**: OVAL 在 training-free 开放词汇方法中 HM3D SR 达到 58.2%（仅次于 Instruct-Nav 的 58.0% 但具备终身能力），MP3D 上 41.1% SR 为总体最高。

### Table IV: Probability Map Ablation (HM3D ObjectNav)

| Footprint | Distance | Semantics | SR | SPL |
|-----------|----------|-----------|------|------|
| ✗ | ✓ | ✓ | 54.3 | 21.4 |
| ✓ | ✗ | ✓ | 56.7 | 22.8 |
| ✓ | ✓ | ✗ | 56.2 | 23.9 |
| ✓ | ✓ | ✓ | **58.2** | **24.5** |

**关键发现**: 足迹因素最重要（去除后 SR 降 3.9%），防止冗余探索；语义因素通过共现关联增强探索方向性；距离因素优先近处区域。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[HM3D]] | 验证集 | 室内 3D 场景 | 终身/标准 ObjectNav |
| [[MP3D]] | 验证集 | 室内 3D 场景，物体类别更复杂多样 | 终身/标准 ObjectNav |

### 实现细节

- **检测**: [[Grounded-SAM2]] (Florence-2-base, DETAILED_CAPTION)
- **特征匹配**: [[SuperGlue]]
- **LLM**: [[GPT-4o]]（验证）, [[GPT-4o-mini]]（同义词生成）
- **硬件**: RTX 4090D (24GB)
- **关键参数**: $A_d=1, A_s=0.5, A_f=1, \sigma_d=\sigma_s=\sigma_f=10^6, \lambda_H=\lambda_X=0.5, \tau_l=0.2, \tau_u=0.8, \tau_M=60, \omega=0.5, k=8$
- **仿真**: [[Habitat]] on HM3D/MP3D
- **成功条件**: STOP 在目标 1m 内，500 步限制
- **FPS**: 1.14（可比 VLFM 1.67, GOAT 1.85）

### 终身 ObjectNav 数据生成

- Episode 按 Scene ID → Floor Height 重排
- 同层继续：保留地图/记忆，传送至上一 episode 终点
- 换层/换场景：清除地图/记忆，标准初始化

### 可视化结果

Figure 4 展示了 OVAL 与 GOAT 在 HM3D 终身 ObjectNav 上的性能对比；Figure 5 展示了不同终身目标数量下性能趋于稳定的趋势。

---

## 批判性思考

### 优点
1. **实用的记忆架构**: HSV 直方图+SuperGlue 的两阶段匹配在效率和准确性间取得平衡
2. **概率地图设计合理**: 三因素融合的思路清晰，消融实验验证了每个因素的贡献
3. **填补终身开放词汇空白**: 同时具备 open-vocabulary 和 lifelong memory 的唯一方法

### 局限性
1. **仅支持 ObjectNav**: 不支持多模态输入（图像、描述、问题等指令）
2. **动态场景识别退化**: 在复杂、动态和快速移动场景中识别准确性下降
3. **预定义语义共现**: 语义因素依赖预定义的物体共现组，不够灵活

### 潜在改进方向
1. 扩展到多模态导航输入
2. 用学习的语义关联替代预定义共现
3. 引入动态场景适应机制

### 可复现性评估
- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（training-free）
- [x] 数据集可获取（公开 benchmark + 生成协议）

---

## 关联笔记

### 基于
- [[VLFM]]: 视觉语言 frontier maps
- [[GOAT]]: GO to Any Thing
- [[OneMap]]: 实时开放词汇映射

### 对比
- [[ESC]]: 软常识约束探索
- [[SemEXP]]: 目标导向语义探索
- [[Instruct-Nav]]: 指令驱动导航
- [[OpenFMNav]]: 开放基础模型导航

### 方法相关
- [[Grounded-SAM2]]: 开放世界接地分割
- [[SuperGlue]]: 图神经网络特征匹配
- [[DBSCAN]]: 密度聚类算法
- [[KMP Algorithm]]: 字符串匹配算法
- [[HSV Color Space]]: 颜色直方图表示

### 硬件/数据相关
- [[HM3D]]: Habitat-Matterport 3D 数据集
- [[MP3D]]: Matterport3D 数据集
- [[Habitat]]: 具身 AI 仿真平台

---

## 速查卡片

> [!summary] OVAL: Open-Vocabulary Augmented Memory for Lifelong ObjectNav
> - **核心**: 开放语义记忆+概率地图探索实现终身开放词汇物体导航
> - **方法**: Grounded-SAM2 检测 + HSV/SuperGlue 实例匹配 + 距离/语义/足迹概率 frontier 选择
> - **结果**: HM3D 终身 68.1% SR, 标准 58.2% SR; MP3D 终身 44.1% SR
> - **代码**: 未公开

---

*笔记创建时间: 2026-04-27*
