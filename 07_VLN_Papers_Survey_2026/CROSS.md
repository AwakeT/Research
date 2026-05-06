---
title: "Change-Robust Online Spatial-Semantic Topological Mapping"
method_name: "CROSS"
authors: [Jiaming Wang, Chen Jizhuo, Liu Diwen, Atharva Ajay Ghotavadekar, Da Jiaxuan, Linh Kästner, Harold Soh]
year: 2026
venue: arXiv
tags: [topological-mapping, spatial-semantic, change-robust, navigation, slam, object-navigation, gaussian-sum-filter, se3]
zotero_collection: ""
image_source: online
arxiv_html: https://arxiv.org/html/2605.02227v1
created: 2026-05-06
---

# 论文笔记：Change-Robust Online Spatial-Semantic Topological Mapping

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | National University of Singapore (NUS) |
| 日期 | May 2026 |
| 项目主页 | 代码待公开 |
| 对比基线 | [[ORB-SLAM3]], [[RTAB-Map]], [[MASt3R-SLAM]], GM, SM, PBU, ABM |
| 链接 | [arXiv](https://arxiv.org/abs/2605.02227) |

---

## 一句话总结

> 提出 CROSS，用在线 pose-aware 拓扑图 + 连续 SE(3) 上的高斯混合信念实现 change-robust 空间语义推理，在严重外观变化下（光照+家具重排）的 object-goal navigation 中大幅超越 SLAM 基线。

---

## 核心贡献

1. **CROSS 拓扑语义表示**: 在线 pose-aware RGB-D 关键帧拓扑图替代全局度量地图，天然抗局部外观变化
2. **SE(3) 上的序列假设检验 (SHT)**: 维护有界高斯混合信念处理感知歧义、回环闭合和"绑架"事件
3. **多假设联合推理**: 测量消息聚合多个检索关键帧的证据，自然处理 tracking loss
4. **真实四足机器人验证**: 光照变化+家具重排下 object-goal navigation SR 70-80%（vs RTAB-Map 30-60%）

---

## 问题背景

### 要解决的问题
自主机器人需要 change-robust 的空间语义推理：在环境变化下决定去哪里、怎么去、在哪里。

### 现有方法的局限
- **度量 SLAM + 语义**: 将语义挂在 SLAM 度量地图上，外观偏移和场景动态下数据关联和重定位退化严重
- **VPR 方法**: 仅做离散地点匹配，缺乏连续位姿估计
- **传统拓扑方法**: 缺乏空间几何推理和不确定性建模

### 本文的动机
用拓扑图替代全局度量地图——拓扑图天然对局部外观变化鲁棒；在连续 SE(3) 上进行多假设位姿推理，严格处理感知歧义。

---

## 方法详解

### 模型架构

CROSS 表示环境为稀疏 pose-aware 拓扑图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$。每个节点 $v_i \in \mathcal{V}$ 存储 RGB-D 关键帧 $z_i$ 和相机位姿 $c_i \in \mathrm{SE}(3)$。机器人状态 $(x_t, n_t)$，$x_t \in \mathrm{SE}(3)$ 为位姿，$n_t$ 索引最近关键帧。

**联合后验分解**:

$$p(x_{1:t}, \mathcal{G} \mid z_{1:t}, u_{1:t-1}) = \underbrace{p(x_{1:t} \mid z_{1:t}, u_{1:t-1}, \mathcal{G})}_{\text{state estimation}} \cdot \underbrace{p(\mathcal{G} \mid z_{1:t}, u_{1:t-1})}_{\text{map management}}$$

两个模块：**状态估计**（橙色）和**地图管理**（蓝色）。

### 核心模块

#### 模块1: SE(3) 上的高斯混合信念

机器人位姿和关键帧位姿均建模为 SE(3) 上的有限高斯混合：

$$p(x_t) \approx \sum_{k=1}^{K} w_t^{(k)} \mathcal{N}_{\mathfrak{se}(3)}\left(\log\big((\mu_t^{(k)})^{-1} x_t\big); \mathbf{0}, \Sigma_t^{(k)}\right)$$

$$p(c_i) \approx \sum_{k=1}^{K} w_i^{(k)} \mathcal{N}_{\mathfrak{se}(3)}\left(\log\big((\mu_i^{(k)})^{-1} c_i\big); \mathbf{0}, \Sigma_i^{(k)}\right)$$

信念通过前向消息传递计算：

$$p(x_t \mid z_{1:t}, u_{1:t-1}, \mathcal{G}) \propto \underbrace{m_t^{\text{meas}}(x_t)}_{\text{measurement}} \underbrace{m_t^{\text{mot}}(x_t)}_{\text{motion}}$$

#### 模块2: 前向运动消息

运动因子: $\phi_t^{\text{mot}}(x_{t-1}, x_t, u_t) \triangleq p(x_t \mid x_{t-1}, u_t)$

$$m_t^{\text{mot}}(x_t) \approx \sum_{k=1}^{K} w_{t-1}^{(k)} \mathcal{N}_{\mathfrak{se}(3)}\left(\log\big((\mu_{t|t-1}^{(k)})^{-1} x_t\big); 0, \Sigma_{t|t-1}^{(k)}\right)$$

- 均值: $\mu_{t|t-1}^{(k)} = \mu_{t-1}^{(k)} u_t$
- 协方差: $\Sigma_{t|t-1}^{(k)} \approx \mathrm{Ad}_{u_t^{-1}} \Sigma_{t-1}^{(k)} \mathrm{Ad}_{u_t^{-1}}^\top + Q_t$
- 权重保持: $w_{t|t-1}^{(k)} = w_{t-1}^{(k)}$

$u_t$ 为里程计输入，$Q_t$ 为 $\mathfrak{se}(3)$ 上的过程噪声协方差。这是 Lie 群上的一阶高斯传输近似。

#### 模块3: 全局测量消息

**局部测量模型**: 通过 PnP 计算当前观测 $z_t$ 与节点 $i$ 的相对位姿：

$$T_{i,t} = f_{\mathrm{PnP}}(z_t, z_i)$$

实现: XFeat 特征提取 + LightGlue 匹配 → 2D-3D 对应 → E-PnP 求解器。

**全局位姿（从关键帧 $i$）**: $x_t \approx c_i T_{i,t}$

由于 $c_i$ 是高斯混合，组合分布：

$$p(x_t \mid v_i) \approx \sum_{k=1}^{K} w_i^{(k)} \mathcal{N}_{\mathfrak{se}(3)}\left(\log\big((\mu_i^{(k)} T_{i,t})^{-1} x_t\big); 0, \Sigma_{i,t}^{(k)}\right)$$

其中 $\Sigma_{i,t}^{(k)} \approx \mathrm{Ad}_{T_{i,t}^{-1}} \Sigma_i^{(k)} \mathrm{Ad}_{T_{i,t}^{-1}}^\top$

**数据关联**: 潜变量 $Y_t \in \{1, \ldots, N_t\}$ 指示哪个关键帧解释 $z_t$：

$$P(i) = \mathrm{softmax}\left(\pi(i) \frac{\mathrm{inlier}(i)}{M}\right)$$

$\pi(i)$ 为 VPR 检索分数（BoQ 模型），$\mathrm{inlier}(i)$ 为 PnP 内点数，$M$ 为最大检测特征数。

**完整全局测量消息**:

$$\mathcal{H}_t(x_t) = \sum_{i=1}^{N_t} P(i) \sum_{k=1}^{K} w_i^{(k)} \mathcal{N}_{\mathfrak{se}(3)}\left(\log\big((\mu_i^{(k)} T_{i,t})^{-1} x_t\big); 0, \Sigma_i^{(k)}\right)$$

**这是核心创新**: 聚合所有检索关键帧的证据，自然处理 tracking loss、回环闭合和绑架事件。

#### 模块4: 假设管理

直接乘积产生最多 $N_t K^2$ 个混合项。产品分量 $(i,j)$ 权重：

$$w_t^{(i,j)} \propto w_{\text{mot}}^{(i)} w_{\text{meas}}^{(j)} C_{ij}$$

$C_{ij}$ 为高斯重叠度。

两阶段管理：
1. **测量聚类**: SE(3)-aware DBSCAN 将所有测量分量聚类为主导模式
2. **融合/出生/剪枝**: 将模式与运动混合融合；为非重叠模式生成新假设；剪枝低权重假设

#### 模块5: 拓扑图管理

**节点创建**: 当最大 VPR 相似度 < 阈值 $\beta$ 时创建新节点。新节点继承高斯混合结构。

**边创建**:
- *里程计边*: 连接新节点到前一节点
- *邻近边*: 连接到空间半径 $d$ 内的现有节点（默认 $d = 0.5\,\text{m}$）

**平滑 (PGO)**: 对每个假设 $h^{(l)}$，最小化：

$$\{c_i^{(l)}\} = \arg\min_{\{\mu_i^{(l)}\}} \left[\sum_{(i,j) \in \mathcal{E}_{\text{odo}}} \left\|\log\big(\hat{T}_{ij}^{-1} (\mu_i^{(l)})^{-1} \mu_j^{(l)}\big)\right\|_{\Sigma_{ij}^{-1}}^2 + \sum_{(i,j) \in \mathcal{E}_{\text{vis}}^{(l)}} \left\|\log\big(\tilde{T}_{ij}^{-1} (\mu_i^{(l)})^{-1} \mu_j^{(l)}\big)\right\|_{\Lambda_{ij}^{-1}}^2\right]$$

通过 GTSAM 独立求解每个假设。

#### 模块6: 回环闭合

对备选假设 $l > 0$：

$$\ell_t^{(l)} = \log \bar{w}_t^{(l)} / \bar{w}_t^{(0)}$$

当滑动窗口 $W$ 内 $\ell_\tau^{(l)} > 0$ 的计数超过阈值 $r$ 时宣布回环闭合。接受后联合 PGO，然后合并假设。

---

## 关键图表

### Figure 1: 系统概览

![Overview](https://arxiv.org/html/2605.02227v1/x1.png)

**说明**: CROSS 系统。橙色：tracking 模块——SE(3) 上的 SHT 状态估计。蓝色：地图管理模块——节点/边管理 + PGO 平滑。

### Figure 3: 多会话重定位可视化

![Multi-session](https://arxiv.org/html/2605.02227v1/figures/exp/all_combined_4_cropped.png)

**说明**: Rover Campus 多会话重定位结果网格。比较 ORB-SLAM3, RTAB-Map, MASt3R-SLAM, CROSS。

### Figure 4: 真实机器人环境变化

| 条件 | 图示 |
|------|------|
| 光照变化 (LC) | ![LC](https://arxiv.org/html/2605.02227v1/figures/exp/real_ssi/LC.png) |
| 物体重排 (OR) | ![OR](https://arxiv.org/html/2605.02227v1/figures/exp/real_ssi/OR.png) |
| 联合变化 (LC+OR) | ![LC+OR](https://arxiv.org/html/2605.02227v1/figures/exp/real_ssi/LC+OR.png) |

---

## 实验

### TABLE I: 外观变化下的重定位成功率 (RS)

**跨序列 (不同 mapping/testing 序列)**:

| Method | RS (OLS) | RS (Rover) | RS (All) |
|--------|----------|------------|----------|
| ORB-SLAM3 | 0.032 | 0.005 | 0.013 |
| RTAB-Map | 0.292 | 0.158 | 0.198 |
| MASt3R-SLAM | 0.173 | 0.014 | 0.062 |
| GM | 0.080 | 0.064 | 0.069 |
| SM | 0.073 | 0.152 | 0.128 |
| PBU | 0.080 | 0.063 | 0.068 |
| ABM | 0.231 | 0.020 | 0.090 |
| **CROSS** | **0.353** | **0.397** | **0.384** |

**含 self-mapping**:

| Method | RS (OLS) | RS (Rover) | RS (All) |
|--------|----------|------------|----------|
| ORB-SLAM3 | 0.003 | 0.048 | 0.040 |
| RTAB-Map | 0.438 | 0.275 | 0.336 |
| MASt3R-SLAM | 0.226 | 0.024 | 0.094 |
| GM | 0.483 | 0.263 | 0.345 |
| SM | 0.479 | 0.333 | 0.387 |
| PBU | 0.482 | 0.262 | 0.344 |
| ABM | 0.569 | 0.229 | 0.356 |
| **CROSS** | **0.478** | **0.494** | **0.488** |

评估: OpenLORIS 745 trials, Rover 1232 trials, 每 trial 200 帧。$r_D = 2\,\text{m}$ (室内), $5\,\text{m}$ (室外)。

### TABLE II: 真实机器人 Object-Goal Navigation 成功率 (%)

| Method | LC | OR | LC+OR |
|--------|----|----|-------|
| Metric+Semantic (ORB-SLAM3) | 30.0 | 30.0 | 20.0 |
| Metric+Semantic (RTAB-Map) | 40.0 | 60.0 | 30.0 |
| **CROSS** | **70.0** | **80.0** | **80.0** |

每设定 10 次独立试验，成功 = 到达目标 1m 内。

### TABLE III: Topo-Bench 感知歧义评估

| Method | A+P | P.O. | A.O. | BLA |
|--------|-----|------|------|-----|
| RTAB-Map | 0.059 | 0.433 | 1 | 0.301 |
| ORB-SLAM3 | 0.059 | 0.183 | 1 | 0.227 |
| ABM | 0.02 | 0.328 | 1 | 0.201 |
| GM | 0.078 | 0.302 | 0.959 | 0.288 |
| SM | 0.118 | 0.187 | 0.959 | 0.281 |
| PBU | 0.078 | 0.302 | 0.959 | 0.288 |
| **CROSS** | **0.275** | **0.336** | **0.99** | **0.452** |

BLA = Balanced Localization Accuracy（A+P, P.O., A.O. 的几何均值）。

### 运行时间

- ~28 ms/step (RTX 4000 GPU)
- 实时运行 > 30 Hz
- 实际部署中，局部里程计在全局更新间以高频传播状态

### 软件工具链

| 工具 | 用途 |
|------|------|
| BoQ | VPR 模型 |
| XFeat | 特征提取 |
| LightGlue | 特征匹配 |
| E-PnP + RANSAC | 位姿估计 |
| GTSAM | PGO 求解 |

---

## 批判性思考

### 优点
1. 理论扎实：SE(3) 上的 GSF 有完整的概率推理框架，不是启发式拼凑
2. 多假设管理优雅处理了 SLAM 中最难的问题——绑架和回环
3. 测量消息聚合多关键帧证据的设计是核心创新，统一了 tracking/relocalization/loop closure
4. 真实机器人实验在极端条件（LC+OR）下仍达 80%，远超 SLAM 基线
5. ~28ms/step 实时可行

### 局限性
1. 依赖 RGB-D 传感器，纯 RGB 场景不适用
2. 假设管理中 DBSCAN 聚类的参数（eps, min_samples）可能需要场景特定调参
3. 10 次/设定的真实机器人评估统计置信度有限
4. 语义层较薄——仅节点级物体标注，缺少细粒度语义推理
5. 大规模环境中高斯混合分量数可能增长

### 潜在改进方向
1. 集成 VLM 增强语义层，支持自然语言目标描述
2. 在 VLN-CE 上评估拓扑图导航性能
3. 研究与 VLN 方法中 topological graph 的结合（如 [[RAGNav]]）
4. 扩展到纯 RGB 设定（用学习的深度估计）

### 可复现性评估
- [ ] 代码开源（待公开）
- [x] Benchmark 标准化 (OpenLORIS, Rover, Topo-Bench)
- [x] 实验设置完整
- [x] 真实机器人实验

---

## 关联笔记

### 基于
- [[Gaussian-Sum Filter]]: 高斯混合滤波器理论基础
- [[SE(3)]]: Lie 群上的位姿表示
- [[GTSAM]]: 因子图优化框架
- [[PnP]]: Perspective-n-Point 位姿估计

### 对比
- [[ORB-SLAM3]]: 特征点 SLAM
- [[RTAB-Map]]: 实时外观基拓扑建图
- [[MASt3R-SLAM]]: 基于 MASt3R 的 SLAM

### 方法相关
- [[Topological Map]]: 拓扑地图
- [[Visual Place Recognition]]: VPR（BoQ 模型）
- [[Object-Goal Navigation]]: 物体目标导航
- [[RAGNav]]: 拓扑推理 + RAG 用于多目标 VLN

---

## 速查卡片

> [!summary] CROSS
> - **核心**: SE(3) 高斯混合信念 + pose-aware 拓扑图，替代 SLAM 度量地图
> - **关键创新**: 测量消息聚合多关键帧证据 → 统一 tracking/relocalization/loop closure
> - **工具链**: BoQ + XFeat + LightGlue + E-PnP + GTSAM
> - **结果**: 重定位 RS 0.384 (跨序列), 真实机器人 LC+OR 80% (vs RTAB-Map 30%)
> - **速度**: ~28ms/step, >30Hz 实时

---

*笔记创建时间: 2026-05-06*
