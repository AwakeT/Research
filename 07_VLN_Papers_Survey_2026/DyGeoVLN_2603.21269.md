---
title: "DyGeoVLN: Infusing Dynamic Geometry Foundation Model into Vision-Language Navigation"
method_name: "DyGeoVLN"
authors: [Xiangchen Liu, Hanghan Zheng, Jeil Jeong, Minsung Yoon, Lin Zhao, Zhide Zhong, Haoang Li, Sung-Eui Yoon]
year: 2026
venue: arXiv
tags: [vision-language-navigation, 3d-geometry, dynamic-scene, token-pruning, depth-estimation, real-robot]
zotero_collection: ""
image_source: mixed
arxiv_html: https://arxiv.org/html/2603.21269v1
created: 2026-04-27
---

# 论文笔记：DyGeoVLN: Infusing Dynamic Geometry Foundation Model into Vision-Language Navigation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | KAIST, HKUST(GZ), JD Explore Academy |
| 日期 | March 2026 |
| 项目主页 | N/A |
| 对比基线 | [[StreamVLN]], [[NaVILA]], [[NaVid]], [[Uni-NaVid]], [[g3D-LF]] |
| 链接 | [arXiv](https://arxiv.org/abs/2603.21269) |

---

## 一句话总结

> 提出动态几何基础模型(DGFM)处理动态场景3D空间推理，通过跨分支特征融合和自适应分辨率token剪枝实现VLN的SOTA。

---

## 核心贡献

1. **DyGeoVLN框架**: 跨分支特征融合架构，将动态几何基础模型注入VLN实现几何感知空间推理
2. **动态几何基础模型 (DGFM)**: 结合 [[Depth Anything|单目深度估计]] 和点图Transformer，通过零均值卷积渐进注入深度引导几何线索
3. **DyHM3D数据集**: ~50,000条动态人体运动轨迹数据，基于 [[HM3D]] 构建
4. **自适应分辨率token剪枝**: 无位姿的占据感知空间token剪枝，利用DGFM信息消除时空冗余

---

## 问题背景

### 要解决的问题
当前VLN系统"主要建立在2D图像-文本预训练上"，"严重缺乏长程轨迹上全局一致的3D空间推理"。更关键的是，在动态场景（如有行人移动）中，几何理解更具挑战性。

### 现有方法的局限
- 使用 [[VGGT]] 等现成模型"在动态环境下性能大幅下降"
- 帧下采样方法"丢弃细粒度时间线索"
- 基于体素的token剪枝"依赖仿真器提供的真值位姿和深度图"，限制真实部署
- 现有3D表示方法"未从根本上优化时空动态的几何表示"

### 本文的动机
开发专门针对动态场景的几何基础模型，并设计无位姿依赖的token压缩策略，使系统仅需单目相机即可部署。

---

## 方法详解

### 模型架构

DyGeoVLN 采用**跨分支2D-3D融合**架构：
- **输入**: 语言指令 $\mathcal{W}$ + 自中心RGB流 $\mathcal{O}_t = \{x_0, \ldots, x_t\}$
- **2D语义分支**: [[Qwen2-VL]] 视觉编码器
- **3D几何分支**: DGFM（冻结）
- **融合**: [[Cross-Attention]]（2D为Query, 3D为KV）
- **LLM骨干**: Qwen2-VL
- **输出**: $a_{t+1} \in \{\text{Move Forward}, \text{Turn Left}, \text{Turn Right}, \text{Stop}\}$

### 核心模块

#### 模块1: 动态几何基础模型 (DGFM)

**设计动机**: 在动态场景中提供鲁棒的3D空间表示

**深度引导局部点图**:
- [[Depth Anything]] 预测密集深度 $D_t$
- 通过相机内参反投影到3D: $p_t(u) = D_t(u) K^{-1} \tilde{u}$

**点图Transformer**:
- 点图分块嵌入后经自注意力处理
- 提取多粒度3D上下文 $G_t^{(1)}, \ldots, G_t^{(S)}$

**动态感知融合**:
- ViT提取2D视觉token $E_t^{(s)}$
- 零均值卷积融合: $\tilde{E}_t^{(s)} = g_{zm}(G_t^{(s)}) + E_t^{(s)}$
- 权重和偏置初始化为零，"逐步学习将深度引导几何线索作为残差注入"

**解码器与动态3D重建**:
- 相机/局部/全局解码器层次化解码
- 基于 [[pi3|pi-cubed]] 初始化

#### 模块2: DyHM3D 数据集

**设计动机**: 现有数据集缺乏动态人体运动场景

**构建方法**:
- 基于 [[HM3D]] 场景
- 骨骼驱动的3D人体模型沿采样路径生成步行动作
- 相机通过线性插值跟随
- ~50,000条轨迹，包含RGB、深度、位姿和内参

#### 模块3: 跨分支特征融合VLN架构

**2D语义token化**:
- Qwen-VL视觉编码器产生 $V_t \in \mathbb{R}^{L_v \times C_v}$
- 多模态投影器映射到 $X_t$

**3D空间几何token化**:
- DGFM产生全局3D特征 $H_t^{glo}$
- 特征对齐投影到 $\tilde{H}_t^{glo}$

**跨分支融合**:
- [[Cross-Attention]]：2D为Query, 3D为Key/Value
- 滑动窗口KV缓存支持长程推理

#### 模块4: 自适应分辨率占据感知token剪枝

**设计动机**: 无位姿依赖地消除时空冗余token

**自适应分辨率体素分组**:
- 深度依赖的尺度因子调整体素大小——近处小、远处大

**占据感知空间token剪枝**:
- 最新规则（Latest rule）: 保留最新token
- 优先级规则（Priority rule）: 时间近+空间近
- 多token规则: 每体素Top-K

**重要性感知token补全**:
- 最小保留比例 $\rho$
- 重要性分数 = 特征幅度 + 范围 + 空间分布 + 时间新近度 的加权和

**时间平滑**:
- 短时间窗口内对二值剪枝掩码进行多数投票

---

## 关键公式

### 深度引导3D反投影

$$
p_t(u) = D_t(u) K^{-1} \tilde{u}
$$

**含义**: 将像素坐标通过深度和相机内参反投影到3D空间

**符号说明**:
- $D_t(u)$: 像素 $u=(u,v)$ 处的深度值
- $K^{-1}$: 相机内参矩阵的逆
- $\tilde{u} = (u,v,1)^\top$: 齐次像素坐标

### 公式1: [[Transformer|点图Transformer]]

$$
G_t^{(1)}, G_t^{(2)}, \ldots, G_t^{(S)} = f_{transformer}(Z_t^{(0)})
$$

**含义**: 自注意力处理点图嵌入，提取多粒度3D上下文

**符号说明**:
- $Z_t^{(0)} = \phi_{patch}(P_t)$: 点图的分块嵌入
- $G_t^{(s)}$: 第 $s$ 粒度级别的3D上下文

### 公式2: 零均值卷积动态融合

$$
\tilde{E}_t^{(s)} = g_{zm}(G_t^{(s)}) + E_t^{(s)}, \quad s = 1, \ldots, S
$$

**含义**: 通过零初始化卷积渐进注入深度引导的几何线索作为残差

**符号说明**:
- $g_{zm}(\cdot)$: 权重和偏置初始化为零的卷积层
- $E_t^{(s)}$: ViT编码器第 $s$ 层的2D视觉特征
- $\tilde{E}_t^{(s)}$: 融合后的特征

### 全局3D特征解码

$$
\{H_t^{glo}\} = \mathcal{D}_{glo}(\{U_t\})
$$

**含义**: 全局解码器将聚合特征解码为全局3D特征

**符号说明**:
- $U_t = \mathcal{A}(H_t^{cam}, H_t^{loc})$: 相机嵌入和局部点嵌入的聚合
- $H_t^{cam} = \mathcal{D}_{cam}(\{H_t\})$: 相机嵌入
- $H_t^{loc} = \mathcal{D}_{loc}(\{H_t\})$: 局部点嵌入

### 公式4: [[Cross-Attention|跨分支2D-3D融合]]

$$
F_t = \text{Cross-Atten}(Q = X_t,\; K = \tilde{H}_t^{glo},\; V = \tilde{H}_t^{glo})
$$

**含义**: 以2D语义token为Query，3D空间几何token为Key和Value的交叉注意力

### 公式5: 滑动窗口KV缓存推理

$$
a_i = \text{Decoder}(\mathcal{I}, \{M_k\}, \{F_t\}_{t \in \Omega_i};\; \text{KV}_i^{window})
$$

**含义**: 在滑动窗口KV缓存下解码动作

**符号说明**:
- $\mathcal{I}$: 指令token
- $M_k$: 空间剪枝后的记忆token
- $\Omega_i$: 当前窗口帧索引

---

## 关键图表

### Figure 1: System Overview / 系统总览

![Figure 1](https://arxiv.org/html/2603.21269v1/x1.png)

**说明**: DyGeoVLN系统总览。给定指令和序列图像，视觉编码器和DGFM分别产生token，通过跨分支特征融合后，LLM整合融合token和指令token，预测机器人动作。

### Figure 2: DGFM Architecture / DGFM架构

![Figure 2](https://arxiv.org/html/2603.21269v1/x2.png)

**说明**: DGFM架构。(a) 3D潜在表示和2D视觉特征的融合与时间对齐。(b) 定性重建对比：DyGeoVLN vs GT, pi-cubed, VGGT。红框突出基线方法在人体重建上的不足。

### Figure 3: DyHM3D Dataset / DyHM3D数据集示例

![Figure 3](https://arxiv.org/html/2603.21269v1/x3.png)

**说明**: DyHM3D数据集示例。展示了室内场景中模拟人体运动的动态轨迹。

### Figure 4: HA-VLN Qualitative / 动态VLN定性对比

![Figure 4](https://arxiv.org/html/2603.21269v1/x4.png)

**说明**: StreamVLN vs DyGeoVLN在动态HA-VLN上的定性对比。StreamVLN在动态人体上下文中失败；DyGeoVLN成功到达目标。

### Figure 5: R2R-CE Qualitative / 静态VLN定性对比

![Figure 5](https://arxiv.org/html/2603.21269v1/x5.png)

**说明**: R2R-CE上的定性对比。DyGeoVLN展示更一致的轨迹；StreamVLN中途停止。

### Figure 6: Real-World SR / 真实世界成功率

![Figure 6](https://arxiv.org/html/2603.21269v1/x6.png)

**说明**: 各真实场景的成功率(%)。DyGeoVLN在所有场景中一致超越基线。

### Figure 7: Real-World Qualitative / 真实世界定性结果

![Figure 7](https://arxiv.org/html/2603.21269v1/x7.png)

**说明**: 真实世界定性结果。机器人按指令导航，避开移动行人。

### Figure 8: DyHM3D Extended / DyHM3D扩展示例

![[DyGeoVLN_fig8.png|600]]

**说明**: DyHM3D数据集扩展示例——多种室内场景中的模拟人体运动。

### Figure 9: Reconstruction Comparison / 重建对比扩展

![Figure 9](https://arxiv.org/html/2603.21269v1/x9.png)

**说明**: 动态场景重建的扩展定性对比（vs pi-cubed, VGGT）。

### Figure 10: HA-VLN Cases 1-2

![Figure 10](https://arxiv.org/html/2603.21269v1/x10.png)

**说明**: 动态HA-VLN定性结果（案例1）。

![Figure 10b](https://arxiv.org/html/2603.21269v1/x11.png)

**说明**: 动态HA-VLN定性结果（案例2）。

### Figure 11: HA-VLN Cases 3-5

![Figure 11a](https://arxiv.org/html/2603.21269v1/x12.png)

**说明**: 动态HA-VLN定性结果（案例3）。

![Figure 11b](https://arxiv.org/html/2603.21269v1/x13.png)

**说明**: 动态HA-VLN定性结果（案例4）。

![Figure 11c](https://arxiv.org/html/2603.21269v1/x14.png)

**说明**: 动态HA-VLN定性结果（案例5）。

### Figure 12: R2R-CE Extended Results

![Figure 12a](https://arxiv.org/html/2603.21269v1/x15.png)

**说明**: R2R-CE扩展定性结果（案例1）。

![Figure 12b](https://arxiv.org/html/2603.21269v1/x16.png)

**说明**: R2R-CE扩展定性结果（案例2）。

![Figure 12c](https://arxiv.org/html/2603.21269v1/x17.png)

**说明**: R2R-CE扩展定性结果（案例3）。

![Figure 12d](https://arxiv.org/html/2603.21269v1/x18.png)

**说明**: R2R-CE扩展定性结果（案例4）。

### Figure 13: Hardware Setup / 硬件配置

![Figure 13](https://arxiv.org/html/2603.21269v1/x19.png)

**说明**: 真实世界实验设置。(a) 四足机器人硬件配置。(b) 部署管道。

### Figure 14: Real-World Extended / 真实世界扩展结果

![Figure 14](https://arxiv.org/html/2603.21269v1/x20.png)

**说明**: 动态和静态场景的真实世界定性结果扩展。

### Table 1: 动态HA-VLN基准

| Methods | Val Seen NE↓ | Val Seen TCR↓ | Val Seen CR↓ | Val Seen SR↑ | Val Unseen NE↓ | Val Unseen TCR↓ | Val Unseen CR↓ | Val Unseen SR↑ |
|---------|-------------|---------------|-------------|-------------|----------------|----------------|----------------|----------------|
| VLN-CMA | 7.63 | 63.09 | 0.75 | 0.04 | 7.34 | 47.06 | 0.77 | 0.07 |
| HA-VLN-VL | 5.02 | 4.44 | 0.52 | 0.20 | 5.25 | 6.63 | 0.59 | 0.14 |
| BEVBert* | 5.53 | 3.64 | 0.46 | 0.27 | 5.51 | 4.71 | 0.55 | 0.21 |
| ETPNav* | 5.17 | 4.07 | 0.43 | 0.24 | 5.43 | 6.94 | 0.58 | 0.17 |
| g3D-LF* | 5.12 | 3.58 | 0.41 | 0.32 | 5.30 | 4.54 | 0.49 | 0.27 |
| NaVid | 6.62 | 6.26 | 0.51 | 0.36 | 7.49 | 6.17 | 0.49 | 0.34 |
| Uni-NaVid | 6.27 | 5.96 | 0.50 | 0.37 | 7.74 | 6.45 | 0.55 | 0.32 |
| NaVILA | 5.95 | 3.86 | 0.42 | 0.33 | 6.39 | 4.42 | 0.45 | 0.32 |
| StreamVLN | 5.52 | 3.72 | 0.41 | 0.34 | 5.59 | 4.03 | 0.42 | 0.33 |
| **DyGeoVLN** | **4.78** | **3.11** | **0.31** | **0.44** | **5.12** | **3.69** | **0.38** | **0.40** |

**说明**: DyGeoVLN在动态场景中大幅领先。Val-Seen SR 0.44 vs g3D-LF 0.32 (+12%)。Val-Unseen SR 0.40 vs StreamVLN 0.33 (+7%)。

### Table 2: R2R-CE静态基准 (Val-Unseen)

| Methods | Simple Data | NE↓ | OSR↑ | SR↑ | SPL↑ |
|---------|------------|-----|------|-----|------|
| GridMM* | No | 5.11 | 61.0 | 49.0 | 41.0 |
| ETPNav* | No | 4.71 | 65.0 | 57.0 | 49.0 |
| g3D-LF* | No | 4.53 | 68.0 | 61.0 | 52.0 |
| NavMorph | Yes | 5.75 | 56.9 | 47.9 | 33.2 |
| NaVid | Yes | 5.47 | 49.1 | 37.4 | 35.9 |
| Uni-NaVid | Yes | 5.58 | 53.3 | 47.0 | 42.7 |
| NaVILA | Yes | 5.22 | 62.5 | 54.0 | 49.0 |
| StreamVLN | Yes | 5.10 | 64.0 | 55.7 | 50.9 |
| NavFoM | Yes | 5.01 | 64.9 | 56.2 | 51.2 |
| **DyGeoVLN** | Yes | **4.41** | **70.1** | **60.8** | **55.8** |

**说明**: DyGeoVLN在单目RGB方法中达到SOTA，SR 60.8%超越StreamVLN 55.7%，甚至超越使用全景RGB-D+里程计的方法。

### Table 3: 消融实验 (HA-VLN Val-Unseen)

| Methods | NE↓ | TCR↓ | CR↓ | SR↑ |
|---------|-----|------|-----|-----|
| **DyGeoVLN (Full)** | **5.12** | **3.69** | **0.38** | **0.40** |
| w/o Visual Semantic | 6.82 | 4.96 | 0.51 | 0.30 |
| w/o Spatial Geometry | 5.54 | 4.03 | 0.43 | 0.34 |
| w/o Dynamic Spatial Injection | 5.37 | 4.10 | 0.43 | 0.36 |
| w/o Spatial Token Pruning | 5.33 | 3.94 | 0.39 | 0.37 |

**关键发现**: 移除视觉语义分支影响最大(SR降10%)，移除空间几何分支次之(SR降6%)，动态空间注入和token剪枝各贡献约3-4%。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| HA-VLN | - | 动态人体活动场景 | 动态基准评估 |
| R2R-CE | 标准 | 静态连续环境 | 静态基准评估 |
| DyHM3D | ~50K轨迹 | 动态人体运动(自建) | DGFM训练 |
| RxR-CE | - | 长指令 | VLN训练 |
| EnvDrop | - | 数据增强 | VLN训练 |
| ScaleVLN | - | 大规模增强 | VLN训练 |

### 实现细节

- **DGFM**: ViT编码器, 基于 [[pi3|pi-cubed]] 初始化, 在DyHM3D上训练后冻结
- **VLN模型**: [[Qwen2-VL]] 视觉编码器+LLM
- **训练数据**: R2R-CE + RxR-CE + EnvDrop + HA-VLN + ScaleVLN + DAgger
- **真实机器人**: [[Unitree Go1]] 四足 + Intel RealSense D435i + NVIDIA Jetson Orin Nano
- **低层控制**: [[Diffusion Policy|扩散动作模块]] + [[MPC]]

### 可视化结果

在真实世界中使用Unitree Go1四足机器人部署，20 episodes/模型/场景。DyGeoVLN在所有真实场景中一致优于基线，尤其在有移动行人的动态场景中优势明显。

---

## 批判性思考

### 优点
1. 动态几何基础模型是针对VLN动态场景的专门设计，不是简单套用VGGT
2. 零均值卷积的渐进注入策略训练友好，避免初始化时的几何噪声干扰
3. 无位姿token剪枝策略使得仅需单目相机即可部署——实用性强
4. 自建DyHM3D数据集填补了动态VLN训练数据空白

### 局限性
1. DGFM需在DyHM3D上预训练然后冻结，泛化到室外或工业场景未验证
2. 真实机器人实验规模较小（20 episodes/场景）
3. 依赖Depth Anything的深度估计质量——在极端光照或无纹理区域可能退化

### 潜在改进方向
1. DGFM端到端微调（当前冻结可能限制适应性）
2. 扩展到更大规模的真实世界评估
3. 探索不依赖深度估计的替代3D表示

### 可复现性评估
- [ ] 代码开源（未提及）
- [ ] 预训练模型
- [x] 训练细节完整
- [ ] DyHM3D数据集可获取（未明确）

---

## 关联笔记

### 基于
- [[Qwen2-VL]]: 视觉语言模型骨干
- [[pi3]]: DGFM初始化
- [[Depth Anything]]: 单目深度估计
- [[HM3D]]: DyHM3D基础环境

### 对比
- [[StreamVLN]]: 流式VLA基线
- [[NaVILA]]: VLN方法
- [[g3D-LF]]: 3D引导方法
- [[VGGT]]: 静态几何基础模型

### 方法相关
- [[Cross-Attention]]: 2D-3D特征融合
- [[Token Pruning]]: 自适应token压缩
- [[Point Cloud]]: 3D点图表示
- [[Zero Initialization]]: 零均值卷积策略

### 硬件/数据相关
- [[Unitree Go1]]: 四足机器人平台
- [[Intel RealSense]]: 深度相机
- [[Jetson Orin Nano]]: 边缘计算平台

---

## 速查卡片

> [!summary] DyGeoVLN
> - **核心**: 动态几何基础模型(DGFM) + 跨分支2D-3D融合处理动态场景VLN
> - **方法**: Depth Anything深度→点图Transformer→零均值卷积融合 + 无位姿token剪枝
> - **结果**: HA-VLN SR 0.40 (+7%), R2R-CE SR 60.8% (+5.1%), 真实机器人四足部署
> - **代码**: 未开源

---

*笔记创建时间: 2026-04-27*
