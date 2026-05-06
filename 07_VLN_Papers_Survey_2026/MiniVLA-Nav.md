---
title: "MiniVLA-Nav v1: A Multi-Scene Simulation Dataset for Language-Conditioned Robot Navigation"
method_name: "MiniVLA-Nav"
authors: [Ali Al-Bustami, Jaerock Kwon]
year: 2026
venue: arXiv
tags: [vision-language-navigation, dataset, language-conditioned, simulation, isaac-sim, object-navigation]
zotero_collection: ""
image_source: online
arxiv_html: https://arxiv.org/html/2605.00397v1
created: 2026-05-06
---

# 论文笔记：MiniVLA-Nav v1: A Multi-Scene Simulation Dataset for Language-Conditioned Robot Navigation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | University of Michigan-Dearborn, Dept. of ECE |
| 日期 | May 2026 |
| 项目主页 | [HuggingFace Dataset](https://huggingface.co/datasets/alibustami/miniVLA-Nav) |
| 对比基线 | [[R2R]], [[VLN-CE]], [[ObjectNav]], [[ALFRED]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.00397) |

---

## 一句话总结

> 提出 MiniVLA-Nav v1，面向 Language-Conditioned Object Approach (LCOA) 导航任务的多场景仿真数据集：1174 个 episode，4 个 Isaac Sim 场景，同步 RGB/深度/分割 + 连续/离散动作标签 @ 60Hz。

---

## 核心贡献

1. **LCOA 任务定义**: 给定自然语言指令，差速驱动机器人导航至目标物体并停在 1m 以内，填补 language-conditioned 连续控制数据集空缺
2. **多场景高真实感数据**: 4 个 Isaac Sim 环境（Office, Hospital, Full Warehouse, Multi-Shelf Warehouse），12 物体类别
3. **双模态动作标签**: 同时提供连续 $(v, \omega)$ 和 $7 \times 7$ 离散化 token，支持传统控制和 VLA-style token 预测
4. **系统化评估分割**: 三层出生距离 + 5 个评估分割（ID / paraphrase-OOD / object-OOD）
5. **完全可复现**: 单种子 seed=42，记录 Git commit hash 和 Isaac Sim 版本

---

## 问题背景

### 要解决的问题
Language-conditioned navigation 缺乏高质量仿真数据集，尤其是同时支持连续控制和语言指令的数据。

### 现有方法的局限
- [[R2R]]: 离散导航图，不适用于连续控制
- [[VLN-CE]]: 连续环境 + 语言，但仅 1 个场景、无 OOD 分割
- [[ObjectNav]]: 连续动作但无语言指令
- [[ALFRED]]: 离散动作 + 语言，仅 1 个场景

### TABLE I: 与现有数据集对比

| Dataset | Sim? | Action | Lang. | Scenes | OOD splits |
|---------|------|--------|-------|--------|------------|
| R2R | ✗ | Discrete | ✓ | 90 | ✗ |
| VLN-CE | ✓ | Cont. | ✓ | 1 | ✗ |
| ObjectNav | ✓ | Discrete | ✗ | 1 | ✗ |
| ALFRED | ✓ | Discrete | ✓ | 1 | ✗ |
| **MiniVLA-Nav v1** | **✓** | **Cont.** | **✓** | **4** | **✓** |

### 本文的动机
构建一个结合语言指令、连续控制、多场景多样性和系统化 OOD 评估的 VLN 仿真数据集。

---

## 方法详解

### 任务定义: LCOA

给定指令 $\ell$ 和观测 $o_t = (\mathbf{I}_t^{\text{RGB}}, \mathbf{D}_t)$，机器人输出动作 $a_t = (v_t, \omega_t)$，使得：

$$\|p_T - p_g\| \leq r_{\text{success}} = 1.0\,\text{m}$$

其中 $p_T$ 为机器人终端位置，$p_g$ 为目标物体包围盒的 3D 质心。

**终止条件**:
- **成功**: 在 $r_{\text{success}}$ 内连续静止 $\geq 5$ 步
- **碰撞**: 命令前进但连续 $\geq 16$ 步无进展（stall detection）
- **超时**: 达到 $T_{\max} = 1000$ 步

### 动作空间

**连续**: $(v, \omega) \in [0, 1]\,\text{m/s} \times [-1.5, 1.5]\,\text{rad/s}$

**离散**: 每个维度量化为 7 个均匀 bin → $7 \times 7 = 49$ token 的联合动作词表，用于 VLA-style token 预测。

### 仿真环境与机器人

- **仿真器**: NVIDIA Isaac Sim 5.1（PhysX 物理引擎 + RTX 渲染）
- **物理时间步**: $\Delta t = 1/60\,\text{s}$（60 Hz）
- **机器人**: NVIDIA Nova Carter，差速驱动，0.52m 轮距，0.14m 轮径
- **相机**: 前向立体相机（front_hawk/right），640×640 像素
- **传感器输出**: 同步 RGB (uint8) + 深度 (float32, metres) + 实例分割 (uint16)

### 场景与目标物体

#### TABLE II: 场景配置

| 场景 | Seen Categories | Held-out Categories |
|------|-----------------|---------------------|
| Office | chair, sofa, table, monitor, plant, trash_can | fire_ext., whiteboard |
| Hospital | chair, trash_can | fire_ext., whiteboard |
| Full Warehouse | shelf, rack | barrel, crate |
| Multi-Shelf | shelf, rack | barrel, crate |

共 **12 个物体类别**。

### 出生距离分层采样 (Tiered Spawn Sampling)

每个 episode 独立采样三层之一：

- **Near (30%)**: $r \sim \mathcal{U}(1.5, 3.5)$m，面向目标 $\pm 25°$ 朝向噪声
- **Mid (40%)**: $r \sim \mathcal{U}(3.5, 7.0)$m，同上
- **Far (30%)**: 全局有效地板位置预计算采样，均匀随机朝向

出生点有效性验证：3 步预热后检查位移，>0.08m 则重试（最多 8 次）。

### 语言指令生成

**训练模板 (18 个，v1 激活 13 个)**:

| ID | 模板 |
|----|------|
| T1 | "Go to the {object}." |
| T2 | "Drive to the {object} and stop." |
| T3 | "Approach the {object}." |
| T5 | "Navigate to the {object}." |
| T9 | "Head to the {object}." |
| T10 | "Move to the {object} and halt." |
| T11 | "Go to the {object} in front of you." |
| T12 | "Drive toward the {object} until you reach it." |
| T13 | "Get to the {object}." |
| T14 | "Your destination is the {object}." |
| T15 | "Locate the {object} and stop in front of it." |
| T17 | "Move all the way to the {object}." |
| T18 | "Stop next to the {object}." |

**释义 OOD 模板 (12 个，v1 激活 10 个)**:

| ID | 模板 |
|----|------|
| O1 | "Make your way to the {object}." |
| O2 | "Proceed to the {object}." |
| O3 | "Find the {object} and come to a stop." |
| O5 | "Go straight to the {object}." |
| O6 | "Close in on the {object}." |
| O7 | "Reach the {object} and stop there." |
| O8 | "Move until you're beside the {object}." |
| O9 | "Find your way to the {object}." |
| O11 | "Get closer to the {object}." |
| O12 | "Park next to the {object}." |

含 `{color}` 槽的模板在 v1 中因所有目标 color=unknown 被抑制。

### 专家控制器

基于像素级目标可见性的**比例控制器**。

#### 目标可见时 (≥32 pixels):

$$\omega_t = \text{clamp}\!\left(-k_\omega\,\frac{c_x - W/2}{W/2},\,\omega_{\min},\omega_{\max}\right)$$

$$v_t = \text{clamp}\!\left(k_v\,(\hat{d}_t - r_{\text{success}}),\,0, v_{\max}\right)$$

- $c_x$: 目标 mask 质心列坐标
- $W = 640$
- $\hat{d}_t$: 目标 mask 像素的中位深度（metres）
- $k_\omega = 1.4$ rad⁻¹, $k_v = 0.7$ s⁻¹

#### 目标不可见时 (bearing-only):

$$\omega_t = \text{clamp}\!\left(k_\omega\,\Delta\psi_t,\,\omega_{\min},\omega_{\max}\right)$$

$$v_t = \text{clamp}\!\left(k_v\,d_t,\,0, v_{\max}\right)$$

$\Delta\psi_t$ 为朝向误差，$d_t$ 为欧氏距离。

#### 障碍物回避
当中心前景裁剪区（rows 55–95%, cols 25–75%）深度 < 0.25m 时钳位 $v$。

---

## 关键图表

### Figure 1: 离散动作空间

![Action Space](https://arxiv.org/html/2605.00397v1/figures/fig11_action_space.png)

**说明**: 7×7 离散动作空间。每个格点为一个有效 $(v, \omega)$ token。标注了原点（stop）和最大前进 token。

### Figure 3: 样本 RGB 观测

![Sample Images](https://arxiv.org/html/2605.00397v1/figures/fig15_sample_images.png)

**说明**: 四个场景、八个物体类别的前向 RGB 样本观测。

### Figure 4: 出生点地图

![Spawn Maps](https://arxiv.org/html/2605.00397v1/figures/fig13_spawn_maps.png)

**说明**: 每个场景的精选出生点分布（各 40 个点）。

---

## 实验

### 数据集统计

#### TABLE V: 每场景 Episode 数

| 场景 | Episodes | 平均 TL (m) |
|------|----------|-------------|
| Office | 700 | 2.66 |
| Hospital | 52 | 2.73 |
| Full Warehouse | 354 | 2.83 |
| Multi-Shelf | 68 | 3.15 |
| **总计** | **1174** | **2.74** |

#### 分割分布 (60/10/10/10/10)

| 分割 | N | 描述 |
|------|---|------|
| train_id | 716 | Seen 物体 + Seen 模板 |
| val_id | 114 | Seen 物体 + Seen 模板（验证）|
| test_id | 121 | Seen 物体 + Seen 模板（测试）|
| test_paraphrase_ood | 122 | Seen 物体 + 释义模板 |
| test_ood_obj | 101 | Held-out 物体 + Seen 模板 |

#### TABLE VI: 各分割 Rollout 统计 (mean / median)

| 分割 | N | NE (m) | TL mean | TL med. | Steps mean/med. |
|------|---|--------|---------|---------|-----------------|
| train_id | 716 | 0.967 | 2.70 | 2.35 | 195.9 / 173 |
| val_id | 114 | 0.969 | 2.93 | 2.70 | 211.1 / 197 |
| test_id | 121 | 0.967 | 2.85 | 2.34 | 210.9 / 172 |
| test_para_ood | 122 | 0.965 | 2.87 | 2.39 | 223.3 / 195 |
| test_ood_obj | 101 | 0.969 | 2.54 | 2.24 | 186.0 / 170 |
| **Overall** | **1174** | **0.967** | **2.74** | **2.38** | **200.9 / 177** |

NE 均值 0.967m 表明机器人通常停在非常接近 1.0m 成功边界处。

#### TABLE VII: 各出生层统计

| 层 | N | Mean NE (m) | Mean TL (m) | Mean Steps |
|----|---|-------------|-------------|------------|
| Near (1.5–3.5m) | 640 | 0.967 | 1.56 | 128.5 |
| Mid (3.5–7.0m) | 520 | 0.967 | 4.14 | 284.3 |
| Far (全局) | 14 | 0.970 | 4.46 | 414.6 |

Far 层 episode 的 TL 是 Near 层的 **2.9 倍**，步数是 **3.2 倍**。

### 关键相关性

出生距离与轨迹长度的 Pearson $r = 0.94$（95% CI: [0.93, 0.95], $p < 0.001$, $N = 1174$, slope $\approx 1.00$ m/m）。

### Episode 数据格式

| 文件 | Shape/格式 | 描述 |
|------|-----------|------|
| rgb_front/{t}.png | 640×640×3, uint8 | 前向 RGB |
| depth_front/{t}.npy | 640×640, float32 | 深度 (m) |
| seg_front/{t}.png | 640×640, uint16 | 实例分割 |
| actions_continuous.npy | T×2, float32 | $(v_t, \omega_t)$ |
| actions_tokens.npy | T×2, int16 | 离散化 token |
| poses.npy | T×7, float32 | $(x,y,z,q_w,q_x,q_y,q_z)$ |
| meta.json | JSON | Episode 元数据 |

### 建议评估指标

- **SR**: 在 1.0m 内停止的比例
- **NE**: 终端位置到目标的平均欧氏距离
- **OP (Oracle Progress)**: 出生距离中已覆盖的比例
- **Collision Rate**: 由 stall detection 终止的比例

---

## 批判性思考

### 优点
1. 填补了 language-conditioned + 连续控制 + 多场景的数据集空缺
2. 双模态动作标签（连续 + 离散 token）支持传统控制和 VLA 方法
3. 系统化的 OOD 评估分割（模板释义 + 物体泛化）
4. 完全可复现（单种子 seed=42 + 版本记录）
5. 数据集公开（HuggingFace）

### 局限性
1. **Far 层严重不足**: 仅 14/1174 (1.2%)，长距导航评估受限
2. **物体类别不平衡**: Monitor (28.9%) + rack (22.0%) + crate (11.2%) = 62.1%
3. **规模较小**: 1174 episodes，相比 R2R (21K+) 数量级差距
4. **仅仿真**: sim-to-real gap 未被验证
5. **语言模板化**: 缺少自然语言多样性，color slot 全部为 unknown
6. **静态场景**: 无动态障碍物
7. **专家控制器有偏**: 比例控制器在狭窄走廊成功率低（40-60%），选择偏差偏向无遮挡路径

### 潜在改进方向
1. 扩展 Far 层采样，增加长距导航数据
2. 引入 LLM 生成更多样化的自然语言指令
3. 添加动态障碍物和多 agent 场景
4. 提供基线模型训练结果

### 可复现性评估
- [x] 数据集公开 (HuggingFace)
- [x] 数据格式文档完整
- [x] 场景配置 + 种子可复现
- [ ] 基线代码

---

## 关联笔记

### 对比
- [[R2R]]: 离散导航图 VLN 数据集
- [[VLN-CE]]: 连续环境 VLN 数据集
- [[ALFRED]]: 交互式具身任务数据集
- [[ObjectNav]]: 物体目标导航任务

### 方法相关
- [[Vision-Language Navigation]]: VLN 任务范式
- [[Isaac Sim]]: NVIDIA 仿真平台
- [[OpenVLA]]: VLA 模型

---

## 速查卡片

> [!summary] MiniVLA-Nav
> - **核心**: LCOA 导航仿真数据集，连续 + 离散动作 @ 60Hz
> - **规模**: 1174 episodes, 4 场景, 12 物体类别, 23 语言模板
> - **数据**: RGB (640×640) + Depth + Segmentation + $(v, \omega)$ + 7×7 tokens
> - **评估**: 5 分割（ID / paraphrase-OOD / object-OOD），Pearson $r=0.94$
> - **数据集**: [HuggingFace](https://huggingface.co/datasets/alibustami/miniVLA-Nav)

---

*笔记创建时间: 2026-05-06*
