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
| 机构 | University of Michigan-Dearborn |
| 日期 | May 2026 |
| 项目主页 | [HuggingFace Dataset](https://huggingface.co/datasets/alibustami/miniVLA-Nav) |
| 对比基线 | [[ObjectNav]], [[ALFRED]], [[RoboVQA]], [[OpenVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.00397) |

---

## 一句话总结

> 提出 MiniVLA-Nav v1 数据集，面向 Language-Conditioned Object Approach (LCOA) 导航任务，在 Isaac Sim 四个高真实感室内场景中提供 1174 个 episode 的多模态导航数据。

---

## 核心贡献

1. **LCOA 任务定义**: 给定自然语言指令，差速驱动机器人导航至目标物体并停在 1m 以内
2. **多场景高真实感数据集**: 4 个 Isaac Sim 环境（Office, Hospital, Full Warehouse, Multi-Shelf Warehouse），1174 个 episode
3. **丰富的多模态数据**: 同步 640x640 RGB、度量深度图（float32）、实例分割 mask，连续 $(v, \omega)$ 和 7x7 离散动作标签（60Hz）
4. **系统化评估设计**: 三层出生距离（near/mid/far）、12 物体类别、18 训练模板 + 12 OOD 释义模板、5 个评估分割

---

## 问题背景

### 要解决的问题
Language-conditioned navigation 缺乏高质量的仿真数据集，现有数据集要么场景单一，要么缺少连续控制标签。

### 现有方法的局限
- 现有 VLN 数据集（R2R, RxR 等）主要面向离散导航图，不适用于连续控制
- 机器人导航数据集缺少语言指令条件
- 场景多样性不足，难以评估泛化能力

### 本文的动机
构建一个结合语言指令、连续控制和多场景多样性的 VLN 仿真数据集，支持 VLA 模型的训练和评估。

---

## 方法详解

### 数据集架构

- **仿真平台**: NVIDIA Isaac Sim
- **机器人**: NVIDIA Nova Carter 差速驱动
- **场景**: Office, Hospital, Full Warehouse, Warehouse with Multiple Shelves
- **传感器**: RGB (640x640), 深度图 (float32, metres), 实例分割 mask

### 核心设计

#### 出生距离分层采样 (Tiered Spawn Sampling)

三层采样策略确保轨迹多样性：
- **Near**: 1.5–3.5m（短距接近）
- **Mid**: 3.5–7.0m（中距导航）
- **Far**: 全局精选点（长距规划）
- 出生距离与轨迹长度 Pearson $r = 0.94$

#### 动作空间

$7 \times 7$ 离散化网格，每个格点对应一组有效的 $(v, \omega)$，支持连续和离散两种控制模式。

#### 语言指令生成

- 18 个训练模板 + 12 个释义 OOD 模板
- 12 个物体类别
- 5 个评估分割：in-distribution accuracy, template-paraphrase robustness, OOD object-category benchmarking

---

## 关键图表

### Figure 1: 离散动作空间

![Action Space](https://arxiv.org/html/2605.00397v1/figures/fig11_action_space.png)

**说明**: 7x7 离散动作空间。每个格点是一个有效的 $(v, \omega)$ token。标注了原点（stop）和最大前进 token。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| MiniVLA-Nav v1 | 1174 episodes | 4场景, 12物体, 多模态 | 训练+评估 |

### 统计

- **Episode 分布**: Train-ID 占 60%，4 个等权重评估分割覆盖 ID 和 OOD 条件
- **数据格式**: 每 episode 包含同步 RGB/Depth/Segmentation + 连续/离散动作标签
- **采集频率**: 60 Hz

---

## 批判性思考

### 优点
1. 完整的语言条件化导航数据集，填补了 VLA-Nav 训练数据的空缺
2. 多模态数据（RGB + depth + segmentation）支持多种方法
3. 系统化的评估分割设计，支持泛化性测试
4. 数据集公开可用（HuggingFace）

### 局限性
1. 仅 1174 个 episode，规模较小
2. 4 个场景多样性有限（均为室内结构化环境）
3. 仅仿真数据，sim-to-real gap 未被验证
4. 语言指令为模板生成，缺少自然语言多样性

### 潜在改进方向
1. 扩展到更多场景和更自然的语言指令
2. 添加 sim-to-real 迁移的验证实验
3. 提供基线模型的训练结果

### 可复现性评估
- [x] 数据集公开
- [ ] 基线代码
- [x] 数据格式文档完整
- [x] 场景配置可获取

---

## 关联笔记

### 对比
- [[ALFRED]]: 交互式具身任务数据集
- [[ObjectNav]]: 物体目标导航任务
- [[RoboVQA]]: 机器人视觉问答数据集

### 方法相关
- [[Vision-Language Navigation]]: VLN 任务范式
- [[Isaac Sim]]: NVIDIA 仿真平台
- [[OpenVLA]]: VLA 模型

---

## 速查卡片

> [!summary] MiniVLA-Nav
> - **核心**: Language-Conditioned Object Approach 导航仿真数据集
> - **规模**: 1174 episodes, 4 场景, 12 物体类别
> - **数据**: RGB + Depth + Segmentation + 连续/离散动作 @ 60Hz
> - **数据集**: [HuggingFace](https://huggingface.co/datasets/alibustami/miniVLA-Nav)

---

*笔记创建时间: 2026-05-06*
