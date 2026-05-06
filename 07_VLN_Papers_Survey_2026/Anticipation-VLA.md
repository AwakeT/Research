---
title: "Anticipation-VLA: Solving Long-Horizon Embodied Tasks via Anticipation-based Subgoal Generation"
method_name: "Anticipation-VLA"
authors: [Zhilong Zhang, Wenyu Luo, Haonan Wang, Yifei Sheng, Yidi Wang, Hanyuan Guo, Haoxiang Ren, Xinghao Du, Yuhan Che, Tongtong Cao, Lei Yuan, Yang Yu]
year: 2026
venue: arXiv
tags: [vision-language-action, long-horizon, subgoal-generation, hierarchical-policy, embodied-intelligence]
zotero_collection: ""
image_source: online
arxiv_html: https://arxiv.org/html/2605.01772v1
created: 2026-05-06
---

# 论文笔记：Anticipation-VLA: Solving Long-Horizon Embodied Tasks via Anticipation-based Subgoal Generation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | (未标注) |
| 日期 | May 2026 |
| 项目主页 | N/A |
| 对比基线 | [[DreamVLA]], [[UniVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.01772) |

---

## 一句话总结

> 提出 Anticipation Model，自适应递归生成未来子目标，结合目标条件化 VLA 策略形成层次化 VLA，解决长程具身任务中的 compounding error 问题。

---

## 核心贡献

1. **Anticipation Model**: 自适应递归子目标生成模型，根据任务展开过程中的动态变化持续调整未来子目标
2. **Anticipation-VLA**: 层次化 VLA 架构——高层 Anticipation Model 生成子目标 + 低层目标条件化 VLA 执行动作
3. **UMM 实现**: 用统一多模态模型（Unified Multimodal Model）实现高层子目标生成，训练高效
4. **仿真+真实验证**: 在 LIBERO、VLABench 仿真和真实机器人任务上均有效

---

## 问题背景

### 要解决的问题
VLA 模型在长程任务中因 compounding error 导致性能急剧下降。

### 现有方法的局限
- 固定粒度任务分解：将长程任务分解为固定大小的子任务，无法适应执行状态的变化复杂度
- 缺乏动态调整能力：不同阶段（简单直行 vs 复杂交互）需要不同粒度的规划

### 本文的动机
通过"预期"（anticipation）机制动态生成子目标——在简单阶段用粗粒度，在复杂阶段用细粒度，实现自适应的层次化规划。

---

## 方法详解

### 模型架构

**层次化 Anticipation-VLA**:
- **高层**: Anticipation Model（基于 UMM）— 生成视觉子目标图像
- **高层辅助**: Value Model — 评估子目标质量
- **低层**: Goal-conditioned VLA — 根据当前观测和子目标执行动作

### 核心模块

#### 模块1: Goal-Conditioned MDP (GMDP)

将任务建模为目标条件化 MDP，Agent 在每步接收当前观测和目标子目标，输出动作。

#### 模块2: Anticipation Model

**设计动机**: 动态适应执行状态的变化复杂度

**关键特性**:
- **自适应**: 根据当前执行状态调整子目标粒度
- **递归**: 连续生成未来子目标，形成规划路径
- **视觉子目标**: 生成目标图像而非文本描述

**实现**: 基于 UMM 微调，输入当前观测 + 语言指令，输出未来子目标图像

#### 模块3: Goal-conditioned VLA

低层执行策略：
- 输入：当前观测 + Anticipation Model 生成的子目标
- 输出：具体动作
- 通过子目标 conditioning 降低长程任务的 compounding error

---

## 关键图表

### Figure 1: 系统概览

![Overview](https://arxiv.org/html/2605.01772v1/x1.png)

**说明**: Anticipation-VLA 架构。高层 Anticipation Model 根据当前状态和指令生成未来子目标图像，低层 Goal-conditioned VLA 根据子目标执行动作。

### Figure 7: 子目标生成可视化

**说明**: 生成的子目标序列随任务展开自适应调整粒度。

---

## 实验

### 仿真实验

| Benchmark | 任务 | 结果 |
|-----------|------|------|
| LIBERO | 长程操作 | 超越基线 |
| VLABench | 多任务操作 | 超越基线 |

### 真实机器人实验

- 多个真实世界机器人任务验证
- 消融实验验证各组件贡献
- 泛化评估验证跨场景能力

### 消融实验

- Anticipation Model 的自适应粒度 vs 固定粒度
- Value Model 的作用
- 递归生成 vs 单步生成

---

## 批判性思考

### 优点
1. 自适应粒度的子目标生成是解决长程任务的优雅方案
2. 层次化架构清晰：规划与执行分离，各司其职
3. 有真实机器人验证，不仅限于仿真
4. UMM 作为子目标生成器的实现路径轻量高效

### 局限性
1. 实验主要集中在操作任务，未直接在导航任务上验证
2. 视觉子目标生成依赖 UMM 的图像生成质量
3. 递归子目标生成的计算开销在长程任务中可能累积
4. 缺少与 VLN 中 waypoint prediction 方法的直接对比

### 潜在改进方向
1. 将 Anticipation Model 应用于 VLN 场景的 waypoint prediction
2. 在 R2R / VLN-CE 上评估
3. 研究文本子目标 vs 视觉子目标的效果差异

### 可复现性评估
- [ ] 代码开源
- [ ] 预训练模型
- [x] 实验设置完整
- [x] Benchmark 标准化

---

## 关联笔记

### 基于
- [[Vision-Language-Action]]: VLA 范式
- [[Goal-Conditioned RL]]: 目标条件化强化学习
- [[Hierarchical Planning]]: 层次化规划

### 对比
- [[DreamVLA]]: VLA + 世界模型
- [[UniVLA]]: 统一 VLA 模型
- [[EmergeNav]]: VLN 中的子目标分解（Plan-Solve-Transition）

### 方法相关
- [[Subgoal Generation]]: 子目标生成
- [[Waypoint Prediction]]: VLN 路点预测
- [[Compounding Error]]: 长程任务误差累积

---

## 速查卡片

> [!summary] Anticipation-VLA
> - **核心**: 自适应递归子目标生成 + 层次化 VLA
> - **方法**: UMM-based Anticipation Model（高层）+ Goal-conditioned VLA（低层）
> - **结果**: LIBERO + VLABench 仿真和真实机器人均有效
> - **代码**: 未开源

---

*笔记创建时间: 2026-05-06*
