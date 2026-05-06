---
title: "What You Think is What You See: Driving Exploration in VLM Agents via Visual-Linguistic Curiosity"
method_name: "GLANCE"
authors: [Haoxi Li, Qinglin Hou, Jianfei Ma, Jinxiang Lai, Tao Han, Sikai Bai, Jingcai Guo, Jie Zhang, Song Guo]
year: 2026
venue: arXiv
tags: [vlm-agent, curiosity-driven-exploration, world-model, reinforcement-learning, visual-linguistic-alignment]
zotero_collection: ""
image_source: online
arxiv_html: https://arxiv.org/html/2605.03782v1
created: 2026-05-06
---

# 论文笔记：What You Think is What You See: Driving Exploration in VLM Agents via Visual-Linguistic Curiosity

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | (未标注) |
| 日期 | May 2026 |
| 项目主页 | N/A |
| 对比基线 | 基于 CoT 推理的 VLM agent |
| 链接 | [arXiv](https://arxiv.org/abs/2605.03782) |

---

## 一句话总结

> 提出 GLANCE 框架，将 VLM agent 的语言世界模型预测与视觉现实之间的差异作为内在好奇心信号，驱动 RL 中的主动探索，解决稀疏奖励下的导航任务。

---

## 核心贡献

1. **Visual-Linguistic Curiosity 信号**: 利用语言预测与视觉现实的 discrepancy 作为内在好奇心驱动，激发主动探索
2. **GLANCE 框架**: 统一推理与探索——将 agent 的语言世界模型 grounding 到演化中的 target network 的稳定视觉表示
3. **"Think = See" 对齐原理**: 证明对齐"agent 认为会看到什么"与"agent 实际看到什么"是解决复杂/稀疏奖励任务的关键

---

## 问题背景

### 要解决的问题
在部分可观察视觉环境中，VLM agent 如何主动探索以获取有效信息？

### 现有方法的局限
- 现有 VLM agent 通过 CoT 推理进行被动的"心理模拟"（mentally simulate futures），但缺乏主动探索的认知驱动力
- 仅靠被动推理不足以处理稀疏奖励任务，无法主动发现"known unknown"
- 缺乏将世界模型不确定性转化为探索信号的机制

### 本文的动机
让 VLM agent 主动寻找能挑战和修正其内部世界模型的信号，通过 curiosity-driven exploration 实现鲁棒泛化。

---

## 方法详解

### 模型架构

GLANCE 框架的核心组件：
- **VLM Policy**: 具有内化世界模型的 VLM agent，通过 CoT 推理预测未来状态
- **Target Network**: 提供稳定视觉表示的演化目标网络
- **Curiosity Signal**: 语言预测与视觉现实的差异作为 RL 内在奖励

### 核心机制

#### 机制1: Grounding Reasoning via Visual Alignment

将 agent 的语言世界模型（linguistic prediction）与目标网络的视觉表示（visual reality）对齐。

#### 机制2: Curiosity as Active Exploration

将语言-视觉差异量化为内在好奇心信号，在 RL 框架中作为内在奖励，引导 agent 探索其内部模型不确定的区域。

### 问题建模

- 将环境建模为 POMDP（部分可观察马尔可夫决策过程）
- VLM agent 的内部世界模型通过显式 CoT 推理实现
- 好奇心信号 = $\|f_{linguistic}(s_t) - f_{visual}(s_t)\|$

---

## 关键图表

### Figure 1: 预测误差可视化

![Prediction Error](https://arxiv.org/html/2605.03782v1/images/intro_1.jpg)

**说明**: 不同环境中的预测误差示例。(a) Sokoban: 预测目标和箱子在同一行。(b) PrimitiveSkill: 预测紫色方块堆叠在绿色方块上。语言预测与视觉现实的差异驱动探索。

---

## 实验

### 任务环境

在多个 agentic 任务上进行验证，包括 Sokoban、PrimitiveSkill 等需要主动探索的稀疏奖励任务。

### 核心结论

- GLANCE 在稀疏奖励任务中显著优于纯被动推理的 VLM agent
- "What the agent thinks" 与 "what the agent sees" 的对齐是解决复杂任务的关键

---

## 批判性思考

### 优点
1. 理论动机清晰：将认知科学的好奇心概念引入 VLM agent，有明确的 information-theoretic 基础
2. 统一了推理和探索：不是两个独立模块，而是通过 alignment gap 自然连接
3. VLM 内在世界模型 + curiosity 的组合是 VLN 探索策略的有前景方向

### 局限性
1. 实验环境（Sokoban, PrimitiveSkill）相对简单，与真实 VLN 场景有差距
2. 依赖 target network 的稳定性，在快速变化环境中可能不够鲁棒
3. 好奇心信号的计算可能增加推理延迟

### 潜在改进方向
1. 在标准 VLN 环境（R2R, VLN-CE）上验证
2. 与现有 VLN 探索方法（frontier-based, learning-based）对比
3. 研究 curiosity signal 在长程导航中的衰减问题

### 可复现性评估
- [ ] 代码开源
- [ ] 预训练模型
- [x] 方法描述完整
- [ ] 标准 VLN benchmark 结果

---

## 关联笔记

### 基于
- [[CoT]]: Chain-of-Thought 推理
- [[POMDP]]: 部分可观察决策过程
- [[Intrinsic Motivation]]: 内在动机驱动探索

### 方法相关
- [[Vision-Language Model]]: VLM 基础
- [[Curiosity-Driven Exploration]]: 好奇心驱动探索
- [[World Model]]: 世界模型
- [[EmergeNav]]: 同期零样本 VLN 方法

---

## 速查卡片

> [!summary] GLANCE
> - **核心**: 用语言预测-视觉现实差异作为 curiosity 信号驱动 VLM agent 主动探索
> - **方法**: Visual-linguistic alignment + intrinsic curiosity reward in RL
> - **关键洞察**: "What you think is what you see" — 对齐思考与所见是解决稀疏奖励任务的关键
> - **代码**: 未开源

---

*笔记创建时间: 2026-05-06*
