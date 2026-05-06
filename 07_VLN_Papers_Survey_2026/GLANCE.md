---
title: "What You Think is What You See: Driving Exploration in VLM Agents via Visual-Linguistic Curiosity"
method_name: "GLANCE"
authors: [Haoxi Li, Qinglin Hou, Jianfei Ma, Jinxiang Lai, Tao Han, Sikai Bai, Jingcai Guo, Jie Zhang, Song Guo]
year: 2026
venue: ICML 2026
tags: [vlm-agent, curiosity-driven-exploration, world-model, reinforcement-learning, visual-linguistic-alignment, pomdp]
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
| 日期 | May 2026 (ICML) |
| 项目主页 | N/A |
| 对比基线 | VaGen, ICM, BYOL-Explore |
| 链接 | [arXiv](https://arxiv.org/abs/2605.03782) |

---

## 一句话总结

> 提出 GLANCE 框架，将 VLM agent 语言世界模型预测与视觉现实的差异作为内在好奇心信号，配合 Curriculum Exploration 防止好奇心衰减，驱动 RL 中的主动探索。

---

## 核心贡献

1. **跨模态好奇心信号**: 利用语言预测（linguistic hypothesis）与视觉现实（visual reality）的 discrepancy 作为 RL 内在奖励
2. **GLANCE 框架**: 统一世界建模与探索——VLM 同时充当世界模型和策略，target network 提供稳定视觉锚点
3. **Curriculum Exploration**: 周期性重初始化投影器防止"好奇心枯竭"（Curiosity Drain），维持自适应课程学习
4. **"Think = See" 对齐原理**: 证明对齐"agent 认为会看到什么"与"agent 实际看到什么"是解决复杂稀疏奖励任务的关键

---

## 问题背景

### 要解决的问题
在部分可观察视觉环境中，VLM agent 如何从被动推理转向主动探索？

### 现有方法的局限
- 现有 VLM agent 通过 CoT 推理进行被动"心理模拟"（mentally simulate futures），但仅在已访问状态上推理
- 稀疏奖励任务中，缺乏主动发现"known unknown"的认知驱动力
- 传统好奇心方法（ICM, BYOL-Explore）使用视觉到视觉的预测，缺乏语义层面的理解
- 缺乏将世界模型不确定性转化为探索信号的机制

### 本文的动机
利用 VLM 内化的世界模型（通过 CoT 推理产生的语言预测）与实际视觉观测之间的差异作为好奇心驱动，实现语义层面的主动探索。

---

## 方法详解

### 问题建模: POMDP

将多轮 VLM agentic 任务建模为 POMDP $(\mathcal{S}, \mathcal{A}, \mathcal{O}, T, R, Z, \gamma)$。

每步 $t$，agent 选择动作 $a_t \sim \pi_{\boldsymbol{\theta}}(a_t | b_t)$（自然语言 token），$b_t$ 为历史 $h_t = (o_0, a_0, \ldots, o_{t-1}, a_{t-1}, o_t)$ 的充分统计量。

**优化目标**:

$$\mathcal{J}(\boldsymbol{\theta}) = \mathbb{E}_{\pi_{\boldsymbol{\theta}}, T, Z}\left[\sum_{t=0}^{\infty} \gamma^t \mathbb{E}_{b_t}[R(s_t, a_t)] \;\bigg|\; b_0\right]$$

### 内化世界模型

VLM 通过结构化 token 序列内化世界建模：

```
<Obs> s_t </Obs> <Res> z_t </Res> <Pred> s_{t+1} </Pred> <Ans> a_t </Ans>
```

策略通过 PPO 优化，外在奖励：

$$r_t^e = r_t^{\text{task}} + r_t^{\text{reason}} + r_t^{\text{format}}$$

使用 **Bi-Level GAE** 进行层次化信用分配（turn 级结果 + token 级生成）。

### 模型架构

#### Online VLM Agent
- 参数 $\boldsymbol{\theta} = (\mathbf{v}, \boldsymbol{\ell})$：视觉编码器 $f_{\mathbf{v}}$ + LLM 骨干 $\Lambda_{\boldsymbol{\ell}}$
- **语言假设状态** $h_{t+1} \in \mathbb{R}^d$：Transformer 最后一层在 `</pred>` token 位置的 hidden state
- **轻量投影器** $g_{\boldsymbol{\psi}}$：桥接语言潜空间到视觉特征空间

#### Momentum Encoder (Target Network)
- 参数 $\boldsymbol{\phi}$，结构同 $f_{\mathbf{v}}$
- **不通过梯度下降更新**，使用 EMA：

$$\boldsymbol{\phi} \leftarrow \alpha \boldsymbol{\phi} + (1 - \alpha)\mathbf{v}$$

$\alpha \in [0,1]$ 为 target decay rate。

- 视觉现实表示：

$$y_{t+1} = \texttt{sg}(f_{\boldsymbol{\phi}}(o_{t+1}))$$

$\texttt{sg}(\cdot)$ 为 stop-gradient 算子。

### 核心模块

#### 模块1: Grounding Reasoning via Visual Alignment

语言假设投影到视觉空间：

$$\widehat{y}_{t+1} = g_{\boldsymbol{\psi}}(h_{t+1})$$

**预测目标**（归一化余弦距离）:

$$\mathcal{L}_{\text{explore}}(\mathbf{v}, \boldsymbol{\psi}, t) = \left\| \frac{\widehat{y}_{t+1}}{\|\widehat{y}_{t+1}\|_2} - \texttt{sg}\left(\frac{y_{t+1}}{\|y_{t+1}\|_2}\right) \right\|_2^2$$

**选择性梯度路由**: LLM 参数冻结（防止语言漂移），但梯度**穿过**骨干反传，更新：
- 投影器 $g_{\boldsymbol{\psi}}$
- 在线视觉编码器 $f_{\mathbf{v}}$

统一损失同时实现三个目标：
1. 塑造视觉编码器捕获语义可操作特征
2. 将内化世界模型 grounding 到物理动态
3. 提供 curiosity-driven 内在奖励

#### 模块2: Curiosity as Active Exploration

**内在奖励**:

$$r_t^i = \beta \cdot \mathcal{L}_{\text{explore}}(\mathbf{v}, \boldsymbol{\psi}, t)$$

**总奖励**:

$$r_t = r_t^e + r_t^i$$

#### 模块3: Curriculum Exploration

**问题: Curiosity Drain（好奇心枯竭）**

轻量投影器快速过拟合到浅层视觉特征（因预训练 LLM 骨干已语义丰富），导致内在奖励过早消失。Agent 因为"错误地相信已掌握环境"而停止探索。

**解决方案**: 周期性重初始化投影器权重，同时保留演化中的视觉编码器。形成自适应课程：
- 随着视觉编码器捕获越来越复杂的语义，重新初始化的投影器揭示新的差异
- 在整个长程学习中维持好奇心驱动

---

## 关键公式

### 公式1: POMDP 优化目标

$$\mathcal{J}(\boldsymbol{\theta}) = \mathbb{E}_{\pi_{\boldsymbol{\theta}}, T, Z}\left[\sum_{t=0}^{\infty} \gamma^t \mathbb{E}_{b_t}[R(s_t, a_t)] \;\bigg|\; b_0\right]$$

### 公式2: 外在奖励

$$r_t^e = r_t^{\text{task}} + r_t^{\text{reason}} + r_t^{\text{format}}$$

### 公式3: EMA 更新

$$\boldsymbol{\phi} \leftarrow \alpha \boldsymbol{\phi} + (1 - \alpha)\mathbf{v}$$

### 公式4: Stop-gradient 视觉表示

$$y_{t+1} = \texttt{sg}(f_{\boldsymbol{\phi}}(o_{t+1}))$$

### 公式5: 跨模态投影

$$\widehat{y}_{t+1} = g_{\boldsymbol{\psi}}(h_{t+1})$$

### 公式6: 探索损失（好奇心信号）

$$\mathcal{L}_{\text{explore}}(\mathbf{v}, \boldsymbol{\psi}, t) = \left\| \frac{\widehat{y}_{t+1}}{\|\widehat{y}_{t+1}\|_2} - \texttt{sg}\left(\frac{y_{t+1}}{\|y_{t+1}\|_2}\right) \right\|_2^2$$

### 公式7: 内在奖励

$$r_t^i = \beta \cdot \mathcal{L}_{\text{explore}}(\mathbf{v}, \boldsymbol{\psi}, t)$$

### 公式8: 总奖励

$$r_t = r_t^e + r_t^i$$

---

## 关键图表

### Figure 1: 预测误差示例

![Prediction Error](https://arxiv.org/html/2605.03782v1/images/intro_1.jpg)

**说明**: (a) Sokoban: 预测"目标和箱子在同一行"。(b) PrimitiveSkill: 预测"紫色方块堆叠在绿色方块上"。语言预测与视觉现实的差异驱动探索。

### Figure 2: GLANCE 框架概览

![GLANCE Framework](https://arxiv.org/html/2605.03782v1/x1.png)

**说明**: 左侧：VLM agent 生成推理轨迹并预测未来状态 $s_{t+1}$；`</pred>` token 的隐状态通过投影器对齐到 momentum target 视觉编码器的输出。预测损失 $\mathcal{L}_{\text{explore}}$ 即好奇心源。梯度穿过冻结 LLM 更新在线视觉编码器。右侧：VLM agent 通过 rollout 交互，由复合奖励 $r_t = r_t^e + \beta \cdot \mathcal{L}_{\text{explore}}$ 引导。Target network 通过 EMA 更新。🔥=可训练；❄️=冻结。

---

## 实验

### Backbone

Qwen2.5-VL-3B

### 任务环境 (5 类 agentic 任务)

1. **Grid Puzzles** (Sokoban)
2. **3D Navigation**
3. **Object Manipulation** (PrimitiveSkill — 方块堆叠)
4. **Geometric Reconstruction** (SVG 重建)
5. 其他 agentic 任务

### 核心结论

- GLANCE 在稀疏奖励任务中**一致性超越**纯 exploitation-based RL 方法（VaGen 等）
- 跨模态好奇心使 VLM agent 即使在无外在奖励时也能学到有效探索策略
- 消融实验确认投影器周期性重初始化是防止"好奇心坍塌"的关键

### 与先前方法的关键区别

| 方面 | 标准好奇心方法 | GLANCE |
|------|--------------|--------|
| 世界模型 | 外部辅助模块 | 通过 CoT 推理内化到策略中 |
| 预测类型 | 视觉→视觉潜空间 | 跨模态：语言→视觉 |
| 探索驱动 | 视觉新颖性 | 语义理解（语言推理）|
| 坍塌预防 | 无 | Curriculum Exploration（投影器重初始化）|

---

## 批判性思考

### 优点
1. 理论优雅：将 VLM 的语言推理能力与好奇心探索统一，不是两个独立模块
2. Curiosity Drain 问题的识别和 Curriculum Exploration 的解决方案有创新性
3. 选择性梯度路由设计合理——冻结 LLM 防止语言漂移，同时更新视觉编码器
4. VLM 同时作为世界模型和策略，架构简洁

### 局限性
1. 实验环境（Sokoban, PrimitiveSkill）相对简单，与真实 VLN 场景有差距
2. 使用 3B 模型（Qwen2.5-VL-3B），更大模型的表现未知
3. HTML 版本中实验表格数据不完整（rendering 问题），难以验证具体数值
4. 好奇心信号计算需要额外的前向传播（投影器 + target network），增加推理延迟
5. 投影器重初始化的最优周期如何确定未深入讨论

### 潜在改进方向
1. 在标准 VLN 环境（R2R, VLN-CE）上验证跨模态好奇心的效果
2. 研究好奇心信号在长程导航中的时间衰减特性
3. 与 VLN 中 frontier-based exploration 结合
4. 在更大 VLM（7B+）上验证 scaling behavior

### 可复现性评估
- [ ] 代码开源
- [ ] 预训练模型
- [x] 方法描述完整（公式 + 架构图）
- [ ] 完整实验数据

---

## 关联笔记

### 基于
- [[CoT]]: Chain-of-Thought 推理
- [[POMDP]]: 部分可观察决策过程
- [[PPO]]: Proximal Policy Optimization
- [[BYOL]]: Bootstrap Your Own Latent（EMA target network 思路）
- [[Qwen2.5-VL]]: VLM 骨干

### 对比
- [[ICM]]: Intrinsic Curiosity Module（视觉→视觉预测）
- [[BYOL-Explore]]: BYOL-style 探索
- VaGen: VLM + RL 方法

### 方法相关
- [[Curiosity-Driven Exploration]]: 好奇心驱动探索
- [[World Model]]: 世界模型
- [[EmergeNav]]: 同期零样本 VLN 方法
- [[Intrinsic Motivation]]: 内在动机理论

---

## 速查卡片

> [!summary] GLANCE
> - **核心**: 跨模态好奇心——语言预测与视觉现实的差异驱动 VLM agent 主动探索
> - **方法**: Online VLM + Momentum Target Network + 轻量投影器 + Curriculum Exploration
> - **关键公式**: $r_t^i = \beta \cdot \|\widehat{y}_{t+1}/\|\widehat{y}_{t+1}\| - \texttt{sg}(y_{t+1}/\|y_{t+1}\|)\|^2$
> - **骨干**: Qwen2.5-VL-3B + PPO + Bi-Level GAE
> - **代码**: 未开源

---

*笔记创建时间: 2026-05-06*
