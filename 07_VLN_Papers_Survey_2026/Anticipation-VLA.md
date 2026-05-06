---
title: "Anticipation-VLA: Solving Long-Horizon Embodied Tasks via Anticipation-based Subgoal Generation"
method_name: "Anticipation-VLA"
authors: [Zhilong Zhang, Wenyu Luo, Haonan Wang, Yifei Sheng, Yidi Wang, Hanyuan Guo, Haoxiang Ren, Xinghao Du, Yuhan Che, Tongtong Cao, Lei Yuan, Yang Yu]
year: 2026
venue: ICML 2026
tags: [vision-language-action, long-horizon, subgoal-generation, hierarchical-policy, embodied-intelligence, flow-matching]
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
| 日期 | May 2026 (ICML) |
| 项目主页 | N/A |
| 对比基线 | [[π₀]], [[UniVLA]], [[DreamVLA]], [[π₀.₅]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.01772) |

---

## 一句话总结

> 提出 Anticipation Model 自适应递归生成视觉+语言子目标，结合 Value Model 判断进展，与目标条件化 VLA 形成层次化架构解决长程具身任务。

---

## 核心贡献

1. **Anticipation Model**: 自适应递归子目标生成——根据 Value Model 的进展反馈动态调整子目标粒度和方向
2. **层次化 Anticipation-VLA**: 高层（Anticipation + Value）生成多模态子目标 → 低层（Goal-conditioned VLA）执行动作
3. **Self-Discriminative Regularization**: 逆动力学模型验证生成子目标的语义一致性
4. **仿真 + 真实机器人验证**: LIBERO 平均 SR 80.8%（+4.0% vs π₀.₅），真实 unseen 场景中唯一非零成功率

---

## 问题背景

### 要解决的问题
VLA 模型在长程任务中因 compounding error 性能急剧下降。

### 现有方法的局限
- **固定粒度分解**: 将长程任务分为固定大小子任务，无法适应执行状态的变化复杂度
- **隐式子目标**: DreamVLA 等使用隐式未来预测，缺乏显式可控的子目标表示
- **缺乏进展监控**: 无法判断当前子目标是否已达成或陷入停滞

### 本文的动机
通过"预期"机制动态生成子目标——简单阶段粗粒度，复杂阶段细粒度，配合栈式管理实现回溯和精细化。

---

## 方法详解

### 模型架构

**三组件层次化架构**:
1. **Anticipation Model $G$**: 递归子目标生成器
2. **Value Function $V^*$**: 进展监控器（3 分类）
3. **Goal-conditioned VLA $\pi$**: 低层动作执行器

### 核心理论

#### Goal-Conditioned MDP (GMDP)

定义为 $(\mathcal{S}, \mathcal{A}, \mathcal{G}, P, r, T)$，目标空间：

$$\mathcal{G} = (\mathcal{L} \cup \{\emptyset_{\mathcal{L}}\}) \times (\mathcal{S} \cup \{\emptyset_{\mathcal{S}}\}) \setminus \{(\emptyset_{\mathcal{L}}, \emptyset_{\mathcal{S}})\}$$

目标 $g = (\ell_g, s_g)$ 至少包含一个非空分量（语言指令或视觉子目标）。

#### Bellman 最优方程

$$V^*(s, g) = \max_a \left[r(s,a,g) + \mathbb{E}_{s' \sim P(\cdot|s,a)}[V^*(s',g)]\right]$$

#### 最优分解性质

$$V^*(s_0, g) = V^*(s_0, g') + V^*(s_{g'}, g)$$

确保从 $s_0$ 到最终目标 $g$ 的最大奖励可通过中间子目标 $g'$ 完美分解。

### 核心模块

#### 模块1: Anticipation Model

映射 $G: \mathcal{S} \times \mathcal{G} \to \mathcal{G}$，输入当前状态 $s$ 和活跃目标 $g$，输出精细化子目标 $g'$。输出可递归反馈为新的活跃目标。

**UMM 实现**（基于 Bagel 统一多模态模型）分两阶段：
1. **语言策略** $l_\theta: \mathcal{S} \times \mathcal{G} \to \mathcal{L}$，预测子目标指令 $\ell_{g'}$
2. **动力学模型** $P_\theta: \mathcal{S} \times \mathcal{L} \to \mathcal{S}$，预测子目标图像 $s_{g'}$

形式化: $G_\theta = (P_\theta \circ l_\theta, l_\theta)$

**Self-Discriminative Regularization**: 生成候选 $s_{g'}$ 后，逆动力学模型 $P_\theta^{-1}: \mathcal{S} \times \mathcal{S} \to \mathcal{L}$ 推断指令 $l'_{\text{inv}}$。若与原始 $l_{g'}$ 语义不等价，则重新生成。

#### 模块2: Value Model

重构为**3 分类**问题（而非回归），避免需要密集奖励：
- **Goal Achievement**: 目标达成
- **Sufficient Progress**: 充分进展
- **Insufficient Progress**: 进展停滞

标签函数：

$$\text{label}(f_1, f_2, g) = \begin{cases} \text{Insufficient Progress} & \text{if } f_2.\text{frame} \in [0, f_1.\text{frame}+\beta) \\ \text{Sufficient Progress} & \text{if } f_2.\text{frame} \in [f_1.\text{frame}+\beta, g.\text{frame}-\gamma) \\ \text{Goal Achievement} & \text{if } f_2.\text{frame} \in [g.\text{frame}-\gamma, \infty) \end{cases}$$

#### 模块3: Dynamic Subgoal Management

使用**栈**管理子目标。每 $K$ 步评估三个条件：

1. **Goal Achievement**: $|V^*(s,g) - V^*(g,g)| < \delta$ → 弹出子目标
2. **Insufficient Progress**: $|V^*(s,g) - V^*(s_{\text{prev}},g)| < \delta$ 且栈未满 → 通过 $G$ 生成精细化子目标并入栈；栈满则回溯（重置）
3. **Sufficient Progress**: $|V^*(s,g) - V^*(s_{\text{prev}},g)| \geq \delta$ → 继续

#### 模块4: Goal-conditioned VLA

基于 **π₀.₅**（flow matching-based VLA）。增强：将子目标图像 $s_g^t$ 追加到相机观测 $\mathbf{s}_o^t = [s_1^t, \ldots, s_n^t]$ 后，接机器人配置 $\mathbf{q}$ 和子目标指令 $\ell_g^t$。

$$\pi_\theta(\mathbf{a}^{t:t+h}|\mathbf{s}_o^t, g) = \pi_\theta(\mathbf{a}^{t:t+h}|\mathbf{s}_o^t, g_t) \cdot G_\theta(g_t|g)$$

训练时随机 mask 目标图像 token 增强鲁棒性。

---

## 关键公式

### 公式4: 策略损失

$$\mathcal{L}_{\text{policy}}(\theta) = \mathbb{E}_{\mathcal{D}_{\text{anti}}}[-\log l_\theta(\ell_{g_{h+1}}|s, g_h)]$$

### 公式5: 前向动力学损失 (flow matching)

$$\mathcal{L}_{\text{dyna}}(\theta) = \mathbb{E}_{t, \mathcal{D}_{\text{anti}}}[\|v_\theta(s, s^t_{g_{h+1}}, \ell_{g_{h+1}}, t) - v\|_2^2]$$

### 公式6: 逆动力学损失

$$\mathcal{L}_{\text{inverse}}(\theta) = \mathbb{E}_{\mathcal{D}_{\text{anti}}}[-\log P_\theta^{-1}(\ell_{g_{h+1}}|s, s_{g_h})]$$

### 公式7: Value 损失

$$\mathcal{L}_{\text{value}}(\theta) = \mathbb{E}_{\mathcal{D}_{\text{value}}}[-\log V_\theta(y|s_1, s_2, g_h)]$$

### 公式8: 总损失

$$\mathcal{L}(\theta) = \lambda_1 \mathcal{L}_{\text{policy}} + \lambda_2 \mathcal{L}_{\text{dyna}} + \lambda_3 \mathcal{L}_{\text{inverse}} + \lambda_4 \mathcal{L}_{\text{value}}$$

---

## 关键图表

### Figure 1: 系统架构

![Architecture](https://arxiv.org/html/2605.01772v1/x1.png)

**说明**: Anticipation-VLA 总体架构。Anticipation Model 根据进展反馈自适应输出多模态子目标，Goal-conditioned VLA 执行低层动作。动态目标栈支持回溯和精细化。

### Figure 2: Anticipation Model 推理流程

![Inference](https://arxiv.org/html/2605.01772v1/x2.png)

**说明**: Value Model 输出进展标签；若检测到停滞，Policy Model 生成文本子目标，Dynamics Model 预测视觉子目标，Inverse Dynamics Model 验证对齐。

### Figure 3: 仿真任务

![Sim Tasks](https://arxiv.org/html/2605.01772v1/x3.png)

### Figure 4: 真实机器人任务与结果

![Real World](https://arxiv.org/html/2605.01772v1/x4.png)

---

## 实验

### 实现细节

- **硬件**: 4× NVIDIA H100
- **Anticipation Model 训练**: ~4–6 小时
- **VLA 训练**: ~1–2 小时
- **UMM 骨干**: Bagel (Deng et al., 2025a)
- **VLA 骨干**: π₀.₅

### Table 1: LIBERO 性能对比 (One-Trajectory SFT)

| Model | Spatial | Object | Goal | Long | Avg |
|-------|---------|--------|------|------|-----|
| π₀ | 70.2 | 80.0 | 70.6 | 37.6 | 64.6 |
| UniVLA | 26.0 | 40.0 | 18.0 | 1.8 | 21.5 |
| DreamVLA | 38.0 | 34.0 | 16.6 | 20.6 | 27.3 |
| π₀.₅ | 78.2 | 88.6 | 85.8 | 54.6 | 76.8 |
| π₀.₅+VLM | 82.0 | 88.0 | 80.8 | 53.2 | 76.0 |
| **Anticipation-VLA** | **81.8** | **91.6** | **86.6** | **63.2** | **80.8** |

**关键发现**: Long-horizon 子集提升最显著（+8.6% vs π₀.₅），验证了自适应子目标生成对长程任务的价值。

### Table 2: VLABench 性能对比

| Model | Process Reward | Success Rate |
|-------|---------------|-------------|
| π₀ | 39.6 | 1.0 |
| UniVLA | 28.1 | 1.0 |
| DreamVLA | 7.3 | 0.0 |
| π₀.₅ | 42.7 | 2.1 |
| π₀.₅+VLM | 47.9 | 2.1 |
| **Anticipation-VLA** | **56.3** | **4.2** |

### Table 3: Anticipation Model 生成质量

| Benchmark | Text Acc. | PSNR | MAE | SSIM | FID |
|-----------|-----------|------|-----|------|-----|
| Libero | 84.4 | 20.4 | 9.4 | 0.85 | 31.0 |
| VLABench | 88.8 | 15.5 | 19.0 | 0.76 | 55.1 |
| Rearrange | 88.1 | 28.0 | 6.1 | 0.93 | 45.1 |
| Spell Words | 98.9 | 26.4 | 6.9 | 0.92 | 34.7 |

### 真实机器人实验

- **平台**: Arx-X5 移动操作臂 + 2× Realsense D435i
- **Rearrange Objects**: 100 demonstrations，图像条件化
- **Spell Words**: 200 demonstrations，语言条件化，3-5 字母
- **评估**: 每任务 40 rollouts（20 seen + 20 unseen）

**关键结果**:
- Unseen 场景提升 **+107%**（vs seen +60%）
- **Anticipation-VLA 是唯一在 unseen Spell Words 任务中达到非零成功率的模型**

### 消融实验 (Figure 5)

| 变体 | 效果 |
|------|------|
| w/o subgoal image | 性能下降，视觉子目标提供物理引导 |
| w/o subgoal text | 性能下降，文本子目标提供语义锚定 |
| w/o recursive | 性能下降，自适应规划支持长程执行 |

### 泛化评估 (Figure 6)

- **Object Generalization**: 新实例（如用字母+数字拼 "H2O"），Rearrange 得分 0.58 vs 标准 unseen 0.59
- **Background Generalization**: 改变表面纹理和光照，得分 0.64
- **Spell Words**: Anticipation-VLA 是唯一非零成功率的策略，所有基线完全失败

---

## 批判性思考

### 优点
1. 栈式动态子目标管理优雅解决了固定粒度的问题——支持递归精细化和回溯
2. Self-Discriminative Regularization（逆动力学验证）是保证子目标质量的巧妙设计
3. Value Model 重构为 3 分类避免了密集奖励需求
4. LIBERO Long-horizon 子集 +8.6%，验证了方法在长程任务上的核心价值
5. 真实机器人验证全面（seen + unseen + object/background generalization）

### 局限性
1. 需要标注子目标的 demonstrations，不是零样本
2. 视觉子目标生成计算开销大，偶尔导致推理暂停
3. 实验集中在操作任务，未在导航任务上验证
4. 基于 Bagel UMM 的图像生成质量在 VLABench 上较低（PSNR 15.5, FID 55.1）

### 潜在改进方向
1. 将 Anticipation Model 应用于 VLN 的 waypoint prediction
2. 研究文本子目标 vs 视觉子目标的分工（导航可能更依赖文本？）
3. 在 R2R / VLN-CE 上评估
4. 探索更轻量的子目标生成器减少推理延迟

### 可复现性评估
- [ ] 代码开源
- [ ] 预训练模型
- [x] 实验设置完整（数据量、硬件、超参数）
- [x] Benchmark 标准化 (LIBERO, VLABench)

---

## 关联笔记

### 基于
- [[π₀.₅]]: Flow matching-based VLA（低层执行器）
- Bagel: 统一多模态模型（高层子目标生成器）
- [[Goal-Conditioned RL]]: 目标条件化 MDP 框架
- [[Flow Matching]]: 动力学模型的生成方法

### 对比
- [[DreamVLA]]: VLA + 隐式世界模型
- [[UniVLA]]: 统一 VLA + 显式未来图像生成
- [[π₀]]: 大规模预训练 VLA
- [[EmergeNav]]: VLN 中的 Plan-Solve-Transition 子目标分解

### 方法相关
- [[Subgoal Generation]]: 子目标生成
- [[Waypoint Prediction]]: VLN 路点预测
- [[Hierarchical Planning]]: 层次化规划
- [[Compounding Error]]: 长程误差累积

---

## 速查卡片

> [!summary] Anticipation-VLA
> - **核心**: 自适应递归子目标生成（栈管理 + Value 监控）+ 层次化 VLA
> - **高层**: Bagel UMM → 文本子目标 + 视觉子目标 + 逆动力学验证
> - **低层**: π₀.₅ Goal-conditioned VLA → 动作执行
> - **结果**: LIBERO Avg 80.8% (+4.0%), Long-horizon 63.2% (+8.6%), 真实机器人 unseen 唯一非零
> - **硬件**: 4× H100, Anticipation 训练 4-6h, VLA 训练 1-2h

---

*笔记创建时间: 2026-05-06*
