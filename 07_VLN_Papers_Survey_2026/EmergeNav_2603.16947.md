---
title: "EmergeNav: Structured Embodied Inference for Zero-Shot Vision-and-Language Navigation in Continuous Environments"
method_name: "EmergeNav"
authors: [Kun Luo, Xiaoguang Ma]
year: 2026
venue: arXiv
tags: [vision-language-navigation, zero-shot, structured-inference, vlm-agent, memory-mechanism]
zotero_collection: ""
image_source: online
arxiv_html: https://arxiv.org/html/2603.16947v1
created: 2026-04-27
---

# 论文笔记：EmergeNav: Structured Embodied Inference for Zero-Shot Vision-and-Language Navigation in Continuous Environments

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Foshan Graduate School of Innovation, Northeastern University |
| 日期 | March 2026 |
| 项目主页 | N/A |
| 对比基线 | [[SmartWay]], [[Open-Nav]], [[InstructNav]], [[MapGPT]], [[DiscussNav]] |
| 链接 | [arXiv](https://arxiv.org/abs/2603.16947) |

---

## 一句话总结

> 提出 Plan-Solve-Transition 结构化推理层级，使开源 [[VLM]] 在零样本VLN-CE中达到竞争力表现，无需训练、地图或路点预测器。

---

## 核心贡献

1. **Plan-Solve-Transition (PST) 层级**: 将长程VLN-CE分解为阶段性结构化推理，显式子目标分解与阶段验证
2. **GIPE (Goal-Conditioned Information-Guided Perceptual Extraction)**: 目标条件化的感知证据提取协议，将原始多视角观测转为紧凑的任务对齐证据
3. **对比式双记忆推理**: 短期记忆(STM)与长期记忆(LTM)对比进展定位，区分前进/停滞/回溯
4. **角色分离的双FOV感知**: Solve阶段用前向三联视角(高频)，Transition阶段用全景(低频)，匹配决策时间尺度

---

## 问题背景

### 要解决的问题
零样本VLN-CE中，瓶颈不是缺乏导航知识（现代VLM已编码大量先验），而是**缺乏将知识组织为指令跟随、感知锚定、时间进展和阶段验证的执行结构**。

### 现有方法的局限
- VLM虽有丰富先验但"不能自然产生稳定的长程行为"
- 现有零样本方法依赖在线语言地图（MapGPT）、多专家讨论（DiscussNav）或路点预测器（SmartWay），增加额外模块
- 缺乏显式的阶段管理和进展验证机制

### 本文的动机
将长程导航形式化为**结构化具身推理**问题，用执行脚手架而非额外模块来提升VLM的导航能力。

---

## 方法详解

### 模型架构

EmergeNav 采用**纯VLM策略 + 结构化推理脚手架**架构：
- **输入**: 语言指令 $x$ + 自中心RGB观测（无深度、无里程计）
- **VLM骨干**: [[Qwen3-VL]]-8B / 32B
- **核心结构**: Plan-Solve-Transition三层层级
- **感知模式**: Solve用前向三联视角，Transition用全景
- **输出**: 连续动作序列 $a_{1:T}$
- **特点**: 完全零样本——无微调、无地图、无路点预测器

### 核心模块

#### 模块1: Plan — 指令分解

**设计动机**: 将自然语言指令转为锚点驱动的子目标序列

**具体实现**:
- VLM将指令 $x$ 分解为子目标序列 $\mathcal{G} = [g_1, \ldots, g_K]$
- 每个子目标对应一个锚点状态转换
- 保持语义完整性

#### 模块2: Solve — 局部控制

**设计动机**: 在当前子目标内执行高频局部控制

**具体实现**:
- 使用前向三联视角（-30度, 0度, +30度, 各72度HFOV）
- [[ReAct]]风格循环：推理证据 -> 选择朝向 -> 执行动作束 -> 更新STM
- 通过GIPE提取紧凑的子目标相关证据
- 生成事实摘要供Transition审计

#### 模块3: Transition — 阶段验证

**设计动机**: 独立于Solve的阶段完成判定

**具体实现**:
- 使用全景观测（6个方向，间隔60度）
- 接收当前/下一子目标、LTM和rollout摘要
- 输出 continue 或 switch
- "Solve可以指示进展，但只有Transition能推进阶段索引"

#### 模块4: GIPE — 目标条件化感知提取

**设计动机**: 将原始多视角观测转为紧凑的任务对齐证据

**具体实现**:
- 以prompt级结构化证据协议实现（非独立模块）
- Solve中：从前向三联视角+STM/LTM提取锚点转换、agent-锚点关系、可通行性、朝向含义
- Transition中：从全景视角提取锚点可见性、阶段边界、下一子目标可行性、真实前进vs漂移
- 要求证据必须有可见像素或记忆支持（grounding约束）

#### 模块5: 对比式双记忆

**设计动机**: 通过STM与LTM对比来定位进展，区分前进/停滞/回溯

**具体实现**:
- STM：密集记录子目标内的前向视角轨迹
- LTM：稀疏存储经验证的进展锚点（仅在阶段切换时更新）
- 进展判断：将演化中的STM序列与LTM基线对比

---

## 关键公式

### 公式1: 局部前向三联观测

$$
\mathcal{O}_t^{loc} = \{o_t^{-30°}, o_t^{0°}, o_t^{+30°}\}
$$

**含义**: Solve阶段的高频前向感知输入

### 公式2: 全景观测

$$
\mathcal{O}_t^{pan} = \{o_t^{0°}, o_t^{60°}, o_t^{120°}, o_t^{180°}, o_t^{240°}, o_t^{300°}\}
$$

**含义**: Transition阶段的低频全景感知输入

### 公式3: 子目标分解

$$
\mathcal{G} = \text{Plan}(x) = [g_1, \ldots, g_K]
$$

**含义**: 将指令分解为锚点驱动的子目标序列

### 公式4: 结构化推理状态

$$
\mathbf{s}_t = (k_t, \mathbf{e}_t, M_t^{STM}, M_t^{LTM}, u_t)
$$

**含义**: 每步的完整推理状态

**符号说明**:
- $k_t$: 当前活跃子目标索引
- $\mathbf{e}_t$: 当前证据状态
- $M_t^{STM}$: 短期记忆
- $M_t^{LTM}$: 长期记忆
- $u_t \in \{\text{continue}, \text{switch}\}$: 阶段控制信号

### 公式5: Solve阶段证据提取

$$
\mathbf{e}_t^{solve} = \text{GIPE}_{solve}(g_{k_t}, \mathcal{O}_t^{loc}, M_t^{STM}, M_t^{LTM})
$$

**含义**: 从局部视角和记忆中提取子目标相关的紧凑证据

### 公式6: 局部控制策略

$$
a_t \sim \pi_{solve}(g_{k_t}, \mathbf{e}_t^{solve})
$$

**含义**: 基于子目标和证据生成动作

### 公式7: Transition阶段证据提取

$$
\mathbf{e}_t^{trans} = \text{GIPE}_{trans}(g_{k_t}, g_{k_t+1}, \mathcal{O}_t^{pan}, M_t^{LTM}, \Sigma_t)
$$

**含义**: 从全景视角、长期记忆和rollout摘要中提取阶段边界证据

**符号说明**:
- $\Sigma_t$: rollout摘要

### 公式8: Transition决策

$$
u_t = \text{Transition}(\mathbf{e}_t^{trans}) \in \{\text{continue}, \text{switch}\}
$$

**含义**: 判断当前子目标是否完成

### 公式9: 记忆状态

$$
M_t = (M_t^{STM}, M_t^{LTM})
$$

**含义**: 双记忆的联合状态

### 公式10: STM更新

$$
M_t^{STM} = \text{Update}_{STM}(M_{t-1}^{STM}, o_t^{0°}, a_t, r_t)
$$

**含义**: 每步更新短期记忆

**符号说明**:
- $o_t^{0°}$: 前向观测
- $a_t$: 动作束
- $r_t$: 结构化记录

### 公式11: LTM更新

$$
M_t^{LTM} = \begin{cases} \text{Update}_{LTM}(M_{t-1}^{LTM}, \Sigma_t), & \text{if } u_t = \text{switch} \\ M_{t-1}^{LTM}, & \text{if } u_t = \text{continue} \end{cases}
$$

**含义**: LTM仅在阶段切换时更新，保持稀疏的经验证进展锚点

### 公式12: 记忆条件化策略

$$
a_t \sim \pi_{solve}(g_{k_t}, \mathbf{e}_t^{solve}, M_t^{STM}, M_t^{LTM})
$$

**含义**: 完整的Solve策略同时条件化在双记忆上

---

## 关键图表

### Figure 1: Overview / 系统概览

![Figure 1](https://arxiv.org/html/2603.16947v1/pic1.png)

**说明**: EmergeNav总览。给定指令，agent先分解为锚点驱动的子目标，然后通过Plan-Solve-Transition层级执行。Solve阶段用前向三联视角执行高频局部rollout；Transition阶段用全景执行低频边界验证。GIPE提供任务对齐的感知证据；双记忆维护密集子目标内轨迹(STM)和稀疏经验证进展锚点(LTM)。

### Figure 2: GIPE Interface / GIPE接口

![Figure 2](https://arxiv.org/html/2603.16947v1/pic2.png)

**说明**: EmergeNav中GIPE接口。Solve阶段：从活跃子目标、前向三联视角和记忆上下文提取紧凑局部证据。Transition阶段：从当前/下一子目标、全景观测、长期记忆和rollout摘要提取边界级证据。

### Figure 3: Dual-Memory Reasoning / 对比式双记忆推理

![Figure 3](https://arxiv.org/html/2603.16947v1/pic3.png)

**说明**: STM存储密集的子目标内前向视角轨迹；LTM存储稀疏的经验证进展锚点。通过将演化中的STM序列与LTM锚点对比来区分前进、停滞和回溯模式。

### Table 1: VLN-CE基线对比

| Method | TL | NE↓ | OSR↑ | SR↑ | SPL↑ |
|--------|-----|-----|------|-----|------|
| **监督方法** | | | | | |
| CMA | 11.08 | 6.92 | 45.0 | 37.0 | 32.17 |
| BEVBert | 13.63 | 5.13 | 57.0 | 60.0 | 53.41 |
| ETPNav | 11.08 | 5.15 | 58.0 | 52.0 | 52.18 |
| StreamVLN | - | 4.98 | 64.2 | 56.9 | 51.9 |
| **零样本方法** | | | | | |
| MapGPT-CE (GPT4o) | 12.63 | 8.16 | 22.0 | 7.0 | 5.04 |
| DiscussNav (GPT4) | 6.27 | 7.77 | 15.0 | 11.0 | 10.51 |
| InstructNav (GPT4) | - | 7.74 | 6.89 | 31.0 | 24.0 |
| Open-Nav (Llama3.1) | 8.07 | 7.25 | 23.0 | 16.0 | 12.90 |
| Open-Nav (GPT4) | 7.68 | 6.70 | 23.0 | 19.0 | 16.10 |
| CA-Nav | - | 8.32 | 39.4 | 23.0 | 11.0 |
| SmartWay | 13.09 | 7.01 | 51.0 | 29.0 | 22.46 |
| Fast-SmartWay | 12.56 | 7.72 | - | 27.75 | 24.95 |
| **EmergeNav (8B)** | 19.50 | 8.38 | 48.00 | **30.00** | 21.26 |
| **EmergeNav (32B)** | 19.22 | 7.60 | **58.00** | **37.00** | 21.33 |

**说明**: 零样本方法中，EmergeNav-8B (30.0 SR) 超越SmartWay (29.0)。扩展到32B后SR达37.0且OSR 58.0，"无需修改执行脚手架即可获益"。

### Table 2: 方法设置对比

| Method | Input | Policy Backbone | Extra Spatial Prior | Waypoint Predictor | Sensing |
|--------|-------|-----------------|--------------------|--------------------|---------|
| MapGPT | RGB | Map-guided GPT | Online linguistic map | No | Panoramic |
| DiscussNav | RGB | Multi-expert | No | No | Panoramic |
| InstructNav | RGB | LLM-guided | Multi-sourced value maps | No | Egocentric |
| Open-Nav | RGB-D | LLM/VLM | No | Yes | Panoramic |
| SmartWay | RGB-D | VLM | No | Yes | Panoramic |
| **EmergeNav** | **RGB** | **VLM** | **No** | **No** | **Role-separated** |

**说明**: EmergeNav是最"干净"的设置——仅RGB输入、无额外先验、无路点预测器

### Table 3: 消融实验

| Variant | SR | OSR | SPL | nDTW | SDTW | Steps | Dist. | Path Len. | Collisions |
|---------|-----|-----|------|------|------|-------|-------|-----------|------------|
| **Full (8B)** | **30.0** | 48.0 | **21.3** | **19.5** | **13.8** | **145.9** | 8.38 | **19.50** | 0.273 |
| w/o GIPE | 12.0 | 32.0 | 6.5 | 9.7 | 3.6 | 166.5 | 9.82 | 22.00 | 0.252 |
| w/o Memory | 17.0 | **51.0** | 6.2 | 4.5 | 1.9 | 219.8 | 9.58 | 30.36 | 0.306 |
| w/o GIPE + Memory | 16.0 | 33.0 | 7.2 | 6.1 | 2.3 | 189.1 | 11.27 | 24.77 | 0.272 |

**关键发现**: 移除GIPE导致SR从30.0暴降到12.0（感知相关性）；移除记忆导致步数从145.9增到219.8、碰撞率上升（时间一致性和执行稳定性）。两者互补。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| VLN-CE (R2R) | 100 episodes | 标准连续环境VLN | 评估 |

### 实现细节

- **Backbone**: Qwen3-VL-8B / Qwen3-VL-32B
- **训练**: 无（完全零样本）
- **硬件**: 8B: 2x RTX 3090; 32B: 2x RTX A6000
- **感知**: Solve: 3视角(72度HFOV), Transition: 6视角全景(60度间隔)

### 可视化结果

方法实现了约30-37%的零样本SR，在零样本VLN-CE方法中具有竞争力，但路径长度偏长（19.50 vs 监督方法的11-14），反映了零样本方法探索效率的局限。

---

## 批判性思考

### 优点
1. 设计哲学清晰：不是给VLM添加更多模块，而是提供结构化执行脚手架
2. 完全零样本，无需任何训练数据或领域特定模块
3. 双FOV感知的角色分离设计合理——匹配Solve和Transition的不同决策频率
4. 对比式双记忆是一个新颖的进展定位机制

### 局限性
1. SPL偏低（~21），路径效率不足——agent倾向于过度探索
2. 仅在100 episodes上评估，统计显著性有限
3. 小模型（8B）的子目标切换敏感度不足，导致临时绕路
4. NE较高（7.60-8.38），说明最终定位精度仍有提升空间

### 潜在改进方向
1. 改善Transition的边界精度，减少false-continue
2. 引入轻量的恢复机制减少绕路
3. 路径效率优化

### 可复现性评估
- [ ] 代码开源
- [ ] 预训练模型（N/A，零样本）
- [x] 实现细节完整（prompt级设计）
- [x] 数据集可获取

---

## 关联笔记

### 基于
- [[ReAct]]: Solve阶段的推理-行动循环
- [[Plan-and-Solve]]: 显式分解范式
- [[Reflexion]]: 语言反馈和记忆
- [[Qwen3-VL]]: VLM骨干

### 对比
- [[SmartWay]]: 零样本VLN-CE with waypoint predictor
- [[Open-Nav]]: LLM/VLM导航
- [[InstructNav]]: 系统性指令跟随
- [[MapGPT]]: 地图引导GPT策略

### 方法相关
- [[Vision-Language Model]]: 基础模型
- [[Structured Agent]]: 结构化智能体范式
- [[Memory Mechanism]]: 记忆机制
- [[Panoramic Observation]]: 全景观测

---

## 速查卡片

> [!summary] EmergeNav
> - **核心**: Plan-Solve-Transition结构化推理层级用于零样本VLN-CE
> - **方法**: GIPE感知提取 + 对比式双记忆 + 角色分离双FOV
> - **结果**: 零样本SR 30.0 (8B) / 37.0 (32B)，超越SmartWay等方法
> - **代码**: 未开源

---

*笔记创建时间: 2026-04-27*
