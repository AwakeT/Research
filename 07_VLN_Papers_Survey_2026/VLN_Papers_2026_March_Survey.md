# 2026年3月 VLN 相关论文汇总（主线关联分析）

> 整理时间：2026-04-27
> 筛选标准：2026年3月发布（arXiv 2603.xxxxx），与 Kinbot NFM 主线设计直接或强相关
> 主线关键词：NFM 27B→4B 蒸馏、空间理解/地图理解、目标 belief/搜索恢复、长程任务编排、世界状态记忆、家庭场景

---

## 主线设计模块速查

| 主线模块 | 优先级 | 描述 |
|----------|--------|------|
| 空间理解 | P0 | 房间识别、家具识别、当前位置判断 |
| 地图理解 | P0 | 视角与地图对齐、区域拓扑关系 |
| 基础空间长期记忆 | P0 | room graph、room-furniture-object relation、target belief map |
| 搜索与恢复 | P1-high | 目标区域排序、搜索顺序、失败恢复 |
| 多视角理解 | P1-high | 多相机语义一致性、遮挡确认 |
| 粗粒度全局定位 | P1-high | 观测到语义地图匹配、区域级重定位 |
| 深度估计与几何约束 | P1-medium | 可通行空间、门口狭窄区域、遮挡关系 |
| RL 避障优化 | P1-low | 搜索效率、主动观察、局部避障 |
| 个性化/行为长期记忆 | P2 | 物品位置先验、人员活动习惯 |
| 27B→4B 蒸馏部署 | 全局 | Teacher-Student 蒸馏、端侧约束 |
| decision_orchestration | 长程任务 | 任务分解、推进与恢复 |
| world_state_memory | 长程任务 | 任务状态持久化、共享状态投影 |

---

## Tier 1：与主线直接对标（5 篇）

### 1. PROSPECT — 统一流式 VLN（空间理解 + NFM 主架构）

- **arXiv**: [2603.03739](https://arxiv.org/abs/2603.03739)
- **发布日期**: 2026-03-05
- **核心方法**: 统一流式 VLA 导航 agent，耦合流式 Vision-Language-Action 策略与潜在预测表征学习
- **技术亮点**:
  - 使用 **CUT3R** 作为流式 3D 空间编码器，产生长上下文、绝对尺度的空间特征
  - 与 **SigLIP** 语义特征通过 cross-attention 融合
  - next-frame prediction 天然兼容流式推理
  - 捕捉语义、空间布局、物理动力学和任务进度之间的时间相关性
- **对标模块**: `空间理解 (P0)` + `多视角理解 (P1-high)` + NFM 主架构设计
- **借鉴价值**: **高**。CUT3R 的 3D 空间编码思路可直接对照你的 `semantic_global_frame` 设计；流式架构与端侧实时推理需求天然匹配。提示：评估 CUT3R 能否作为 27B teacher 的空间编码器候选。

---

### 2. EmergeNav — 零样本连续环境 VLN（搜索恢复 + 双记忆）

- **arXiv**: [2603.16947](https://arxiv.org/abs/2603.16947)
- **发布日期**: 2026-03-16
- **核心方法**: 将连续 VLN 建模为结构化 embodied inference
- **技术亮点**:
  - **Plan-Solve-Transition** 层级执行结构
  - **GIPE** 目标条件感知提取
  - **对比双记忆推理**：短期环境感知 + 长期全局任务记忆
  - **双视野感知** (Dual-FOV)：时间对齐的局部控制 + 边界验证
  - 使用 Qwen3-VL-8B 达 30.00 SR，Qwen3-VL-32B 达 37.00 SR（零样本，无地图/图搜索）
- **对标模块**: `搜索与恢复 (P1-high)` + `基础空间记忆 (P0)`
- **借鉴价值**: **高**。双记忆架构直接对标你的 NFM 短期/长期记忆分离设计；Qwen3-VL-8B/32B 的零样本表现为你评估 backbone 提供了基线参考。Plan-Solve-Transition 与 `decision_orchestration` 的任务分解-执行-恢复链路有结构对应。

---

### 3. RAGNav — 检索增强拓扑推理 VLN（空间长期记忆）

- **arXiv**: [2603.03745](https://arxiv.org/abs/2603.03745)
- **发布日期**: 2026-03-04
- **核心方法**: 检索增强的多目标 VLN 框架
- **技术亮点**:
  - **双基记忆系统** (Dual-Basis Memory):
    - 底层：**拓扑图** 维护物理连通性
    - 高层：**语义森林** 进行层级环境抽象
  - 解决 LLM 导航中的空间幻觉和规划漂移问题
  - 桥接语义推理与物理结构的鸿沟
- **对标模块**: `基础空间长期记忆 (P0)` — 直接对标 `room graph` + `room-furniture-object relation`
- **借鉴价值**: **高**。拓扑图 + 语义森林的双层结构与你设计的 `semantic_global_frame`（语义层）+ 拓扑关系（物理连通层）高度吻合。语义森林的层级抽象思路可借鉴到你的 room→furniture→object 三层记忆设计。

---

### 4. Embodied Foundation Models at the Edge — 边端部署综述（蒸馏部署）

- **arXiv**: [2603.16952](https://arxiv.org/abs/2603.16952)
- **发布日期**: 2026-03-16
- **文档类型**: Survey
- **核心内容**:
  - 提出 **Deployment Gauntlet**：8 大耦合部署壁垒
  - autoregressive VLA 瓶颈在**内存带宽**，diffusion controller 瓶颈在**计算延迟**
  - **NanoVLA** 双层架构：重量级 VLM 做高层语义推理 + 轻量级 visual policy 做高频控制
  - **OneDP** 单步蒸馏：多步扩散压缩为单步生成，1.5Hz → 60Hz
  - 可靠部署依赖系统级协同设计（内存、调度、通信、模型架构）
- **对标模块**: `27B→4B 蒸馏部署 (全局)`
- **借鉴价值**: **极高**。NanoVLA 的"重型语义推理 + 轻型控制"分层与你的 "27B teacher 认知 + 4B student 执行"完全同构。8 大部署壁垒清单可直接用于评估你的 PDCP Student 方案可行性。

---

### 5. DyGeoVLN — 动态几何基础模型注入 VLN（深度/几何）

- **arXiv**: [2603.21269](https://arxiv.org/abs/2603.21269)
- **发布日期**: 2026-03-22
- **来源**: KAIST, HKUST(GZ), JD Explore Academy
- **核心方法**: 将动态几何 foundation model 通过跨分支特征融合注入 VLN
- **技术亮点**:
  - 动态几何 FM 增强前馈几何编码器 + 单目深度估计
  - 显式 3D 空间线索注入解码器
  - 细粒度 3D 重建 + 动态场景表征
  - 多 benchmark SOTA，真实环境验证鲁棒性
- **对标模块**: `深度估计与几何约束 (P1-medium)` + `空间理解 (P0)`
- **借鉴价值**: **高**。动态几何 FM 的 3D 空间线索注入方式，可参考设计你的 P1-medium 阶段深度/几何模块如何接入 NFM 主干。跨分支融合策略提供了一种不破坏已有语义能力的几何增强路径。

---

## Tier 2：与主线强相关（5 篇）

### 6. SPAN-Nav — 通用空间感知 VLN（占据预测）

- **arXiv**: [2603.09163](https://arxiv.org/abs/2603.09163)
- **发布日期**: 2026-03-10
- **核心方法**: 端到端 FM，通过 RGB 视频流注入通用 3D 空间感知
- **技术亮点**:
  - 通过 **occupancy prediction** 任务提取跨场景空间先验
  - 发现**单个 token** 即可编码导航所需的粗粒度空间线索
  - 不依赖显式深度传感器或 3D 重建
- **对标模块**: `空间理解 (P0)` + `可通行空间判断 (P1-medium)`
- **借鉴价值**: 单 token 空间编码对 4B student 模型的输入压缩有直接参考；occupancy prediction 作为预训练任务可考虑加入 teacher 训练。

---

### 7. CMMR-VLN — 持续多模态记忆检索 VLN（经验记忆）

- **arXiv**: [2603.07997](https://arxiv.org/abs/2603.07997)
- **发布日期**: 2026-03-10
- **核心方法**: 跨模态映射，从语言到空间导航的持续记忆检索
- **技术亮点**:
  - 模仿人类导航者的经验回忆能力
  - 在导航过程中识别并应用相关先验知识
  - 解决 LLM-based VLN agent 缺乏经验复用的问题
- **对标模块**: `个性化/行为长期记忆 (P2)` + `基础空间记忆 (P0)`
- **借鉴价值**: 经验检索机制可参考设计你的 P2 阶段显式记忆库的检索策略（物品位置先验、历史搜索结果回写）。

---

### 8. PiJEPA — 策略引导世界模型规划（世界模型）

- **arXiv**: [2603.25981](https://arxiv.org/abs/2603.25981)
- **发布日期**: 2026-03-25
- **核心方法**: 两阶段框架，结合学习策略 + 潜在世界模型进行指令条件视觉导航
- **技术亮点**:
  - 策略引导的 action initialization 解决高维动作空间搜索难题
  - 系统对比 **DINOv2** vs **V-JEPA-2** 作为视觉编码器
  - 真实世界导航任务验证
- **对标模块**: `world_state_memory` + 规划能力
- **借鉴价值**: 世界模型的 "想象-规划-执行" 循环与你的 `world_state_memory → decision_orchestration → NFM` 链路有结构对应；DINOv2/V-JEPA-2 对比结论可直接用于你的视觉编码器选型。

---

### 9. MA-CoNav — 主从多 Agent 长程 VLN（任务编排）

- **arXiv**: [2603.03024](https://arxiv.org/abs/2603.03024)
- **发布日期**: 2026-03-03
- **核心方法**: Master-Slave 多 Agent 框架 + 层级协作 + 双层反思
- **技术亮点**:
  - **Plan-Perceive-Act-Evaluate** 闭环
  - Master Agent 接收指令 → Task Planning Agent 生成子任务序列 → Observation Agent 获取环境语义 → Control Execution Agent 执行
  - 双层反思机制纠正执行偏差
- **对标模块**: `decision_orchestration` 任务分解 + 推进 + 恢复
- **借鉴价值**: Plan-Perceive-Act-Evaluate 闭环直接对标你的 `decision_orchestration` 设计；子 Agent 分工模式可参考设计你的 cloud/edge 协同链路中不同模块的职责划分。

---

### 10. Spatially Grounded Long-Horizon Planning（长程规划）

- **arXiv**: [2603.13433](https://arxiv.org/abs/2603.13433)
- **发布日期**: 2026-03-13
- **核心方法**: 空间可执行性约束下的长程动作规划
- **技术亮点**:
  - 定义 **grounded planning** 任务：规划必须同时满足语义正确 + 空间可执行
  - **GroundedPlanBench** benchmark 联合评估分层子动作规划 + 空间动作接地
  - **V2GP** 自动数据生成框架，从真实机器人视频示教生成训练数据
- **对标模块**: 长程任务规划 + 空间可执行性验证
- **借鉴价值**: 你的"跨步骤复合任务"（如去药盒→找爷爷→提醒服药）需要每步规划都空间可执行，GroundedPlanBench 的评估思路可直接借鉴。

---

## Tier 3：参考价值（4 篇）

| # | 论文 | arXiv | 对标模块 | 一句话亮点 |
|---|------|-------|----------|-----------|
| 11 | [Implicit Geometry VLN from Web Videos](https://arxiv.org/abs/2603.09259) | 2603.09259 | 训练数据扩展 | 从房间巡游视频提取隐式几何，无需 3D 重建，解决数据稀缺 |
| 12 | [Language-Conditioned World Modeling](https://arxiv.org/abs/2603.26741) | 2603.26741 | world_state_memory | 扩散世界模型 + actor-critic，39k 轨迹 117k 指令数据集 |
| 13 | [NavTrust: Trustworthiness Benchmark](https://arxiv.org/abs/2603.19229) | 2603.19229 | 蒸馏鲁棒性 | Teacher-Student 蒸馏 vs 数据增强 vs adapter 系统对比 |
| 14 | [Large Reward Models](https://arxiv.org/abs/2603.16065) | 2603.16065 | RL 避障优化 (P1-low) | VLM 作 reward 生成器：过程奖励 + 完成奖励 + 时间对比奖励 |

---

## 主线模块 → 论文速查表

| 主线模块 | Tier 1 | Tier 2 | Tier 3 |
|----------|--------|--------|--------|
| **P0 空间理解** | PROSPECT, DyGeoVLN | SPAN-Nav | — |
| **P0 地图理解** | RAGNav | — | — |
| **P0 基础空间记忆** | RAGNav | CMMR-VLN | — |
| **P1-high 搜索恢复** | EmergeNav | — | — |
| **P1-high 多视角** | PROSPECT | — | — |
| **P1-medium 深度/几何** | DyGeoVLN | SPAN-Nav | — |
| **P1-low RL 优化** | — | — | Large Reward Models |
| **P2 个性化记忆** | — | CMMR-VLN | — |
| **27B→4B 蒸馏** | Embodied FM at Edge | — | NavTrust |
| **decision_orchestration** | — | MA-CoNav, Grounded Planning | — |
| **world_state_memory** | — | PiJEPA | Lang-Cond World Model |
| **训练数据** | — | — | Implicit Geometry VLN |
