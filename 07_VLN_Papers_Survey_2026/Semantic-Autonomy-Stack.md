---
title: "A Semantic Autonomy Framework for VLM-Integrated Indoor Mobile Robots: Hybrid Deterministic Reasoning and Cross-Robot Adaptive Memory"
method_name: "Semantic Autonomy Stack"
authors: [Bogdan Felician Abaza, Andrei-Alexandru Staicu, Cristian Vasile Doicin]
year: 2026
venue: arXiv
tags: [vlm-navigation, indoor-navigation, semantic-reasoning, edge-deployment, cross-robot-transfer, ros2, dual-process-reasoning, semantic-memory]
zotero_collection: ""
image_source: online
arxiv_html: ""
created: 2026-05-06
---

# 论文笔记：A Semantic Autonomy Framework for VLM-Integrated Indoor Mobile Robots

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Faculty of Industrial Engineering and Robotics (FIIR), National University of Science and Technology POLITEHNICA Bucharest |
| 日期 | May 2026 |
| 项目主页 | [GitHub (数据集)](https://github.com/bogdan-abaza/nav2-sas-vlm-memory) |
| 对比基线 | [[IROS]], [[Hydra-Nav]], [[CausalNav]], [[ROSClaw]], [[ROS-LLM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.02525) |

---

## 一句话总结

> 提出六层 Semantic Autonomy Stack (SAS)，用七步参数化确定性解析器处理 88% 导航指令（<0.1ms，无 VLM/相机/GPU），仅歧义指令升级 VLM；五类语义记忆 (M1-M5) + 编译摘要实现跨会话学习和跨机器人知识迁移（103,000x 延迟降低），在 2 台物理机器人上验证 100% 语义迁移准确率。

---

## 核心贡献

1. **Semantic Autonomy Stack (SAS)**: 六层参考框架 (L0-L5)，将平台硬件到运营智能分层解耦，每层定义能力而非实现，支持组件替换
2. **混合确定性-VLM 推理 (L3a/L3b)**: 七步参数化解析器（含 M3 偏好匹配）处理 88% 指令无需 VLM；仅歧义指令升级到 L3b VLM 推理
3. **跨机器人自适应记忆 (L5)**: 五类语义记忆 M1-M5 + 显式作用域分类（全局/每操作员/每机器人），M3 偏好提升机制 + 编译摘要实现跨机器人迁移
4. **物理多机器人验证**: 2 台异构差速驱动机器人（RPi5），82 场景级决策，3 会话，100% 语义迁移准确率，零训练数据

---

## 问题背景

### 要解决的问题
室内移动机器人可用 ROS 2 Nav2 导航到度量坐标，但无法理解"带我去可以坐下休息的地方"这类表达意图的自然语言指令。

### 现有方法的三大挑战
- **Challenge 1 - 推理延迟**: VLM 推理在消费级硬件上需 2-9 秒/决策，对"去实验室"等明确指令来说完全不必要
- **Challenge 2 - 会话失忆**: VLM 系统每次会话从零开始，运行中学到的知识（操作员偏好、成功路线）在重启后丢失
- **Challenge 3 - 缺乏跨机器人迁移**: 物理多机器人验证稀少，学到的语义关联无法在共享地图的机器人间传递

### 本文的动机
基于认知科学的双过程理论（System 1 快速自动 vs System 2 慢速审慎），用分层架构将 VLM 使用最小化：大多数指令通过确定性级联快速解析，仅真正歧义指令调用 VLM；通过语义记忆将 VLM 交互经验提升为确定性规则，实现跨会话和跨机器人复用。

---

## 方法详解

### 系统架构: 六层 Semantic Autonomy Stack

| 层 | 名称 | 职责 | 本文实现 |
|----|------|------|---------|
| L0 | Mobile Platform | 运动、传感器、电源、计算 | Xplorer-B (单 RPi5) / Xplorer-C (双 RPi5), 4WD 差速驱动 |
| L1 | Navigation Execution | 定位、路径规划、避障 | Nav2 Jazzy 1.3.10, AMCL + EKF @ 20Hz, Route Server (24 节点, 60 边), MPPI |
| L2 | Perception | 物体检测、传感器融合、语义地图 | 单目相机 + 2D LiDAR, YOLOv26n (C++ NCNN, 5.5Hz), Angular Sector Fusion @ 3Hz, 18 静态 POI |
| L3 | Semantic Reasoning | 意图解析、目标选择 | **L3a**: 七步确定性解析 (<0.1ms) / **L3b**: VLM 推理 (Qwen 3.5:4b, 2-9s) |
| L4 | Mission Interface | 外部指令接入 | `/vlm_instruction` ROS 2 topic + OpenClaw TUI |
| L5 | Operational Intelligence | 跨会话学习、记忆管理 | M1-M5 语义记忆 + 编译摘要 (≤500 tokens) |

**设计原则**: 每层定义 I/O 接口和能力要求，不规定具体算法/模型/数据格式。替换 L2 的 2D LiDAR 为 3D 深度相机、或 L3 的 Qwen 为 Gemma，均不影响相邻层。

### TABLE 1: 与现有架构对比

| 层 | Commercial AMRs | Research VLN | LLM-ROS frameworks | SAS (本文) |
|----|----------------|-------------|-------------------|-----------|
| L5 | Fleet analytics, KPIs | - | Generic memory plugins | M1-M5 with scope taxonomy |
| L4 | VDA 5050, REST APIs | Natural language | Generic tool-calling | NL + TUI + extensible |
| L3 | - | LLM/VLM (full pipeline) | LLM with tool-use | **Hybrid L3a/L3b** |
| L2 | Proprietary, optimized | Simulated | Standard ROS topics | ASF + YOLO |
| L1 | Optimized, proprietary | Simulated Nav2 | Standard Nav2 | Nav2 + Route Server |
| Cross-robot memory | Fleet-level coordination | Not demonstrated | Not demonstrated | **Demonstrated (100%)** |
| Physical validation | Production environments | Predominantly simulation | Lab demonstrations | **Multi-robot, multi-session** |

### 核心模块

#### 模块1: L3a 七步确定性解析器

接收自然语言指令，尝试在导航图 $G$、静态 POI 集 $S$、机器人位姿 $P$ 和编译偏好摘要 $D$ 上匹配。七步按固定顺序评估，首个匹配步骤立即返回；无匹配则升级到 L3b。

**预处理 - 语言归一化**: 转为小写，下划线替换为空格，去除变音符号（支持罗马尼亚语），合并空白。

**Algorithm 1: L3a Deterministic Instruction Resolver**

```
输入: I (指令), G (导航图), S (静态 POI), P (机器人位姿 x,y,yaw), D (M3 编译摘要)
输出: (node_id, method) 或 ESCALATE_TO_L3B

Step 0 - Preference match (M3):
    tokens ← tokenize(normalize(I)) \ STOPWORDS
    for each preference p in D.promotions:
        for each example e in p.examples:
            j ← jaccard(tokens, tokenize(e))
            if j ≥ 0.6:
                candidates ← candidates ∪ {(j, p.node_id)}
    if single unambiguous match:
        return (node_id, "L3a_m3_preference")

Step 1 - Explicit node ID:
    if regex matches "node (N)" or "nodul (N)" in I:
        if N exists in G:
            return (N, "L3a_deterministic")

Step 2 - Node name match:
    matches ← {n ∈ G | normalize(n.name) ⊂ I or I ⊂ normalize(n.name)}
    if |matches| = 1:
        return (matches[0], "L3a_deterministic")

Step 3 - Object ID match:
    matches ← {o ∈ S | normalize(o.obj_id) ⊂ I}
    if |matches| = 1:
        return (o.nearest_node, "L3a_deterministic")

Step 4 - Attribute scoring:
    for each o ∈ S:
        score(o) ← count of attribute values matching words in I
    if single highest-scoring object:
        return (o.nearest_node, "L3a_deterministic")

Step 5 - Single-class match:
    classes ← {c | c.name appears in I, at least one o ∈ S has class c}
    if |classes| = 1:
        objs ← static objects of that class
        if |objs| = 1:
            return (o.nearest_node, "L3a_deterministic")

Step 6 - Proximity + class:
    if proximity keyword in I ("closest", "nearest", "cel mai apropiat"):
        if |classes| = 1 and |objs| > 1:
            nearest ← argmin_{o ∈ objs} distance(P, o)
            return (nearest.nearest_node, "L3a_deterministic")

return ESCALATE_TO_L3B
```

**关键性质**: 给定相同的图、POI 集、位姿和摘要，解析器总是产生相同结果（确定性）。无需训练数据、无 GPU、无网络。

#### 模块2: L3b VLM 视觉-语义推理

仅当 L3a 七步均无法产生明确匹配时激活。

**流程**: (i) 通过 HTTP bridge 从机器人获取相机图像 → (ii) 构建结构化 prompt → (iii) 送入 VLM 推理

**VLM 配置**:
- 模型: Qwen 3.5:4b via Ollama (localhost:11434)
- 硬件: NVIDIA RTX 5050 GPU (操作员工作站)
- 参数: temperature=0.3, num_predict=500, num_ctx=16384, think=false

**Prompt 结构** (6 节):
1. **Memory prefix** (0-500 tokens): 编译摘要中的 top entities + spatial patterns
2. **Current state**: 机器人 AMCL 位姿 (x, y, yaw)
3. **Navigation graph**: 所有节点 ID、名称和坐标
4. **Semantic objects by zone**: 按最近图节点分组的物体，含属性、来源 (static/YOLO)、距离
5. **Designated seating/resting locations**: 具有 seating=true 或 use=resting_spot 属性的物体专区
6. **Task and rules**: 指令 + 显式规则（优先静态物体、图内节点选择等）

**输出**: JSON `{node_id: int, reason: string}`，经 markdown code block 和 chain-of-thought 标签清洗后解析。

#### 模块3: 视觉确认 (Post-Arrival)

机器人到达 Nav2 FollowPath 报告的目标坐标后，执行视觉验证确认目标物体在场。

**TABLE 3: 确认类型路由**

| Signature Type | 方法 | 使用相机 | 示例 POI | 理由 |
|---------------|------|--------|---------|------|
| *landmark* | VLM 验证命名物体 | Yes | radiator_main, plant_5 | 到达时可见的显著物体 |
| *contextual* | VLM 验证周边线索 | Yes | plant_2 (near doorway) | 物体本身不显著，上下文标识位置 |
| *architectural* | VLM 验证空间布局 | Yes | corridor intersection (future) | 无特定物体，布局确认位置 |
| *none* | AMCL 位姿 + 半径检查 | No | restroom_men, restroom_women | 无视觉签名（墙壁标志在相机高度外） |

确认结果独立于导航结果记录：`mission_complete` 基于 Nav2 FollowPath 成功，`confirmation` 是审计日志中的独立字段。

#### 模块4: Executive Contract ⟨A, O, V, L⟩

采自 ROSClaw 框架，特化用于语义导航：

- **[A] Affordance Manifest**: 每个决策周期构建——导航图节点 + 合并语义物体 (static POI + YOLO) + 安全策略允许的动作 (ComputeRoute, FollowPath, NavigateToPose, Spin)
- **[O] Observation Normalizer**: 统一 AMCL 位姿 + 语义物体清单 + 可选相机图像 (base64 JPEG, 仅 L3b)
- **[V] Action Validator**: 3 步安全检查——(1) 动作类型在白名单内, (2) 目标节点存在于图中, (3) 目标距离 < 50m。实验中所有 VLM 提议的节点均有效（零幻觉）
- **[L] Audit Logger**: JSONL 结构化条目——时间戳、指令、解析方法、目标节点、验证结果、导航结果、计时分解、平台 ID、确认数据

#### 模块5: 五类语义记忆 (M1-M5)

**TABLE 4: 记忆类别与作用域分类**

| 类别 | 内容 | 作用域 | 共享策略 | 更新源 | 存储 |
|------|------|--------|---------|--------|------|
| M1 Environment | 实体访问统计: 计数、成功率、导航时间、置信度、位置分布 | Global (per-map) | 同地图所有机器人 | 审计日志: node_id + confirmation | JSONL, per POI |
| M2 Temporal | 重复视觉观察按关键词重叠聚类 (Jaccard ≥ 0.5) | Global (per-map) | 所有机器人 | 审计日志: confirmation.scene_context | JSONL, per pattern |
| M3 Operator prefs | 指令→node_id 映射，含频率、一致性、方法分布 | Global (per-operator) | 同操作员所有机器人 | 审计日志: instruction + node_id + method | JSONL, per instruction group |
| M4 Platform caps | 硬件配置、软件版本、传感器、计算架构 | Per-robot | 不共享 | navigator_startup 事件 | JSONL, per platform |
| M5 Task history | 每决策摘要: 指令、解析、结果、异常标志 | Per-robot (aggregatable) | 按 platform_id 归属 | 所有决策事件 | JSONL, all decisions |

**M3 提升机制** (核心学习循环):

对每个唯一指令（归一化后），记忆提取器追踪:
- **Frequency**: 该指令被给出的次数（跨所有会话和平台）
- **Dominant node**: 最频繁解析到的目标节点
- **Consistency**: 解析到主导节点的比例
- **Method distribution**: L3a_deterministic / L3a_m3_preference / L3b_vlm 的计数

**提升为 L3a 偏好的三个条件**（全部满足时 `ready_for_l3a_promotion = true`）:

$$\text{promote} \iff \begin{cases} \text{frequency} \geq 3 \\ \text{consistency} \geq 0.80 \\ \text{method\_counts}[\text{L3b\_vlm}] \geq 1 \end{cases}$$

条件 3 确保只有真正需要 VLM 语义推理的指令被提升（"go to node 5" 这类不应创建 M3 偏好）。

#### 模块6: 编译摘要 (Compiled Digest)

运行时工件，将 M1-M5 跨会话知识压缩为单个 JSON 文件（≤500 tokens / ≤2000 字符），在 VLM 上下文窗口内注入。

**四节结构**:
1. **Top entities** (from M1): 5 个最常访问 POI + 访问次数 + 成功率 + 签名类型
2. **Top patterns** (from M2): 3 个最高置信度空间模式（如"node 20 附近，相机通常看到 floor, plant, tiled window"）
3. **L3a promotions** (from M3): 已提升的指令→节点映射，被 L3a Step 0 Jaccard 匹配消费
4. **Global statistics** (from M5): 总任务数 + 整体成功率

**大小管理**: 超过 2000 字符限制时，优先裁剪 entities → patterns；promotions 永不裁剪。

#### 模块7: 跨机器人迁移机制

**9 步流水线**:
1. 操作员向 Robot C 发出指令
2. L3b VLM 解析 → 目标节点
3. 决策记录到审计日志 (platform_id=xplorer-c)
4. 记忆提取器运行 → M3 提升 → 编译摘要更新
5. Robot B 启动时加载同一编译摘要
6. 同一指令到达 Robot B → L3a Step 0 Jaccard 匹配
7. 确定性解析 <0.1 ms, 无 VLM, 无相机

**结果**: 6,733 ms (VLM on C) → 0.065 ms (M3 on B) = **102,923x 加速**

---

## 关键图表

### Figure 1: SAS 六层参考框架

**说明**: L0 (Mobile Platform) → L1 (Navigation Execution) → L2 (Perception) → L3 (Semantic Reasoning, split L3a/L3b) → L4 (Mission Interface) → L5 (Operational Intelligence & Memory, M1-M5)。实线为运行时数据流，虚线为离线记忆循环。

### Figure 2: 混合推理流程图

**说明**: 自然语言指令首先经 L3a 七步确定性级联（快速路径 <0.1ms）。首个匹配步骤立即返回。若无步骤匹配，升级到 L3b（慢速路径 2-9s）：获取相机 → 构建 prompt → VLM 推理 → 解析 node_id。两条路径汇聚于 Executive Contract (A,O,V,L) → Nav2 执行 → 到达后视觉确认 → 审计日志。

### Figure 3: M1-M5 记忆作用域 + 跨机器人迁移

**说明**: (a) 全局共享 (M1 Environment, M2 Temporal, M3 Operator prefs) vs 每机器人 (M4 Capabilities, M5 Task history)。Memory extractor 离线重建 → Compiled digest (≤500 tokens)。(b) 迁移流程: Xplorer-C VLM 学习 → 审计日志 → M3 提升 → 更新摘要 → Xplorer-B 启动加载 → 确定性解析。6,733 ms → 0.065 ms。

### Figure 5: 导航图拓扑

**说明**: FIIR 二楼走廊。24 节点 + 60 有向边（30 双向对）。POI 节点按语义类别着色：蓝=基础设施（实验室、洗手间），红=安全（紧急出口、消防设备），绿=环境（植物、窗户）。

### Figure 7: 多机器人物理部署

**说明**: Xplorer-B (单 RPi5) 和 Xplorer-C (双 RPi5) 同时运行。L0-L2 在各机器人 RPi5 上 (CPU-only)；L3/L5 在共享 GPU 工作站上。蓝色箭头=HTTP 上下文请求，绿色箭头=ROS 2 DDS 导航命令，紫色虚线=离线记忆循环。两台机器人共享同一编译摘要。

---

## 实验

### 实验设置

#### TABLE 5: 平台规格

| 组件 | Xplorer-B | Xplorer-C |
|------|-----------|-----------|
| 导航计算 | Raspberry Pi 5, 16GB, ARM Cortex-A76 | 同 |
| 推理计算 | 同一 RPi5 (单板) | 第二 RPi5 via Ethernet (双板) |
| 相机 | Logitech C920 HD, 640×480, USB | 同 |
| 2D LiDAR | STL-19P (LD19), 360°, ~503 rays, 10Hz | 同 |
| IMU | BNO055, 9-DoF, I2C | - |
| 驱动 | 4WD 差速, goBILDA, 正交编码器 | 4WD 差速, 霍尔编码器 |
| 电机控制 | 2× RoboClaw 2×15A, UART | 2× L298N + ESP32 |
| 架构标签 | *single-rpi5* | *dual-rpi5* |
| VLM 工作站 | Laptop: Intel Core Ultra 7 255H, NVIDIA RTX 5050, 32GB RAM, Ubuntu 24.04, ROS 2 Jazzy | 同 (共享) |

#### 环境: FIIR 二楼走廊

- 导航图: 24 节点, 60 有向边 (30 双向对), GeoJSON 编码
- 18 个静态 POI, 8 个物体类别

#### TABLE 7: 静态 POI 摘要

| 类别 | 数量 | 签名类型 | 关键属性 |
|------|------|---------|---------|
| potted plant | 5 | 4 landmark, 1 contextual | near_window |
| laboratory | 4 | 4 landmark | room_number, department, access, entrance |
| restroom | 2 | 2 none | type (men/women) |
| emergency_exit | 2 | 2 landmark | type (entrance/exit) |
| window | 2 | 2 landmark | opens, side (north/south) |
| fire hydrant | 1 | 1 landmark | type (wall_mounted) |
| radiator | 1 | 1 landmark | - |
| chair | 1 | 1 landmark | seating=true, use=resting spot |
| **总计** | **18** | **15 landmark, 2 none, 1 contextual** | - |

### 实验设计: 三会话结构

#### TABLE 8: 实验会话与场景

| Session | Robot | 目的 | 日期 | Navigator | 场景 (N) |
|---------|-------|------|------|-----------|---------|
| A | Xplorer-C | M3 确认 + S3new 学习循环 | Apr 22-23, 2026 | v4.7.4→v4.8 | S1(6), S2(5), S3old(5), S3new(7), S4(3), S5(11) |
| B | Xplorer-B | 跨机器人迁移验证 | Apr 23, 2026 | v4.8 | S1(14), S2(10), S3old(6), S3new(3), S4(5), S5(3) |
| C | Both | 并发操作可行性 | Apr 24, 2026 | v4.8 | 2 pairs × 2 robots = 4 |
| **Total** | | | | | **82 scenario-level decisions** |

**五个语义场景**:
- **S1 (restroom)**: "go to a place to take a short break for personal needs" → L3a_m3_preference → node 0
- **S2 (fire safety)**: "the washroom got on fire, help me find something to stop the fire" → L3a_m3_preference → node 8
- **S3old (fresh air)**: "it is too hot in here, take me somewhere I can get some fresh air" → L3a_m3_preference → node 14
- **S3new (learning cycle)**: "Take me somewhere I can sit and relax" → L3b_vlm (C) → L3a_m3_preference (B after promotion)
- **S4 (deterministic)**: "go to lab_cb204" → L3a_deterministic (Step 2, node name match) → node 5
- **S5 (proximity)**: "Take me to the closest plant" → L3a_deterministic (Step 6, proximity + class) → varies by pose

### 核心结果

#### 7.1 学习循环: VLM → 确定性解析

**TABLE 10: S3new 指令 VLM 学习过程 (Xplorer-C)**

| 决策 | 目标节点 | 正确? | VLM 时间 (ms) | 导航结果 |
|------|---------|-------|-------------|---------|
| 1 | 9 (radiator_main) | ✓ | 8,552 | mission_complete |
| 2 | 9 (radiator_main) | ✓ | 8,203 | mission_complete |
| 3 | 9 (radiator_main) | ✓ | 8,935 | missed (nav) |
| 4 | 9 (radiator_main) | ✓ | 2,566 | mission_complete |
| 5 | 15 (lab_chair) | ✗ | 7,921 | mission_complete |
| 6 | 9 (radiator_main) | ✓ | 5,317 | mission_complete |
| 7 | 9 (radiator_main) | ✓ | 5,634 | mission_complete |
| **Summary** | | **6/7 (86%)** | **6,733 ± 2,148** | |

提升条件验证: frequency=8 ≥ 3 ✓, consistency=6/7=0.86 ≥ 0.80 ✓, L3b_vlm count ≥ 1 ✓ → 提升到 M3 digest

#### 7.2 跨机器人记忆迁移

**TABLE 11: Session B - Xplorer-B 迁移结果**

| 类别 | 指令 | N | 解析方法 | 目标节点 | 语义准确率 | Resolve (ms) | Nav 成功 |
|------|------|---|--------|---------|---------|-------------|---------|
| Seed-promoted | "...short break for personal needs" | 14 | L3a_m3_preference | 0 (toilet_m) | 14/14 | 0.063 ± 0.008 | 10/14 (71%) |
| Seed-promoted | "...washroom got on fire..." | 10 | L3a_m3_preference | 8 (cb202) | 10/10 | 0.070 ± 0.010 | 10/10 (100%) |
| Seed-promoted | "...too hot...fresh air..." | 6 | L3a_m3_preference | 14 (plant_5) | 6/6 | 0.066 ± 0.010 | 5/6 (83%) |
| **Learning cycle** | **"...sit and relax"** | **3** | **L3a_m3_preference** | **9 (radiator)** | **3/3** | **0.060 ± 0.005** | **3/3 (100%)** |
| Deterministic | "go to lab_cb204" | 5 | L3a_deterministic | 5 (lab_cb204) | 5/5 | 0.101 ± 0.041 | 5/5 (100%) |
| Deterministic | "Take me to the closest plant" | 3 | L3a_deterministic | 19 (plant_3) | 3/3 | 0.400 ± 0.054 | 3/3 (100%) |
| **All M3** | | **33** | | | **33/33 (100%)** | **0.065 ± 0.009** | **28/33 (85%)** |
| **All deterministic** | | **8** | | | **8/8 (100%)** | **0.213 ± 0.124** | **8/8 (100%)** |
| **Total** | | **41** | | | **41/41 (100%)** | | **36/41 (88%)** |

**关键发现**: 语义解析准确率 41/41 = 100%，但导航完成率 36/41 = 88%。5 次导航失败的 XY 误差均值 3.77m (vs 成功的 0.30m)，归因于 AMCL 定位精度而非 L3 目标选择错误。Fisher's exact test (p=0.563) 确认解析方法与导航成功率无显著关联。

#### 7.3 确定性一致性

- **S4** ("go to lab_cb204"): 两台机器人均通过 L3a Step 2 解析到 node 5 (5/5 + 3/3)
- **S5** ("closest plant"): 两台机器人均通过 L3a Step 6 解析到 node 19 (plant_3) (via proximity 计算)
- 确定性解析时间显著高于 M3 偏好（Mann-Whitney U=254, p<0.001），因 M3 在 Step 0 短路，而确定性解析需遍历 Steps 1-6

#### 7.4 并发操作

**TABLE 12: Session C - 并发结果**

| Pair | Xplorer-B | B node | B outcome | Xplorer-C | C node | C outcome |
|------|-----------|--------|-----------|-----------|--------|-----------|
| 1 | S2 (fire) | 8 ✓ | mission_complete | S1 (restroom) | 0 ✓ | mission_complete |
| 2 | cb203 entrance | 22 ✓ | mission_complete | lab_cb204 | 5 ✓ | mission_complete |

两台机器人使用相同 digest (MD5: 97241265)，隔离 DDS domain (默认 vs 67)，无跨域串扰。

### 统计检验汇总

**TABLE 13: 统计检验结果**

| 检验 | 变量 | 统计量 | p值 | 解释 |
|------|------|--------|-----|------|
| Exact binomial (Clopper-Pearson) | M3 迁移准确率: 33/33 | - | - | 100%, 95% CI [0.894, 1.000] |
| Mann-Whitney U | L3b 时间 (C, n=7) vs L3a 时间 (B, n=33) | U=231 | 2.12×10⁻³ | 高度显著延迟降低 |
| Cohen's d | L3b vs L3a 时间 | d=7.30 | - | 超大效应量 |
| Shapiro-Wilk | M3 解析时间 (B, n=33) | W=0.977 | 0.683 | 正态分布 |
| Shapiro-Wilk | 确定性解析 (B, n=8) | W=0.809 | 0.036 | 非正态 (Step 6 proximity 计算代价) |
| Fisher's exact | M3 vs Det 导航成功率 (B) | - | 0.563 | 无显著关联 |
| Mann-Whitney U | 确定性 vs M3 解析时间 (B) | U=254 | <0.001 | M3 (Step 0) 显著快于 Steps 1-6 |

### 延迟对比 (Figure 10)

| 指标 | L3b VLM (C) | L3a M3 (B) |
|------|------------|------------|
| n | 7 | 33 |
| Mean (ms) | 6,733 | 0.065 |
| SD (ms) | 2,320 | 0.010 |
| Min (ms) | 2,566 | 0.047 |
| Max (ms) | 8,935 | 0.089 |
| **Speedup** | | **102,923×** |
| Mann-Whitney U | 231 | |
| p-value | 2.12×10⁻³ | |
| Cohen's d | 7.30 | |
| Accuracy | | 33/33 |
| 95% CI | | [0.894, 1.0] |

### 与相关工作对比

**TABLE 14: 与相关系统定性对比**

| 维度 | IROS | Hydra-Nav | CausalNav | ROSClaw | ROS-LLM | Sensors [7] | **本文** |
|------|------|-----------|-----------|---------|---------|-------------|---------|
| 快速解析机制 | 训练 RL 分类器 | Adaptive CoT in VLM | - | - | - | N/A (L1-L2) | **7-step parametric cascade** |
| 快速路径率 | 53.6% | Not reported | - | - | - | - | **88%** |
| VLM/LLM 角色 | With bypass | Unified (single VLM) | Implicit (RAG) | Generic tool-use | Task programming | - | **Selective (L3b only when L3a fails)** |
| 记忆类别 | - | - | Causal graph (1) | Generic plugin | - | - | **M1-M5 (5 with scope taxonomy)** |
| 跨会话学习 | - | - | Yes (graph update) | Plugin-dependent | - | - | **Yes (M3 promotion)** |
| 跨机器人迁移 | - | - | - | - | - | - | **Yes (33/33 = 100%)** |
| 物理验证 | 5 buildings | Simulation | Sim + real | 3 platforms, 115 legs | Lab tasks | 3 robots, 115 legs | **2 robots, 82 decisions, 3 sessions** |
| 训练需求 | RL | RL fine-tuning | Model-specific | None | Imitation learning | None | **None** |

---

## 批判性思考

### 优点
1. 双过程架构设计精妙: 88% 指令在 <0.1ms 内确定性解析，极大降低了 VLM 部署的实际门槛
2. M3 提升机制的三条件设计（frequency ≥ 3, consistency ≥ 0.80, L3b count ≥ 1）确保只有真正需要语义推理的指令被提升
3. 跨机器人迁移 100% (33/33) 且零代码修改/零训练，仅需共享编译摘要 + 兼容图/POI
4. 语义准确率 (100%) 与导航完成率 (88%) 的分离分析优雅验证了分层架构: L3 正确性独立于 L1 执行
5. 在 Raspberry Pi 5 (无 GPU) 上运行，对端侧 VLN 部署有直接参考价值
6. Executive Contract ⟨A,O,V,L⟩ 提供模型无关性——更换 VLM 仅需修改一个常量
7. 审计日志设计完善（JSONL + digest MD5 hash），支持完全可复现

### 局限性
1. **单一环境**: 仅 FIIR 二楼走廊（24 节点, 18 POI），未验证更大/多层/多房间环境的可扩展性
2. **Jaccard 词法匹配**: M3 偏好匹配基于 token 集合相似度，语义等价但词汇不同的释义（"I need to use the bathroom" vs "go to a place to take a short break"）可能无法匹配
3. **VLM 非确定性**: S3new 学习循环中 86% 一致性（6/7），接近 0.80 阈值的指令可能在提升/非提升间振荡
4. **相机视野限制**: 前向相机安装高度 16cm，无法确认墙壁安装物体（如 1.5m 高消防栓），所有 S2 确认均报告 confirmed=false
5. **仅 Qwen 3.5:4b**: Executive Contract 声称模型无关，但未在其他模型族 (7B, 14B, Gemma, LLaVA) 上实验验证
6. **未测试并发 VLM 序列化**: 82 决策中仅 12% 调用 L3b，并发会话中未出现同时 VLM 升级
7. **82 决策的统计规模有限**: 每场景仅 3-14 次试验

### 潜在改进方向
1. 用轻量级句子嵌入模型（如 all-MiniLM-L6-v2, 22M 参数, <10ms on CPU）替换 Jaccard 进行 M3 匹配
2. 在标准 VLN 数据集上评估确定性-VLM 混合推理策略的泛化能力
3. 扩展到 Unitree Go2 四足机器人验证 L0 平台无关性
4. 将 L3 迁移到 NVIDIA Jetson Orin 实现完全边缘部署
5. 研究 M1-M5 记忆在异构环境（不同地图/不同 POI 集）间的迁移策略
6. 实现基于 REST API 的共享记忆服务器支持舰队级部署

### 可复现性评估
- [x] 实验数据集公开 (GitHub: 审计日志, M1-M5 记忆, 编译摘要, 分析脚本)
- [ ] 导航器/合约/记忆提取器源码（需联系作者）
- [x] 实验设置完整（平台规格, 图拓扑, POI 配置, 会话设计）
- [x] 物理多机器人验证
- [x] 统计检验预注册 + 完整报告

---

## 关联笔记

### 基于
- [[ROS 2 Nav2]]: 度量导航框架 (L1)
- [[ROSClaw]]: Executive Contract ⟨A,O,V,L⟩ 模板
- [[Dual Process Theory]]: Kahneman System 1/System 2（L3a/L3b 设计理论基础）
- [[YOLO]]: YOLOv26n 物体检测 (L2)
- [[Qwen]]: Qwen 3.5:4b VLM (L3b)

### 对比
- [[IROS]]: 双过程 VLM 导航（RL 分类器路由, 53.6% 快速路径率）
- [[Hydra-Nav]]: 自适应 CoT 切换（单 VLM 内, RL 训练）
- [[CausalNav]]: 因果语义图 + LLM-RAG 导航
- [[EmergeNav]]: 零样本 VLN（纯 VLM 路线）
- [[CROSS]]: 拓扑语义地图（表示学习路线）

### 方法相关
- [[Vision-Language Model]]: VLM 推理
- [[Semantic Memory]]: 语义记忆系统
- [[Cross-Robot Transfer]]: 跨机器人知识迁移
- [[Edge Deployment]]: 端侧部署
- [[Topological Map]]: 拓扑导航图

---

## 速查卡片

> [!summary] Semantic Autonomy Stack
> - **核心**: 六层 SAS (L0-L5) + 七步确定性解析器 (L3a) + 选择性 VLM (L3b) + 五类记忆 (M1-M5)
> - **快速路径**: 88% 指令 <0.1ms 确定性解析, 无 VLM/相机/GPU
> - **学习循环**: L3b VLM → M3 promotion → L3a Step 0 Jaccard → 103,000x 加速
> - **迁移**: 100% 语义准确率 (33/33, 95% CI [0.894, 1.000])
> - **硬件**: 2× RPi5 机器人 (无 GPU) + 共享 RTX 5050 工作站
> - **结果**: 82 decisions, 3 sessions, 语义 100%, 导航 88%, Cohen's d=7.30
> - **数据**: [GitHub](https://github.com/bogdan-abaza/nav2-sas-vlm-memory)

---

*笔记创建时间: 2026-05-06*
