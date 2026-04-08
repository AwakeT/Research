# 家庭语义记忆系统——历史信息快速查询与模型数据注入策略

> 基于《地图分区及房间语义地图演进》第五节「家庭语义记忆系统」L1-L4 四层架构，进一步分析当前存储结构下的查询效率问题，并给出让 LLM/VLM 更高效接收记忆数据的具体方案。本文进一步与 Kinbot 后续 VLN 主线对齐：采用 `27B research teacher 持续迭代 + 多轮蒸馏 4B student` 的双轨路线，并将长期记忆明确定位为显式记忆库能力，而不是直接压入主模型参数。

---

## 0. 在 Kinbot VLN 主线中的定位

### 0.1 核心路线

Kinbot 后续 VLN 主线建议收敛为：

- **最终目标**：4B 左右 student 模型满足量产部署约束，作为端侧长期演进版本。
- **当前主策略**：27B teacher 承担能力上限探索、训练数据生成、补标与蒸馏教师职责。
- **能力演进方式**：每一项新能力先在 27B teacher 上稳定，再蒸馏为 4B 新版本，而不是先在 4B 上定义能力边界。

对应到记忆系统，这份文档讨论的不只是“怎么存”，而是**如何让显式家居记忆结构持续为 teacher 训练、student 蒸馏和线上推理提供统一的数据底座**。

### 0.2 这套记忆系统在整体方案中的职责

这套 L1-L4 记忆系统的职责应限定为：

- 为 P0 阶段提供空间理解、地图理解、区域排序、搜索恢复所需的结构化监督与推理上下文。
- 为 P1 阶段提供多视角一致性摘要、深度/几何中间约束摘要的统一接入点。
- 为 P1-low 阶段提供可供 RL 使用的结构化状态、搜索顺序与恢复结果摘要，用于效率优化而非替代主认知层。
- 为 P2 阶段提供显式长期记忆库，包括位置先验、活动区域先验、历史失败回写、记忆新鲜度与可信度管理。
- 作为训练数据组织层与推理注入层，服务 VLM/LLM，但**不替代**现有 SLAM、局部规划与安全链。

### 0.3 当前主线能力边界

当前阶段的主训练目标应限定为高层语义导航认知，而不是端到端底盘控制。也就是说，这套记忆系统主要支持以下输出：

- 指令到目标对象、目标区域、任务约束的理解
- 当前视角中的空间语义理解
- 当前观测与地图区域、地图拓扑的对齐理解
- 目标候选区域排序
- 搜索顺序建议
- 搜索失败后的恢复建议

当前阶段**不应**把以下内容作为主线目标：

- 直接输出底盘连续控制
- 替代现有 SLAM + 局部规划 + 安全链
- 将多视角、深度几何、长期记忆、强化学习在同一阶段统一混训
- 在 4B 主线能力尚未稳定前推进极限模型压缩

### 0.4 Teacher / Student 的输入契约

这套记忆系统需要同时服务两套不同约束：

| 角色 | 输入契约 | 目标 |
|---|---|---|
| 27B Research Teacher | 可使用更丰富传感器配置、多视角、更高分辨率、深度真值、更宽松 token 预算 | 探索能力上限，生成高质量判断与蒸馏监督 |
| 4B PDCP Student | 必须对齐量产输入基线，仅使用量产相机子集与受限 token 预算 | 满足端侧时延、功耗、频率与稳定性约束 |

因此，L3 的输出不能默认按“最强输入”构造，而必须显式区分：

- **Teacher 视图**：允许信息更丰富，面向能力探索、补标、教师监督。
- **Student 视图**：只保留量产输入可承载的字段，面向部署与蒸馏对齐。

---

## 1. 当前架构速览

```
┌─────────────────────────────────────────┐
│           L4：用户展示层                 │  ← 面向人，语义布局草图/房间卡片
├─────────────────────────────────────────┤
│           L3：记忆服务层                 │  ← 翻译层，按需从 L1+L2 组装
├─────────────────────────────────────────┤
│           L2：动态事件层                 │  ← 高频写入，Observation/ObjectState/Transition
├─────────────────────────────────────────┤
│           L1：语义结构层                 │  ← 低频写入，Room/Zone/AnchorObj/MovableObj
└─────────────────────────────────────────┘
```

L1 存节点与直接归属关系（JSON），L2 存时序事件（JSON），L3 不持久化，L4 由 L3 驱动生成。

---

## 2. 当前存储下的查询瓶颈分析

### 2.1 L1 关系跳转需要全量遍历

一个典型的"找药箱"查询链路：

```
Movable Object (medicine_box)
  → habitual_locations → room_004, room_005
    → Room Node (room_004) → 下属 Zone Nodes
      → Zone Node → remembered_objects 是否含 medicine_box
        → Anchor Objects → position_relative 辅助描述
```

当前 L1 以 flat JSON 数组存储所有节点，**每次跳转都是 O(n) 线性遍历 + ID 匹配**。当房间数 ~10、物品数 ~50 时尚可接受；当扩展到多层住宅或物品数上百时，查询延迟将线性增长。

### 2.2 L2 时间/空间过滤无索引

"最近 3 天客厅出现过什么？"的执行过程：

```
遍历全部 Observation Events
  → 过滤 timestamp ∈ [T-3d, T]
    → 过滤 zone ∈ room_001 下属 zones
      → 聚合 visible_objects
```

所有事件存在同一个列表中，**无空间分区、无时间索引**。7 天事件窗口内，假设每 30 秒采样一次、覆盖 5 个房间，事件量约 `7×24×60×2×5 ≈ 100,000` 条，全量扫描代价显著。

### 2.3 L3 每次查询重复计算

L3 定位为"不持久化的翻译层"。这意味着：
- 每次"找物"都要重新聚合 `habitual_locations` + `remembered_objects` + 最近对象状态事件
- 每次"巡逻异常"都要重新对比当次观测 vs L1 基线
- 没有缓存或预计算，热路径上的重复开销大

### 2.4 跨层联合查询缺乏统一接口

L3 的三个任务场景（找物/找人/巡逻异常）各自从 L1+L2 手动拼装，没有通用的查询原语。添加新场景需要重写查询逻辑。

---

## 3. 查询优化方案

### 3.1 L1 倒排索引（P0）

在 L1 JSON 之外，由 L2 沉淀流程同步维护以下索引文件：

**物品→位置索引** `l1_index/object_to_location.json`

```json
{
  "medicine_box": {
    "type": "movable",
    "habitual_rooms": [
      {"room": "room_004", "evidence": 12},
      {"room": "room_005", "evidence": 6}
    ],
    "last_confirmed": {"room": "room_005", "ts": "2026-03-28T20:15:00Z", "source": "vlm"}
  },
  "remote_control": {
    "type": "movable",
    "habitual_rooms": [
      {"room": "room_001", "evidence": 8}
    ],
    "last_confirmed": {"room": "room_001", "ts": "2026-03-28T14:52:01Z", "source": "vlm"}
  },
  "sofa": {
    "type": "anchor",
    "canonical_room": "room_001",
    "canonical_zone": "zone_sofa_area"
  }
}
```

**房间→内容一级展开索引** `l1_index/room_summary.json`

```json
{
  "room_001": {
    "type": "living_room",
    "neighbors": ["room_002", "room_003"],
    "zones": {
      "zone_sofa_area": {
        "anchors": ["sofa", "coffee_table"],
        "remembered": ["remote_control", "pillow"]
      },
      "zone_tv_area": {
        "anchors": ["tv", "tv_cabinet"],
        "remembered": []
      }
    },
    "last_confirmed": "2026-03-28T10:00:00Z"
  }
}
```

**邻接图索引** `l1_index/adjacency.json`

```json
{
  "room_001": ["room_002", "room_003"],
  "room_002": ["room_001", "room_003"],
  "room_003": ["room_002", "room_004", "room_009"]
}
```

> **维护时机**：L2→L1 沉淀时同步更新索引；L1 被用户纠错时同步更新索引。索引可从 L1 全量重建，丢失不致命。

**收益**："找药箱在哪"从遍历全部 Movable Object 节点 → 单次 key 查找 O(1)；"客厅有什么"从遍历 Room→Zone→Objects 链 → 单次 key 查找。

### 3.2 L2 事件按房间+日期分片存储（P1）

改造 L2 存储目录结构：

```
l2_events/
├── observations/
│   ├── room_001/
│   │   ├── 2026-03-28.jsonl
│   │   ├── 2026-03-27.jsonl
│   │   └── ...
│   └── room_002/
│       └── ...
├── object_states/
│   ├── 2026-03-28.jsonl      # 全局，按时间分片即可
│   └── ...
└── transitions/
    ├── 2026-03-28.jsonl
    └── ...
```

每个文件为 JSONL 格式（每行一条事件），支持追加写入和流式读取。

**查询示例**：

| 查询 | 访问路径 | 复杂度 |
|------|--------|--------|
| 客厅今天发生了什么 | `observations/room_001/2026-03-28.jsonl` | 单文件顺序读 |
| 药箱最后在哪 | `l1_index/object_to_location.json["medicine_box"]` | O(1) |
| 最近 3 天所有房间切换 | `transitions/2026-03-{26,27,28}.jsonl` | 3 文件 |
| 过期清理 | 删除 7 天前的 `.jsonl` 文件 | 文件级操作 |

### 3.3 L3 热路径缓存（P2）

对高频查询结果引入短期缓存（内存 or 文件），避免重复聚合：

```python
class L3Cache:
    """L3 查询缓存，由 L2 写入事件触发失效"""
    
    # 全屋摘要：L1 结构变化时失效
    home_summary: str | None
    
    # 房间摘要：对应房间的 L1 节点或 L2 事件更新时失效
    room_summaries: dict[str, str]  # room_id → 结构化文本
    
    # 物品位置：对应物品的 Object State Event 写入时失效
    object_locations: dict[str, str]  # object_category → 结构化文本
```

缓存粒度为**单个房间/单个物品**，局部失效不影响全局。

### 3.4 统一查询接口（P2）

为 L3 定义通用查询原语，新增任务场景时只需组合原语：

```python
class MemoryQuery:
    def locate_object(self, category: str) -> ObjectLocationResult:
        """找物：聚合 L1 habitual + L2 recent state"""
        
    def room_snapshot(self, room_id: str, time_range: tuple) -> RoomSnapshot:
        """房间快照：L1 结构 + L2 时间窗口内事件"""
        
    def navigation_hint(self, from_room: str, to_room: str) -> list[str]:
        """路径提示：基于 L1 adjacency 的 BFS 最短路径"""
        
    def anomaly_diff(self, room_id: str, observation: ObservationEvent) -> AnomalyReport:
        """异常检测：当次观测 vs L1 基线 diff"""
        
    def home_summary(self) -> str:
        """全屋摘要：预生成的结构化文本"""
```

---

### 3.5 面向能力阶段的查询视图

为了和 Kinbot 的能力接入顺序对齐，L3 除了通用原语，还应提供按阶段裁剪的任务视图：

```python
class MemoryQuery:
    def map_alignment_view(self, current_observation) -> MapAlignmentView:
        """P0：当前观测与房间/区域/拓扑的对齐结果"""

    def region_ranking_view(self, target, current_room: str) -> RegionRankingView:
        """P0：目标候选区域排序与搜索顺序建议"""

    def recovery_view(self, target, failed_regions: list[str]) -> RecoveryView:
        """P0：搜索失败后的恢复建议"""

    def multiview_consistency_view(self, observation_bundle) -> MultiViewView:
        """P1-high：多相机语义一致性与遮挡确认摘要"""

    def geometry_constraint_view(self, current_room: str, target=None) -> GeometryView:
        """P1-medium：通行空间、门口、狭窄区域、遮挡关系摘要"""

    def long_term_prior_view(self, target) -> LongTermPriorView:
        """P2：显式长期先验，含 freshness / confidence / writeback 状态"""
```

其设计意图是：

- **P0** 只输出语义判断，不输出连续动作控制。
- **P1-high / P1-medium** 在 teacher 侧先稳定，再决定 student 是否继承对应字段。
- **P2** 的长期记忆通过显式查询视图接入，而不是直接压进 student 参数。

---

## 4. 模型数据注入策略

这是整个记忆系统最关键的产出环节。L3 组装结果最终会进入 teacher/student 的训练样本、蒸馏样本与推理 prompt，因此**context window、运行频率、量产输入约束必须一起设计**。

### 4.1 核心原则：结构化自然语言 > 原始 JSON

模型理解自然语言的效率远高于解析嵌套 JSON。L3 输出应当是**面向模型可读性优化的结构化文本**。

**反面示例（直接喂 JSON）**：
```json
{"id":"obj_medicine_box","category":"medicine_box","habitual_locations":[{"room":"room_004","evidence_count":12},{"room":"room_005","evidence_count":6}],"last_confirmed_location":{"room":"room_005","source":"vlm"},"last_confirmed_at":"2026-03-28T20:15:00Z"}
```
~95 tokens，模型需要自行解析嵌套结构、理解 key 含义、推断 room_004 是什么房间。

**正面示例（结构化自然语言）**：
```
药箱(medicine_box):
  常见位置: 主卫(置信度高, 观测12次) > 次卫(观测6次)
  最后确认: 次卫, 3月28日 20:15, 来源VLM
  当前状态: 主卫连续3次未观测到, 可能已被移走
```
~65 tokens，模型直接理解语义，无需额外推理。**token 更少，理解更准**。

### 4.2 双轨模型契约：Teacher 追求上限，Student 对齐部署

同一套记忆数据需要输出两种版本：

| 维度 | Teacher 注入视图 | Student 注入视图 |
|---|---|---|
| 目标 | 能力探索、补标、蒸馏教师 | 量产部署、时延控制、功耗控制 |
| 输入 | 可包含多视角、更高分辨率、深度真值、更多中间摘要 | 仅保留量产输入可获得的信息 |
| 记忆注入策略 | 可注入更丰富的结构、证据链与中间判断 | 只注入当前决策真正需要的字段 |
| 与长期记忆关系 | 可更充分使用显式记忆与历史证据 | 只消费经裁剪的显式记忆摘要 |

因此，L3 不应只输出一个“通用 prompt 片段”，而应明确生成：

- `teacher_rich_context`
- `student_deploy_context`
- `distillation_target_view`

其中 `distillation_target_view` 负责把 teacher 的高质量判断压缩成 student 可学习的结构化目标。

### 4.3 运行频率与 token 预算必须前置约束

Research teacher 可以使用更宽松的 token 预算探索能力上限，但 PDCP student 必须严格控制输入/输出预算。建议以 student 预算作为系统设计硬约束，以 teacher 预算作为研究上限。

| 任务层级 | Student 总 token 预算目标 | 主要用途 |
|---|---|---|
| 房间级导航 | 450-650 | 当前房间判断、地图对齐、区域排序 |
| 家具级找物 | 600-850 | 家具/支撑面级候选区域搜索 |
| 人物级找人 | 650-900 | 人员搜索、区域优先级、恢复建议 |
| 高频轻量层 | 120-260 | 高频运行的轻量判断或守护层 |

约束含义如下：

- 输出 token 应压缩到 **20-50 tokens**，只保留核心字段。
- 输入 token 占总 token 的绝大部分，因此优化重点应放在输入组织而不是输出措辞。
- 同一任务若需要更大上下文，应优先采用**分轮注入**而不是单轮塞满。

### 4.4 Student 降 token 策略

为确保 4B student 在量产相机配置下可部署，L3 必须提供一套默认的裁剪规则。

**图像输入优化**

- 每次推理只取主视角 + 辅视角各 1 张关键帧，不输入连续视频流。
- 分辨率控制在 `336×336` 或 `448×448`。
- 只有在需要精细判断时才额外加入 1 个局部 ROI。
- 不反复回灌历史原图，历史信息改用结构化摘要表达。

**地图输入优化**

- 不输入完整地图图像，只输入当前位置附近的局部裁剪。
- 拓扑关系优先转为文本摘要，例如“当前在客厅，左侧卧室，前方厨房”。
- 候选区域以结构化列表表达，不重复渲染完整地图。

**历史信息优化**

- 不回灌完整轨迹图像，只保留近 3-5 步文本摘要。
- 摘要仅保留已搜索区域、已排除区域、当前搜索策略。
- 多视角一致性结果转为摘要，不反复注入全部原始视角。

**任务信息优化**

- 指令采用简洁文本，不保留冗长自然语言噪声。
- 任务约束使用结构化字段表达，如 `{"target":"person","constraint":"close_approach"}`。
- 位姿信息只保留当前位置、朝向、相对目标距离等关键字段。

### 4.5 分阶段注入：先支撑 P0，再逐层接入增强能力

这套记忆系统在模型注入侧也应遵循与能力研发一致的接入顺序。

**P0：空间理解 + 地图理解**

- 输出当前所处房间/区域判断
- 输出当前观测与地图拓扑的对齐关系
- 输出目标候选区域排序
- 输出搜索顺序建议与失败恢复建议
- 不输出底盘连续控制

**P1-high：多视角理解**

- 输出多视角一致性判断
- 输出遮挡确认与视角切换建议
- 输出跨相机目标持续识别结果

**P1-medium：深度与几何约束**

- 输出可通行空间摘要
- 输出门口、狭窄区域、遮挡关系约束
- 输出靠近目标时的局部安全性判断

**P1-low：强化学习策略优化（后接入）**

- 将区域排序、搜索结果、不确定性与恢复结果转成可供 RL 消费的结构化状态
- 优化搜索效率、主动观察顺序与局部避障提示
- 作为策略效率增强层，而不是替代 P0/P1 的语义判断层

**P2：长期记忆**

- 输出物品位置先验、人员活动区域先验
- 输出记忆新鲜度、可信度、最近写回状态
- 作为显式记忆库接入，而不是将长期先验直接写入 student 参数

### 4.6 任务级 Prompt / Supervision 模板

为不同阶段定义固定模板，L3 负责填充。

#### P0 语义导航模板

```
## 任务上下文
当前房间: {current_room}
当前区域: {current_zone}
目标: {target}
任务约束: {task_constraint}

### 地图对齐
{map_alignment}

### 候选区域排序
{ranked_regions}

### 搜索与恢复
{search_order}
{recovery_hint}
```

这个模板对应第一阶段的主训练目标：**只输出语义判断，不输出连续动作**。

#### 找物模板

```
## 记忆上下文
当前位置: {current_room}({current_zone})
目标物品: {target_object}

### 位置推断
候选房间(按历史概率排序):
{ranked_rooms}

### 区域内优先探索
{zone_hints}

### 最近动态
{recent_state_events}

### 路径建议
{navigation_path}
```

**填充示例**：

```
## 记忆上下文
当前位置: 客厅(沙发区)
目标物品: 药箱

### 位置推断
候选房间(按历史概率排序):
  1. 主卫 (60%, 上次观测: 3月28日 09:30)
  2. 次卫 (30%, 上次观测: 3月27日 14:10)
排除房间: 客厅、卧室 (历史概率 < 5%)

### 区域内优先探索
  主卫 → 镜柜区 (remembered_objects 含 medicine_box)

### 最近动态
  主卫: 药箱连续3次未观测到, 可能已被移走
  次卫: 药箱 3月28日 20:15 最后确认

### 路径建议
  客厅 → 走廊 → 次卫 (优先, 最后确认位置)
  客厅 → 走廊 → 主卫 (备选)
```

#### 巡逻异常模板

```
## 巡逻上下文
巡逻区域: {room_name}
基线来源: L1 Zone remembered_objects

### 异常报告
{anomaly_items}

### 建议动作
{suggested_actions}
```

**填充示例**：

```
## 巡逻上下文
巡逻区域: 客厅

### 异常报告
客厅茶几区:
  新增物品: bottle, hat (不在 remembered_objects 中)
  消失物品: newspaper (长期常见, 本次 unseen)
  → 异常等级: 低 (临时物品变化)

主卫:
  medicine_box 未见 (历史常见, 连续3次 unseen)
  → 异常等级: 中 (可能已被移走, 触发对象状态更新)

儿童房:
  检测到清洁剂 (命中 not_expected_in 规则)
  → 异常等级: 高 (安全告警, 需通知用户)

### 建议动作
  1. [高] 通知用户: 儿童房发现清洁剂
  2. [中] 更新药箱状态: 标记主卫位置为不确定
  3. [低] 忽略: 客厅临时物品变化属正常范围
```

#### 全屋概览模板

```
## 家庭概览
{home_structure_summary}

### 今日状态
{today_anomalies}
{recent_activity_zones}
```

**填充示例**：

```
## 家庭概览
共9个房间: 客厅(沙发区+电视区), 厨房, 餐厅, 主卧, 次卧, 书房, 主卫, 次卫, 阳台
走廊连接全部房间, 入户门位于走廊东端

### 今日状态
异常:
  - [高] 儿童房发现清洁剂 (安全告警)
  - [中] 主卫药箱可能被移走
最近活动集中: 客厅沙发区, 餐厅
今日未覆盖: 书房, 阳台
```

### 4.7 输出压缩原则

原始设计中，模型输出若包含大量解释性文本，很容易达到 `100-180 tokens`。对于 student 部署，这一设计过重。建议统一改为只保留核心字段：

- 当前所处位置
- 下一步目标区域
- 备选目标区域
- 当前不确定性或恢复建议

压缩后单次输出可控制在 **20-50 tokens**，更适合高频运行场景。

### 4.8 全屋摘要预生成与增量更新

全屋摘要是多个任务共享的基础上下文，应当**预生成并缓存**，而非每次查询现场拼装。

```
触发更新的条件:
  - L1 新增/删除房间节点
  - L1 房间类型变更
  - L1 邻接关系变更
  - 每日首次查询时刷新"今日状态"部分
```

摘要控制在 **~150 tokens**，作为多种任务 prompt 的公共前缀。

### 4.9 多轮对话中的渐进式展开

对于需要先全局、再局部聚焦的任务，采用**两阶段注入**：

```
第一轮 prompt (概览, ~150 tokens):
  注入全屋摘要 → 模型决定目标房间

第二轮 prompt (细节, ~200-350 tokens):
  注入目标房间的 Zone 详情 + 最近事件 → 模型做出具体判断
```

避免一次性注入所有房间的全部细节导致 context 爆炸。

### 4.10 进阶：语义 Embedding 索引（P3）

当物品种类增长到数百时，基于 key 的精确匹配无法覆盖模糊查询。例如：

- 用户说"充电的地方" → 需匹配"书房桌面区，有充电器"
- 用户说"放吃的地方" → 需匹配"厨房" + "餐厅"

方案：对 L1 每个 Zone 节点生成描述文本 → embedding → 存入向量索引。查询时先语义检索 top-K zone，再注入对应结构化文本。

```python
# Zone 描述生成示例
zone_descriptions = {
    "zone_sofa_area": "客厅沙发区，有沙发和茶几，常见遥控器和抱枕",
    "zone_kitchen_counter": "厨房操作台区域，有灶台、水池和冰箱",
    "zone_desk_area": "书房桌面区，有电脑、充电器和台灯"
}

# 查询 "充电的地方"
results = embedding_search("充电的地方", zone_descriptions)
# → [("zone_desk_area", 0.87), ("zone_tv_area", 0.52), ...]
```

---

## 5. 完整查询流程示例

以"找药箱"为例，展示优化后的完整链路：

```
用户指令: "去找药箱"
           │
           ▼
    ┌─── L3 记忆服务层 ───┐
    │                      │
    │  1. 查索引            │
    │     l1_index/object_to_location.json    ← O(1)
    │     → medicine_box: 主卫(60%), 次卫(30%)
    │     → last_confirmed: 次卫, 3月28日 20:15
    │                      │
    │  2. 查最近状态         │
    │     l2_events/object_states/2026-03-28.jsonl   ← 单文件grep
    │     → 主卫连续3次 unseen
    │                      │
    │  3. 查路径            │
    │     l1_index/adjacency.json    ← BFS
    │     → 客厅→走廊→次卫 / 客厅→走廊→主卫
    │                      │
    │  4. 填充模板           │
    │     OBJECT_SEARCH_TEMPLATE.format(...)
    │     → 结构化自然语言, ~200 tokens
    │                      │
    └──────────┬───────────┘
               │
               ▼
    ┌─── LLM / VLM ────┐
    │                    │
    │  接收 ~200 tokens   │
    │  记忆上下文          │
    │  + 当前视觉观测      │
    │  → 输出导航决策      │
    │                    │
    └────────────────────┘
```

**对比优化前**：
- 查索引 O(1) vs 遍历全部 Movable Objects O(n)
- 单文件 grep vs 遍历全部 L2 事件 O(m)
- 结构化文本 ~200 tokens vs 原始 JSON dump ~800+ tokens

---

## 6. 存储结构总览

优化后的完整目录结构：

```
home_semantic_memory/
├── l1_structure/                    # L1 语义结构层（源数据）
│   ├── rooms.json                   # 所有 Room Node
│   ├── zones.json                   # 所有 Zone Node
│   ├── anchor_objects.json          # 所有 Anchor Object Node
│   └── movable_objects.json         # 所有 Movable Object Node
│
├── l1_index/                        # L1 查询索引（可从 l1_structure 全量重建）
│   ├── object_to_location.json      # 物品 → 位置倒排索引
│   ├── room_summary.json            # 房间 → 内容一级展开
│   └── adjacency.json               # 房间邻接图
│
├── l2_events/                       # L2 动态事件层（按房间+日期分片）
│   ├── observations/
│   │   └── {room_id}/
│   │       └── {YYYY-MM-DD}.jsonl
│   ├── object_states/
│   │   └── {YYYY-MM-DD}.jsonl
│   └── transitions/
│       └── {YYYY-MM-DD}.jsonl
│
├── l3_cache/                        # L3 查询缓存（可丢失重建）
│   ├── home_summary.txt             # 全屋摘要文本
│   ├── room_{id}_summary.txt        # 单房间摘要文本
│   └── object_{category}_location.txt  # 单物品位置文本
│
└── l3_templates/                    # L3 Prompt 模板
    ├── vln_p0_navigation.txt
    ├── object_search.txt
    ├── person_search.txt
    ├── patrol_anomaly.txt
    ├── home_overview.txt
    ├── multiview_consistency.txt
    └── geometry_constraints.txt
```

---

## 7. 优先级与实施建议

| 优先级 | 改进项 | 改动范围 | 收益 |
|-------|--------|---------|------|
| **P0** | L3 输出改为结构化自然语言 | L3 组装逻辑 | 模型理解效率提升最大，token 消耗减少 ~60% |
| **P0** | 新增 `vln_p0_navigation` 视图 | L3 查询与模板层 | 直接支撑空间理解、地图理解、区域排序、恢复建议 |
| **P0** | 按 Teacher / Student 双契约裁剪注入内容 | L3 模板 + 数据视图层 | 避免 teacher/student 共用一套过重上下文 |
| **P1** | L1 增加倒排索引文件 | 新增 `l1_index` 目录 + 沉淀流程同步更新 | 查询从 O(n) → O(1) |
| **P1** | L2 事件按 room+date 分片 | L2 写入路径改造 | 时间/空间过滤零开销，过期清理简单 |
| **P1** | 为多视角/几何能力增加派生摘要视图 | L3 查询层 | 给 teacher 稳定接入 P1-high / P1-medium 能力 |
| **P2** | L3 热路径缓存 | 新增 `l3_cache` + 失效逻辑 | 避免重复聚合计算 |
| **P2** | 统一查询接口 `MemoryQuery` | L3 API 层 | 新增任务场景的开发成本降低 |
| **P2** | 长期记忆 freshness / confidence / writeback 管理 | L2→L1 沉淀与 L3 注入层 | 控制脏记忆扩散，支撑最终长期记忆接入 |
| **P3** | Zone 描述 embedding 索引 | 新增向量检索模块 | 支持模糊语义查询 |

建议将能力路线和存储路线按如下节奏协同推进：

1. **第一轮（P0）**：先让记忆系统稳定服务空间理解与地图理解，只输出高层语义判断；随后蒸馏 4B v1。
2. **第二轮（P1）**：在 teacher 上接入多视角与几何约束对应的查询视图和模板，再将结构化状态提供给 RL 做效率优化，最终蒸馏出 4B v2 / v3 / v4。
3. **第三轮（P2）**：最后接入长期记忆，保持显式记忆库形态，完成 freshness / confidence / writeback 管理后，再蒸馏完整 student 版本。

**核心思路**：L1/L2 的存储格式为**机器查询效率**优化（索引 + 分片），L3 的输出格式为**模型理解效率**优化（结构化自然语言 + 模板 + 裁剪），而 Teacher / Student 双轨契约决定了这套系统必须能同时服务能力探索与量产部署。长期记忆应被视为最终阶段接入的显式能力，而不是在早期就压入模型参数。
