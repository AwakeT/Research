# Thinking 模式对 VLN→NFM 演进全流程的影响分析

---

文档版本：v1.1
创建日期：2026-04-08
更新日期：2026-04-08（v1.1：新增 Habitat VLN 导航实测数据分析）
文档性质：技术分析
适用范围：Qwen3.5 系列模型在 Kinbot VLN→NFM 各阶段中 thinking/no-thinking 模式的选择策略

---

## 1. 核心结论

Thinking 模式的价值**高度依赖任务类型**，不存在"一律开启"或"一律关闭"的通用策略。

| 任务性质 | 推荐模式 | 原因 |
|---------|---------|------|
| 感知密集型（检测、定位、枚举） | **No-Thinking** | thinking 产生"过度推理"，干扰快速感知输出 |
| 推理密集型（空间推理、搜索规划、恢复策略） | **Thinking（仅限 Teacher）** | 推理链提升复杂空间推理，但 Student 端不可用 |
| 结构化短输出（bbox JSON、区域排序列表） | **No-Thinking** | 输出格式更稳定，不会被推理链干扰 |
| 闭环导航动作输出 | **No-Thinking** | VLN 实测证明 thinking 导致过早停止、格式溢出、上下文污染 |
| 端侧实时部署 | **锁定 No-Thinking** | 延迟从 2.4s 降至 0.5s，满足实时性约束 |
| Teacher 数据生成 | **按任务切换** | 检测数据用 no-thinking，推理数据用 thinking |

---

## 2. Thinking 模式的机制与行为差异

### 2.1 机制说明

Qwen3.5 的 thinking 模式通过 `enable_thinking=true/false` 控制：

- **Thinking 开启**：模型在生成最终答案前，先输出一段 `<think>...</think>` 推理链，展示中间推理过程，然后再输出结论
- **Thinking 关闭**：模型直接输出结论，不生成推理链

### 2.2 行为差异的实测数据

以下数据均来自 homeobjects_mini_v1 基准（352 个任务），同一模型（Qwen3.5-4B）、同一数据、仅切换 thinking 开关：

**输出行为差异**：

| 维度 | Thinking | No-Thinking | 差异倍数 |
|------|---------|-------------|---------|
| 平均推理耗时 | 2.422s/task | 0.529s/task | **4.6x** |
| 总推理耗时 | 852.7s | 186.1s | **4.6x** |
| MMBench 总输出 token | 5,315,540 | 1,801,190 | **2.95x** |

**能力差异（homeobjects 实测）**：

| 指标 | Thinking | No-Thinking | 变化 | 胜出 |
|------|---------|-------------|------|------|
| Detection F1 | 0.7734 | **0.8164** | +5.6% | No-Thinking |
| Detection Recall | 0.7111 | **0.8278** | +16.4% | No-Thinking |
| Table 召回 | 0.6288 | **0.7727** | +22.9% | No-Thinking |
| Referring Acc@0.5 | 0.7021 | 0.7021 | 0% | 持平 |
| Referring Mean IoU | 0.6352 | **0.6610** | +4.1% | No-Thinking |
| 几何关系准确率 | 0.7021 | **0.8298** | +18.2% | No-Thinking |
| 房间分类准确率 | 0.0000 | **0.7059** | — | No-Thinking |
| 推理延迟 | 2.422s | **0.529s** | -78.2% | No-Thinking |

**能力差异（MMBench-EN Dev 通用基准）**：

| 指标 | Thinking | No-Thinking | 变化 | 胜出 |
|------|---------|-------------|------|------|
| 总分 | **81.79%** | 73.20% | -8.59 | Thinking |
| 空间关系 | **71.11%** | 46.67% | -24.44 | Thinking |
| 逻辑推理 | **72.88%** | 57.63% | -15.25 | Thinking |
| 属性比较 | **88.64%** | 72.73% | -15.91 | Thinking |
| 场景识别 | 92.31% | **90.38%** | -1.93 | 接近 |
| OCR | 82.05% | **76.92%** | -5.13 | Thinking |

### 2.3 关键发现

**发现一：thinking 的价值因任务性质分化**

- 在需要多步推理的通用 benchmark（MMBench）上，thinking 大幅领先（+8.59 总分，空间关系 +24.44）
- 在实际感知任务（homeobjects 检测/定位/几何）上，no-thinking 全面胜出

**发现二：thinking 产生"过度推理"伤害感知任务**

Thinking 模式下，模型为检测任务生成了大量不必要的推理链（如"这个物体可能是桌子，因为它有四条腿和一个平面……"），这些推理链：
- 增加了输出中出现格式错误的概率（如房间分类退化为 100% unknown）
- 干扰了多目标枚举的完整性（table 召回从 77.27% 降至 62.88%）
- 推理链中的中间结论可能误导最终输出

**发现三：两套 benchmark 的矛盾说明"空间关系"能力有两种形态**

- MMBench 空间关系（71.11% thinking vs 46.67% no-thinking）：考查"描述性空间推理"，需要文字化推理过程
- homeobjects 几何关系（82.98% no-thinking vs 70.21% thinking）：考查"感知性空间判断"，需要直接从视觉特征中输出判断

这两种空间能力对应 VLN→NFM 演进中的不同阶段需求。

### 2.4 VLN 导航实测数据（Habitat Instance ImageNav）

除了上述检测/感知基准外，`model_analysis.md` 中记录了一组更接近真实导航场景的 VLN 闭环测试数据。该测试在 Habitat 仿真器上运行，使用 HM3D 数据集 val 集，每个模型运行 50 个 episode，最大步数 500 步，停止成功阈值 5.0m。模型需要**直接输出导航动作指令**（前进、转向、停止），而非仅输出语义判断。

#### VLN 导航性能总览

| 模型 | 成功率 | 平均 SPL | 平均步数 | 解析错误数 | Thinking |
|------|--------|---------|---------|-----------|---------|
| Qwen3-VL-4B | **24.0%** | 0.123 | 296.7 | 0 | 无 |
| Qwen3.5-2B | 22.0% | 0.051 | 253.4 | 1020 | 有（部分） |
| Qwen3-VL-8B | 18.0% | 0.126 | 230.6 | 0 | 无 |
| **Qwen3.5-4B** | **16.0%（最低）** | 0.137 | **59.9** | **615** | **有（97.6% 步骤）** |

#### Thinking 模式在 VLN 中的四大致命问题

**问题一：Chain-of-Thought 溢出（最严重）**

Qwen3.5-4B 在 97.6% 的步骤中输出了完整的 `<think>...</think>` 推理链。推理文本经常**溢出到动作输出区域**，导致解析失败：

```
实际响应示例（Episode 342, Step 0）：
"需要找到椅子，从当前的视角看过去，左侧有一个门洞通向卧室...
因此，左转是更合理的的第一步。
</think>

右转30度"
```

问题清单：
- 推理文字溢出到动作输出（"左转30，直行0.25。这意味着我转了左边，然后..."）
- 推理结论与最终动作**自相矛盾**（思考说"应左转"，实际输出"右转"）
- 思考过程出现**语言混乱**（如 Step 6 输出泰语）
- 615 次解析错误中的绝大部分由推理链溢出导致

**问题二：上下文窗口被推理链占据**

每步都生成的推理链文本大量占用有限的上下文窗口，导致：
- 历史动作窗口被压缩，模型对自身导航状态的追踪能力下降
- 多步导航时，早期的重要观测信息被推理链文本"挤出"上下文
- 模型逐步失去对全局导航进度的感知

**问题三：过早停止导致探索深度极浅**

| 模型 | 平均步数 | 达到 500 步上限 | 30 步内过早停止 |
|------|---------|---------------|---------------|
| Qwen3-VL-4B | 296.7 | 20 | 4 |
| Qwen3-VL-8B | 230.6 | 15 | 10 |
| Qwen3.5-2B | 253.4 | 13 | 6 |
| **Qwen3.5-4B** | **59.9** | **0** | **13** |

Qwen3.5-4B 的平均步数仅 59.9（其他模型均 200+步），**没有任何一个 episode 达到 500 步上限**。13 个 episode 在 30 步以内就停止失败。原因：thinking 推理链使模型在探索初期就"自信"地判断已找到目标，频繁触发错误停止动作。

**问题四：能力与成功率倒挂**

Qwen3.5-4B 的 SPL（0.137）是所有模型中最高的，说明**一旦成功，路径效率最优**。但成功率（16%）却最低。这验证了一个核心矛盾：

> **Thinking 模式提升了推理质量，但严重损害了动作执行的稳定性。**

模型"想得更清楚"（路径规划更优），但"做得更差"（推理链干扰动作输出、过早停止、格式溢出）。在闭环导航任务中，**执行稳定性比推理质量更重要**。

#### VLN 实测的关键结论

1. **Thinking 模式不仅在检测任务中有害，在闭环导航任务中更加致命**。检测任务中 thinking 导致 F1 下降 5.6%；VLN 任务中 thinking 导致成功率从对标的 24%（Qwen3-VL-4B no-thinking）降至 16%（-33%）
2. **推理链溢出是结构性问题，不是 prompt 调优能解决的**。97.6% 的步骤都输出推理链，说明 thinking 模式下模型的生成行为已经固化，无法通过 prompt 约束可靠地抑制
3. **上下文污染是多步任务的独特风险**。单次检测/推理中，thinking 只增加单次延迟；多步导航中，thinking 的推理链文本会**累积性地污染上下文**，导致能力随步数递减
4. **"更大模型不一定更好"的规律在 VLN 中同样成立**：Qwen3-VL-8B（18%）< Qwen3-VL-4B（24%），Qwen3.5-4B（16%）< Qwen3.5-2B（22%）。更强的模型产生更复杂的输出，反而干扰了动作指令的遵从性

#### 与检测任务数据的交叉验证

| 问题类型 | 检测任务表现 | VLN 导航任务表现 | 交叉验证结论 |
|---------|-----------|----------------|-------------|
| 输出格式稳定性 | thinking 导致房间分类 100% 输出 unknown | thinking 导致 615 次解析错误 | **Thinking 系统性地破坏结构化输出** |
| 过度推理 | thinking 降低 table 召回（77.27%→62.88%） | thinking 导致过早错误停止（13 个 episode <30 步） | **Thinking 的"自信"判断在执行层面是有害的** |
| 推理质量 | thinking 在 MMBench 空间关系上更优（71.11%） | thinking 的 SPL 最高（0.137），路径规划更好 | **Thinking 确实提升推理质量，但提升被执行层面的损害完全抵消** |
| 延迟/效率 | thinking 延迟 4.6x（2.4s vs 0.5s） | thinking 平均步数仅 59.9（vs 200+步） | **Thinking 在时间维度和步数维度都大幅降低效率** |

---

## 3. 对 VLN→NFM 分阶段设计的逐阶段影响

### 3.1 T0 阶段：训练对象与接口定义

**影响等级**：低

T0 是接口定义阶段，不涉及模型训练。但需要在此阶段**预先确定输出格式规范**：

| 输出类型 | 推荐模式 | 理由 |
|---------|---------|------|
| 结构化 JSON 输出（bbox、区域列表、置信度） | No-Thinking | 格式稳定性更高 |
| 推理链 + 结论混合输出（搜索策略解释） | Thinking | 推理过程本身是训练信号 |

**建议**：T0 阶段定义两套输出模板：
- **感知模板**（no-thinking）：直接输出结构化 JSON，无推理链
- **认知模板**（thinking）：输出 `<think>推理过程</think>` + 结构化 JSON 结论

### 3.2 T1 阶段：27B Teacher 基础能力形成（P0）

**影响等级**：高 — 这是 thinking 模式影响最复杂的阶段

T1 的训练目标包含两类性质截然不同的能力：

#### 感知类能力（推荐 No-Thinking 训练）

| 能力 | 优先级 | 推荐模式 | 理由 |
|------|--------|---------|------|
| 房间识别与属性理解 | P0-high | No-Thinking | 直接视觉判断，不需推理链 |
| 当前视角与地图区域对齐 | P0-high | No-Thinking | 视觉匹配任务，推理链反而干扰 |
| 当前所处位置判断 | P0-high | No-Thinking | 已验证 no-thinking 房间分类 70.59% vs thinking 0.00% |
| 家具和支撑面识别 | P0-medium | No-Thinking | 检测枚举任务 |

实测依据：
- 房间分类：no-thinking 70.59% vs thinking 0.00%（**系统性崩溃**）
- 检测 F1：no-thinking 0.8164 vs thinking 0.7734
- 多目标枚举（table 召回）：no-thinking 77.27% vs thinking 62.88%

#### 认知类能力（推荐 Thinking 训练）

| 能力 | 优先级 | 推荐模式 | 理由 |
|------|--------|---------|------|
| 观测到语义地图的匹配假设生成 | P0-high | Thinking | 需要多步推理：当前观测 → 语义特征提取 → 地图候选匹配 → 假设排序 |
| 目标候选区域排序 | P0-medium | Thinking | 需要综合多维信息推理优先级 |
| target belief 更新 | P0-medium | Thinking | 涉及概率推理和多源信息融合 |
| room graph / furniture relation 写回 | P0-medium | Thinking | 需要从观测中抽象出结构化关系 |
| 搜索顺序建议 | P0-low | Thinking | 需要策略推理 |
| 搜索失败恢复建议 | P0-low | Thinking | 需要反事实推理（"如果这里找不到，应该去哪"） |

推理依据：
- MMBench 空间关系：thinking 71.11% vs no-thinking 46.67%（+24.44），说明复杂空间推理依赖推理链
- 逻辑推理：thinking 72.88% vs no-thinking 57.63%（+15.25）

#### T1 训练策略建议

```
T1 Teacher 训练数据组成（按 thinking 模式分）：

感知类数据（no-thinking）：~40%
  ├── 房间识别：enable_thinking=false, 直接输出房间类别
  ├── 家具检测：enable_thinking=false, 输出 bbox JSON
  ├── 位置判断：enable_thinking=false, 输出位置结论
  └── 视角-地图对齐：enable_thinking=false, 输出对齐结果

认知类数据（thinking）：~45%
  ├── 匹配假设生成：enable_thinking=true, <think>推理过程</think> + 假设列表
  ├── 区域排序：enable_thinking=true, <think>推理过程</think> + 排序结果
  ├── belief 更新：enable_thinking=true, <think>推理过程</think> + 更新后 belief
  ├── 搜索策略：enable_thinking=true, <think>推理过程</think> + 策略建议
  └── 恢复建议：enable_thinking=true, <think>推理过程</think> + 恢复方案

通用防遗忘数据：~15%
  └── 混合 thinking/no-thinking
```

**关键风险**：混合 thinking/no-thinking 数据训练时，需要确保模型能根据 `enable_thinking` 标志正确切换行为模式，避免两种模式相互干扰。

### 3.3 T2 阶段：首次蒸馏到 4B（P0）

**影响等级**：非常高 — thinking 模式是蒸馏设计的核心变量

#### 核心问题：Student 是否需要 thinking 能力？

这是 T2 阶段最关键的决策。分析如下：

**方案 A：Student 锁定 No-Thinking（推荐）**

```
Teacher (27B, thinking) ─── 蒸馏 ──→ Student (4B, no-thinking only)
                                         │
                     Teacher 的推理能力被"压缩"为直觉判断
                     Student 只学习 Teacher 的结论，不学推理过程
```

优势：
- 输出 token 少（20-50 tokens），满足端侧 token 预算（450-900 tokens/次）
- 延迟可控（~0.5s → INT8 量化后 <100ms）
- 输出格式稳定，系统集成简单
- 已被实测验证为检测/定位/几何任务的最优模式

劣势：
- 丢失 Teacher 的显式推理能力
- 复杂空间推理场景可能能力不足（MMBench 空间关系从 71.11% 降至 46.67%）
- 搜索策略和恢复建议的质量可能下降

**方案 B：Student 保留 Thinking 能力**

```
Teacher (27B, thinking) ─── 蒸馏 ──→ Student (4B, thinking + no-thinking)
                                         │
                     Student 同时具备两种模式
                     部署时根据任务类型动态切换
```

优势：
- 复杂推理场景可激活 thinking 获得更好结果
- 更完整地保留 Teacher 能力

劣势：
- Thinking 模式下延迟暴增（2.4s → 不满足实时性要求）
- 输出 token 暴增（thinking 模式输出 ~1500 tokens），远超 token 预算
- 需要部署时实现模式切换逻辑，增加系统复杂度
- 4B 模型的 thinking 质量可能远不及 27B（推理链能力与模型尺寸强相关）
- 已被实测证明 thinking 在检测任务上有害

**推荐方案 A**，理由：

1. **端侧部署约束**：token 预算 450-900 tokens/次，thinking 输出（~1500 tokens）严重超标
2. **延迟约束**：0.25-0.40 Hz 的 Student 稳态频率要求单次推理 <2.5s，thinking 模式（2.4s）刚好卡线且没有余量
3. **实测验证**：在实际感知任务中 no-thinking 全面优于 thinking
4. **蒸馏本质**：蒸馏的核心目标就是把 Teacher 的"慢思考"压缩为 Student 的"快直觉"

#### 蒸馏策略中 Thinking 模式的角色

即使 Student 锁定 no-thinking，**Teacher 的 thinking 能力在蒸馏过程中仍然重要**：

```
蒸馏数据生成流程：

1. Teacher (thinking mode) 处理训练样本
   ├── 生成推理链：<think>当前视角看到客厅沙发，地图显示客厅在东侧...</think>
   └── 生成结论：{"loc": "客厅", "next": "卧室-床头柜", "conf": 0.92}

2. 提取蒸馏目标
   ├── 结论作为 Student 的输出监督信号（output distillation）
   ├── 中间层特征作为 Student 的特征对齐目标（feature distillation）
   └── 推理链可用于数据增强（但不作为 Student 的训练目标）

3. Student (no-thinking mode) 训练
   ├── 输入：与 Teacher 相同的量产配置输入
   └── 输出目标：Teacher 的结论（不含推理链）
```

**关键原则**：Teacher 用 thinking 生成高质量判断，Student 用 no-thinking 学习直接输出这些判断。推理链是 Teacher 内部的"脚手架"，不需要传递给 Student。

#### 蒸馏损失设计中的 Thinking 考量

```python
# 蒸馏损失设计（Student 锁定 no-thinking）
distillation_loss = (
    # Teacher no-thinking 输出对齐（主损失）
    0.4 * output_alignment_loss(
        student_output,  # Student no-thinking 输出
        teacher_output_nothinking  # Teacher no-thinking 输出（格式对齐）
    ) +
    # Teacher thinking 结论对齐（辅助损失）
    0.3 * conclusion_alignment_loss(
        student_output,  # Student no-thinking 输出
        teacher_thinking_conclusion  # Teacher thinking 推理链的最终结论
    ) +
    # 中间特征对齐
    0.2 * feature_alignment_loss(
        student_features,
        teacher_features
    ) +
    # 置信度校准
    0.1 * confidence_calibration_loss(
        student_confidence,
        teacher_confidence
    )
)
```

**说明**：
- `output_alignment_loss`：让 Student 的 no-thinking 输出对齐 Teacher 的 no-thinking 输出（格式一致，最易对齐）
- `conclusion_alignment_loss`：让 Student 的 no-thinking 输出也能对齐 Teacher thinking 模式下的最终结论（利用 thinking 的推理质量，但不要求 Student 生成推理链）
- 两种损失互补：no-thinking 对齐保证格式稳定，thinking 结论对齐利用更高质量的判断

### 3.4 T3 阶段：多视角理解（P1-high）

**影响等级**：中

多视角理解涉及两类子任务：

| 子任务 | 性质 | 推荐模式 | 理由 |
|--------|------|---------|------|
| 跨视角语义一致性判断 | 感知 | No-Thinking | 直接从多图特征中判断一致性 |
| 目标在不同视角的重识别 | 感知 | No-Thinking | 视觉匹配任务 |
| 遮挡后的重新确认 | 感知+推理 | 混合 | 简单遮挡用 no-thinking，复杂遮挡推理用 thinking |
| 视角切换决策 | 推理 | Thinking | 需要推理"哪个视角更可能看到目标" |

**跨视角一致性摘要生成（推荐 No-Thinking）**：

```json
// No-Thinking 模式下的跨视角摘要输出（~50 tokens）
{
  "main_view": "客厅-沙发区",
  "aux_view": "客厅-门口",
  "consistency": "high",
  "target_visible": "main_only"
}
```

```json
// Thinking 模式下的跨视角摘要输出（~300+ tokens）
// <think>
// 主视角看到沙发区域，有一个茶几，上面有遥控器...
// 辅助视角看到的是门口区域，可以看到玄关柜...
// 两个视角的空间关系一致，都属于客厅...
// 目标遥控器只在主视角可见...
// </think>
{
  "main_view": "客厅-沙发区",
  "aux_view": "客厅-门口",
  "consistency": "high",
  "target_visible": "main_only"
}
```

No-thinking 输出 ~50 tokens，thinking 输出 ~350 tokens，最终结论相同。在 token 预算严格的多视角场景中（基础模式 ~580 tokens），推理链的 token 开销不可接受。

### 3.5 T4 阶段：深度与几何增强（P1-medium）

**影响等级**：中

| 子任务 | 推荐模式 | 理由 |
|--------|---------|------|
| 门口/狭窄通道判断 | No-Thinking | 视觉空间判断，homeobjects 几何关系已证明 no-thinking 更优 |
| 家具边缘/遮挡关系 | No-Thinking | 视觉匹配判断 |
| 局部可达性判断 | Thinking | 需要综合深度信息、空间关系和历史轨迹推理 |
| 靠近目标时的安全性评估 | Thinking | 需要推理潜在风险 |

**注意**：深度图作为额外输入（~80-120 tokens）会进一步压缩输出 token 预算。Thinking 模式的额外输出开销在此阶段更难以承受。

### 3.6 T5 阶段：RL 策略优化（P1-low）

**影响等级**：高 — RL 与 thinking 有特殊交互

#### RL 训练中 Thinking 的矛盾

RL 的 reward 信号基于**最终行为结果**（搜索成功率、碰撞率、搜索效率），而非推理过程。

**Thinking 模式在 RL 中的问题**：

1. **Reward 延迟**：Thinking 生成推理链需要额外时间，但 RL 的 reward 只评价最终动作。推理链的时间成本不会被 reward 机制考虑
2. **探索效率低**：Thinking 模式每次推理耗时 4.6x，RL 的 rollout 数据采集速度大幅下降
3. **Credit assignment 更难**：推理链使输出序列更长，RL 更难确定哪部分推理对最终 reward 有贡献
4. **推理链可能被 RL "hack"**：模型可能学会生成看似合理但实际无意义的推理链来获取 reward

**建议**：

```
RL 阶段统一使用 No-Thinking 模式

理由：
1. RL 关注行为结果，不关注推理过程
2. No-thinking 推理速度快 4.6x，rollout 采样效率大幅提高
3. Student 最终部署也是 no-thinking，RL 应在部署模式上直接优化
4. 避免 RL 对推理链的 reward hacking
```

### 3.7 T6 阶段：个性化/行为长期记忆（P2）

**影响等级**：高

长期记忆的读取和利用涉及复杂推理：

| 子任务 | 推荐模式 | 理由 |
|--------|---------|------|
| 记忆库检索 | No-Thinking | 基于关键词/语义匹配的直接检索 |
| 记忆与当前观测的冲突判断 | Thinking | 需要推理新旧信息的可信度 |
| 记忆更新决策 | Thinking | 需要推理是否应该更新、如何更新 |
| 基于记忆的搜索优先级调整 | Thinking | 需要综合记忆 + 当前观测 + 任务约束进行多步推理 |

**但考虑到 Student 端侧部署约束**：

个性化长期记忆功能如果要在 4B Student 上运行，仍需锁定 no-thinking。解决方案：

```
方案：记忆推理前置到 Teacher 或离线模块

运行时流程：
1. 显式记忆库提供候选记忆条目
2. 轻量规则引擎做初步筛选和冲突检测
3. Student (no-thinking) 接收筛选后的记忆摘要作为输入 context
4. Student 基于 context 直接输出决策（无需自己做记忆推理）

训练时流程：
1. Teacher (thinking) 处理完整记忆推理
2. 生成"记忆摘要 → 最终决策"的训练对
3. Student 学习从记忆摘要直接输出决策
```

---

## 4. 对 Token 预算的影响

### 4.1 输出 Token 对比

| 输出格式 | No-Thinking | Thinking | 差异 |
|---------|-------------|---------|------|
| 标准输出（Teacher 训练） | 100-150 tokens | 400-600 tokens（含推理链） | 3-4x |
| 压缩输出（Student 部署） | 20-40 tokens | 不适用（Student 锁定 no-thinking） | — |
| 极简输出（极低 TPS） | 15-20 tokens | 不适用 | — |

### 4.2 对总 Token 预算的影响

设计文档中的 Student token 预算（输入+输出）：

| 任务级别 | 输入 Token | No-Thinking 输出 | Thinking 输出 | No-Thinking 总计 | Thinking 总计 |
|---------|-----------|-----------------|--------------|-----------------|--------------|
| 房间级导航 | 430-630 | 20-30 | 400-600 | **450-660** | 830-1230 ❌ |
| 家具级找物 | 570-820 | 30-40 | 400-600 | **600-860** | 970-1420 ❌ |
| 人物级找人 | 620-870 | 30-50 | 400-600 | **650-920** | 1020-1470 ❌ |
| 高频轻量层 | 100-240 | 20-30 | 200-400 | **120-270** | 300-640 ❌ |

**结论**：Thinking 模式的输出 token 使所有任务级别的总 token 超出预算 50-100%，完全不可行。

### 4.3 对 TPS 需求的影响

| 任务级别 | 频率 | No-Thinking 每秒总 Token | Thinking 每秒总 Token |
|---------|------|------------------------|---------------------|
| 房间级 | 0.33 Hz | 150-220 | 274-406 |
| 家具级 | 0.25 Hz | 150-215 | 243-355 |
| 人物级 | 0.25 Hz | 163-230 | 255-368 |
| 高频轻量层 | 4-5 Hz | 480-1350 | 1200-3200 ❌ |

Thinking 模式在高频轻量层下 TPS 需求完全失控。

---

## 5. 对蒸馏流程的影响

### 5.1 蒸馏数据生成策略

Teacher 使用 thinking 模式生成的数据，需要经过处理才能用于 Student 蒸馏：

```
Teacher Thinking 输出示例：

<think>
当前视角显示一个开放的客厅空间。
左侧有一个三人沙发，前方有一个木质茶几。
地图显示客厅位于公寓东侧，与卧室相邻。
目标"遥控器"通常放在茶几上或沙发扶手旁。
茶几上可以看到一个黑色长方形物体，可能是遥控器。
候选区域排序：客厅-茶几 > 客厅-沙发扶手 > 卧室-床头柜。
</think>

{
  "loc": "客厅",
  "conf": 0.92,
  "targets": [
    {"r": "客厅-茶几", "p": 0.85},
    {"r": "客厅-沙发扶手", "p": 0.65},
    {"r": "卧室-床头柜", "p": 0.42}
  ],
  "next": "客厅-茶几",
  "alt": "客厅-沙发扶手"
}
```

```
蒸馏数据处理：

方式 A — 只提取结论（推荐，用于输出对齐蒸馏）：
  Student 训练目标 = 上述 JSON 结论部分
  推理链被丢弃，Student 不需要学习推理过程

方式 B — 推理链作为数据增强（辅助，用于困难样本）：
  将推理链中的关键判断转为结构化标注
  例如：{"reasoning_hints": ["茶几上有黑色长方形物体", "可能是遥控器"]}
  作为额外监督信号，但不要求 Student 生成推理链

方式 C — 不推荐：
  让 Student 也学习生成推理链
  原因：4B 模型推理链质量差、输出 token 超标、延迟不可接受
```

### 5.2 三条蒸馏路线中 Thinking 的角色

| 蒸馏路线 | Teacher 模式 | Student 模式 | Thinking 的作用 |
|---------|-------------|-------------|----------------|
| 路线 A：Data Distillation | Thinking 生成高质量结论 | No-Thinking 学习结论 | Teacher 内部推理工具，不传递给 Student |
| 路线 B：Logit KD | 两种模式分别对齐 | No-Thinking | Teacher thinking 的 logit 可作为辅助对齐目标 |
| 路线 C：Progressive（27B→9B→4B） | 27B Thinking | 9B 可选 → 4B No-Thinking | 中间 9B 可保留轻量 thinking，最终 4B 锁定 no-thinking |

### 5.3 蒸馏质量的 Thinking 依赖风险

**风险**：如果 Teacher 的某些高质量判断**只在 thinking 模式下才能产生**，而 no-thinking 模式下判断质量显著下降，那么 Student 从 Teacher no-thinking 输出学习时可能无法获得最佳监督信号。

**验证方法**：

```
对比实验：
1. Teacher (thinking) 生成结论 → Student 学习 → 评估 Student 能力
2. Teacher (no-thinking) 生成结论 → Student 学习 → 评估 Student 能力
3. 对比两组 Student 的能力差异

如果差异显著（>5%）：
  → 蒸馏数据应使用 Teacher thinking 结论
  → 即 "Teacher 用 thinking 推理，Student 用 no-thinking 输出"

如果差异不显著（<3%）：
  → 可以统一使用 Teacher no-thinking，简化蒸馏流程
```

---

## 6. 对部署架构的影响

### 6.1 Student 部署配置（推荐）

```yaml
# Student 端侧部署配置
model: Qwen3.5-4B
quantization: INT8
enable_thinking: false  # 锁定关闭，不允许运行时切换
temperature: 0.0  # 确定性输出
max_new_tokens: 50  # 压缩输出格式
output_format: compressed_json

# 不需要模式切换逻辑
# 不需要推理链解析
# 不需要 thinking token 预算
```

### 6.2 如果保留 Thinking 的部署复杂度

如果选择在 Student 端保留 thinking 能力，部署复杂度将显著增加：

```yaml
# 假设保留 thinking 的部署配置（不推荐）
model: Qwen3.5-4B
quantization: INT8

# 需要实现模式切换逻辑
thinking_mode_router:
  detection_tasks: no-thinking  # 感知任务关闭
  search_planning: thinking  # 搜索策略开启
  recovery_suggestion: thinking  # 恢复建议开启
  memory_reasoning: thinking  # 记忆推理开启
  default: no-thinking

# 需要两套 token 预算
token_budget:
  no_thinking: {max_input: 850, max_output: 50}
  thinking: {max_input: 600, max_output: 400}  # 输入要压缩以腾出输出空间

# 需要两套延迟预算
latency_budget:
  no_thinking: 500ms
  thinking: 2500ms  # 且不满足高频场景

# 需要推理链解析器
thinking_parser:
  extract_conclusion: true
  log_reasoning: true  # 调试用
  fallback_on_parse_error: no_thinking_mode
```

**结论**：保留 thinking 的部署方案增加了模式路由、双预算管理、推理链解析、错误回退等工程复杂度，但收益不确定。在端侧 4B 模型约束下不推荐。

---

## 7. 总结与推荐策略

### 7.1 分角色策略

| 角色 | Thinking 策略 | 理由 |
|------|-------------|------|
| **Teacher 训练** | 按任务类型混合 | 感知任务 no-thinking，认知推理任务 thinking |
| **Teacher 数据生成** | 以 Thinking 为主 | 生成最高质量的判断结论供蒸馏 |
| **蒸馏过程** | Teacher thinking 结论 → Student no-thinking 输出 | "慢思考"压缩为"快直觉" |
| **Student 训练** | 锁定 No-Thinking | 全部训练数据使用 no-thinking 模式 |
| **Student 部署** | 锁定 No-Thinking | 满足延迟、token、格式稳定性约束 |
| **RL 优化** | No-Thinking | RL 关注行为结果，thinking 降低采样效率 |

### 7.2 分阶段策略总览

| 阶段 | Teacher Thinking 占比 | Student Thinking | 说明 |
|------|---------------------|-----------------|------|
| T0 接口定义 | — | — | 定义两套输出模板 |
| T1 基础能力（P0） | 45% thinking + 40% no-thinking + 15% 通用 | — | 感知与认知类能力分开训练 |
| T2 首次蒸馏 | Thinking 生成蒸馏数据 | **锁定 No-Thinking** | 推理链是脚手架，不传递 |
| T3 多视角（P1-high） | 30% thinking + 55% no-thinking + 15% 通用 | No-Thinking | 多视角以感知为主 |
| T4 深度几何（P1-medium） | 40% thinking + 45% no-thinking + 15% 通用 | No-Thinking | 可达性判断需要推理 |
| T5 RL 优化（P1-low） | No-Thinking | No-Thinking | RL 全程 no-thinking |
| T6 长期记忆（P2） | 60% thinking + 25% no-thinking + 15% 通用 | No-Thinking | 记忆推理前置到 Teacher |

### 7.3 VLN 导航数据对策略的强化

Habitat VLN 实测数据进一步强化了 "Student 锁定 no-thinking" 的结论，且揭示了之前检测任务数据未覆盖的**多步闭环场景特有风险**：

| 风险类型 | 检测任务中是否暴露 | VLN 导航中是否暴露 | 严重程度 |
|---------|----------------|-----------------|---------|
| 输出格式被推理链破坏 | 是（房间分类 0.00%） | 是（615 次解析错误） | 高 |
| 过度推理导致误判 | 是（table 召回下降） | 是（过早停止，成功率最低） | 高 |
| 推理链占用上下文窗口 | 否（单次推理无累积效应） | **是（多步导航中推理链累积污染上下文）** | **极高** |
| 推理结论与实际动作矛盾 | 否（检测不涉及动作执行） | **是（思考"左转"但输出"右转"）** | **极高** |
| 多步效率下降 | 否（单次推理场景） | **是（平均仅 59.9 步 vs 200+步）** | **极高** |

**对策略的调整**：

原分析中认为 "推理密集型任务可在 Teacher 端使用 thinking"，这个结论仍然成立，但需要补充一个重要限定：

> **即使在 Teacher 端，如果任务涉及多步闭环执行（如 VLN rollout、RL 探索），也应锁定 no-thinking。Thinking 仅限于单次/少次推理的离线数据生成场景。**

修正后的 Teacher thinking 使用范围：
- **可用 thinking**：离线数据标注、单次语义判断生成、困难样本分析、蒸馏数据生成
- **不可用 thinking**：VLN rollout 数据采集、RL 环境探索、在线导航闭环测试、任何多步序贯决策场景

### 7.4 三类数据源的一致性验证

本文档的结论基于三类独立数据源，结论高度一致：

| 数据源 | 任务类型 | Thinking 影响 | 结论 |
|--------|---------|-------------|------|
| **homeobjects 检测基准**（352 任务） | 单次感知（检测、定位、几何） | F1 下降 5.6%，几何关系下降 18.2%，房间分类崩溃 | No-thinking 全面优于 thinking |
| **MMBench-EN Dev**（通用 VLM） | 单次推理（描述性空间推理、逻辑） | 总分提升 8.59%，空间关系提升 24.44% | Thinking 在描述性推理上有优势 |
| **Habitat VLN 导航**（50 episodes） | 多步闭环执行（导航动作序列） | 成功率从对标的 24% 降至 16%，步数从 200+降至 59.9 | Thinking 在闭环导航中**灾难性有害** |

三类数据一致指向同一结论：**Thinking 的推理质量收益无法弥补其在执行稳定性、输出格式、延迟和上下文效率上的系统性损害，尤其在需要结构化输出和多步执行的场景中。**

### 7.5 一句话总结

**Teacher 用 thinking 来"想清楚"（仅限离线单次推理），Student 用 no-thinking 来"做快速"。蒸馏是把"想清楚"的结论教给"做快速"的模型。推理链是 Teacher 的内部工具，不是 Student 的输出目标。任何涉及多步闭环执行的场景（包括 Teacher 端的 VLN rollout 和 RL 探索），都必须锁定 no-thinking。**
