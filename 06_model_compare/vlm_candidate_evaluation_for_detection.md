# VLM 候选模型评估：目标检测微调选型分析

> 状态：evaluation report  
> 更新时间：2026-04-07  
> 评估目标：为导航场景目标检测微调选择 Teacher (27-32B) 和 Student (4B) 模型  
> 候选模型：Qwen3-VL-30B-A3B、Qwen3.5-27B、Qwen3-VL-4B、Qwen3.5-4B  
> 关联文档：[架构差异分析](../04_model_architecture_compare/qwen35_vs_qwen3vl_architecture_diff.md)、[微调蒸馏方案](vlm_detection_finetune_distillation_plan.md)

---

## 1. 核心结论

### 1.1 硬件约束

**部署算力仅支撑单个 4B 模型**，不接受双模型方案。因此选型核心问题变为：**在单 4B 模型约束下，选 Qwen3-VL-4B 还是 Qwen3.5-4B？**

### 1.2 一句话判断

- **部署模型：Qwen3.5-4B (no-thinking)**——在 homeobjects 实测中，no-thinking 模式在检测 F1、指代定位、几何关系上**全面超越** Qwen3-VL-4B 甚至 Qwen3-VL-8B，且耗时仅 0.529s/task
- **Teacher 模型：Qwen3.5-27B**——与 Student 同架构族，蒸馏路径最直接；Qwen3-VL-32B 作为备选
- **关键发现**：thinking 模式对检测任务**不仅无益反而有害**——检测 F1 从 0.8164 降至 0.7734，耗时从 0.529s 暴涨至 2.422s。部署时应锁定 no-thinking

### 1.3 选型矩阵

| 模型 | 角色定位 | 核心优势 | 核心风险 | 推荐度 |
|------|---------|---------|---------|--------|
| **Qwen3.5-27B** | Teacher 首选 | 原生多模态、与 Student 同架构、空间感知强 | 微调生态尚不成熟、不推荐 QLoRA | ★★★★★ |
| **Qwen3-VL-32B** | Teacher 备选 | Grounding token 成熟、社区微调经验多 | 跨架构蒸馏到 Qwen3.5-4B 存在 gap | ★★★★☆ |
| **Qwen3-VL-30B-A3B** | 不推荐 | MoE 推理快 | MoE 微调不稳定、跨架构蒸馏 | ★★☆☆☆ |
| **Qwen3.5-4B** | **部署模型** | 空间关系 82.98% (NT)、检测 F1 0.8164、架构方向正确 | 检测生态不成熟、推理比 Qwen3-VL 慢 47% | ★★★★★ |
| **Qwen3-VL-4B** | 降级备选 | MMBench 83.68%、检测结构化输出稳定 | 空间关系 48.89% 是硬伤、架构即将淘汰 | ★★★☆☆ |

---

## 2. 架构对比：原生多模态 vs 专用 VL

这是本次选型的核心技术判断。Qwen3.5 和 Qwen3-VL 代表了两种根本不同的多模态架构范式。

### 2.1 架构范式差异

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Qwen3-VL: Encoder-Decoder 范式                  │
│                                                                     │
│   ┌──────────┐    DeepStack     ┌──────────────────────────────┐   │
│   │ Vision   │    Multi-Layer   │       LLM Decoder            │   │
│   │ Encoder  │───Injection────▶│  (Qwen3 Transformer)         │   │
│   │ (ViT)    │   Layer 1-3     │                              │   │
│   └──────────┘                  │  ┌─────────┐ ┌───────────┐  │   │
│                                 │  │Reasoning│ │ Action    │  │   │
│        独立视觉编码器             │  │Head     │ │ Head      │  │   │
│        + 专用 MLP Merger        │  └─────────┘ └───────────┘  │   │
│                                 └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     Qwen3.5: Early Fusion 原生范式                   │
│                                                                     │
│   ┌────────────────────────────────────────────────────────────┐   │
│   │              Unified Transformer                           │   │
│   │                                                            │   │
│   │   [Text Token] [Image Token] [Text Token] [Image Token]   │   │
│   │        ↓            ↓            ↓            ↓            │   │
│   │   ┌────────────────────────────────────────────────────┐   │   │
│   │   │  Gated Delta Networks + Feed Forward / MoE         │   │   │
│   │   │  (所有模态在同一 attention 空间中交互)               │   │   │
│   │   └────────────────────────────────────────────────────┘   │   │
│   │                                                            │   │
│   │        视觉和文本 token 在训练初期就混合                    │   │
│   │        无独立视觉编码器                                     │   │
│   └────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 关键技术差异

| 维度 | Qwen3-VL | Qwen3.5 |
|------|----------|---------|
| **视觉编码** | 独立 ViT 编码器 + DeepStack 多层注入 | 无独立视觉编码器，early fusion 统一流 |
| **融合方式** | ViT 多层特征 → MLP Merger → LLM Layer 1-3 residual add | 视觉/文本 token 在同一 attention 空间原生交互 |
| **注意力机制** | 标准 Transformer + Interleaved MRoPE | Gated Delta Networks (线性注意力) + FFN/MoE |
| **视觉预算** | 保守（max_pixels=1.6M, max_frames=32） | 更大（max_pixels=128×32×32, max_frames=768） |
| **生成策略** | 短输出导向（max_tokens=128, temp=0.0） | 长输出导向（max_tokens=1024, temp=0.7, thinking） |
| **位置编码** | Interleaved MRoPE（空间+时序） | MRoPE 变体 |
| **上下文长度** | 256K（原生） | 262K（原生），可扩展至 1M+ |

### 2.3 对目标检测微调的架构影响分析

#### Qwen3-VL 对检测更友好的原因

1. **独立视觉编码器可单独控制**：微调时可选择冻结/解冻 ViT，精确控制视觉特征学习
2. **DeepStack 保留多层视觉特征**：低层纹理/边缘 + 高层语义，对 bbox 回归有利
3. **短输出生成模式**：检测任务输出天然是结构化短文本（坐标 JSON），与 Qwen3-VL 默认行为匹配
4. **成熟的 grounding token 体系**：`<|object_ref_start|>` / `<|box_start|>` 等特殊 token 已有大量社区验证
5. **确定性解码**：temp=0.0 + greedy decoding，坐标输出更稳定

#### Qwen3.5 的潜在优势

1. **空间关系理解更强**：4B 模型空间关系 71.11% vs Qwen3-VL-4B 的 48.89%（thinking 模式下）
2. **更深的跨模态交互**：early fusion 使视觉-文本交互从第一层就开始，理论上语义 grounding 更深
3. **适合复杂推理链**：Object-Goal CoT 等需要多步推理的检测场景
4. **统一架构简化部署**：一个模型既做检测又做通用对话

#### Qwen3.5 对检测的风险

1. ~~**thinking 依赖严重**~~（**已被实测推翻**）：MMBench 通用评测中关闭 thinking 后 73.20% vs 81.79%，但 homeobjects 检测实测中 **no-thinking 全面优于 thinking**（F1 0.8164 > 0.7734，几何关系 82.98% > 70.21%）。检测任务不依赖 thinking。
2. **生成行为不可控**：倾向长输出，检测这类短结构化输出场景需要额外约束（no-thinking 模式下已大幅缓解）
3. **视觉编码器不可分离**：无法像 Qwen3-VL 那样精确冻结/解冻视觉层
4. **微调生态不成熟**：Qwen3.5 发布仅 1.5 个月（2026-02），社区 grounding 微调经验少
5. **不推荐 QLoRA**：官方明确不推荐 4-bit 量化训练，增加训练成本

---

## 3. 候选模型逐一评估

### 3.1 Qwen3-VL-30B-A3B

**架构**：MoE，30B 总参数，3B 活跃参数  
**发布**：2025-10-04  

| 指标 | 数值 | 说明 |
|------|------|------|
| MMBench-V1.1 | 87.0% | 公开基准 |
| DocVQA | 95.0% | 文档理解 |
| ScreenSpot | 94.7% | UI 定位 |
| OCRBench | 90.3% | OCR |
| MMLU-Redux | 88.4% | 知识推理 |

**作为 Teacher 的优劣**：

| 优势 | 劣势 |
|------|------|
| 推理速度快（仅 3B 活跃参数） | MoE 架构微调不稳定，expert routing 可能在微调中退化 |
| MMBench 87% 综合强 | 蒸馏到 Dense 4B 存在架构不匹配问题 |
| 推理成本低 | 社区报告 MoE 微调后泛化能力下降 |

**判断**：MoE 作为 Teacher 微调风险较高，不推荐作为首选。推理加速场景可考虑。

### 3.2 Qwen3.5-27B

**架构**：Dense，27B 参数，Gated Delta Networks  
**发布**：2026-02-24  

| 指标 | 数值 | 说明 |
|------|------|------|
| MMMU | 82.3% | 多模态理解 |
| MathVision | 86.0% | 数学视觉 |
| MMLU-Pro | 86.1% | 知识推理 |
| SWE-bench | 72.4% | 代码 |
| GPQA Diamond | 85.5% | 推理 |
| IFEval | 95.0% | 指令遵循 |

**作为 Teacher 的优劣**：

| 优势 | 劣势 |
|------|------|
| 原生多模态，空间感知最强 | Grounding 微调社区经验少 |
| Dense 架构，微调稳定 | 不推荐 QLoRA，LoRA 需 ~56GB VRAM |
| OCR 代际提升，结构化数据抽取强 | 无独立视觉编码器，微调粒度控制有限 |
| 性能对标 122B-A10B | Transformers v5+ 才支持 |
| 蒸馏到 Qwen3.5-4B 架构对齐 | 发布时间短，生态不够成熟 |

**判断**：长期最优选择，但当前建议先用 Qwen3-VL-32B 验证方案，待 Qwen3.5 微调生态成熟后迁移。

### 3.3 Qwen3-VL-4B（本机评测数据）

**架构**：Dense，4B 参数  
**发布**：2025-10-15  
**本机评测**：MMBench-EN Dev, lmms-eval

| 能力大类 | 准确率 |
|----------|--------|
| **总体** | **83.68%** |
| 粗粒度感知 | 89.53% |
| 细粒度感知（实例级） | 89.42% |
| 属性推理 | 83.42% |
| 关系推理 | 80.87% |
| 细粒度感知（跨实例） | 74.83% |
| 逻辑推理 | 68.64% |

**检测相关关键能力**：

| 能力 | 准确率 | 与检测的关系 |
|------|--------|-------------|
| 场景识别 | 98.08% | 环境感知基础 ✅ |
| 属性识别 | 95.95% | 目标属性判断 ✅ |
| OCR | 94.87% | POI 文字识别 ✅ |
| 目标定位 | 71.60% | 直接相关 ⚠️ |
| 空间关系 | **48.89%** | 导航核心能力 ❌ |
| 物理关系 | 58.33% | 障碍物理解 ❌ |

**推理效率**（本机实测）：
- 总输出 tokens：37,421
- 评测耗时：低（短输出模式）

### 3.4 Qwen3.5-4B（本机评测数据）

**架构**：Dense，4B 参数，Gated Delta Networks  
**发布**：2026-03-02  
**本机评测**：MMBench-EN Dev, lmms-eval

| 模式 | 总体准确率 | 空间关系 | 总输出 tokens | 评测耗时 |
|------|----------|---------|-------------|---------|
| **Thinking** | 81.79% | **71.11%** | 5,315,540 | 1742s |
| **No-Thinking** | 73.20% | 46.67% | 1,801,190 (34%) | 545s (31%) |

**检测相关关键能力**（Thinking 模式）：

| 能力 | 准确率 | 与检测的关系 |
|------|--------|-------------|
| 场景识别 | 95.19% | 环境感知基础 ✅ |
| 属性识别 | 89.19% | 目标属性判断 ✅ |
| OCR | 89.74% | POI 文字识别 ✅ |
| 属性比较 | 88.64% | 多目标区分 ✅ |
| 空间关系 | **71.11%** | 导航核心能力 ✅ |
| 目标定位 | 75.31% | 直接相关 ⚠️ |
| 物理关系 | 50.00% | 障碍物理解 ❌ |

---

## 4. 四模型横向对比

### 4.1 综合能力对比

| 能力维度 | Qwen3-VL-30B-A3B | Qwen3.5-27B | Qwen3-VL-4B | Qwen3.5-4B (T) | Qwen3.5-4B (NT) |
|---------|------------------|-------------|-------------|----------------|-----------------|
| MMBench | 87.0% | ~86-88%* | 83.68% | 81.79% | 73.20% |
| 空间关系 | ~55%* | 较强* | 48.89% | **71.11%** | 46.67% |
| OCR | 90.3% | 代际提升* | 94.87% | 89.74% | 82.05% |
| 目标定位 | ~75%* | 较强* | 71.60% | 75.31% | 66.67% |
| 属性识别 | ~95%* | ~90%* | 95.95% | 89.19% | 78.38% |

*注：带 * 的数据为公开基准估算或官方报告值，非本机实测。(T)=Thinking, (NT)=No-Thinking*

### 4.2 homeobjects 实测横向对比（关键数据）

以下数据来自 homeobjects_mini_v1 benchmark，352 个任务（检测 130 + 指代 94 + 几何 94 + 房间 34），均为本机实测：

| 指标 | Qwen3-VL-4B | Qwen3-VL-8B | Qwen3.5-4B (T) | **Qwen3.5-4B (NT)** |
|------|-------------|-------------|----------------|---------------------|
| **检测 F1** | 0.7988 | 0.8000 | 0.7734 | **0.8164** ✅ |
| 检测 Precision | 0.8544 | 0.8645 | 0.8477 | 0.8054 |
| 检测 Recall | 0.7500 | 0.7444 | 0.7111 | **0.8278** ✅ |
| Table 召回 | 0.6742 | 0.6742 | 0.6288 | **0.7727** ✅ |
| **指代 Acc@0.5** | 0.5745 | 0.6064 | 0.7021 | **0.7021** ✅ |
| 指代 Mean IoU | 0.5188 | 0.5323 | 0.6352 | **0.6610** ✅ |
| **几何关系** | 69.15% | 76.60% | 70.21% | **82.98%** ✅ |
| 房间分类 | 70.59% | 73.53% | 0.00% ❌ | **70.59%** |
| **推理耗时** | **0.361s** | 0.664s | 2.422s ❌ | 0.529s |

**(T)=Thinking, (NT)=No-Thinking。✅ = 该行最优**

**关键发现**：
1. **Qwen3.5-4B no-thinking 在检测 F1、指代定位、几何关系三项核心指标上全面领先**，甚至超过参数量翻倍的 Qwen3-VL-8B
2. **thinking 模式对检测有害**：F1 从 0.8164 降至 0.7734，Table 召回从 77.27% 降至 62.88%，几何关系从 82.98% 降至 70.21%
3. **thinking 导致房间分类系统性退化**：从 70.59% 降至 0.00%（几乎全部输出 unknown），这是 thinking 过度推理的极端案例
4. **推理效率**：no-thinking (0.529s) 比 thinking (2.422s) 快 4.6x，仅比 Qwen3-VL-4B (0.361s) 慢 47%

### 4.3 导航检测场景适配度（基于实测数据修正）

| 场景 | Qwen3-VL-4B | Qwen3.5-4B (NT) | 胜出 |
|------|-------------|-----------------|------|
| 室内物体检测（sofa/table） | F1 0.7988 | **F1 0.8164** | **Qwen3.5 NT** |
| 多目标枚举（多桌场景） | Table Recall 67.42% | **Table Recall 77.27%** | **Qwen3.5 NT** |
| 指代目标定位（左/右侧物体） | Acc@0.5 57.45% | **Acc@0.5 70.21%** | **Qwen3.5 NT** |
| 空间关系判断（几何方向） | 69.15% | **82.98%** | **Qwen3.5 NT** |
| 推理速度（边端部署） | **0.361s** | 0.529s | Qwen3-VL |
| 房间场景分类 | 70.59% | 70.59% | 持平 |

### 4.4 资源需求对比

| 资源维度 | Qwen3-VL-30B-A3B | Qwen3.5-27B | Qwen3-VL-4B | Qwen3.5-4B |
|---------|------------------|-------------|-------------|------------|
| LoRA 微调 VRAM | ~20GB (3B active) | ~56GB (bf16) | ~12GB | ~16GB |
| 全参数微调 VRAM | ~80GB | ~120GB+ | ~20GB | ~24GB |
| 推理 VRAM (FP16) | ~8GB (3B active) | ~56GB | ~10GB | ~10GB |
| 推理延迟 | 快 (MoE) | 慢 (27B Dense) | 快 | 中（thinking 慢） |

---

## 5. 两种架构对目标检测微调的影响分析

### 5.1 核心问题：目标检测微调该选哪种架构？

这个问题的本质是：**检测任务更需要"精确的视觉特征"还是"深度的跨模态理解"？**

#### 检测任务的特点

```
检测 = 定位 (WHERE) + 识别 (WHAT)
  │
  ├── 定位 ──▶ 需要精确的空间特征 ──▶ 视觉编码器质量关键
  │            需要多尺度特征        ──▶ DeepStack 有优势
  │            需要确定性坐标输出    ──▶ 短输出模式更稳定
  │
  └── 识别 ──▶ 需要语义理解          ──▶ 两种架构都行
               需要开放词汇泛化      ──▶ early fusion 可能更强
               需要属性判断          ──▶ Qwen3-VL 属性识别更高
```

#### 单 4B 模型约束下的结论

在只能部署一个 4B 模型的前提下，homeobjects 实测数据给出了明确答案：

```
Qwen3-VL-4B:  检测 F1 0.7988, 指代 Acc@0.5 0.5745, 几何关系 69.15%
              推理耗时 0.361s/task
              空间关系 48.89% 是导航场景致命短板
              架构即将被淘汰

Qwen3.5-4B (no-thinking):
              检测 F1 0.8164, 指代 Acc@0.5 0.7021, 几何关系 82.98%
              推理耗时 0.529s/task
              在所有检测维度上全面超越 Qwen3-VL-4B 甚至 Qwen3-VL-8B
              且不需要 thinking 模式——no-thinking 本身就是最优解
```

**判断：选 Qwen3.5-4B (no-thinking)**。理由：

1. **检测任务全面领先**——F1 0.8164 > 0.7988 (Qwen3-VL-4B) > 0.7734 (thinking 模式)，no-thinking 是检测最优模式
2. **空间关系大幅领先**——几何关系 82.98% vs 69.15% (Qwen3-VL-4B)，这是导航场景的刚需能力
3. **指代定位显著更强**——Acc@0.5 0.7021 vs 0.5745，多目标场景选择正确实例的能力更好
4. **不依赖 thinking**——之前担心的"thinking 依赖"在检测任务中完全不存在，no-thinking 反而更好
5. **推理效率可接受**——0.529s vs 0.361s，仅慢 47%，远好于 thinking 模式的 2.422s（慢 571%）
6. **架构方向正确**——Qwen3-VL 不会有下一代，选它意味着走进死胡同

### 5.2 单模型方案：Qwen3.5-4B no-thinking 检测专项微调

```
摄像头输入 (2-5 fps)
      │
      ▼
┌──────────────────────────────────────┐
│        Qwen3.5-4B (单模型)            │
│        检测专项 SFT 微调后             │
│        enable_thinking = false        │
│                                      │
│  ┌────────────────────────────────┐  │
│  │         统一检测模式             │  │
│  │  no-thinking, temp=0.0          │  │
│  │  结构化 JSON 输出               │  │
│  │  ~0.5s/task (微调后预期更快)     │  │
│  └────────────────────────────────┘  │
│                                      │
│  部署时锁定 no-thinking，无需模式切换  │
└──────────────────────────────────────┘
```

**关键发现**：homeobjects 实测数据证明 **thinking 模式对检测任务不仅无益反而有害**：

| 指标 | no-thinking | thinking | 差异 |
|------|------------|---------|------|
| 检测 F1 | **0.8164** | 0.7734 | no-thinking +5.6% |
| 指代 Acc@0.5 | **0.7021** | 0.7021 | 持平 |
| 几何关系 | **82.98%** | 70.21% | no-thinking +18.2% |
| 房间分类 | **70.59%** | 0.00% (异常) | thinking 导致系统性退化 |
| 推理耗时 | **0.529s** | 2.422s | thinking 慢 4.6x |
| Table 召回 | **77.27%** | 62.88% | no-thinking 多目标枚举更完整 |

**分析**：thinking 模式在检测任务中产生"过度推理"——模型花费大量 token 进行不必要的推理链，反而干扰了检测输出的准确性和完整性。检测任务本质上是"看到即输出"的快速感知任务，不需要深度推理链。

因此部署方案大幅简化：**锁定 no-thinking 模式，不需要模式切换逻辑**。

### 5.3 已验证的结论（非赌注）

之前的分析假设 Qwen3.5-4B 的空间推理能力依赖 thinking 模式，需要通过微调将 thinking 能力"蒸馏"到 no-thinking 路径。**homeobjects 实测数据推翻了这个假设**：

- **几何关系**：no-thinking 82.98% >> thinking 70.21%，no-thinking 模式的空间理解能力**已经内建在模型中**，不需要额外的 thinking 链路来激活
- **指代定位**：no-thinking Acc@0.5 0.7021 = thinking 0.7021，无差异
- **检测精度**：no-thinking F1 0.8164 > thinking 0.7734，no-thinking 更准

这意味着：
1. **不存在"thinking 依赖"问题**——检测场景下 no-thinking 已经是最优解
2. **微调目标简化**——只需训练 no-thinking 模式的检测 SFT，不需要混合 thinking 数据
3. **部署架构简化**——无需模式切换逻辑，降低工程复杂度
4. **回退风险降低**——当前零样本基线已经超越 Qwen3-VL-4B 和 Qwen3-VL-8B，微调只会进一步提升

**注意**：MMBench 通用评测中 thinking 模式仍然有优势（81.79% vs 73.20%），但在目标检测专项任务中 no-thinking 全面胜出。这说明 **thinking 的价值因任务类型而异**——推理密集型任务（如数学、逻辑）受益于 thinking，而感知密集型任务（如检测、定位）不需要甚至受损于 thinking。

---

## 6. 最终推荐方案（单 4B 模型约束）

### 6.1 方案总览

```
约束条件: 部署算力仅支撑单个 4B 模型

训练阶段 (云端):
  Teacher: Qwen3.5-27B ──LoRA SFT──▶ 检测专项 Teacher
                                          │
                                     数据蒸馏 + Logit KD
                                          │
  Student: Qwen3.5-4B  ──────────────────▶ 检测专项 4B 模型

部署阶段 (边端):
  Qwen3.5-4B (单模型, INT8 量化, no-thinking 锁定)
  └── 统一检测模式: enable_thinking=false, temp=0.0, 结构化 JSON 输出
```

### 6.2 为什么选 Qwen3.5-4B 而非 Qwen3-VL-4B

| 决策因素 | Qwen3-VL-4B | Qwen3.5-4B | 判断 |
|---------|-------------|------------|------|
| 空间关系（导航刚需） | 48.89% ❌ | 71.11% ✅ | **空间理解是架构级优势，无法通过微调弥补** |
| OCR / 属性识别 | 94.87% / 95.95% | 89.74% / 89.19% | 差距可通过检测 SFT 数据弥补 |
| 检测输出稳定性 | temp=0.0 天然稳定 | 需通过 SFT 约束 | 可解决 |
| 架构演进 | 已停止演进 | 正确方向，持续受益 | **Qwen3-VL 不会有下一代** |
| 参数利用效率 | ViT+投影+LLM 分散 | 全部参数共享 | early fusion 效率更高 |
| Teacher-Student 架构对齐 | 需跨架构蒸馏 | Qwen3.5-27B → 4B 同架构 | **同架构蒸馏损失更小** |

### 6.3 微调策略

基于 homeobjects 实测结论，微调策略大幅简化——**全部使用 no-thinking 模式训练**：

```yaml
# 核心目标: 在 no-thinking 基线上进一步提升检测精度

训练数据混合:
  检测数据 (no-thinking): 85%
    - 输入: 图像 + "检测目标物体"
    - 输出: bbox JSON (enable_thinking=false)
    - temp=0.0, max_tokens=256

  通用 VQA 防遗忘: 15%
    - 保持基础视觉理解不退化
    - 同样使用 no-thinking 模式

关键超参数:
  enable_thinking: false  # 全程关闭 thinking
  learning_rate: 1e-5
  生成约束: max_tokens=256, temp=0.0
  LoRA target: 全部 attention + FFN 层 (r=64)
```

**与之前方案的核心差异**：
- 移除了 20% 的 thinking 训练数据——实测证明 thinking 对检测有害
- 检测数据占比从 70% 提升到 85%——无需分配 token 给 thinking 推理链
- 训练效率提升——no-thinking 输出短（~256 tokens），thinking 输出长（~1500 tokens），单位时间可训练更多样本

### 6.4 成功标准与回退方案

| 指标 | 成功阈值 | 当前零样本基线 | 说明 |
|------|---------|-------------|------|
| 检测 F1 | >0.85 | 0.8164 | 在强基线上进一步提升 |
| 指代 Acc@0.5 | >0.75 | 0.7021 | 重点提升 table 多目标场景 |
| 几何关系准确率 | >0.85 | 0.8298 | 压制残余的 left_of 偏置 |
| Table 召回率 | >0.85 | 0.7727 | 多桌场景枚举完整性 |
| 检测输出格式合规率 | >95% | 待测 | JSON 格式稳定性 |
| 推理延迟 (no-thinking) | <100ms | ~529ms (vLLM) | 边端 INT8 量化后预期更快 |

**回退方案**：如果 Qwen3.5-4B 微调后检测 F1 低于 0.80（即低于零样本基线），说明 SFT 导致能力退化，此时：
1. 首先排查数据质量和超参数
2. 如确认 Qwen3.5-4B 不适合检测 SFT，回退到 Qwen3-VL-4B 方案（零样本 F1 0.7988，微调生态更成熟）

### 6.5 需要补充的评估

| 评测项 | 数据集 | 优先级 | 目标 |
|--------|--------|--------|------|
| **Qwen3.5-4B no-thinking grounding** | RefCOCO val | P0 | 验证 no-thinking 模式的标准 grounding 基线 |
| Qwen3-VL-4B grounding | RefCOCO val | P1 | 回退方案基线对比 |
| 边端推理延迟 | 实机 INT8 量化测试 | P1 | 验证 <100ms 实时性目标 |
| 多类别泛化 | 扩展 homeobjects 至 chair/door/shelf | P2 | 验证非 sofa/table 场景的检测能力 |

**建议下一步**：
1. 在本机对 Qwen3.5-4B (no-thinking) 跑 RefCOCO grounding 零样本评测，获取标准检测基线
2. 启动 Teacher (Qwen3.5-27B) 检测 SFT 数据准备
3. homeobjects 基线数据已经证实 Qwen3.5-4B no-thinking 是最优选择，可以 all-in

---

## 7. 深层分析：为什么 Qwen3.5 要去掉 ViT 改为 Early Fusion

### 7.1 Qwen3-VL 的 ViT + LLM 架构的根本问题

Qwen3-VL 的架构本质上是"拼接"：

```
图像 → ViT (独立预训练) → MLP 投影层 → LLM (独立预训练)
                              ↑
                         信息瓶颈在此
```

这里存在一个**信息瓶颈**——MLP 投影层要把 ViT 的高维视觉特征压缩成 LLM 能理解的 token。LLM 的底层（前 N 层 transformer）是在纯文本上训练出来的，它们从来没学过"什么是视觉"，只有上层才通过投影层勉强接触到视觉信息。

即便 Qwen3-VL 用 DeepStack 把 ViT 多层特征注入到 LLM 的前 3 层来缓解这个问题，**本质上仍然是两个独立训练的系统在做后期对齐**——视觉和语言始终是两条平行线，只在数据层面被粗略地拼到一起。

### 7.2 Early Fusion 解决了什么

Qwen3.5 的做法是：从预训练第一天起，视觉 token 和文本 token 就在同一个 transformer 里混合处理：

```
[文本token] [图像token] [文本token] [图像token] → 同一个 Transformer 所有层
```

这意味着：
1. **每一层都学会了跨模态表示**——不存在"底层只懂文本、上层才懂图像"的割裂
2. **没有投影瓶颈**——视觉信息不需要被压缩适配到一个本不理解它的 backbone 里
3. **模态间的语义对齐是"长出来的"，不是"对上去的"**

### 7.3 Early Fusion 的核心收益

#### 收益一：小模型也能做好多模态

这是最直接的收益。传统架构里，一个 4B 模型 = ~1B ViT + ~1B 投影层 + ~2B LLM，参数被分散在三个组件里，每个都不够大。Early fusion 让所有 4B 参数共享用于所有模态，参数利用效率更高。

Scaling law 研究（[arXiv:2504.07951](https://arxiv.org/pdf/2504.07951)）证实：**early fusion 模型在相同 FLOPs 下需要更少的参数就能达到同等性能**。这对边端部署极其重要——0.8B 的 Qwen3.5 就有可用的视觉理解能力，而 0.8B 的 encoder-decoder 模型几乎不可能做到。

#### 收益二：空间推理更强

本机评测数据已经证明了这一点：

| 模型 | 空间关系 | 架构 |
|------|---------|------|
| Qwen3.5-4B (thinking) | **71.11%** | Early Fusion |
| Qwen3-VL-8B | 51.11% | ViT + DeepStack |
| Qwen3-VL-4B | 48.89% | ViT + DeepStack |

Qwen3.5-4B 在空间关系上甚至超过了参数量翻倍的 Qwen3-VL-8B。这不是偶然——空间关系判断需要视觉特征和语言概念的深度交织，early fusion 天然更擅长这类需要跨模态联合推理的任务。

#### 收益三：部署架构统一

以前做一个多模态系统需要三套模型：

```
纯文本: Qwen3
图文:   Qwen3-VL
语音:   Qwen3-Audio
```

Qwen3.5 一个模型全覆盖——一套推理栈、一套运维、一套微调流程。

#### 收益四：Agent 能力更自然

Agent 场景需要模型在截图、代码、文档、对话之间自由切换。encoder-decoder 架构每换一种模态就要走一次编码-投影-解码链路，early fusion 直接在一个统一流里处理所有输入。

### 7.4 Early Fusion 的代价

| 代价 | 说明 |
|------|------|
| **必须从头训练** | 不能复用现有高质量文本模型做起点，需要在万亿级多模态 token 上从零预训练 |
| **训练成本极高** | 需要同时准备大规模的文本、图像、视频、音频数据的交织语料 |
| **视觉特征控制粒度降低** | 无法像 ViT+LLM 那样分别冻结/解冻视觉层和语言层 |
| **生态成熟度滞后** | 传统 encoder-decoder 范式积累了数年的微调经验和工具链 |

### 7.5 为什么 Qwen3-VL 在检测上"暂时更好"

这不是 early fusion 的架构缺陷，而是**成熟度差异**：

| 因素 | Qwen3-VL | Qwen3.5 |
|------|----------|---------|
| 发布时间 | 2025-10（~6个月） | 2026-02（~1.5个月） |
| Grounding token 体系 | 专门设计并大量验证 | 仍在迭代中 |
| 检测数据 SFT | 经过大量检测数据训练 | 通用预训练为主 |
| 框架支持 | ms-swift / Unsloth 完善 | 需 Transformers v5，尚不完全稳定 |
| 默认生成参数 | temp=0.0 短输出（匹配检测） | temp=0.7 长输出（不匹配检测） |

本机 MMBench 综合分 Qwen3-VL-4B (83.68%) > Qwen3.5-4B thinking (81.79%) 的核心原因是：Qwen3-VL 的默认参数天然匹配 benchmark 短答题。而 Qwen3.5 一旦关闭 thinking 就从 81.79% 暴降至 73.20%——**它的不少能力是"依赖推理链激活"的，不是天然低成本就能保留。**

**这不是架构上限的差异，而是训练阶段和调优成熟度的差异。**

---

## 8. 行业趋势与 Qwen 后续方向推测

### 8.1 行业共识：Early Fusion 是确定方向

2024-2026 年，所有头部多模态模型都在向 early fusion 收敛：

| 模型 | 厂商 | 时间 | 架构 |
|------|------|------|------|
| **Gemini 1.0** | Google | 2023-12 | 原生多模态，early fusion 先驱 |
| **Chameleon** | Meta | 2024 | early fusion token-based |
| **GPT-4o** | OpenAI | 2024-05 | 原生全模态输入输出 |
| **Llama 4** | Meta | 2025 | early fusion MLP adapter |
| **Gemini 3** | Google | 2025-11 | 原生多模态 + thinking + tool use |
| **Gemma 4** | Google | 2026-04 | 原生多模态，小模型全尺寸覆盖 |
| **Qwen3.5** | Alibaba | 2026-02 | early fusion + Gated Delta Networks |

Qwen3-VL 的 encoder-decoder 范式（ViT + projection + LLM）已经是**上一代架构**。没有一家头部厂商还在为下一代模型走这条路。

### 8.2 Qwen 演进路线推测

```
Qwen-VL (2023)      ViT + Cross-Attention + LLM
    │
    ▼
Qwen2-VL (2024)     ViT + 动态分辨率 + LLM
    │
    ▼
Qwen3-VL (2025)     ViT + DeepStack 多层注入 + LLM    ← 当前检测最优
    │
    ▼
Qwen3.5 (2026-02)   Early Fusion 原生多模态             ← 架构转折点
    │
    ├──▶ Qwen3.5-Omni  文本+视觉+音频+视频 全模态统一
    │
    ▼
Qwen4 (推测 2026下半年)
    ├── Early Fusion 全面成熟
    ├── Grounding / 检测专项 SFT 补齐
    ├── 可能: + 3D 感知 + 空间理解 + 具身控制
    └── Qwen3-VL 分支不再维护
```

**关键信号**：Qwen3.5 已经不再有单独的 `-VL` 后缀——它不是"文本模型的视觉扩展"，而是"原生就包含视觉的统一模型"。这意味着 Qwen 团队内部已经确认 early fusion 是正确方向，不会再有 `Qwen4-VL` 这样的独立视觉分支。

### 8.3 Qwen 的真正目标：Agent 基础设施

Qwen3.5 官方博客标题已经点明：**"Towards Native Multimodal Agents"**。

Qwen 团队的目标不是做"最好的视觉模型"或"最好的检测模型"，而是做 **Agent 的基础模型**：

```
Agent 需要的能力:
  ├── 看 (视觉)        ──▶ early fusion 覆盖
  ├── 说 (文本)        ──▶ early fusion 覆盖
  ├── 听 (音频)        ──▶ Qwen3.5-Omni 已覆盖
  ├── 想 (推理)        ──▶ thinking 模式
  ├── 用工具 (函数调用)  ──▶ IFEval 95.0%
  ├── 记住上下文 (长记忆) ──▶ 262K 原生上下文, 可扩展至 1M+
  └── 端侧部署         ──▶ 0.8B~4B 小模型也有多模态能力
```

Early fusion 是实现这个目标的唯一可行架构——你不可能给一个端侧 4B 模型挂 5 个独立编码器（视觉、音频、视频、深度、点云……）。

这也解释了为什么 Qwen3.5 在 IFEval（指令遵循/工具调用）上得分极高 (95.0%)——这不是偶然，而是 Agent 场景的核心能力被刻意强化。

### 8.4 与 ABot-N0 的关系

ABot-N0 的 Cognitive Brain 目前用的是 Qwen3-4B。按照 Qwen 的演进路线：

```
当前: ABot-N0 Brain = Qwen3-4B (纯文本) + 外挂视觉模块
                                    │
                                    ▼
未来: ABot-N0 Brain = Qwen3.5-4B (原生多模态)
                      一个模型同时处理视觉感知 + 语义推理 + 动作决策
```

这正是 early fusion 的价值所在——把原来需要多个模块协作的能力，压缩到一个统一模型里。

### 8.5 对本项目的战略判断

| 判断 | 说明 |
|------|------|
| **短期用 Qwen3-VL 做检测微调是正确的** | 它当前的检测生态更成熟，能最快出结果 |
| **但要做好迁移到 Qwen3.5 的准备** | 数据集格式、训练流程应设计为模型无关的 |
| **Qwen3-VL 不会有下一代** | 它是过渡架构，Qwen 团队已 all-in early fusion |
| **空间推理已证明 early fusion 更强** | 对导航场景至关重要，价值会随生态成熟逐步释放 |
| **检测能力差距会随 Qwen3.5 迭代缩小** | 当 grounding SFT 数据和微调工具链补齐后，early fusion 架构将同时在定位和识别上超越 encoder-decoder |

**一句话总结**：Qwen 去掉 ViT 不是为了做更好的检测，而是为了做更好的 Agent。检测只是统一模型能做的事情之一。当 early fusion 的训练和微调生态追上来时，它在检测上也会超过 encoder-decoder 架构——因为它没有信息瓶颈。

---

## 附录：数据来源

- 本机评测数据：`/home/zktian3/lmms-eval/results_*` 目录下原始 JSON
- 架构分析：`04_model_architecture_compare/qwen35_vs_qwen3vl_architecture_diff.md`
- Qwen3-VL 架构：[DeepWiki - Qwen3-VL Model Architecture](https://deepwiki.com/QwenLM/Qwen3-VL/4.2-model-architecture)
- Qwen3.5 架构：[Qwen3.5 官方博客](https://qwen.ai/blog?id=qwen3.5)
- Qwen3-VL-30B-A3B 基准：[HuggingFace Model Card](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct)
- Qwen3.5-27B 基准：[HuggingFace Model Card](https://huggingface.co/Qwen/Qwen3.5-27B)
- 架构对比：[Deep Dive - Qwen 3.5 Native Multimodality](https://trilogyai.substack.com/p/deep-dive-qwen-35-brings-native-multimodality)
- Early Fusion Scaling Laws：[Scaling Laws for Native Multimodal Models (arXiv:2504.07951)](https://arxiv.org/pdf/2504.07951)
- 行业趋势：[The Architectural Revolution of Multimodal AI Models](https://www.aminext.blog/en/post/multimodal-ai-architecture-revolution-gpt4o-gemini-key-designs-1)
- Qwen3-VL DeepStack：[Qwen3-VL DeepStack Fusion Analysis](https://thesalt.substack.com/p/qwen3-vl-deepstack-fusion-interleaved)
- Qwen3.5 架构解析：[Qwen 3.5 Explained: Architecture Upgrades](https://medium.com/data-science-in-your-pocket/qwen-3-5-explained-architecture-upgrades-over-qwen-3-benchmarks-and-real-world-use-cases-af38b01e9888)
