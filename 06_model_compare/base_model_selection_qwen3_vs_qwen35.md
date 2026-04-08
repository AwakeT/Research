# 基座模型选型分析：Qwen3 vs Qwen3.5

---

文档版本：v1.0
创建日期：2026-04-08
文档性质：模型选型分析
适用范围：Kinbot VLN → NFM 演进过程中 Teacher/Student 基座模型选择

---

## 1. 选型结论

| 角色 | 推荐模型 | 备选模型 | 理由 |
|------|---------|---------|------|
| **Teacher (27B级)** | **Qwen3.5-27B** | Qwen3-VL-32B | 原生多模态、同架构族蒸馏路径最直接 |
| **Student (4B级)** | **Qwen3.5-4B (no-thinking)** | Qwen3-VL-4B | homeobjects 实测全面领先，空间关系理解大幅优于 Qwen3-VL-8B |

---

## 2. 备选模型系列概览

### 2.1 Qwen3 系列

- **发布时间**：2025年4月29日
- **Dense 模型**：0.6B, 1.7B, 4B, 8B, 14B, 32B
- **MoE 模型**：Qwen3-30B-A3B（30B 总参 / 3B 激活）、Qwen3-235B-A22B（235B 总参 / 22B 激活）
- **训练数据**：~36万亿 token（网页、书籍、PDF、合成代码/数学）
- **上下文**：32K 原生，2507 更新后扩展至 1M
- **语言**：119 种

**Qwen3-VL（视觉语言）**：
- **发布时间**：2025年9-10月
- **尺寸**：2B, 4B, 8B, 32B（Dense）；30B-A3B, 235B-A22B（MoE），均有 Instruct 和 Thinking 变体
- **上下文**：256K（文本 + 交错图像/视频）
- **关键能力**：3D Grounding（可输出 3D bounding box）、2D Grounding（相对坐标）、视觉 Agent、32 语言 OCR
- **架构**：Encoder-Decoder 范式（独立 ViT + DeepStack 多层特征融合 + LLM），Interleaved-MRoPE 时空编码

### 2.2 Qwen3.5 系列

- **发布时间**：2026年2-3月
- **旗舰**：Qwen3.5-397B-A17B（MoE）
- **中型模型**（2月24日）：Qwen3.5-122B-A10B、Qwen3.5-35B-A3B、**Qwen3.5-27B（Dense）**、Qwen3.5-Flash
- **小型模型**（3月2日）：Qwen3.5-0.8B、2B、**4B**、9B
- **上下文**：262K 原生，可扩展至 1M
- **语言**：201 种
- **许可**：全部 Apache 2.0

**架构重大变化**：
- **原生多模态**：不存在单独的 "Qwen3.5-VL" 产品线，所有 Qwen3.5 模型均为统一视觉-语言模型，训练阶段即采用 Early Fusion 处理多模态 token
- **Gated Delta Networks + Gated Attention 混合架构**：8x(3xDeltaNet→FFN→1xAttention→FFN) 模式 + 稀疏 MoE
- **训练效率**：FP8 训练管线，多模态训练效率接近纯文本
- **RL 训练**：百万级 Agent 环境下的大规模强化学习

---

## 3. 架构对比：Qwen3-VL vs Qwen3.5

| 维度 | Qwen3-VL | Qwen3.5 |
|------|---------|---------|
| **架构范式** | Encoder-Decoder（独立 ViT + MLP 投影 + LLM） | Early Fusion（视觉和文本 token 在统一 Transformer 中处理） |
| **视觉处理** | 独立 ViT 编码器 + DeepStack 多层融合 | 原生多模态，无独立视觉编码器 |
| **模型类** | `Qwen3VLForConditionalGeneration` | `Qwen3_5ForConditionalGeneration` / `Qwen3_5MoeForConditionalGeneration` |
| **Thinking 模式** | 无显式 toggle | 支持 `enable_thinking` |
| **视觉预算** | 保守（max_pixels=1.6M, max_frames=32） | 更大（max_pixels=128x32x32, max_frames=768） |
| **输出风格** | 短输出（max_new_tokens=128, temp=0.0） | 长输出（max_new_tokens=1024, temp=0.7） |
| **3D Grounding** | 显式支持（3D bounding box） | 原生多模态空间能力，但无专项文档 |
| **演进方向** | 已停止演进（官方不再发布新版本） | 长期主线方向 |

**核心判断**：Qwen3-VL 的 Encoder-Decoder 架构是过渡方案，Qwen3.5 的 Early Fusion 是长期方向。选择 Qwen3.5 系列可以避免后续被迫迁移架构的风险。

---

## 4. 实测数据对比

### 4.1 homeobjects 基准测试（352 个任务）

数据来源：`vlm_candidate_evaluation_for_detection.md`（2026-04-07）

| 指标 | Qwen3-VL-4B | Qwen3-VL-8B | Qwen3.5-4B (Thinking) | **Qwen3.5-4B (No-Thinking)** |
|------|------------|------------|----------------------|------------------------------|
| Detection F1 | 0.7988 | 0.8000 | 0.7734 | **0.8164** |
| Referring Acc@0.5 | 0.5745 | 0.6064 | 0.7021 | **0.7021** |
| 空间关系准确率 | 69.15% | 76.60% | 70.21% | **82.98%** |
| 房间分类准确率 | 70.59% | 73.53% | 0.00%（系统性失败） | **70.59%** |
| 推理延迟 | **0.361s** | 0.664s | 2.422s | 0.529s |

**关键发现**：
- Qwen3.5-4B (no-thinking) 在空间关系准确率上全面碾压 Qwen3-VL-8B（82.98% vs 76.60%），4B 模型超越 8B 模型
- Referring grounding 能力提升显著（0.7021 vs 0.6064），说明 Early Fusion 对细粒度空间定位有结构性优势
- 推理延迟可控（0.529s），满足部署要求

### 4.2 MMBench-EN Dev 通用基准

数据来源：`06_model_compare/` 目录下各 eval report

| 指标 | Qwen3-VL-4B | Qwen3-VL-8B | Qwen3.5-4B (Thinking) | Qwen3.5-4B (No-Thinking) |
|------|------------|------------|----------------------|--------------------------|
| 总分 | 83.68% | **84.97%** | 81.79% | 73.20% |
| 空间关系 | 48.89% | 51.11% | **71.11%** | 46.67% |
| 场景识别 | **98.08%** | 96.15% | 92.31% | 90.38% |
| OCR | **94.87%** | 87.18% | 82.05% | 76.92% |

**关键发现**：
- 通用 benchmark 上 Qwen3-VL-8B 总分最高，但**空间关系能力**（导航核心指标）Qwen3.5-4B thinking 模式大幅领先（71.11% vs 51.11%）
- No-thinking 模式在通用 benchmark 上显著弱于 thinking 模式，但在实际任务（homeobjects）上反而更优，说明通用 benchmark 不能完全代表任务表现

### 4.3 Thinking vs No-Thinking 模式结论

| 场景 | 推荐模式 | 理由 |
|------|---------|------|
| 检测/导航/部署任务 | **No-Thinking** | F1 更高、延迟更低、输出格式更稳定 |
| 复杂推理/难例审核 | Thinking | 空间推理能力更强（71.11% vs 46.67%） |
| Teacher 研究阶段 | 按需切换 | 数据生成用 thinking，蒸馏对齐用 no-thinking |
| Student 端侧部署 | **锁定 No-Thinking** | 延迟从 2.4s 降至 0.5s，F1 反而更优 |

---

## 5. 对 VLN → NFM 分阶段设计的映射

结合 `kinbot_vln_model_detailed_design.md` 的分阶段设计：

### 5.1 各阶段模型配置

| 阶段 | Teacher | Student | 蒸馏说明 |
|------|---------|---------|---------|
| **P0** 空间+地图+基础空间记忆 | Qwen3.5-27B | Qwen3.5-4B v1 | 同架构族，data distillation 为主 |
| **P1-high** 多视角+粗粒度定位 | Qwen3.5-27B | Qwen3.5-4B v2 | Early Fusion 天然支持多图输入扩展 |
| **P1-medium** 深度几何增强 | Qwen3.5-27B | Qwen3.5-4B v3 | 深度图作为额外 token 输入 |
| **P1-low** RL 策略优化 | Qwen3.5-27B / 4B | Qwen3.5-4B v4 | RL 可在 4B 上直接微调 |
| **P2** 个性化长期记忆 | Qwen3.5-27B | Qwen3.5-4B v5 | 显式记忆库方案不依赖架构变更 |

### 5.2 蒸馏路径优势

选择同系列（Qwen3.5）Teacher/Student 的核心优势：

1. **同架构族**：Early Fusion 统一架构，中间表示对齐更容易，避免 Encoder-Decoder → Early Fusion 跨架构蒸馏的额外复杂度
2. **原生多模态**：无需单独处理视觉编码器的蒸馏，视觉 token 和文本 token 在同一空间中
3. **可复用蒸馏流程**：每轮新能力稳定后的蒸馏流程一致（27B → 4B 或 27B → 9B → 4B）

已规划的三条蒸馏路线（来自 `vlm_detection_finetune_distillation_plan.md`）：
- **路线 A（推荐首选）**：Data Distillation — Teacher 生成结构化判断结果，Student 学习
- **路线 B**：Logit-level KD — 输出概率分布对齐
- **路线 C**：Progressive Distillation — 27B → 9B → 4B 逐级蒸馏

---

## 6. 27B Teacher 模型详细对比

### 6.1 Qwen3.5-27B（推荐）

- **参数量**：27.8B Dense
- **架构**：64 层，5120 hidden dim，Early Fusion
- **原生多模态**：视觉和文本统一处理
- **上下文**：262K
- **部署**：单卡 A100 80GB 可跑 BF16；4-bit 量化后可在消费级 GPU 上运行
- **许可**：Apache 2.0
- **Benchmark**：MMLU-Pro 86.1%、GPQA Diamond 85.5%、MMMU 82.3%、MathVision 86.0%

### 6.2 Qwen3-VL-32B（备选）

- **参数量**：32B Dense
- **架构**：Encoder-Decoder（ViT + DeepStack + LLM）
- **上下文**：256K
- **显式 3D Grounding**：可输出 3D bounding box
- **Instruct / Thinking 双变体**

### 6.3 Qwen3.5-35B-A3B（关注）

- **参数量**：35B 总参 / 3B 激活（MoE）
- **推理成本**：显著低于 27B Dense
- **性能**：官方声称超越 Qwen3-235B-A22B（22B 激活）
- **注意**：MoE 路由增加部署复杂度，边缘硬件兼容性需验证

### 6.4 Teacher 选型建议

**主线选择 Qwen3.5-27B 的理由**：
1. Dense 架构部署简单，无 MoE 路由开销
2. 与 Student（Qwen3.5-4B）同架构族，蒸馏路径最直接
3. 原生多模态，无需维护独立视觉编码器
4. 单卡 A100 可跑，研究迭代效率高

**保留 Qwen3-VL-32B 为备选的理由**：
1. 显式 3D Grounding 能力可能在 P1-medium 深度几何阶段有价值
2. 如果 Qwen3.5-27B 在特定空间推理任务上表现不足，可作为对比验证

---

## 7. 4B Student 模型详细对比

### 7.1 Qwen3.5-4B（推荐）

- **参数量**：~4B Dense
- **架构**：Early Fusion，原生多模态
- **部署**：INT8 ~5GB，INT4 ~3GB
- **推理速度**：215.9 tokens/s（中位数）
- **许可**：Apache 2.0
- **homeobjects 实测**：全面领先（详见第 4 节）

### 7.2 Qwen3-VL-4B（备选）

- **参数量**：~4B Dense
- **架构**：Encoder-Decoder
- **显式 3D Grounding**：有
- **homeobjects 实测**：Detection F1 0.7988，空间关系 69.15%，均弱于 Qwen3.5-4B

### 7.3 Student 选型建议

**Qwen3.5-4B (no-thinking) 全面占优**：
- Detection F1：0.8164 vs 0.7988（+2.2%）
- Referring Acc@0.5：0.7021 vs 0.5745（+22.2%）
- 空间关系：82.98% vs 69.15%（+20.0%）
- 推理延迟：0.529s vs 0.361s（略慢但在预算内）

---

## 8. 边缘部署可行性

### 8.1 量化支持

| 量化方案 | 支持情况 | 说明 |
|---------|---------|------|
| INT8（BitsAndBytes） | 已验证 | `load_in_8bit=True` |
| INT4/NF4（BitsAndBytes） | 已验证 | 双量化支持 |
| GPTQ (INT4) | 已验证 | AWQ 报告为压缩/质量最佳平衡 |
| FP8 | 已验证 | 静态和动态变体 |
| GGUF (llama.cpp) | 官方支持 | 含层感知量化（重要层上调至 8/16-bit） |
| ExecuTorch | 已验证 | INT8-INT4 混合量化，iPhone 15 Pro 实测 14.8 tok/s |

### 8.2 目标硬件适配

| 硬件 | Qwen3.5-4B 可行性 | 说明 |
|------|-------------------|------|
| 地瓜 S100 Pro (128 TOPS) | 主线目标 | 需实际验证 TTFT/TPS |
| RK3588 + 协处理器 | 备线目标 | INT4 ~3GB 可适配 |
| Jetson Orin NX | 研究验证 | INT8 已验证，<100ms/frame |
| Jetson Orin AGX | Teacher 研究 | BF16 可跑 27B |

### 8.3 Token 预算对齐

设计文档中的 token 预算与 Qwen3.5-4B 的推理能力匹配：

| 任务级别 | Token 预算 | Student 频率 | 可行性 |
|---------|-----------|-------------|--------|
| 房间级导航 | 450-650 token | 0.25-0.40 Hz | 满足 |
| 家具级找物 | 600-850 token | 0.20-0.33 Hz | 满足 |
| 人物级找人 | 650-900 token | 0.25-0.33 Hz | 满足 |
| 高频轻量层 | 120-260 token | 4-5 Hz | 满足 |

---

## 9. 风险与关注点

### 9.1 Qwen3.5 已确认的风险

| 风险 | 等级 | 应对策略 |
|------|------|---------|
| Thinking 模式在检测任务上有害 | **高** | 部署锁定 no-thinking；训练阶段按需切换 |
| Qwen3.5-4B thinking 模式房间分类系统性失败（0.00%） | **高** | 确认为 thinking 模式特有问题，no-thinking 模式正常（70.59%） |
| No-thinking 模式通用 benchmark 偏弱（73.20% vs 81.79%） | **中** | 实际任务表现更重要；Teacher 阶段可用 thinking 生成高质量数据 |

### 9.2 需要后续验证的风险

| 风险 | 等级 | 验证方式 |
|------|------|---------|
| Qwen3.5-27B 的 3D Grounding 能力未专项验证 | **中** | P1-medium 阶段前需在 3D 空间推理 benchmark 上验证 |
| 跨架构蒸馏降级方案（Qwen3-VL-32B → Qwen3.5-4B）可行性 | **低** | 仅在主线方案失败时需要，可延后验证 |
| 地瓜 S100 Pro 上的实际 TTFT/TPS | **高** | 需在真实硬件上按统一口径验证 |
| Qwen3.5-35B-A3B（MoE）作为 Teacher 的性价比 | **低** | 3B 激活参数推理成本远低于 27B Dense，如研究算力紧张可评估 |

---

## 10. 总结

1. **Qwen3.5 系列是正确的长期选择**：Early Fusion 架构是官方主线方向，Qwen3-VL 已停止演进。
2. **实测数据已充分验证**：Qwen3.5-4B (no-thinking) 在 homeobjects 基准上全面领先，空间关系能力（82.98%）是导航任务的关键差异化指标。
3. **同架构族蒸馏路径最优**：Qwen3.5-27B → Qwen3.5-4B 避免跨架构蒸馏的额外复杂度。
4. **部署锁定 no-thinking 模式**：检测/导航任务中 thinking 模式有害（延迟高、F1 低、房间分类崩溃）。
5. **保留 Qwen3-VL-32B 为备选 Teacher**：其显式 3D Grounding 能力可能在 P1-medium 深度几何阶段有补充价值。
