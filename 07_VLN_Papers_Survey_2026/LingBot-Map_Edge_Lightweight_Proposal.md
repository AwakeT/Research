# LingBot-Map 位姿+深度能力端侧轻量化方案

> 基于 LingBot-Map (arXiv:2604.14141) 的位姿估计与深度预测能力，设计端侧部署方案，与 Qwen3-VL-4B VLM 共存于手机级 NPU。

文档版本：v0.2
变更记录：
- v0.2 | 2026-04-28 | 新增"大模型验证 → 小模型蒸馏"两阶段研发路线；明确 VLM 主线零影响约束，所有方案中 ViT 冻结不微调；更新方案推荐排序
- v0.1 | 2026-04-28 | 初始版本

---

## 1. 背景与目标

### 1.1 目标

从 LingBot-Map（~1.19B 参数）中提取**单目相机相对位姿估计 + 逐像素深度预测**能力，部署到手机级 NPU 上，与 Qwen3-VL-4B VLM 共存。

### 1.2 核心约束

- Qwen3-VL-4B（INT8 ~5GB）已占用大部分 NPU 预算
- 手机级 NPU（骁龙/天玑级），总内存 8-12 GB
- 不需要点云（Point Head 可删除），只需要 6DoF 位姿 + 深度图
- VLM 语义决策频率受端侧设备算力限制（0.25-0.4 Hz）；位姿模型在服务器侧（lingbot-map）可达 15-20 Hz，端侧部署目标 10-15 Hz
- **VLM 主线零影响**：位姿/深度能力的引入不得影响 VLM 的语义决策性能（不微调 VLM 任何参数）

### 1.3 研发资源与路线

- **验证阶段**可使用大尺寸 VLM（如 Qwen3-VL-30B），在云端/A100 上验证方案可行性
- **部署阶段**才需要蒸馏到端侧 4B 级别
- 这意味着研发路线是"**大模型验证 → 确认可行 → 蒸馏到端侧**"，而非直接在小模型上试错

### 1.4 关键发现：Qwen3-VL ViT 与 DINOv2 ViT-L 高度同构

| 参数 | Qwen3-VL-4B ViT | LingBot-Map DINOv2 ViT-L |
|---|---|---|
| hidden_dim | 1024 | 1024 |
| 层数 | 24 | 24 |
| 注意力头数 | 16 | 16 |
| head_dim | 64 | 64 |
| MLP 中间维度 | 4096 | 4096 |
| patch_size | 16 | 14 |
| 参数量 | ~374M | ~304M |
| 中间特征提取层 | DeepStack [5,11,17] | selected_idx [4,11,17,23] |
| 训练目标 | 图文对齐（语义） | 自监督对比学习（几何） |

两者在维度、层数、注意力结构上**完全一致**，中间特征提取点也高度吻合。Qwen3-VL 有**可分离的独立 ViT**，可以作为共享视觉骨干。这意味着可以复用 VLM 已有的 ViT，而不必额外引入 DINOv2。

### 1.5 为什么不能使用 Qwen3.5 架构

本方案基于 Qwen3-VL 而非更新的 Qwen3.5，原因有二：

1. **ViT 不可分离**：Qwen3.5 采用 early fusion 统一多模态架构，视觉编码器与语言模型通过早期融合训练深度耦合，不再像 Qwen3-VL 那样拥有独立可分离的 ViT 模块。我们的方案核心前提是"冻结共享 ViT、仅训练下游头"，这要求 ViT 能被独立提取和复用，Qwen3.5 的架构无法满足这一点。

2. **架构不再同构**：Qwen3.5 的骨干从标准 Transformer 换成了 Gated DeltaNet + sparse MoE 混合架构，视觉特征的表示与处理方式也随之改变。LingBot-Map 的 GCA、Camera Head、DPT Head 均基于标准 ViT 特征设计（与 DINOv2 ViT-L 同构），Qwen3.5 的特征空间与之不再兼容，无法直接复用现有模块。

综上，Qwen3-VL 的"独立 ViT + 后融合"架构是本方案可行的关键前提。

---

## 2. LingBot-Map 原始参数分布

| 组件 | 参数量 | 占比 | 说明 |
|---|---|---|---|
| DINOv2 ViT-Large | 304M | 25.5% | 24层, dim=1024, patch_size=14 |
| frame_blocks | 302M | 25.4% | 24层帧内自注意力 |
| global_blocks | 302M | 25.4% | 24层跨帧 GCA 注意力 |
| Camera Head | 216M | 18.1% | 4层 CameraBlock, dim=2048, 4次迭代精炼 |
| DPT Depth Head | 33M | 2.8% | 4级 refinenet, features=256 |
| DPT Point Head | 33M | 2.8% | 同 Depth Head 结构（**可删除**） |
| **合计** | **~1,190M** | | |

---

## 3. 三种轻量化方案

### 设计原则

所有方案共同遵循以下约束：

1. **VLM 主线零影响**：Qwen3-VL 的所有参数（ViT + LLM）完全冻结，不做任何微调。位姿/深度模块是纯增量添加，仅训练新增参数。
2. **两阶段研发路线**：先在 Qwen3-VL-30B（云端 A100）上验证方案可行性，确认效果后再蒸馏到 4B 端侧部署。避免在小模型上反复试错。
3. **Point Head 删除**：只保留位姿 + 深度输出，不需要点云。

### 3.1 方案 A（推荐）：共享 Qwen3-VL ViT + 轻量 GCA + 精简头

**核心思路**：不新增视觉编码器，复用 Qwen3-VL 已有的 ViT（完全冻结），只增加几何推理层和预测头。

```
Qwen3-VL ViT（已有，374M，完全冻结）
    │
    ├──→ DeepStack merger → LLM → 语义决策（原有 VLM 流程，完全不变）
    │
    └──→ [新增] 投影层 + 6层GCA + Camera Head + DPT Depth Head → 位姿+深度
```

#### 新增参数明细

| 新增组件 | 配置 | 参数量 |
|---|---|---|
| 投影适配层 | Linear(1024→1024) + LayerNorm | ~1M |
| frame_blocks | 6 层, dim=1024, 16 heads | 75.6M |
| global_blocks | 6 层, dim=1024, 16 heads（SDPA） | 75.6M |
| Camera Head Lite | 1 层 CameraBlock, dim=2048, 2 次迭代 | ~54M |
| DPT Depth Head Lite | features=128, dim_in=2048 | ~8M |
| Special tokens | camera + 4 register + scale, 2 variants | ~12K |
| **新增合计** | | **~214M** |

#### 多尺度特征复用

Qwen3-VL DeepStack 在 ViT 层 [5,11,17] 提取中间特征，加上最终层 [23]，可直接映射为 LingBot-Map 的 4 级多尺度特征供 DPT Head 使用：

| DPT 级别 | LingBot-Map 原版取自 | 方案 A 取自 Qwen3-VL ViT 层 |
|---|---|---|
| 浅层（纹理/边缘） | 第 4 组 | 第 5 层（DeepStack 层 1） |
| 中层（结构） | 第 11 组 | 第 11 层（DeepStack 层 2） |
| 深层（物体） | 第 17 组 | 第 17 层（DeepStack 层 3） |
| 最终（语义） | 第 23 组 | 第 23 层（最终输出） |

#### 频率解耦设计

```
相机帧 @ 5-15 Hz
    │
    ▼
Qwen3-VL ViT（共享，冻结，在位姿频率运行）
    │
    ├──→ 特征缓存 ──→ VLM LLM @ 0.25 Hz（按需读取缓存特征）
    │
    └──→ 6层GCA + 预测头 ──→ pose + depth → world_state_memory
```

ViT 在高频（5-15 Hz）运行为位姿服务，VLM 的 LLM 部分（~3.6B，占总参数 90%）仍然低频运行。ViT 完全冻结，仅做前向推理，不存在训练干扰风险。

#### 优劣分析

| 优势 | 劣势 |
|---|---|
| ViT 不重复，省 300M+ 计算 | ViT 在高频运行，增加功耗 |
| dim=1024 全程保持，精度最高 | 工程复杂度高（需改造推理流） |
| DeepStack 特征直接复用 | Qwen3-VL ViT 偏语义，需 GCA 补偿几何能力 |
| ViT 冻结，VLM 性能零影响 | 新增 214M 参数，INT8 约 214MB |

---

### 3.2 方案 B：共享 ViT + 降维 GCA（极致轻量）

在方案 A 基础上，ViT 输出（冻结）后加一层 Linear(1024→384) 降维，后续 GCA 在 dim=384 上运行。

| 新增组件 | 配置 | 参数量 |
|---|---|---|
| 降维投影 | Linear(1024→384) + LayerNorm | ~0.4M |
| frame_blocks | 6 层, dim=384, 6 heads | 10.7M |
| global_blocks | 6 层, dim=384, 6 heads | 10.7M |
| Camera Head Lite | 2 层, dim=768, 2 次迭代 | 16.3M |
| DPT Depth Head Lite | features=128, dim_in=768 | 7.8M |
| **新增合计** | | **~46M** |

#### 优劣分析

| 优势 | 劣势 |
|---|---|
| 新增参数极少，INT8 仅 ~46MB | 1024→384 降维丢失大量几何细节 |
| NPU 占用最小 | 位姿/深度精度下降明显（预估 +40-60%） |
| ViT 冻结，VLM 性能零影响 | |

---

### 3.3 方案 C：独立 DINOv2 ViT-Small（完全解耦）

不共享 ViT，使用独立的 DINOv2 ViT-Small 作为视觉骨干，与 VLM 完全解耦。

| 组件 | 配置 | 参数量 |
|---|---|---|
| DINOv2 ViT-Small | dim=384, 12 层, 6 heads, patch_size=14 | 22M |
| frame_blocks | 6 层, dim=384 | 10.7M |
| global_blocks | 6 层, dim=384 | 10.7M |
| Camera Head Lite | 2 层, dim=768, 2 次迭代 | 16.3M |
| DPT Depth Head Lite | features=128, dim_in=768 | 7.8M |
| **合计** | | **~67M** |

```
相机帧 @ 5-15 Hz ──→ DINOv2 ViT-S + 6层GCA + 预测头 ──→ pose + depth
相机帧 @ 0.25 Hz ──→ Qwen3-VL 完整管线（完全独立）     ──→ 语义决策
```

#### 优劣分析

| 优势 | 劣势 |
|---|---|
| 完全独立，VLM 零影响（天然解耦） | 额外 67M 参数+计算 |
| DINOv2 天然擅长几何匹配 | dim=384 限制特征表达能力 |
| 频率完全自由（5-30 Hz） | 无法利用 VLM 已有的视觉特征 |
| 工程复杂度最低 | |

---

## 4. 方案对比总表

| 维度 | 方案A（共享ViT） | 方案B（共享+降维） | 方案C（独立ViT-S） |
|---|---|---|---|
| 新增参数 | ~214M | ~46M | 67M |
| INT8 新增模型大小 | ~214MB | ~46MB | ~67MB |
| ViT 计算 | 共享（不重复） | 共享（不重复） | 重复（额外 22M ViT） |
| GCA 特征维度 | 1024（全精度） | 384（降维后） | 384（ViT-S 原生） |
| 位姿精度（预估） | 最高（原版 +30-50%） | 中低（原版 +50-80%） | 中等（原版 +30-50%） |
| 深度精度（预估） | 最高（原版 +20-40%） | 中低（原版 +40-70%） | 中等（原版 +20-40%） |
| 频率灵活性 | 受 ViT 频率约束 | 受 ViT 频率约束 | 完全独立 |
| 工程复杂度 | 高 | 高 | 低 |
| 对 VLM 的影响 | 零（ViT 冻结） | 零（ViT 冻结） | 零（完全独立） |

**选择建议**：

- **优先精度** → 方案 A（共享冻结 ViT，dim=1024 全程）
- **优先部署简单性** → 方案 C（独立 ViT-S，解耦清晰）
- **NPU 预算极度紧张** → 方案 B（共享+降维，最小开销）

---

## 5. 实施路线

### Phase 0：大模型云端验证（1-2 周）

在 Qwen3-VL-30B（云端 A100）上验证方案可行性，避免在小模型上反复试错：

- 使用 Qwen3-VL-30B 的 ViT（dim=1152, 27 层）作为冻结视觉骨干
- 在其上训练 GCA + 预测头，验证语义 ViT 能否支撑位姿/深度任务
- 在 ScanNet 上评估 ATE 和 AbsRel，确认精度在可接受范围内
- **通过/失败判定**：位姿 ATE 达到 LingBot-Map 原版的 70% 以上 → 方案可行，进入 Phase 1

### Phase 1：特征质量验证（1 周）

验证 Qwen3-VL-4B ViT 的几何特征质量，决定走方案 A 还是方案 C：
- 冻结 Qwen3-VL-4B ViT，取 [5,11,17,23] 层特征
- 在 ScanNet 数据集上计算相邻帧同位置 patch 的余弦相似度
- 与 DINOv2 ViT-L 对比：相似度差距 <20% → 方案 A 可行；否则回退方案 C

### Phase 2：构建轻量架构（1 周）

- 新建 `GCTStreamLite` 类，6 层 GCA + 精简头
- Camera Head 精简为 1-2 层 CameraBlock + 2 次迭代精炼
- DPT Depth Head: features=128
- 删除 Point Head
- 只用 SDPA 后端，滑动窗口 16 帧
- 多尺度特征取自 [5,11,17,23] 层（方案 A）或 [0,2,4,5] 组（方案 C）

### Phase 3：知识蒸馏训练（2-3 周）

**教师**：LingBot-Map 完整模型（1.19B）
**学生**：轻量版（ViT 完全冻结，仅训练新增的 GCA + 头）

| 阶段 | 步数 | 策略 | Loss |
|---|---|---|---|
| Warmup | 5K | 冻结 ViT，训练 GCA + 头 | 0.5×特征蒸馏 + 0.3×位姿蒸馏 + 0.2×深度蒸馏 |
| Fine-tune | 20K | ViT 保持冻结，加入 GT 监督 | 0.3×特征 + 0.3×位姿 + 0.3×GT深度 + 0.1×深度蒸馏 |
| End-to-end | 10K | ViT 保持冻结，GCA+头全参数微调 | GT only（位姿 + 深度） |

蒸馏 Loss 说明：
- **特征蒸馏**：学生 GCA 中间特征 → 线性投影 → 与教师对应层对齐（cosine similarity）
- **位姿蒸馏**：学生 vs 教师的 9D pose encoding L2 loss
- **深度蒸馏**：SILog（尺度不变对数深度误差）+ gradient matching

> 注：所有训练阶段 ViT 均完全冻结，确保 VLM 主线零影响。

### Phase 4：INT8 量化与部署（1 周）

- 新增模块（GCA + 头）线性层和卷积 INT8 量化
- Camera Head 的 `exp/atan/sin/cos` 保留 FP16（数值敏感）
- DPT Head 已内置 `nn.quantized.FloatFunctional()`，量化友好
- 导出 ONNX → 目标 NPU SDK
- KV 缓存：滑动窗口 16 帧，INT8

### Phase 5：系统集成与验证（1 周）

- 与 Qwen3-VL-4B 集成（ViT 共享但冻结）
- ScanNet / TUM-RGBD 上对比 ATE 和深度 AbsRel
- 目标 NPU 上实测推理延迟和峰值内存
- 验证与 VLM 同时运行无资源冲突
- **关键验证**：VLM 语义决策性能在集成前后无变化（VLM benchmark 回归测试）

---

## 6. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|---|---|---|
| GCA 24→6 层导致跨帧推理能力大幅下降 | 位姿累积漂移增大 | 知识蒸馏 + 降低滑动窗口后用 IMU 插值 |
| Qwen3-VL ViT 几何特征不足 | 方案 A 不可行 | Phase 0 大模型验证 + Phase 1 特征质量验证，不可行则回退方案 C |
| 大模型（30B）验证通过但蒸馏到 4B 后精度大幅下降 | 部署版精度不可用 | 对比 30B→4B 精度衰减，必要时增大 GCA 层数或保留更多特征维度 |
| INT8 量化导致位姿精度劣化 | 几何一致性下降 | Camera Head 关键运算保留 FP16 |
| 手机 NPU 不支持某些算子 | 部署失败 | 用 SDPA 后端，避免 FlashInfer；RoPE 可预计算 |

---

*本文档基于 LingBot-Map 代码分析（/home/zktian3/lingbot-map/）和 Qwen3-VL-4B 架构分析生成。*
