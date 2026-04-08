# VLM 导航目标识别微调与蒸馏方案

> 状态：architecture design draft  
> 更新时间：2026-04-07  
> 关联论文：ABot-N0 (arXiv:2602.11598v1)  
> 目标：基于 Qwen3.5-27B 微调目标检测能力，蒸馏至 4B 级别模型用于机器人端侧部署

---

## 1. 整体架构

### 1.1 双阶段 Teacher-Student 流水线

```
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 1: Teacher 微调                         │
│                                                                 │
│  ┌──────────┐     ┌──────────────┐     ┌───────────────────┐   │
│  │ 数据引擎  │────▶│ Qwen3.5-27B  │────▶│ 微调后 27B 模型    │   │
│  │ (Sec.2)  │     │ LoRA SFT     │     │ (Detection Expert)│   │
│  └──────────┘     └──────────────┘     └─────────┬─────────┘   │
│                                                   │             │
├───────────────────────────────────────────────────┼─────────────┤
│                    Stage 2: 蒸馏                   │             │
│                                                   ▼             │
│  ┌───────────────────┐     ┌──────────────┐     ┌───────────┐  │
│  │ Teacher 推理生成   │────▶│ KD + SFT     │────▶│Qwen3.5-4B │  │
│  │ 高质量伪标注       │     │ 联合训练      │     │(部署模型)  │  │
│  └───────────────────┘     └──────────────┘     └───────────┘  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    Stage 3: 部署                                │
│                                                                 │
│  Qwen3.5-4B ──▶ ONNX/TensorRT ──▶ Jetson Orin NX / 边端设备   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 模型选型

| 角色 | 模型 | 参数量 | 理由 |
|------|------|--------|------|
| **Teacher** | Qwen3.5-27B | 27B Dense | 原生多模态早期融合架构，支持 2D/3D grounding，性能对标 Qwen3.5-122B-A10B |
| **Student** | Qwen3.5-4B | 4B Dense | 同架构族，token embedding 兼容，适合边端部署 |
| **备选 Teacher** | Qwen3-VL-32B | 32B Dense | 更成熟的 grounding token 体系（`<|object_ref_start|>` 等），社区微调经验更丰富 |
| **备选 Student** | Qwen3-VL-4B | 4B Dense | 与 Qwen3-VL-32B 同架构，蒸馏路径更直接 |

**选型决策**：
- 若追求最新架构和最强基础能力 → Qwen3.5-27B → Qwen3.5-4B
- 若追求更稳定的 grounding 微调经验和社区支持 → Qwen3-VL-32B → Qwen3-VL-4B
- 建议先用 **Qwen3-VL-32B** 做原型验证（grounding 微调更成熟），确认方案可行后迁移到 Qwen3.5

### 1.3 与 ABot-N0 系统的关系

在 ABot-N0 的 Agentic Navigation System 中，本方案微调的检测模型定位为：

```
高层指令（"找到沙发"）
    │
    ▼
Agentic Planner ──────────────────────────┐
    │                                      │
    ▼                                      ▼
┌─────────────────────┐    ┌──────────────────────────┐
│ ABot-N0 Brain       │    │ 本方案微调的检测模型       │
│ (导航决策 + 轨迹生成) │◀───│ (目标识别 + 空间定位)      │
│                     │    │ Qwen3.5-4B Detection     │
└─────────────────────┘    └──────────────────────────┘
```

核心价值：为导航系统提供实时、精确的目标检测 grounding 能力，替代或增强 ABot-N0 中 Cognitive Brain 的 Object-Goal reasoning 模块。

---

## 2. 数据集准备方案

### 2.1 数据集总体规划

面向导航目标识别场景，数据集需覆盖三个层次：

```
Layer 1: 通用检测/Grounding 基础能力
  ├── RefCOCO/+/g (14.2万表达式)
  ├── COCO Detection (118K图, 80类)
  └── Objects365 子集 (按需采样)

Layer 2: 导航场景专项数据
  ├── 室内物体识别 (家具、门、电器、POI)
  ├── 室外场景理解 (人行道、路口、店铺入口)
  └── 动态目标追踪 (行人、宠物)

Layer 3: 认知推理增强数据 (参考 ABot-N0)
  ├── Object-Goal CoT (目标可见性 + 空间关系推理)
  ├── 可通行区域分析
  └── 通用 VQA 防遗忘数据
```

### 2.2 现成公开数据集选用

#### 第一层：通用 Grounding 基础

| 数据集 | 选用规模 | 任务类型 | 用途 |
|--------|---------|---------|------|
| **Ref-L4**（清洗版 RefCOCO/+/g） | 全量 ~14万条 | 指代表达定位 | Grounding 核心能力，修复了原版 14-24% 标注错误 |
| **COCO Detection** | train2017 全量 118K | 多类别检测 | 通用 bbox 检测基础 |
| **Visual Genome** | region descriptions 子集 ~100K | 区域描述+bbox | 密集描述能力，导航场景描述 |
| **Flickr30k Entities** | 全量 ~27万 bbox | 短语-区域对齐 | Grounded captioning |

#### 第二层：导航场景专项

| 数据集 | 规模 | 说明 |
|--------|------|------|
| **HM3D ObjectNav** | ~1.8M 轨迹样本（含目标识别） | ABot-N0 同款，室内开放词汇目标搜索 |
| **ScanQA** | ~41K 问答对 | 3D 室内场景理解 |
| **InteriorGS** | 1000 场景，700+ 物体类 | 高保真室内渲染，含 bbox 标注 |
| **BridgeNav POI** | 街景图像+POI入口坐标 | 室外 POI 入口定位 |
| **自建数据**（见 2.3） | 目标 5K-20K | 面向实际部署场景 |

#### 第三层：认知推理 + 防遗忘

| 数据集 | 规模 | 用途 |
|--------|------|------|
| **LLaVA-Instruct-150K** | 150K | 通用对话能力保持 |
| **ShareGPT4V** | 100K 选用 | 高质量图像描述 |
| **导航推理 CoT**（自建） | 10K-50K | 参考 ABot-N0 Object-Goal CoT 格式 |

### 2.3 自建数据集流水线

#### 整体流程

```
Step 1: 场景图像采集
  │  来源: 机器人实拍 / 3D场景渲染 / 互联网室内外图像
  ▼
Step 2: 自动标注
  │  Grounding DINO 2.0 ──▶ 高质量 bbox
  │  Qwen3.5-27B (预训练版) ──▶ 目标描述 + 属性 + 空间关系
  ▼
Step 3: 质量过滤
  │  ├── bbox 置信度 > 0.65 (参考 GrIT 做法)
  │  ├── 描述-bbox 一致性校验 (CLIP score > 阈值)
  │  └── 人工抽样审核 (10% 样本)
  ▼
Step 4: 格式转换
  │  转为目标模型的对话式 JSON 格式
  ▼
Step 5: 数据增强
  │  ├── 图像增强: 翻转/旋转/缩放/颜色抖动 (同步变换 bbox)
  │  └── 文本多样化: 同一目标用多种描述方式
  ▼
Step 6: 数据混合
     按比例混合三层数据
```

#### 导航场景目标类别体系

基于 ABot-N0 论文的五类导航任务，定义核心检测类别：

```yaml
室内物体 (Object-Goal):
  家具: [沙发, 床, 桌子, 椅子, 柜子, 书架, 电视柜]
  电器: [电视, 冰箱, 微波炉, 洗衣机, 空调]
  功能区标识: [门, 窗, 楼梯, 走廊, 厨房台面]
  小物体: [杯子, 遥控器, 书本, 花瓶, 时钟]

室外场景 (POI-Goal):
  建筑入口: [店铺门口, 大楼入口, 电梯口, 地铁口]
  交通设施: [人行道, 斑马线, 红绿灯, 路标]
  商业POI: [咖啡店, 超市, 药店, 银行ATM]

动态目标 (Person-Following):
  人物: [行人, 特定目标人物]
  属性: [衣着颜色, 姿态, 行进方向]
```

#### 导航推理 CoT 数据格式（参考 ABot-N0 Section 3.3.3）

```json
{
  "image": "scene_001.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n你正在寻找 wardrobe。根据输入视角和目标名称，生成推理链和导航决策。"
    },
    {
      "from": "gpt",
      "value": "[CoT] 当前视野中可以看到客厅，前方是厨房区域。右侧有一扇半开的门，通向卧室方向。Wardrobe 通常出现在卧室中。[Status] Invisible [Decision] Turn Right, Forward"
    }
  ]
}
```

### 2.4 数据混合比例

最终训练数据的推荐混合比例：

| 数据类型 | 占比 | 规模估算 | 作用 |
|---------|------|---------|------|
| 检测/Grounding 数据 | 45% | ~200K | 核心目标检测能力 |
| 导航场景专项数据 | 25% | ~110K | 领域适配 |
| 认知推理 CoT | 10% | ~45K | 空间理解和推理能力 |
| 通用 VQA / 对话 | 20% | ~90K | 防止灾难性遗忘 |
| **合计** | 100% | **~450K** | |

---

## 3. Teacher 微调策略

### 3.1 训练阶段设计

参考 ABot-N0 的三阶段训练 recipe，采用两阶段渐进式微调：

```
Phase 1: Cognitive Grounding Warm-up
  │  目标: 建立检测/grounding 基础认知
  │  数据: Layer 1 通用 grounding 数据 + Layer 3 VQA 数据
  │  冻结: Vision Encoder
  │  训练: LLM 层 LoRA
  │  Epoch: 1-2
  ▼
Phase 2: Navigation Detection SFT
  │  目标: 导航场景专项检测能力
  │  数据: 全量混合数据 (含 Layer 2 导航专项)
  │  冻结: 无 (或仅冻结 Vision Encoder)
  │  训练: LLM 全参数 或 高秩 LoRA (r=64)
  │  Epoch: 2-3
  │  Replay: Layer 1 数据以 20% 比例回放 (参考 ABot-N0 Phase 2)
  ▼
输出: Navigation Detection Teacher (27B)
```

### 3.2 微调方式选择

| 方式 | VRAM 需求 | 训练速度 | Grounding 精度 | 推荐场景 |
|------|----------|---------|--------------|---------|
| **LoRA (r=8)** | ~56GB (bf16) | 快 | 良好 | Phase 1 warm-up、快速验证 |
| **LoRA (r=64)** | ~70GB (bf16) | 中 | 很好 | Phase 2 主训练，推荐 |
| **全参数 SFT** | ~120GB+ (需多卡) | 慢 | 最优 | 追求极致精度 |
| ~~QLoRA (4-bit)~~ | ~28GB | 快 | 差 | **不推荐**，Qwen3.5 量化损失大 |

**推荐路径**：Phase 1 用 LoRA (r=8) 快速 warm-up → Phase 2 用 LoRA (r=64) 深度训练。

### 3.3 关键超参数

```yaml
# Phase 1: Cognitive Grounding Warm-up
phase1:
  model: Qwen3.5-27B 或 Qwen3-VL-32B
  lora_r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  lora_target_modules: [q_proj, v_proj, gate_proj, up_proj, down_proj]
  learning_rate: 2e-5
  lr_scheduler: cosine
  warmup_ratio: 0.03
  batch_size: 4  # per GPU
  gradient_accumulation: 8
  precision: bf16
  max_epochs: 2
  freeze_vision_encoder: true

# Phase 2: Navigation Detection SFT
phase2:
  lora_r: 64
  lora_alpha: 128
  learning_rate: 1e-5  # 降低学习率
  vision_encoder_lr: 2e-6  # 视觉编码器学习率为 LLM 的 1/5
  batch_size: 2  # per GPU
  gradient_accumulation: 16
  max_epochs: 3
  freeze_vision_encoder: false
  replay_ratio: 0.2  # 20% Phase 1 数据回放
```

### 3.4 数据格式（以 Qwen3-VL 为例）

检测任务的输入输出格式：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是一个导航场景目标检测助手。请识别图像中的目标物体并给出精确的边界框坐标。"
    },
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "indoor_scene_042.jpg"},
        {"type": "text", "text": "检测图中所有可识别的导航相关目标物体。"}
      ]
    },
    {
      "role": "assistant",
      "content": "[{\"bbox_2d\": [120, 45, 380, 290], \"label\": \"sofa\"}, {\"bbox_2d\": [410, 180, 520, 350], \"label\": \"door\"}, {\"bbox_2d\": [50, 200, 150, 400], \"label\": \"bookshelf\"}]"
    }
  ]
}
```

Grounding 任务格式：

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "scene.jpg"},
        {"type": "text", "text": "找到<|object_ref_start|>靠窗的白色沙发<|object_ref_end|>"}
      ]
    },
    {
      "role": "assistant",
      "content": "<|box_start|>(120,45),(380,290)<|box_end|>"
    }
  ]
}
```

### 3.5 微调框架选择

| 框架 | 优势 | 适用场景 |
|------|------|---------|
| **Unsloth** | 1.5x 加速，50% VRAM 节省，原生支持 Qwen3.5 | 单机/少卡，LoRA 微调首选 |
| **ms-swift** | 阿里官方出品，Qwen 系列适配最完善，支持坐标自动转换 | Qwen 系列的最佳选择 |
| **LLaMA-Factory** | 社区活跃，支持多种训练策略 | 需要灵活配置 |
| **Transformers + PEFT** | 最底层控制，灵活度最高 | 需要深度定制训练流程 |

**推荐**：ms-swift（与 Qwen 生态最契合，自动处理坐标格式转换）。

---

## 4. 蒸馏策略

### 4.1 蒸馏方案概览

采用 **数据蒸馏 + 特征蒸馏** 双路线：

```
                    ┌────────────────────────┐
                    │  微调后 Teacher (27B)   │
                    └──────┬─────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ 路线 A    │    │ 路线 B   │    │ 路线 C   │
    │ 数据蒸馏  │    │ Logit KD │    │ 渐进蒸馏  │
    │ (推荐起步) │    │ (进阶)   │    │ (最优)   │
    └──────────┘    └──────────┘    └──────────┘
```

### 4.2 路线 A：数据蒸馏（推荐首选）

**原理**：用 Teacher 模型对大量未标注图像做推理，生成高质量伪标注，再用这些数据 SFT 训练 Student。

**优势**：
- 实现最简单，不需要修改训练框架
- Teacher 和 Student 可以是不同架构
- 可离线生成伪标注，训练与推理解耦

**流程**：

```
Step 1: Teacher 推理生成伪标注
  │  输入: 大量未标注的导航场景图像 (50K-200K)
  │  输出: 每张图的检测结果 JSON + 推理 CoT
  │  过滤: 保留高置信度结果 (score > 0.7)
  ▼
Step 2: 伪标注与真实标注混合
  │  比例: 真实标注 40% + Teacher 伪标注 60%
  ▼
Step 3: Student SFT 训练
  │  模型: Qwen3.5-4B
  │  策略: 全参数 SFT (4B 模型可全参数训练)
  │  数据: 混合数据集 ~300K
  ▼
Student Detection Model (4B)
```

**关键配置**：

```yaml
student_sft:
  model: Qwen3.5-4B
  training_type: full_parameter  # 4B 模型可全参数训练
  learning_rate: 5e-6
  lr_scheduler: cosine
  warmup_ratio: 0.05
  batch_size: 8  # per GPU
  gradient_accumulation: 4
  precision: bf16
  max_epochs: 3
  # VRAM: ~32GB (bf16 全参数)
```

### 4.3 路线 B：Logit-level 知识蒸馏

**原理**：在训练过程中，Student 同时学习 ground-truth 标签和 Teacher 的 soft logit 分布。

**损失函数**：

```
L_total = α * L_SFT(student, ground_truth) 
        + β * L_KL(student_logits, teacher_logits, T)
        + γ * L_feature(student_hidden, teacher_hidden)

其中:
  L_SFT     = 标准交叉熵损失
  L_KL      = KL 散度 (soft label 蒸馏)
  L_feature = 中间层特征对齐 (可选)
  T         = 温度参数 (推荐 T=2.0~4.0)
  α=0.5, β=0.4, γ=0.1
```

**优势**：比纯数据蒸馏保留更多 Teacher 的决策分布信息
**劣势**：需要 Teacher 和 Student 同时在线，显存要求高；需要修改训练代码

### 4.4 路线 C：渐进式蒸馏（最优效果）

**原理**：分步骤逐级蒸馏，每步缩减适度。

```
Qwen3.5-27B (Teacher)
    │
    │  Step 1: 数据蒸馏 + Logit KD
    ▼
Qwen3.5-9B (中间模型)
    │
    │  Step 2: 数据蒸馏 + Logit KD
    ▼
Qwen3.5-4B (Student)
```

**优势**：每步能力损失更小，最终 Student 质量更高
**劣势**：耗时长，需要训练两次

### 4.5 蒸馏路线推荐

| 阶段 | 推荐路线 | 理由 |
|------|---------|------|
| 原型验证 | **路线 A（数据蒸馏）** | 最快出结果，验证可行性 |
| 质量优化 | **路线 B（Logit KD）** | 在 A 基础上提升 2-5% |
| 极致性能 | **路线 C（渐进蒸馏）** | 27B → 9B → 4B，效果最优 |

---

## 5. 评估方案

### 5.1 评估指标体系

```yaml
检测精度:
  - mAP@0.5        # 标准检测精度
  - mAP@0.5:0.95   # 严格检测精度
  - Acc@0.5 (IoU)   # Grounding 准确率 (RefCOCO 标准)

导航场景专项:
  - 目标类别召回率   # 导航关键物体是否被召回
  - POI 入口定位精度  # 参考 BridgeNav 的 SR@0.1m/0.2m/0.3m
  - 动态目标追踪成功率 # Person-Following 场景

效率指标:
  - 推理延迟 (ms/image)    # 端侧实时性
  - 吞吐量 (images/sec)    # 批量处理能力
  - 显存占用 (GB)          # 部署硬件要求

蒸馏质量:
  - Teacher-Student 精度差  # 目标: < 5% mAP 下降
  - 能力保持率              # Student / Teacher 指标比值
```

### 5.2 评估基准

| 基准 | 评估能力 | 指标 |
|------|---------|------|
| RefCOCO/+/g val | 通用 grounding | Acc@0.5 |
| COCO val2017 | 通用检测 | mAP |
| HM3D-OVON | 开放词汇目标搜索 | SR, SPL |
| BridgeNav | POI 入口定位 | SR@0.1m |
| 自建导航测试集 | 领域适配 | 类别召回率 |

### 5.3 消融实验设计

为了避免 ABot-N0 论文中"消融不足"的问题，建议设计以下对比实验：

| 实验 | 对比内容 | 验证目标 |
|------|---------|---------|
| E1 | 有/无 Phase 1 warm-up | 渐进训练的价值 |
| E2 | 有/无导航专项数据 | 领域适配的收益 |
| E3 | 有/无 CoT reasoning 数据 | 认知推理对检测的增益 |
| E4 | 有/无 VQA 防遗忘数据 | 灾难性遗忘的影响 |
| E5 | LoRA r=8 vs r=64 vs 全参数 | 微调深度与精度的关系 |
| E6 | 数据蒸馏 vs Logit KD vs 渐进 | 蒸馏路线效果对比 |
| E7 | Teacher 直推 vs Student | 蒸馏精度损失量化 |

---

## 6. 部署方案

### 6.1 边端部署链路

```
Qwen3.5-4B (PyTorch)
    │
    ├──▶ ONNX 导出
    │       │
    │       ├──▶ TensorRT (NVIDIA Jetson Orin NX)
    │       │     目标: <100ms/帧, INT8 量化
    │       │
    │       └──▶ ONNX Runtime (通用 GPU/CPU)
    │
    └──▶ llama.cpp / vLLM (服务端备选)
           目标: 高吞吐量批处理
```

### 6.2 硬件需求估算

| 部署方式 | 硬件 | 显存需求 | 预估延迟 |
|---------|------|---------|---------|
| 4B INT8 | Jetson Orin NX (16GB) | ~6GB | ~200ms |
| 4B FP16 | RTX 4090 (24GB) | ~10GB | ~80ms |
| 4B INT4 (GPTQ/AWQ) | Jetson Orin NX (8GB) | ~4GB | ~300ms |
| 27B FP16 | 2×A100 (160GB) | ~60GB | ~500ms |

### 6.3 与导航系统集成

参考 ABot-N0 的 2Hz VLA 推理 + 10Hz 控制器设计：

```
Camera (30fps) ──▶ 关键帧采样 (2-5fps)
                         │
                         ▼
                  Detection Model (4B)
                  推理: ~200ms/帧
                         │
                         ▼
                  检测结果缓存
                  (目标类别, bbox, 置信度, 描述)
                         │
                  ┌──────┴──────┐
                  ▼              ▼
           Navigation Brain   Topo-Memory
           (导航决策)          (空间记忆更新)
```

---

## 7. 时间线与里程碑

```
Week 1-2:  数据准备
  ├── 收集/清洗公开数据集
  ├── 搭建自动标注流水线 (Grounding DINO + Qwen3.5-27B)
  └── 生成导航场景专项数据

Week 3-4:  Teacher 微调
  ├── Phase 1: Cognitive Warm-up (1-2天)
  ├── Phase 2: Navigation Detection SFT (3-5天)
  └── 评估 Teacher 在各基准上的表现

Week 5-6:  蒸馏
  ├── 路线 A: 数据蒸馏快速验证 (2-3天)
  ├── 对比 Teacher vs Student 精度差
  └── 如需要，尝试路线 B/C 优化

Week 7-8:  部署与集成
  ├── 模型量化与导出
  ├── 边端推理性能测试
  └── 与导航系统集成测试
```

---

## 8. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| Qwen3.5 grounding 微调经验不足 | 训练不收敛或精度不达标 | 先用 Qwen3-VL-32B 做原型验证 |
| 蒸馏精度损失过大 (>10%) | Student 模型不可用 | 采用渐进蒸馏 27B→9B→4B |
| 导航场景数据不足 | 领域适配不够 | 扩大自动标注规模，加入 3D 场景合成 |
| 灾难性遗忘 | 通用能力退化 | 严格保持 20% VQA 数据回放比例 |
| 边端推理延迟不满足 | 无法实时使用 | INT4 量化 + 关键帧采样降频 |
| Transformers v5 兼容性 | Qwen3.5 需新版本 | 锁定框架版本，提前验证依赖 |

---

## 附录 A: 参考资源

- [Qwen3.5 官方博客](https://qwen.ai/blog?id=qwen3.5)
- [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3.5 Unsloth 微调指南](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- [ms-swift Qwen3-VL 最佳实践](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3-VL-Best-Practice.html)
- [ABot-N0 论文](https://arxiv.org/abs/2602.11598)
- [Ref-L4 清洗版 RefCOCO](https://github.com/JierunChen/Ref-L4)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Autodistill 自动标注](https://github.com/autodistill/autodistill)
- [HuggingFace VLM Detection 微调指南](https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_object_detection_grounding)
