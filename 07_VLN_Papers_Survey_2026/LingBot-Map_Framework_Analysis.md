# LingBot-Map: Geometric Context Transformer 框架深度解析

> 本文结合论文 *"Geometric Context Transformer for Streaming 3D Reconstruction"* (arXiv:2604.14141) 与代码实现，全面、深入浅出地解析 LingBot-Map 的整体框架流程。

---

## 目录

1. [项目背景与核心问题](#1-项目背景与核心问题)
2. [整体架构总览](#2-整体架构总览)
3. [Stage 1: 图像编码 — DINOv2 视觉骨干网络](#3-stage-1-图像编码--dinov2-视觉骨干网络)
4. [Stage 2: 特殊 Token 系统 — 几何语义的载体](#4-stage-2-特殊-token-系统--几何语义的载体)
5. [Stage 3: 交替注意力层 — Frame Attention 与 GCA](#5-stage-3-交替注意力层--frame-attention-与-gca)
6. [核心创新: Geometric Context Attention (GCA)](#6-核心创新-geometric-context-attention-gca)
7. [Stage 4: 预测头 — 相机位姿与稠密几何](#7-stage-4-预测头--相机位姿与稠密几何)
8. [推理系统设计 — 分页 KV 缓存](#8-推理系统设计--分页-kv-缓存)
9. [位置编码 — 2D RoPE 与 3D Video RoPE](#9-位置编码--2d-rope-与-3d-video-rope)
10. [推理模式 — Direct Output 与 VO 模式](#10-推理模式--direct-output-与-vo-模式)
11. [训练策略](#11-训练策略)
12. [实验结果与性能分析](#12-实验结果与性能分析)
13. [代码结构速查](#13-代码结构速查)

---

## 1. 项目背景与核心问题

### 1.1 什么是流式 3D 重建？

想象你拿着手机拍摄一栋建筑，视频帧源源不断地产生。**流式 3D 重建** (Streaming 3D Reconstruction) 要求模型在每一帧到来时，**仅依据当前帧和历史帧**（不能看到未来），实时输出：

- **相机位姿 (Camera Pose)**：这一帧的相机在三维空间中的位置和朝向
- **深度图 (Depth Map)**：每个像素离相机的距离
- **3D 点云 (Point Cloud)**：将深度图反投影得到的三维世界坐标

这本质上就是经典 SLAM (Simultaneous Localization and Mapping) 要解决的问题，但 LingBot-Map 用一个纯前馈的 Transformer 模型取代了传统 SLAM 中手工设计的特征提取、特征匹配、束调整 (Bundle Adjustment) 等模块。

### 1.2 核心挑战：流式上下文管理

流式推理的核心难题在于**如何管理历史上下文**——模型需要足够的历史信息来保持全局一致性，又不能让状态无限增长。现有方法的不同策略各有缺陷：

| 方法类型 | 代表工作 | 策略 | 问题 |
|---------|---------|------|------|
| 循环压缩 | CUT3R | 固定大小的循环状态 | 激进压缩导致几何先验遗忘 |
| 因果缓存 | StreamVGGT, Stream3R | 保留所有历史 K/V | 显存线性增长，冗余信息混杂 |
| SLAM 混合 | VGGT-SLAM, MASt3R-SLAM | 学习模型 + 传统优化 | 依赖手工启发式，实时性受限 |

LingBot-Map 的核心洞察是：**经典 SLAM 系统之所以有效，是因为它们维护了三种不同类型的空间上下文**——一个用于坐标定标的参考帧、一个用于局部几何的滑动窗口、一个用于全局漂移校正的地图。GCA 将这一结构化思路引入注意力机制，用端到端学习取代手工优化。

---

## 2. 整体架构总览

```
输入视频流: I_1, I_2, ..., I_t
    │
    │  每帧独立编码
    ▼
┌──────────────────────────────────────────────┐
│         DINOv2 ViT-Large Backbone            │
│    图像 → patch tokens [B*S, M, 1024]        │
└──────────────────┬───────────────────────────┘
                   │
                   │  每帧前插入 6 个特殊 token
                   │  [camera, reg×4, scale, patch_0, ..., patch_M]
                   ▼
┌──────────────────────────────────────────────┐
│      24 组交替注意力层 (共 48 个 Block)          │
│                                              │
│   ┌──────────────┐   ┌───────────────────┐   │
│   │Frame Attention│ → │  GCA (Global)     │   │  × 24 组
│   │  帧内自注意力  │   │  跨帧因果注意力     │   │
│   │  + 2D RoPE   │   │  + 3D Video RoPE  │   │
│   └──────────────┘   │  + 分页 KV 缓存    │   │
│                      └───────────────────┘   │
│                                              │
│   在第 [4, 11, 17, 23] 组提取多尺度特征         │
│   frame_feat ⊕ global_feat → [B,S,P,2048]   │
└──────────┬──────────────────┬────────────────┘
           │                  │
     camera token          patch tokens
     (位置 0)              (位置 6~P)
           │                  │
           ▼                  ▼
┌──────────────────┐  ┌──────────────────────┐
│   Camera Head    │  │      DPT Head        │
│  迭代精炼 × 4     │  │  多尺度特征融合        │
│  → 9D 位姿编码   │  │  → 深度图 / 3D 点云   │
└────────┬─────────┘  └──────────┬───────────┘
         │                       │
         ▼                       ▼
   [T_x, T_y, T_z,         [depth, x, y, z,
    q_i, q_j, q_k, q_w,     confidence]
    fov_h, fov_w]
         │                       │
         ▼                       ▼
  外参 [R|t] + 内参 K      3D 世界坐标点云
```

**对应代码入口**：
- 整体推理流程：`demo.py` → `GCTStream.inference_streaming()` (`models/gct_stream.py:290-420`)
- 单帧前向传播：`GCTBase.forward()` (`models/gct_base.py:291-359`)

---

## 3. Stage 1: 图像编码 — DINOv2 视觉骨干网络

### 3.1 为什么选择 DINOv2？

DINOv2 是 Meta 训练的自监督视觉基础模型，它在没有任何标注的情况下学会了强大的视觉特征表示。LingBot-Map 选择 DINOv2 ViT-Large 作为骨干网络，因为：

1. **丰富的几何先验**：自监督训练让 DINOv2 学会了深度、法线等隐式几何知识
2. **强泛化能力**：在各种场景（室内/室外/合成/真实）中表现稳定
3. **标准化的 patch 表示**：14×14 的 patch 划分提供了统一的空间分辨率

### 3.2 编码过程

```python
# aggregator/base.py:369-392

# 1. ImageNet 标准化
images = (images - mean) / std    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

# 2. 展平 batch 维度
images = images.reshape(B * S, 3, H, W)

# 3. DINOv2 前向传播
patch_tokens = self.patch_embed(images)["x_norm_patchtokens"]
# 输出: [B*S, M, 1024]
# 其中 M = (H/14) × (W/14)
# 例如 518×294 的图像 → 37×21 = 777 个 patch token
```

每个 patch token 是一个 1024 维的向量，它编码了对应 14×14 像素区域的视觉特征。

---

## 4. Stage 2: 特殊 Token 系统 — 几何语义的载体

这是 LingBot-Map 设计中最精妙的部分之一。在每帧的 patch tokens 前面，模型会插入 **6 个特殊 token**，每个承担不同的几何功能：

```
每帧 token 序列:
┌────────┬─────┬─────┬─────┬─────┬───────┬────────┬────────┬───┐
│ camera │reg_0│reg_1│reg_2│reg_3│ scale │patch_0 │patch_1 │...│
│  idx=0 │  1  │  2  │  3  │  4  │   5   │   6    │   7    │   │
└────────┴─────┴─────┴─────┴─────┴───────┴────────┴────────┴───┘
  ← 6 个特殊 token (num_special_tokens) →   ← M 个 patch tokens →
```

### 4.1 各 Token 的角色

| Token | 数量 | 形状 | 功能 |
|-------|------|------|------|
| **Camera Token** | 1 | `[1, 2, 1, 1024]` | 聚合场景信息，最终输入 Camera Head 预测位姿 |
| **Register Token** | 4 | `[1, 2, 4, 1024]` | DINOv2 风格的辅助 token，防止注意力 sink 现象 |
| **Scale Token** | 1 | `[1, 2, 1, 1024]` | 标识当前帧是否为"锚定帧"，建立尺度参考 |

### 4.2 双变体机制

注意形状中的 `dim=1` 是 **2**——每种特殊 token 都有两个变体：

- **变体 0**（scale frame 版本）：用于前 n 个锚定帧
- **变体 1**（normal frame 版本）：用于后续的流式帧

```python
# aggregator/base.py:30-57 — slice_expand_and_flatten()
# 功能：根据帧的角色（锚定帧 vs 普通帧）选择对应的 token 变体

# 前 first_num_frame 帧使用变体 0，其余使用变体 1
first_tokens = token[:, 0:1, :, :]   # 锚定帧版本
rest_tokens  = token[:, 1:2, :, :]   # 普通帧版本
```

这种设计让网络能自动区分"建立坐标系的参考帧"和"需要定位的新帧"。

### 4.3 代码实现

```python
# aggregator/stream.py:151-177

self.camera_token = nn.Parameter(torch.randn(1, 2, 1, self.embed_dim))
self.register_token = nn.Parameter(torch.randn(1, 2, self.num_register_tokens, self.embed_dim))
self.scale_token = nn.Parameter(torch.ones(1, 2, 1, self.embed_dim))

# 初始化为极小值，确保初始阶段不干扰 patch 特征
nn.init.normal_(self.camera_token, std=1e-6)
nn.init.normal_(self.register_token, std=1e-6)
nn.init.normal_(self.scale_token, std=1e-6)

self.patch_start_idx = 1 + self.num_register_tokens + 1  # = 6
```

---

## 5. Stage 3: 交替注意力层 — Frame Attention 与 GCA

### 5.1 交替结构

LingBot-Map 使用 24 组交替注意力层，每组包含两个 Block：

```
Group 0:  [Frame Block 0] → [Global Block 0]
Group 1:  [Frame Block 1] → [Global Block 1]
  ...
Group 23: [Frame Block 23] → [Global Block 23]
```

**共计 48 个 Transformer Block**（24 个 Frame + 24 个 Global），两种 Block 轮流执行。

```python
# aggregator/base.py:582-607 — forward() 主循环

for block_group_idx in range(self.aa_block_num):     # 24 组
    for aa_type in self.aa_order:                    # ["frame", "global"]
        if aa_type == "frame":
            # 帧内自注意力，每帧独立处理
            tokens, frame_intermediates = self._process_frame_attention(...)
        elif aa_type == "global":
            # 跨帧因果注意力 (GCA)，通过 KV 缓存连接历史
            tokens, global_intermediates = self._process_global_attention(...)

    # 在选定层提取多尺度特征
    if block_group_idx in selected_idx:  # [4, 11, 17, 23]
        output = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
        # 拼接结果维度: [B, S, P, 2048]
```

### 5.2 Frame Attention（帧内注意力）

Frame Attention 在**每帧内部**独立执行标准自注意力：

```
帧 t 的 tokens: [camera, reg×4, scale, patch_0, ..., patch_M]
                 ↕ 全连接自注意力（帧内所有 token 互相关注）↕
```

- **作用**：提取单帧的视觉特征，让 camera token 从本帧的 patch 中收集几何信息
- **位置编码**：使用 2D RoPE，让每个 patch 感知自己在图像中的空间位置
- **特点**：没有跨帧通信，计算量固定，与序列长度无关

### 5.3 Global Attention (GCA)（跨帧注意力）

Global Attention 是 LingBot-Map 的核心创新——**Geometric Context Attention**，在下一节详细展开。

---

## 6. 核心创新: Geometric Context Attention (GCA)

### 6.1 设计动机

传统因果注意力（如 GPT）让每个新 token 关注**所有**历史 token，但这对流式 3D 重建有两个严重问题：

1. **显存线性增长**：每帧约 500+ 个 patch token，10,000 帧就是 500 万个 token
2. **噪声淹没信号**：远距离帧的 patch token 与当前帧视觉重叠极小，反而引入噪声

GCA 的核心思想是：**不是所有历史信息都同等重要，不同类型的上下文承担不同的几何功能。**

### 6.2 三级上下文结构

GCA 将流式上下文分解为三个互补的层次，直接借鉴了经典 SLAM 的设计哲学：

```
时间轴: ──────────────────────────────────────────────────→
帧序列: T_1 T_2 ... T_n | T_{n+1} ... T_{t-k-1} | T_{t-k} ... T_{t-1} | T_t
        └──────────────┘  └──────────────────────┘  └───────────────────┘  ↑
         Anchor Context      Trajectory Memory      Pose-Reference Window  当前帧
        (锚定上下文)          (轨迹记忆)              (位姿参考窗口)

保留内容:  全部 token          仅 6 个特殊 token       全部 token
          (camera+reg+scale  (camera+reg+scale       (camera+reg+scale
           +所有 patch)        丢弃 patch)              +所有 patch)

作用:     坐标系定标            全局漂移校正            局部几何对齐
          尺度锚定              时序排列感知            密集视觉匹配
```

#### (a) Anchor Context（锚定上下文）

**是什么**：序列开头的 n 帧（默认 n=8），保留全部 token（特殊 + patch）。

**为什么需要**：单目重建天然存在尺度模糊——从一张图片无法判断是 1 米远的小盒子还是 100 米远的大楼。锚定帧通过**双向注意力**（相互关注）建立一个一致的坐标系和尺度参考。所有后续帧都以这些锚定帧为参考来确定自己的位置。

**代码实现**：
```python
# models/gct_stream.py:355-364 — Phase 1: Scale frames
scale_output = self.forward(
    scale_images,                       # 前 n 帧
    num_frame_for_scale=scale_frames,
    num_frame_per_block=scale_frames,   # 所有 scale 帧作为一个 block 处理
    causal_inference=True,              # 但彼此之间是双向注意力
)
```

**训练时的尺度归一化**：
```
s = (1 / |X^anchor|) * Σ_{x ∈ X^anchor} ||x||_2
```
所有 ground-truth 深度和平移都除以 s，归一化到以锚定帧点云为参考的标准尺度。

#### (b) Local Pose-Reference Window（位姿参考窗口）

**是什么**：最近 k 帧（默认 k=64）的**完整** token（特殊 + patch）。

**为什么需要**：仅靠远距离的锚定帧无法提供当前帧与邻近帧之间的密集视觉对应关系。当相机连续移动时，相邻帧之间有大量视觉重叠，保留它们的 patch token 让模型能执行隐式的特征匹配来精确推断相对运动。

**代码实现**：
```python
# layers/flashinfer_cache.py:229-254 — evict_frames()

# 当滑动窗口中的帧数超过 k 时，驱逐最旧的 patch page
while len(state["live_window_patch_pages"]) > self.sliding_window:
    old_page = state["live_window_patch_pages"].popleft()
    state["free_patch_pages"].append(old_page)  # 回收给空闲池
# 注意：只驱逐 patch page，特殊 token 永不驱逐！
```

#### (c) Trajectory Memory（轨迹记忆）

**是什么**：介于锚定帧和滑窗之间的所有历史帧，只保留 **6 个特殊 token**（camera + register×4 + scale），**丢弃** patch token。

**为什么需要**：仅靠锚定帧和局部窗口，中间帧的位姿误差会无法检查地累积，导致长距离漂移。轨迹记忆以极低的成本（每帧仅 6 个 token vs ~500 个 patch token）保留了完整观测历史的**压缩摘要**，结合 3D Video RoPE 的时序编码，让模型能推理出完整的轨迹结构，从而校正累积漂移。

**关键洞察**：camera token 在经过 24 层交替注意力后，已经充分聚合了该帧的几何信息——它是该帧"在哪里看到了什么"的高效压缩表示。

### 6.3 复杂度分析

对于 T 帧序列（n 个锚定帧，k 个窗口帧，M 个 patch/帧）：

| 方法 | 每帧注意力上下文大小 | T=10,000 时的总 token 数 |
|------|-------------------|------------------------|
| **Full Causal** | T × (M + 6) | ~5,000,000 |
| **Sliding Window** | k × (M + 6) | ~32,000（但丢失长程信息）|
| **GCA (Ours)** | (n+k) × (M+6) + 6T | ~70,000 |

GCA 将每帧的上下文增长率从 ~500 token/帧降低到仅 **6 token/帧**（约 **80 倍压缩**），同时保留了长程信息。

### 6.4 注意力掩码对比

论文 Figure 3 展示了四种注意力模式的对比：

```
(a) Full Attention        (b) Causal Attention      (c) Sliding Window       (d) GCA (Ours)
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│█████████████████│      │█                │      │█                │      │█ █   █          │
│█████████████████│      │██               │      │ █               │      │█ ██  █          │
│█████████████████│      │███              │      │  ██             │      │█  ██ █          │
│█████████████████│      │████             │      │  ███            │      │█  ███ █         │
│█████████████████│      │█████            │      │   ████          │      │█   ██████       │
│█████████████████│      │██████           │      │    █████        │      │█    █████████   │
│█████████████████│      │███████          │      │     ██████      │      │█     ██████████ │
│█████████████████│      │████████         │      │      ███████    │      │█      ██████████│
└─────────────────┘      └─────────────────┘      └─────────────────┘      └─────────────────┘
 所有帧互相关注           每帧只看历史          只看最近 k 帧          锚定 + 轨迹记忆 + 窗口
 不能流式               显存线性增长          丢失长程信息           近乎恒定的每帧开销
```

---

## 7. Stage 4: 预测头 — 相机位姿与稠密几何

交替注意力层输出多尺度特征后，分两路送入不同的预测头。

### 7.1 Camera Head — 迭代精炼位姿

Camera Head 从 camera token 预测 9D 位姿编码，是定位的核心模块。

#### 9D 位姿编码

```
pose_enc = [T_x, T_y, T_z, q_i, q_j, q_k, q_w, fov_h, fov_w]
             └─────────┘  └────────────────────┘  └──────────┘
              3D 平移          4D 四元数旋转           2D 视场角
```

- **平移 T**：相机在世界坐标系中的位置
- **四元数 (i,j,k,w)**：相机的旋转姿态（scalar-last 格式）
- **视场角 FoV**：等价于内参，`f = (H/2) / tan(fov/2)`

#### 迭代精炼过程

Camera Head 不是一步预测位姿，而是通过 **4 次迭代精炼** 逐步收敛：

```
迭代 0: 输入=零向量（空位姿种子）  →  预测 Δpose_0  →  pose = Δpose_0
迭代 1: 输入=embed(pose.detach()) →  预测 Δpose_1  →  pose += Δpose_1
迭代 2: 输入=embed(pose.detach()) →  预测 Δpose_2  →  pose += Δpose_2
迭代 3: 输入=embed(pose.detach()) →  预测 Δpose_3  →  pose += Δpose_3 → 最终输出
```

每次迭代的内部流程：

```python
# heads/camera_head.py:360-395 — trunk_fn() 迭代循环

for iteration in range(num_iterations):
    # 1. 将当前位姿估计嵌入为条件信号
    if iteration == 0:
        module_input = self.embed_pose(self.empty_pose_tokens)  # 零初始化
    else:
        module_input = self.embed_pose(pred_pose_enc.detach())  # 截断梯度

    # 2. AdaLN 调制：用位姿估计来调制 camera token
    shift, scale, gate = self.poseLN_modulation(module_input).chunk(3, dim=-1)
    modulated = gate * (adaln_norm(camera_tokens) * (1 + scale) + shift)
    modulated = modulated + camera_tokens  # 残差连接

    # 3. 通过 4 层因果 Transformer Block（每次迭代有独立的 KV 缓存）
    for block in self.trunk:
        modulated = block(modulated, kv_cache=iter_cache, rope=rope_3d)

    # 4. 预测位姿残差并累加
    delta = self.pose_branch(self.trunk_norm(modulated))  # MLP: 2048 → 1024 → 9
    pred_pose_enc = pred_pose_enc + delta  # (首次迭代直接赋值)

    # 5. 激活函数
    pred_pose_enc = activate_pose(pred_pose_enc)
    # T: identity, quat: identity, FoV: ReLU (确保正值)
```

**为什么用 AdaLN？** 灵感来自 DiT (Diffusion Transformer)。AdaLN 让当前的位姿估计作为"条件信号"调制注意力计算——类似于扩散模型中的去噪过程，每次迭代都在上一步估计的基础上修正。

**为什么 detach？** 截断梯度流防止通过迭代路径的梯度爆炸，同时让每次迭代都独立优化当前残差。

#### 位姿编码的转换

```python
# utils/pose_enc.py:72-140 — pose_encoding_to_extri_intri()

# 四元数 → 旋转矩阵
R = quat_to_mat(quaternion)  # [B, S, 3, 3]

# 组成外参矩阵
extrinsics = [R | T]  # [B, S, 3, 4], world-to-camera

# FoV → 内参
fy = (H / 2) / tan(fov_h / 2)
fx = (W / 2) / tan(fov_w / 2)
K = [[fx, 0, W/2],
     [0, fy, H/2],
     [0,  0,  1 ]]
```

### 7.2 DPT Head — 稠密深度与点云

DPT (Dense Prediction Transformer) Head 使用多尺度特征融合来预测像素级的深度和 3D 坐标。

#### 多尺度特征金字塔

从交替注意力层的第 [4, 11, 17, 23] 组提取 4 个尺度的特征：

```
Scale 4 (最深层):  [B*S, 2048, h, w] → project → [B*S, 1024, h, w] → ↓2x 下采样
Scale 3:          [B*S, 2048, h, w] → project → [B*S, 1024, h, w] → 保持
Scale 2:          [B*S, 2048, h, w] → project → [B*S, 512,  h, w] → ↑2x 上采样
Scale 1 (最浅层):  [B*S, 2048, h, w] → project → [B*S, 256,  h, w] → ↑4x 上采样
```

#### 自底向上融合

```python
# heads/dpt_head.py:264-294 — scratch_forward()

path_4 = refinenet4(layer_4)                    # 最粗尺度
path_3 = refinenet3(upsample(path_4), layer_3)  # 融合第 3 层
path_2 = refinenet2(upsample(path_3), layer_2)  # 融合第 2 层
path_1 = refinenet1(upsample(path_2), layer_1)  # 融合第 1 层

# 最终卷积
output = output_conv(path_1)  # [B*S, 4, H, W]
# 其中 4 = 3 (xyz 或 depth) + 1 (confidence)
```

#### 两种预测模式

| Head | 输出 | 激活函数 | 用途 |
|------|------|---------|------|
| **Depth Head** | 1D 深度 + 置信度 | `inv_log`: sign(y)·(exp(\|y\|)-1) | 密度深度图 |
| **Point Head** | 3D 世界坐标 + 置信度 | `inv_log` | 直接预测 3D 点云 |

**深度反投影**：对于使用 depth 模式的情况，模型通过以下流程将深度图转换为 3D 点云：

```python
# models/gct_base.py:254-289 — _unproject_depth_to_world()

# 1. 9D 编码 → 外参/内参
extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc)

# 2. 外参求逆：cam-from-world → world-from-cam
c2w = closed_form_inverse_se3(extrinsics)  # R^T, -R^T·t

# 3. 针孔反投影
x_cam = (u - cx) * depth / fx
y_cam = (v - cy) * depth / fy
z_cam = depth

# 4. 相机坐标 → 世界坐标
world_points = c2w @ [x_cam, y_cam, z_cam, 1]^T
```

---

## 8. 推理系统设计 — 分页 KV 缓存

### 8.1 为什么需要分页 KV 缓存？

标准的 KV 缓存使用连续内存布局——每次驱逐旧帧或添加新帧都需要内存重新分配和数据搬移，这在高帧率推理中成为性能瓶颈。LingBot-Map 借鉴了 LLM 推理系统中的 **PagedAttention** 思想（来自 vLLM），使用分页布局让内存操作变为 O(1)。

### 8.2 双流设计

这是实现 GCA 三级上下文的工程核心：

```
物理内存布局 (每个 Transformer Block 一个):
kv_caches[block_idx]: [max_num_pages, 2, page_size, num_heads, head_dim]

┌─────────────────────────────────────────────────────────────────┐
│                     Patch Page Pool (可回收)                     │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐       ┌──────┐           │
│  │Page 0│ │Page 1│ │Page 2│ │Page 3│  ...  │Page N│           │
│  │scale │ │scale │ │window│ │window│       │ free │           │
│  │frame0│ │frame1│ │frame │ │frame │       │      │           │
│  └──────┘ └──────┘ └──────┘ └──────┘       └──────┘           │
├─────────────────────────────────────────────────────────────────┤
│                   Special Page Pool (只追加)                     │
│  ┌──────┐ ┌──────┐ ┌──────┐                                    │
│  │Page A│ │Page B│ │Page C│  ...                               │
│  │42帧的 │ │42帧的 │ │部分  │                                    │
│  │特殊tok│ │特殊tok│ │填充  │                                    │
│  └──────┘ └──────┘ └──────┘                                    │
└─────────────────────────────────────────────────────────────────┘
```

#### Patch 流（可回收）

- 每帧的 patch token 占用**恰好 1 个 page**（page_size = patches_per_frame）
- `scale_patch_pages` 队列：锚定帧的 patch page，**永不驱逐**
- `live_window_patch_pages` 队列：滑动窗口内的 patch page，超过 k 帧后 FIFO 回收
- 回收的 page 归还 `free_patch_pages` 池，供新帧复用

#### Special 流（只追加，永不回收）

- 每帧 6 个特殊 token，连续打包到共享 page 中
- 一个 page 可容纳 `floor(page_size / 6)` 帧的特殊 token（如 page_size=256 时约 42 帧）
- **永不驱逐**——这正是"轨迹记忆"的实现：patch 被回收后，特殊 token 仍然保留

### 8.3 注意力计算

```python
# layers/flashinfer_cache.py:417-429 — build_visible_page_table()

# 构建当前帧可见的所有 page（按顺序排列）
visible_pages = (
    list(scale_patch_pages)     # 锚定帧的 patch
    + list(live_window_patch_pages)  # 窗口内的 patch
    + list(all_special_pages)        # 所有帧的特殊 token（放最后）
)

# FlashInfer 使用 page table 执行高效的 paged attention
prefill_wrapper.plan(
    qo_indptr, paged_kv_indptr, paged_kv_indices,
    paged_kv_last_page_len, ...
)
output = prefill_wrapper.run(query, kv_caches[block_idx])
```

### 8.4 关键帧机制

当序列超过训练时的最大长度（320 帧）时，启用关键帧策略来控制 KV 缓存增长：

```python
# models/gct_stream.py:250-264 — _set_skip_append()

# 非关键帧：推理时正常使用 KV 缓存做注意力，但不将自己的 K/V 写入缓存
if not is_keyframe:
    for manager in all_kv_cache_managers:
        manager._skip_append = True  # 只用不存
```

效果：每 m 帧只有 1 帧写入 KV 缓存，内存增长率降低 m 倍。非关键帧仍然能读取缓存中的历史信息来做预测。

---

## 9. 位置编码 — 2D RoPE 与 3D Video RoPE

### 9.1 2D RoPE（帧内空间编码）

用于 **Frame Attention**，让每个 patch 知道自己在图像中的位置：

```
patch 坐标: (row, col) = (y, x)
RoPE 维度分配: 前半 → y 方向, 后半 → x 方向

例如 head_dim=64:
  dim[0:32]  编码 y 位置
  dim[32:64] 编码 x 位置
```

```python
# layers/rope.py:135-154 — _apply_1d_rope()

# 标准旋转位置编码：将特征对视为复数，乘以位置相关的旋转因子
x_complex = torch.view_as_complex(x.reshape(..., head_dim//2, 2))
rotated = x_complex * freqs_complex  # 复数乘法 = 旋转
output = torch.view_as_real(rotated).flatten(-2)
```

### 9.2 3D Video RoPE（跨帧时空编码）

用于 **GCA (Global Attention)** 和 **Camera Head**，在时间和空间三个维度上编码位置：

```
3D 位置: (t, h, w) = (帧序号, 行, 列)

head_dim 分配（以 camera head 为例, head_dim=128）:
  dim[0:40]   → 时间维度 t
  dim[40:84]  → 高度维度 h
  dim[84:128] → 宽度维度 w
```

**特殊 token 的位置分配**：

```python
# layers/rope.py:386-405

# 特殊 token 放在对角线位置，避免与 patch 冲突
# camera:  (f, 0, 0)
# reg_0:   (f, 1, 1)
# reg_1:   (f, 2, 2)
# reg_2:   (f, 3, 3)
# reg_3:   (f, 4, 4)
# scale:   (f, 5, 5)
# patch_0: (f, patch_start_idx + row, patch_start_idx + col)
```

**流式推理的关键**：通过 `f_start` 追踪全局帧偏移，确保第 1000 帧的时间编码正确反映其在序列中的真实位置，而不是每次 forward 都从 0 开始。

```python
# layers/rope.py:377-379

# 因果推理模式：使用全局帧计数器
f_start = f_start  # 例如已处理 999 帧，当前帧 f_start=999
f_end = f_start + S  # S=1 for streaming
```

### 9.3 为什么 Video RoPE 如此重要？

消融实验（论文 Table 6）显示，加入 Video RoPE 带来了最大的单项 ATE 改善（7.46 → 5.98, -1.48）。原因是：没有时序位置编码时，轨迹记忆中的 6 个 token 虽然携带几何信息，但**缺乏时间顺序感**——模型不知道哪个 token 是 100 帧前的、哪个是 1000 帧前的。Video RoPE 将时间信息直接注入注意力计算，让模型能推理出轨迹的时序结构，从而有效校正长距离漂移。

---

## 10. 推理模式 — Direct Output 与 VO 模式

### 10.1 Direct Output 模式（默认）

```python
# models/gct_stream.py:290-420 — inference_streaming()

# Phase 1: 处理锚定帧（双向注意力）
scale_output = model.forward(scale_images, num_frame_per_block=n)

# Phase 2: 逐帧流式推理
for frame_idx in range(n, total_frames):
    frame_output = model.forward(
        frame_image,
        num_frame_per_block=1,  # 单帧处理
        causal_inference=True
    )
    predictions.append(frame_output)
```

- GCA 的三级上下文持续累积，不重置
- 适用于 ≤3,000 帧的序列
- 无额外对齐误差，轨迹最准确

### 10.2 Visual Odometry (VO) 模式

```python
# models/gct_stream_window.py — inference_windowed()

# 将长序列切分为重叠窗口
for window in sliding_windows(frames, window_size=128, overlap=16):
    # 每个窗口独立推理（重置 KV 缓存）
    window_predictions = model.inference_streaming(window)

    # 通过重叠区域的 Sim(3) 对齐拼接窗口
    # 计算相对 scale, rotation, translation
    aligned = sim3_align(window_predictions, previous_predictions, overlap)
```

- 适用于超长序列（>3,000 帧，如城市级驾驶视频）
- 每个窗口独立处理，显存固定
- 窗口边界处引入对齐误差（Sim(3) 对齐非精确）

### 10.3 两种模式的权衡

| 特性 | Direct Output | VO Mode |
|------|:---:|:---:|
| 适用长度 | ≤ ~3,000 帧 | 任意长 |
| 轨迹精度 | 更高（无拼接误差）| 略低（窗口对齐误差累积）|
| 显存占用 | 线性增长（特殊 token 累积）| 固定 |
| 实现复杂度 | 简单 | 需要跨窗口对齐 |

---

## 11. 训练策略

### 11.1 两阶段训练

```
Stage 1: Base Model Training (离线, 全局注意力)
├── 初始化: DINOv2 ViT-Large 预训练权重
├── 注意力: 标准全局双向注意力（非因果）
├── 数据: 29 个数据集, 2-24 视图/样本, 无时序约束
├── 采样: nearby sampler (随机参考帧 + 空间邻近帧)
├── 训练: 160K 迭代, lr=2×10⁻⁴, AdamW
├── 分布式: FSDP, ~21,500 GPU hours
└── 目标: 学习鲁棒的几何先验

        ↓ 权重迁移（Q/K/V 投影直接复用）

Stage 2: Streaming Model Training (因果 GCA)
├── 初始化: Stage 1 权重 + GCA 替换全局注意力
├── 注意力: Geometric Context Attention (因果)
├── 数据: 偏重长轨迹视频数据集
├── 采样: foldback video sampler (折叠采样, 避免退化振荡)
├── 渐进训练: 视图数 24 → 320 线性增加
│   窗口大小 k: 16 → 64 随机采样
├── 训练: 160K 迭代, lr=5×10⁻⁴
├── 分布式: Ulysses 上下文并行, ~15,360 GPU hours
└── 目标: 学习流式因果推理能力
```

### 11.2 损失函数

```
L = λ_depth · L_depth + λ_abs-pose · L_abs-pose + λ_rel-pose · L_rel-pose
```

| 损失项 | 公式 | 作用 |
|-------|------|------|
| **L_depth** | 加权 L1 + 梯度 L1 - 不确定性正则 | 深度预测精度 + 边缘锐利度 |
| **L_abs-pose** | \|\|P̂ᵢ - Pᵢ\|\|₆ (camera-to-world) | 全局坐标系中的绝对位姿 |
| **L_rel-pose** | 窗口内所有帧对的相对旋转 + 平移误差 | 局部几何一致性 |

**相对位姿损失的作用**：消融实验显示，去掉相对损失后 RPE-rot 从 1.93 恶化到 5.35（2.4 倍），说明它对局部旋转一致性至关重要。

### 11.3 渐进式视图训练

直接在长序列上训练会导致梯度不稳——早期帧的位姿误差沿轨迹传播并放大。渐进策略让模型先在短片段上学好局部几何，再逐步扩展到长轨迹：

```
训练进度 0%   → 视图数=24,  窗口 k=16
训练进度 50%  → 视图数=172, 窗口 k=40
训练进度 100% → 视图数=320, 窗口 k=64
```

### 11.4 Foldback Video Sampler

从长视频中采样训练子序列的策略，避免只采样单方向运动：

```
从随机帧开始 → 随机步长前进 → 到达边界 → 反转方向(新步长) → ...
```

这产生了帧率自然变化、无前向时间偏差的训练子序列，帮助模型学习各种运动模式。

---

## 12. 实验结果与性能分析

### 12.1 位姿估计 — Oxford Spires (Sparse, 320 帧)

| 方法 | 类型 | AUC@15 ↑ | ATE ↓ |
|------|------|---------|------|
| DA3 | 离线 | 49.84 | 12.87 |
| VGGT | 离线 | 23.84 | 24.78 |
| VIPE | 优化 | 45.35 | 10.52 |
| CUT3R | 在线 | 5.98 | 18.16 |
| Wint3R | 在线 | 11.61 | 23.42 |
| **LingBot-Map** | **在线** | **61.64** | **6.42** |

LingBot-Map 作为在线方法，不仅超越所有在线竞争者（位姿精度 10 倍于 CUT3R），甚至超过最强的离线方法 DA3 和优化方法 VIPE。

### 12.2 长序列鲁棒性 — Oxford Spires (Dense, 3840 帧)

| 方法 | ATE_sparse | ATE_dense | ΔATE |
|------|-----------|----------|------|
| CUT3R | 18.16 | 32.47 | +14.31 |
| Wint3R | 21.10 | 32.90 | +11.80 |
| **LingBot-Map** | **6.42** | **7.11** | **+0.69** |

当序列从 320 帧扩展到 3840 帧时，竞争方法的 ATE 增长 60-80%，而 LingBot-Map 仅增长 10.7%，验证了 GCA 三级上下文结构的长程一致性。

### 12.3 3D 重建质量

| 方法 | ETH3D F1 ↑ | 7-Scenes F1 ↑ | NRGBD F1 ↑ |
|------|-----------|--------------|-----------|
| Wint3R | 77.28 | 78.81 | 56.96 |
| Stream3R | 72.87 | 78.79 | 54.07 |
| **LingBot-Map** | **98.98** | **80.39** | **64.26** |

### 12.4 推理效率

| 配置 | FPS | 显存 |
|------|-----|------|
| GCA + FlashInfer (窗口=64) | ~20 | 13.28 GB |
| Full Causal + FlashInfer | ~12 | 36.06 GB |
| GCA + SDPA Fallback | ~10.5 | ~15 GB |

GCA + FlashInfer 实现了 1.7 倍加速和 2.7 倍显存降低，同时轨迹精度反而更好（ATE 5.98 vs 6.60）。

### 12.5 消融实验总结

| 组件 | AUC@3 变化 | ATE 变化 | 核心作用 |
|------|-----------|---------|---------|
| Anchor Init. | +3.83 | -0.71 | 坐标系定标，尺度锚定 |
| Context Tokens | +2.12 | -0.42 | 长程漂移校正 |
| Relative Loss | - | -0.79 | 局部旋转一致性 |
| Video RoPE | +0.64 | **-1.48** | 时序感知，最大单项改善 |

---

## 13. 代码结构速查

```
lingbot-map/
├── demo.py                          # 推理入口: 加载图像/视频 → 模型推理 → 3D 可视化
├── gct_profile.py                   # FPS 性能基准测试
├── lingbot-map-long.pt              # 预训练权重 (4.6 GB)
│
└── lingbot_map/                     # 核心 Python 包
    ├── models/
    │   ├── gct_base.py              # GCTBase: 模型基类, forward() 主流程
    │   ├── gct_stream.py            # GCTStream: 流式推理, inference_streaming()
    │   └── gct_stream_window.py     # GCTStream 扩展: 窗口化 VO 模式
    │
    ├── aggregator/
    │   ├── base.py                  # AggregatorBase: DINOv2 编码, 交替注意力循环
    │   └── stream.py                # AggregatorStream: 因果 GCA 实现
    │
    ├── heads/
    │   ├── camera_head.py           # CameraCausalHead: 迭代精炼位姿预测
    │   ├── dpt_head.py              # DPTHead: 多尺度深度/点云预测
    │   ├── head_act.py              # 激活函数: inv_log, relu, expp1
    │   └── utils.py                 # UV 网格, 位置编码工具
    │
    ├── layers/
    │   ├── attention.py             # 多头注意力 (SDPA/FlashInfer)
    │   ├── block.py                 # Transformer Block: Block, FlashInferBlock, CameraBlock
    │   ├── flashinfer_cache.py      # 双流分页 KV 缓存管理器
    │   ├── rope.py                  # 2D RoPE + 3D Video RoPE
    │   ├── vision_transformer.py    # DINOv2 ViT 工厂函数
    │   └── ...                      # MLP, SwiGLU, DropPath, PatchEmbed 等
    │
    ├── utils/
    │   ├── geometry.py              # 3D 几何: 反投影, SE(3) 逆, Umeyama 对齐
    │   ├── pose_enc.py              # 9D 编码 ↔ 外参/内参 转换
    │   ├── rotation.py              # 四元数 ↔ 旋转矩阵
    │   └── load_fn.py              # 图像加载与预处理
    │
    └── vis/
        ├── point_cloud_viewer.py    # viser 3D 交互可视化
        ├── glb_export.py            # GLB 格式导出
        ├── sky_segmentation.py      # ONNX 天空分割
        └── ...
```

---

## 附录: 一句话总结

**LingBot-Map = DINOv2 视觉编码 + GCA 三级因果注意力（锚定 + 窗口 + 轨迹记忆）+ 迭代精炼 Camera Head + 分页 KV 缓存**，用一个纯前馈 Transformer 实现了 ~20 FPS 的实时流式 3D 重建，在多个基准上超越离线方法和优化方法。

---

*本文档基于论文 arXiv:2604.14141 及代码仓库 https://github.com/robbyant/lingbot-map 生成。*
