# 双目深度估计在板端 / 边缘设备上的部署约束与硬件路线调研

> 结论先行：**如果目标是“稳定可量产、功耗可控、时延可预估”**，板端双目首先应把 **classical stereo（SGBM/SGM/BM + Census/AD-Census + WLS/中值/左右一致性检查）** 当成基线方案；只有在**低纹理、重复纹理、弱光、反光、跨域鲁棒性**确实成为核心瓶颈时，再考虑 neural stereo。神经双目里又应优先区分：
> - **轻量 2D/局部相关模型**：有机会在 Jetson GPU / 少数 NPU 上落地；
> - **重型 3D cost volume 模型**（PSMNet、GC-Net 一类）：在板端通常不现实，除非接受很低分辨率、很低帧率、较高功耗，或转向更强的边缘 GPU/FPGA/专用加速卡。

---

## 1. 问题背景：为什么双目在板端部署很“吃系统工程”

双目深度估计不是只看“模型能不能跑”。板端真正受限的是一整条链路：

1. **输入分辨率**：双目通常至少两路同步输入，640×480、720p、甚至 1080p；
2. **视差搜索范围**：最大视差越大，匹配代价体越大；
3. **目标帧率**：机器人/AGV/无人机往往要求 15–30 FPS，避障甚至更高；
4. **端到端时延**：不仅是推理，还包括 ISP、去畸变、校正、内存搬运、后处理；
5. **功耗/散热**：边缘盒子和移动机器人常常只有 5–15W 或 10–25W 预算；
6. **算子与工具链**：NPU 不一定支持 correlation / grid_sample / dynamic shape / 3D conv；
7. **内存带宽**：双目比单目更容易卡在 cost volume 构建与访存，而不是纯算力；
8. **量产维护**：板卡寿命、SDK 稳定性、量化精度、标定流程、相机同步、供应链。

所以，板端双目本质上是一个 **“视觉算法 × 芯片架构 × 工程工具链 × 功耗热设计”** 的联合优化问题。

---

## 2. 典型硬件路线：Jetson、RK3588 与常见边缘 NPU/SoC 的部署现实

## 2.1 NVIDIA Jetson 路线：目前最现实的“神经双目落地平台”

NVIDIA Jetson 的优势不只是峰值 TOPS，而是 **CUDA + TensorRT + cuDNN + VPI/OpenCV CUDA/PVA（部分平台）** 的整体生态。对于双目这类既有传统视觉又可能有深度网络的任务，Jetson 仍然是最稳的一条路线。

### Jetson Nano

根据公开规格页，Jetson Nano 属于入门级平台，典型信息包括：
- 4GB 内存
- 内存带宽约 **25.6 GB/s**
- 低功耗工作模式常见在 **5W** 级别
- AI 算力远低于 Xavier / Orin，常被描述为亚 TOPS 级（经典资料里约 472 GFLOPS FP16 量级）

**部署现实：**
- **classical stereo：可做**。640×480 / 720p 下用 OpenCV StereoBM/SGBM、CUDA 版本 SGM、或者较轻的 Census + SGM，仍有落地空间；
- **轻量 neural stereo：非常吃力**。只能跑非常小的 backbone、低分辨率、低帧率版本；
- **重型 cost volume 网络：基本不建议**；
- Nano 更适合做：
  - 传统双目
  - 双目 + 小型检测/跟踪
  - 原型验证，而不是重型神经立体匹配量产

**一句话判断：** Nano 能做双目，但更像“传统算法板”，不是“神经双目板”。

### Jetson Xavier NX

公开资料常见规格：
- **21 TOPS** 级 AI 算力
- **8GB / 16GB** 内存版本（常见资料中以 8GB 为主）
- 内存带宽约 **59.7 GB/s**
- 常见功耗档位 **10W / 15W**

**部署现实：**
- classical stereo 已经比较从容，可在更高分辨率、更大视差范围下工作；
- 轻量神经双目开始可用，例如较轻量的 2D encoder-decoder、局部相关、迭代 refinement 类模型；
- 对 PSMNet/GC-Net/RAFT-Stereo/CREStereo 这类模型，通常仍要**大幅降分辨率 + TensorRT + FP16 + 小 batch=1**，才能勉强做到可接受时延；
- Xavier NX 最大问题不是“跑不起来”，而是**带宽和显存余量紧张**，同时还要留资源给 ROS、感知融合、控制、编码等任务。

**适合场景：**
- 中端机器人/AMR/无人机
- 需要神经双目但帧率要求不极端
- 720p 以下更现实

### Jetson Orin Nano / Orin NX / AGX Orin

NVIDIA Orin 系列明显把板端神经感知门槛往前推了一大截。公开规格页常见信息：
- **Jetson Orin Nano**：约 **67 TOPS** 级别，常见 **4GB / 8GB**，低功耗档约 **7W–25W**；
- **Jetson Orin NX**：约 **157 TOPS** 级别，常见 **8GB / 16GB**，常见 **10W–40W**；
- **Jetson AGX Orin**：最高约 **275 TOPS**，常见 **32GB / 64GB**，功耗可到 **15W–60W**；
- 内存带宽明显提升：如 Orin Nano 4GB 约 **51 GB/s**，8GB 约 **102 GB/s**；更高型号可到 **204.8 GB/s** 级别。

**部署现实：**
- **Orin Nano**：已经是“能认真做神经双目”的入门板；但若模型包含大 cost volume、长迭代链、复杂后处理，仍容易受限于显存与带宽；
- **Orin NX**：是目前很多机器人/工业视觉里最均衡的点——性能、功耗、体积、生态都不错；
- **AGX Orin**：适合多传感器融合、多网络并行、较高分辨率神经双目、复杂后处理或多相机系统。

**关键现实判断：**
- 对 neural stereo 而言，Jetson 的真正优势是 **TensorRT 路线成熟**；
- 但注意：很多 stereo 模型不是标准“分类/检测网络”，而是涉及：
  - correlation / cost volume build
  - 3D convolution
  - iterative update / GRU-like blocks
  - warp / grid_sample
  - soft argmin / differentiable regression
  这些算子并不都能顺滑转 TensorRT，需要插件、改图、简化结构，甚至重写部分算子。

因此，**Jetson 能部署神经双目，不代表任何论文 stereo 网络都能“一键 TensorRT”**。真正量产可落地的，一定是被“工程改造”过的版本。

---

## 2.2 RK3588 路线：更适合“传统双目 + 轻量 AI 辅助”，不适合重型神经双目主干

RK3588 / RK3588S 是国内板端很常见的一条路线，优势是：
- 成本通常比 Jetson 更友好；
- 8 核 CPU（A76 + A55）整体通用算力不错；
- 集成 Mali-G610 GPU；
- 集成 **约 6 TOPS** 级 NPU（INT8 口径）；
- 板卡生态丰富（Radxa、Firefly、FriendlyElec 等）；
- 多路视频编解码和 I/O 资源比较强。

**部署现实：**
1. **传统双目很适合**：
   - CPU NEON / OpenCL / Vulkan / GPU 可做一定加速；
   - 适合 SGBM/SGM、 Census、块匹配等；
2. **神经双目受限明显**：
   - RKNN/NPU 对网络结构支持有边界；
   - 对 3D cost volume、动态形状、复杂相关算子、grid_sample、迭代更新这类结构不友好；
   - 即便模型“能转”，也常出现 **部分子图落 CPU/GPU，整体反而更慢**；
3. **量化约束更强**：
   - RK3588 的 6TOPS 一般是 INT8 宣传口径；
   - stereo 回归任务对量化噪声比分类任务更敏感；
   - 视差回归、亚像素精度、softmax/soft-argmin 在低比特下可能明显掉精度；
4. **工具链投入较大**：
   - 要围绕 RKNN 支持的算子表去“逆向设计模型”；
   - 很多论文模型要结构重写，不能照搬。

**一句话判断：**
RK3588 更像是 **“传统视觉 SoC + 轻量感知 NPU”**，而不是通用神经立体匹配平台。若核心卖点是“神经双目高质量深度”，RK3588 往往不如 Orin 路线省心；但如果你要的是 **低成本、国产链路、传统双目足够用、偶尔再加一点轻量网络辅助**，RK3588 很有竞争力。

---

## 2.3 常见边缘 NPU / SoC 的总体现实

除了 Jetson 和 RK3588，板端还常见：
- **Qualcomm QCS/QRB/SA 系列**：CPU + Adreno GPU + Hexagon DSP/NPU；
- **瑞芯微 / 全志 / Amlogic / 海思等 SoC**：偏视频/IPC/边缘盒子；
- **地平线 Journey、寒武纪 MLU 边缘模块、爱芯、算能 SOPHON** 等国产 AI 芯片；
- **Intel CPU/iGPU/VPU（OpenVINO 路线）**；
- **FPGA / SoC FPGA**（Xilinx/AMD Zynq、Intel FPGA）；
- **Google Coral / Edge TPU** 这类固定算子风格的低功耗边缘加速器。

这些路线在双目上有一个共同现实：

### 现实 1：宣传 TOPS 往往不能直接映射到 stereo 能力

很多 NPU 的 TOPS 指标是：
- INT8
- 稠密卷积
- 理想算子组合
- 大 batch 或理想访存条件

但 stereo 常见瓶颈并不完全是“卷积 MAC 数”：
- cost volume 构建带来**大访存**；
- 3D conv 带来**显存/带宽爆炸**；
- correlation / warping / gather/scatter 这类操作对 NPU 不友好；
- batch=1、低延迟模式下，利用率常常不好。

所以一个“10 TOPS NPU”不一定比一个“算力更低但 CUDA 生态成熟的 GPU”更适合双目。

### 现实 2：双目比普通检测/分类更依赖算子支持与内存体系

做目标检测时，NPU 很容易发挥作用，因为模型主要是 2D conv / activation / concat / upsample；
但 stereo 网络经常有：
- 相关性计算
- 代价体拼接
- 3D 卷积
- 多尺度迭代 refinement
- 亚像素回归

这会让很多 NPU 在图转换阶段就卡住，或者只能把最重部分丢回 CPU/GPU。

### 现实 3：DSP / FPGA 常在“确定性、低功耗、固定流程”上更有价值

如果你的算法路径非常稳定，比如：
- rectification
- Census transform
- SGM
- median / bilateral / LR check

那么 DSP/FPGA 往往能做出比通用 NPU 更好的：
- 确定性时延
- 每瓦性能
- 实时性
- 量产一致性

这也是为什么很多工业双目、车载双目、深度相机模组内部仍然大量使用 classical / semi-classical pipeline 或硬件专用 stereo block。

---

## 3. Classical stereo vs Neural stereo：板端部署差异拆解

这是最关键的一部分。

## 3.1 算力特征

### Classical stereo

典型流程：
- 标定/校正
- 代价计算（SAD / Census / AD-Census）
- 代价聚合（box filter / SGM）
- 视差选择（winner-take-all）
- 左右一致性检查
- hole filling / WLS / 中值滤波

**算力特点：**
- 更偏 **规则计算 + 内存访问 + 整数/定点运算**；
- 对 SIMD、DSP、FPGA、ISP/VPU 邻近模块友好；
- 不一定需要高 TOPS，但需要高效访存和流水线；
- 延迟通常比较可控。

### Neural stereo

典型流程：
- 左右图特征提取（CNN/Transformer backbone）
- cost volume 构建（concat/correlation/group-wise correlation）
- 代价聚合（2D/3D CNN, recurrent update）
- disparity regression / refinement

**算力特点：**
- 高度依赖 MAC 算力，但也非常依赖**显存/带宽**；
- 3D cost volume 容易从“算力问题”变成“内存问题”；
- 小板端 batch=1 时，GPU/NPU 利用率未必理想；
- 复杂模型很容易出现“峰值 TOPS 看起来够，实际 FPS 上不去”。

**直观判断：**
- 如果你在意**可解释、低功耗、确定性**：classical stereo 更优；
- 如果你在意**困难纹理场景鲁棒性与最终精度上限**：neural stereo 更优，但代价高很多。

---

## 3.2 内存与带宽差异

### Classical stereo 的内存行为

classical stereo 也吃内存，但它的中间表示往往相对规整、可流式处理，适合：
- line buffer
- tile-based processing
- 定点压缩
- FPGA / DSP pipeline

例如 SGM 的内存消耗会随分辨率和最大视差上升，但仍比重型 3D CNN 更容易做工程优化。

### Neural stereo 的内存行为

neural stereo 的最大板端痛点往往是 **cost volume**。

粗略看，一个体素维度可能接近：
- H × W × D × C 或 H × W × D
- 其中 D 是 disparity range，C 是通道数

只要：
- 分辨率上去（例如 720p）
- disparity range 上去（例如 192 / 256）
- feature channel 上去
- 再叠 3D conv

显存与带宽压力会迅速失控。

这也是为什么很多板端可部署神经双目模型会采用：
- 更低分辨率
- 分层 coarse-to-fine
- group-wise correlation 替代 full concat
- recurrent update 替代大规模 3D conv stack
- 局部窗口/稀疏代价体

**一句话：板端 neural stereo 先受限的通常不是理论算力，而是 cost volume 引发的显存和带宽。**

---

## 3.3 功耗差异

### Classical stereo

- 可在 CPU/DSP/FPGA 上稳定低功耗运行；
- 对于 5W–10W 等级平台更友好；
- 长时间恒定负载下，热行为更可预测。

### Neural stereo

- GPU/NPU 持续高负载，功耗更高；
- 如果还要叠加检测、分割、SLAM，会迅速把系统推到热瓶颈；
- 对无风扇/密闭边缘盒子尤其敏感；
- NPU 虽然名义上更省电，但前提是**核心子图真的能完整落在 NPU**。

实际项目里常见现象：
- 论文模型在实验室里能跑；
- 上车/上板后因为热 throttling，10 分钟后 FPS 掉一截；
- 于是最后又回退到更轻的模型或 classical baseline。

---

## 3.4 延迟与实时性差异

### Classical stereo

- 通常更容易做到**确定性时延**；
- 对实时控制闭环更友好；
- 最坏情况时延（worst-case latency）更容易评估。

### Neural stereo

- 端到端 latency 受模型结构、kernel launch、内存碎片、异构调度影响更大；
- 若有 CPU/GPU/NPU 混合执行，抖动会更明显；
- 对 ROS/多线程系统，要特别小心 pipeline 抖动。

所以在避障、抓取、飞控等强实时场景，**平均 FPS 不够，必须看 P95/P99 latency**。

---

## 3.5 量化难度差异

### Classical stereo

- 本来就适合定点化；
- 许多模块天然适合 int8/int16/定制 bit-width；
- 对 FPGA / DSP 极友好。

### Neural stereo

量化比分类/检测更难，原因包括：
1. **回归任务比分类更怕量化误差**；
2. soft argmin / disparity regression 对数值分布敏感；
3. 小视差变化就会导致深度误差放大，尤其远距离时更明显；
4. cost volume 内部动态范围复杂；
5. 多尺度迭代 refinement 对低比特误差累积敏感。

因此：
- **FP16** 往往是 Jetson 上最实际的甜点；
- **INT8** 只有在模型专门为量化友好设计、且有高质量校准数据时才值得做；
- 对很多 stereo 网络，INT8 可能“能跑，但深度边缘/细结构/远距离显著变差”。

---

## 3.6 算子支持差异

### Classical stereo

实现形式灵活：
- CPU C/C++
- OpenCV
- CUDA
- DSP kernel
- FPGA RTL/HLS
- ISP / 专用硬件 block

### Neural stereo

最常见的部署障碍不是“模型太大”，而是**算子不受支持**。

重点风险算子：
- 3D convolution / 3D deconv
- correlation / cost volume custom op
- grid_sample / warp / remap
- dynamic loop / recurrent update
- soft argmin / custom regression head
- 某些 layer norm / attention 变体

这也是为什么：
- 许多论文模型在 PyTorch 能训能测；
- 到 ONNX/TensorRT/RKNN/OpenVINO/TFLite 时要么图断裂，要么性能很差。

**部署经验规律：**
- 标准 2D CNN 越多，越好部署；
- 自定义 volume / iterative / 3D op 越多，越难部署；
- 如果目标是 NPU，模型结构必须“围着工具链设计”，不能先论文后迁移。

---

## 4. 哪些模型更适合 GPU / NPU / CPU / DSP / FPGA

下面不是“论文精度榜”，而是板端部署导向的分类。

## 4.1 更适合 CPU / DSP / FPGA 的路线

### 代表算法
- StereoBM
- StereoSGBM / SGM
- Census / AD-Census + SGM
- ELAS（部分场景）
- 局部块匹配 + 后处理

### 原因
- 算法结构规则；
- 可做定点；
- 内存访问模式可优化；
- 易于做低功耗、确定性实现；
- 适合量产与长生命周期维护。

### 适合场景
- 工业避障
- AGV/AMR
- 固定场景测距
- 深度相机模组
- 功耗特别敏感的设备

**我的判断：如果目标是“先交付、先量产、先稳定”，classical stereo 永远值得先做一版。**

---

## 4.2 更适合 GPU 的神经双目路线

### 代表模型族
- **AnyNet**：coarse-to-fine、强调实时性，历史上就面向移动/嵌入式友好；
- **HITNet**：主打实时与轻量，工业界关注度高；
- **AANet**：通过自适应聚合减轻传统 3D 体开销；
- **一些轻量 RAFT-style stereo 变体**：迭代 refinement 替代重型 3D CNN；
- 经过工程裁剪的小型 hourglass / 2D aggregation stereo 网络。

### 原因
- GPU 对 2D conv、相关性、部分自定义算子更灵活；
- TensorRT/FP16 可较好发挥；
- 对 batch=1、低延迟推理经验更成熟。

### 不太适合板端 GPU 的模型
- **PSMNet / GC-Net**：经典但重，3D cost volume + 3D conv 开销大；
- **RAFT-Stereo / CREStereo 原始大模型**：精度强，但工程落地常需明显裁剪；
- Transformer 化、全局相关很重的 stereo 结构：对板端更不友好。

**落地建议：**
在 Jetson 上，优先考虑：
1. 小分辨率训练/推理版本；
2. 去掉昂贵 3D 模块；
3. FP16 优先，INT8 谨慎；
4. 尽量避免依赖 TensorRT 难支持的自定义算子。

---

## 4.3 更适合 NPU 的模型形态

这里要强调：**不是“哪个论文模型适合 NPU”，而是哪种网络形态适合 NPU”。**

### 更适合 NPU 的特征
- 以 **2D CNN** 为主；
- 少用或不用 3D conv；
- 少用动态迭代循环；
- 少用 grid_sample / deformable / custom correlation；
- 张量形状静态；
- 使用标准 conv / depthwise conv / relu / concat / upsample；
- 视差估计改写为更规则的分类/回归头。

### 适合思路
- “**双目特征提取 + 简化相关性 + 2D refinement**”
- “**classical stereo 产出初值 + 小网络做 refinement / confidence / hole filling**”
- “**边缘检测/语义辅助 + 传统匹配**”

### 不适合 NPU 的思路
- 大型 3D cost volume
- RAFT 风格多轮迭代更新（若工具链不支持循环展开）
- 高精度亚像素 soft argmin 重度依赖浮点细节

因此在 RK3588、Journey、Hexagon、Edge TPU 这类平台上，**最现实的不是“纯神经 stereo 主干”，而是“传统双目主干 + 轻量神经修补件”**。

---

## 4.4 FPGA 的位置

FPGA 不是最灵活，但在双目这个问题上非常有存在感。

### FPGA 适合什么
- rectification
- census transform
- cost computation
- SGM 路径聚合
- LR consistency
- filter / post-process
- 深度后处理 pipeline

### FPGA 的优势
- 低延迟、强确定性；
- 高每瓦性能；
- 适合高吞吐视频流；
- 适合车载、工业、安防、深度相机模组。

### FPGA 的劣势
- 开发门槛高；
- 迭代速度慢；
- 复杂神经网络适配不如 GPU/NPU 灵活。

**如果产品已进入量产后期，算法也基本稳定，FPGA/专用 ASIC stereo pipeline 往往比“继续堆大网络”更像正确方向。**

---

## 5. 按硬件平台给出实际部署建议

## 5.1 Jetson Nano：建议路线

**推荐：**
- OpenCV StereoSGBM / CUDA SGM
- 低分辨率 classical stereo
- 如需学习方法，只做 very lightweight refinement

**不推荐：**
- 任何重型 3D CNN stereo
- 想同时兼顾高分辨率 + 高帧率 + 高质量 neural stereo

**典型定位：**
教育、原型、低成本边缘盒子、轻量机器人。

---

## 5.2 Xavier NX：建议路线

**推荐：**
- classical stereo 作为生产方案
- 轻量神经双目作为增强方案
- TensorRT FP16
- 分辨率控制在 640×480 或 640×384 / 768×384 这类更现实的范围

**谨慎：**
- 720p 以上 + 大 disparity + 神经双目 + 多任务并行
- 原始 RAFT-Stereo / CREStereo 级模型直接上板

**典型定位：**
中端机器人、视觉导航、较复杂边缘感知。

---

## 5.3 Orin Nano / NX / AGX：建议路线

**推荐：**
- Orin Nano：轻量 neural stereo 起步平台
- Orin NX：主力部署平台，性价比高
- AGX Orin：多传感器、复杂融合、高分辨率部署

**推荐策略：**
- 先做 classical baseline；
- 再做 1–2 个轻量 neural stereo 候选模型；
- 必须做 TensorRT profiling，而不是只看 PyTorch latency；
- 优先 FP16，必要时再试 INT8；
- 看显存峰值、带宽占用、P95 latency、长稳态温度。

---

## 5.4 RK3588：建议路线

**推荐：**
- 传统双目做主干
- 小网络做 confidence / refine / mask / segmentation assist
- 若必须上神经 stereo，优先设计 **NPU-friendly 2D CNN 小模型**，不要直接搬论文 SOTA

**不推荐：**
- 把 RK3588 当成 Jetson 替代品去跑重型 neural stereo
- 依赖复杂 ONNX → RKNN 自动转换的侥幸心理

**典型定位：**
低成本边缘网关、国产化项目、轻量机器人、工业终端。

---

## 5.5 其他边缘 NPU/SoC：建议路线

如果平台是 Qualcomm / Journey / SOPHON / Coral / 其它 NPU：

**优先问 4 个问题：**
1. 支不支持 3D conv？
2. 支不支持 correlation / custom volume？
3. 支不支持 grid_sample / warp？
4. 不支持时，fallback 到 CPU/GPU 的代价多大？

如果这 4 个问题里有 2 个以上答案不理想，基本就该转向：
- classical stereo
- 或 classical + neural refinement hybrid

---

## 6. 一个实用的“板端部署决策框架”

下面给一个我认为最实用的决策框架，不是按论文指标，而是按工程约束推进。

## 第一步：先定义你的系统边界

至少明确 8 个参数：
1. 输入分辨率：640×480 / 720p / 1080p？
2. 最大视差范围：64 / 128 / 192 / 256？
3. 最低帧率要求：10 / 15 / 30 FPS？
4. 最坏时延预算：50ms / 100ms / 200ms？
5. 功耗预算：5W / 10W / 15W / 30W+？
6. 深度精度更看重近距还是远距？
7. 场景是结构化工业环境还是开放道路/室内复杂场景？
8. 是否还要同时跑检测、分割、SLAM、语义？

如果这些都不清楚，讨论“选什么模型/板子”几乎没有意义。

---

## 第二步：先做 classical baseline

先实现：
- 标定 + rectification
- StereoSGBM / Census-SGM
- LR check
- WLS / median / speckle removal

测 4 个指标：
- 精度够不够
- 低纹理失败率
- 时延
- 功耗

**如果 classical 已满足 80% 需求，就不要轻易引入神经双目。**
这不是保守，而是工程理性。

---

## 第三步：确认瓶颈到底是不是“匹配能力”而不是系统问题

很多项目误以为“要上神经双目”，实际问题可能是：
- 标定不好
- 曝光不同步
- 镜头畸变校正误差
- baseline / 焦距设计不合理
- ISP 参数破坏匹配稳定性
- 视差搜索范围设置不合理
- 后处理不足

这些问题不解决，换模型只是在放大工程噪声。

---

## 第四步：如果确实要 neural stereo，先按硬件选模型，不要反过来

### 如果是 Jetson
- 优先轻量、2D-friendly、TensorRT-friendly 模型；
- 避免从重型 3D cost volume 网络直接开始；
- 优先 FP16；
- 优先在 Orin NX/AGX 上验证，再下探 Nano/Xavier。

### 如果是 RK3588 / 常见 NPU
- 先查算子表；
- 只选标准 2D CNN 小模型；
- 最好采用 hybrid：traditional stereo + neural refine/confidence。

### 如果是 CPU/DSP/FPGA 主导
- 以 classical 为主；
- 如果要学习方法，只把神经网络放在边缘修补环节，而不是全流程主干。

---

## 第五步：评估时不要只看 mIoU / EPE / D1-all，要看板端四指标

对板端真正关键的是：
1. **P95 / P99 latency**
2. **稳态功耗与温升**
3. **峰值内存/显存占用**
4. **工具链可维护性**

我会把这 4 项看得和精度同等重要，甚至更重要。

---

## 第六步：做“三级方案”而不是单点押注

一个靠谱项目通常同时保留：

### A 案：classical 生产保底
- 最稳、最低功耗、确定性最好

### B 案：轻量 neural 增强
- 在难场景提升质量

### C 案：高性能上限方案
- 只在高端板卡/演示版/下一代产品上启用

这样能避免项目因为单一神经模型迟迟无法量产而整体卡死。

---

## 7. 最后给出一个简明选型建议

### 7.1 如果你追求“最稳妥量产”
选：
- **算法**：Classical stereo（SGBM/SGM/Census）
- **硬件**：RK3588 / Xavier NX / FPGA / DSP 路线
- **原因**：低风险、低功耗、确定性强、开发周期可控

### 7.2 如果你追求“在边缘板上把神经双目真正做出来”
选：
- **硬件**：**Jetson Orin NX** 优先，其次 AGX Orin，预算很紧再看 Orin Nano
- **算法**：轻量 neural stereo 或 hybrid stereo
- **原因**：Jetson 生态对这类任务最成熟

### 7.3 如果你追求“国产低成本 + 工程可交付”
选：
- **硬件**：RK3588
- **算法**：traditional stereo + tiny neural refinement
- **原因**：纯神经主干不划算，hybrid 更现实

### 7.4 如果你追求“极低功耗 + 确定性实时”
选：
- **硬件**：DSP / FPGA / 专用 stereo 硬件管线
- **算法**：classical stereo
- **原因**：每瓦性能和时延稳定性最好

---

## 8. 我的主观结论

如果让我给一句非常直接的建议：

1. **不要把所有双目任务都神经化。** 很多板端项目最后会发现 classical 已经够用；
2. **Jetson 是神经双目最现实的平台，尤其 Orin NX。** 因为它给的是“可工作的工程生态”，不只是 TOPS；
3. **RK3588 更适合 hybrid，而不是纯重型 neural stereo。** 它在成本和国产化上好，但不是 stereo NPU 天堂；
4. **决定能不能落地的关键不是论文精度，而是 cost volume、算子支持、内存带宽、热设计。**
5. **先做 classical baseline，再做 lightweight neural upgrade，是板端双目最靠谱的路线。**

---

## 9. 供主方案讨论时直接复用的表格

| 维度 | Classical stereo | Lightweight neural stereo | Heavy neural stereo |
|---|---|---|---|
| 典型代表 | SGBM / SGM / Census | AnyNet / HITNet / 小型 AANet / 轻量迭代式 | PSMNet / GC-Net / 大型 RAFT-Stereo / CREStereo |
| 算力需求 | 中低 | 中 | 高到很高 |
| 带宽/显存压力 | 中 | 中高 | 很高 |
| 功耗 | 低到中 | 中 | 中高到高 |
| 时延确定性 | 高 | 中 | 低到中 |
| 量化友好度 | 高 | 中 | 低 |
| 算子支持风险 | 低 | 中 | 高 |
| 适合硬件 | CPU / DSP / FPGA / GPU | Jetson GPU / 部分 NPU | 高端 GPU / 部分 FPGA / 强边缘 GPU |
| RK3588 适配性 | 高 | 中（需重设计） | 低 |
| Jetson Orin 适配性 | 高 | 高 | 中 |
| 量产风险 | 低 | 中 | 高 |

---

## 10. 参考依据说明

本报告基于以下几类信息综合形成：
- NVIDIA Jetson Nano / Xavier NX / Orin 系列公开规格页中的典型算力、内存、功耗档信息；
- RK3588 系列公开规格与开发板资料中的典型 NPU/GPU/CPU能力信息；
- 双目经典算法与主流神经 stereo 网络（如 GC-Net、PSMNet、AnyNet、HITNet、RAFT-Stereo、CREStereo 等）的结构共性与部署约束；
- 板端推理的一般工程经验：TensorRT / RKNN / ONNX / OpenVINO 路线下的算子支持、量化、带宽与热设计问题。

由于当前研究会话内的公开网页抓取能力受限（Brave API key 未配置，部分网页抓取被网络策略拦截），本报告对具体板卡数值采用公开常见规格口径，并重点强调**部署规律与工程判断**而非追求逐项参数表复刻。
