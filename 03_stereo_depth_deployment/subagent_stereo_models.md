# 可板端部署的双目深度估计方案调研

> 目标：面向“板端/边缘端/NPU+GPU SoC”场景，梳理双目深度估计（stereo matching）在**算法路线**与**部署可行性**上的选择。重点区分 classical stereo 与轻量学习型 stereo，并给出板端推荐梯队。
>
> 说明：当前环境下外部网页抓取与搜索工具存在访问限制，下面内容基于公开论文/开源项目的通行认知、社区部署经验与工程常识进行归纳，个别参数量/延迟为“论文或社区常见量级”，部署前仍建议按目标芯片（Jetson Orin/Xavier、RK3588、Ascend、地瓜/J5/J6、TI TDA4、高通 QRB/QCS 等）做一次实测基线。

---

## 1. 先给结论：板端双目方案怎么选

如果目标是**稳定可上线、资源可控、实时性优先**，我的判断很明确：

1. **超轻量实时优先**：优先 classical stereo（SGM/BM/改进 census + cost aggregation）或 **HitNet / AnyNet 一类轻量网络**。
2. **平衡型方案**：优先 **MobileStereoNet / 轻量 RAFT-Stereo / HitNet 高分辨率版本**。
3. **高精度但较重**：可考虑 **RAFT-Stereo、IGEV-Stereo、PSMNet/GwcNet 的轻量化裁剪版**，但通常更适合“带较强 GPU/NPU 的边缘盒子”，不太适合极低功耗板端。

如果是**工业/机器人/无人系统**，且对纹理稀疏、弱光、重复纹理、远距离鲁棒性有要求，通常最终会落在：

- **传统 SGM 系**作为保底和基线
- **HitNet / MobileStereoNet**作为主力候选
- **RAFT-Stereo/IGEV**只在资源允许、精度收益明确时采用

一句话：

- **能不用重型 3D cost volume，就尽量不用**；
- **板端最怕的是体积成本（H×W×D）爆炸，而不是单纯参数量大**；
- stereo 的部署瓶颈往往不是“模型参数”，而是**代价体/迭代更新带来的显存、带宽和延迟波动**。

---

## 2. 双目深度估计的板端约束

在板端部署时，通常要同时看 5 件事：

### 2.1 算力类型是否匹配
- **CPU/DSP 友好**：BM、SGM、ELAS、Census+SGM、局部匹配+后处理
- **GPU/NPU 友好**：轻量 CNN stereo、2D encoder + correlation / 小型 cost volume / iterative refinement
- **NPU 受限点**：动态 shape、循环迭代、grid_sample、soft argmin、大规模 3D conv、可变长 cost volume 往往不友好

### 2.2 显存/内存占用
stereo 和普通单目网络不同，最容易炸的是：

- disparity 搜索范围 D
- 输入分辨率 H×W
- 3D cost volume
- 多尺度堆叠
- 多次 iterative update

一个经验判断：

- **classical stereo**：参数几乎没有，内存主要是图像缓存和代价缓存，整体可控
- **2D 轻量网络**：参数小，但若构建 full cost volume，内存仍可能偏大
- **3D hourglass / stacked volume**：板端高风险
- **iterative refinement**：参数未必大，但延迟与循环次数强相关

### 2.3 实时性定义
板端常见档位：

- **30 FPS+**：无人车避障、近距感知、机械臂高速引导
- **10~20 FPS**：导航、AGV、巡检
- **1~10 FPS**：高精地图更新、离线增强感知

### 2.4 分辨率适配
双目很依赖分辨率：

- 低分辨率：快，但远距离精度下降、边界糊
- 高分辨率：有利于小目标/细边界，但 D 和显存急剧上升

工程上常见折中：

- 输入做 **640×360 / 640×480 / 768×384 / 960×540**
- 视差搜索在低尺度完成，再做 refinement
- ROI stereo（只对感兴趣区域高精）

### 2.5 部署框架兼容性
优先级一般是：

1. **TensorRT**（NVIDIA 板端最优）
2. **ONNX Runtime**（通用）
3. **ncnn / MNN / TNN**（ARM/移动端友好）
4. **TVM**（可做专项优化）
5. 厂商工具链（RKNN、SNPE、Ascend ATC、Horizon BPU toolchain）

但 stereo 模型常见坑点：

- 相关性/correlation 算子没有现成 kernel
- 3D conv 在部分 NPU 上性能很差
- soft-argmin、grid_sample、循环 refine 不一定好转
- 自定义 CUDA op 很难跨平台迁移

所以**“论文能跑”≠“板端能落”**。

---

## 3. 路线一：Classical Stereo（传统双目）

这类方法的共同特点：

- 不依赖大模型参数
- 可解释性好
- 对 CPU/DSP/FPGA 友好
- 延迟稳定，易于硬实时
- 在低纹理、反光、重复纹理、遮挡区域通常需要精心调参

### 3.1 BM（Block Matching）

#### 核心思路
以局部窗口做匹配代价计算（SAD/SSD/NCC/census 等），在视差范围内搜索最优匹配。

#### 优点
- 最容易部署
- CPU 上即可跑
- OpenCV StereoBM 可直接用
- 内存占用低，适合低端板子快速验证

#### 缺点
- 对纹理弱、光照变化、遮挡不鲁棒
- 边界易糊
- 对窗口大小敏感
- 精度通常不适合高要求场景

#### 板端可部署性
- **参数量**：无
- **内存**：低
- **延迟**：低，CPU 可实时（取决于分辨率/视差范围）
- **分辨率适配**：较好，但高分辨率下速度会明显下降
- **部署经验**：OpenCV / embedded C++ 极成熟

#### 适用结论
仅建议作为：
- 最低成本 baseline
- 资源极其受限的保底方案
- 教学/原型验证

---

### 3.2 SGM / Semi-Global Matching

#### 核心思路
先构建匹配代价（常见 census/BT/AD-census），再沿多个方向做半全局路径聚合，平衡局部与全局一致性。

#### 优点
- 工业界使用非常广泛
- 精度/速度/鲁棒性综合表现强于 BM
- 很适合 FPGA / DSP / CPU SIMD / CUDA 优化
- 对板端实时部署非常成熟

#### 缺点
- 参数调节比较工程化
- 对强反光、无纹理、重复纹理仍有困难
- 亚像素、孔洞填充、左右一致性检查要配合后处理

#### 板端可部署性
- **参数量**：无
- **内存**：中等，主要看 cost 存储与路径聚合实现
- **延迟**：中低；优化后可实时
- **分辨率适配**：好，可根据视差范围线性扩展，但 D 大时成本上升明显
- **部署经验**：极成熟；OpenCV StereoSGBM、CUDA SGM、FPGA SGM、车载 SoC 大量落地
- **框架依赖**：无需深度学习框架

#### 工程建议
若大哥的目标是**板端第一版可用系统**，SGM 一定要做基线：

- census/AD-census 作为匹配代价
- 左右一致性检查
- speckle removal
- WLS / edge-aware filter
- 亚像素拟合

#### 适用结论
- **板端首选 baseline**
- 若算力弱、实时性要求高、可接受“传统算法上限”，SGM 往往是最稳的选择

---

### 3.3 ELAS（Efficient Large-scale Stereo Matching）

#### 核心思路
通过稀疏支持点建立先验，再进行稠密匹配，兼顾速度与一定的结构先验。

#### 优点
- 比纯局部匹配更聪明
- 历史上在 CPU 实时 stereo 中较有代表性
- 对一些结构化场景表现不错

#### 缺点
- 现在工程生态不如 SGM 主流
- 复杂场景鲁棒性通常不如现代学习法
- 实现和维护成本相比 OpenCV SGM 略高

#### 板端可部署性
- **参数量**：无
- **内存**：低到中等
- **延迟**：较快
- **部署经验**：有一定历史经验，但新项目采用率不如 SGM

#### 适用结论
可以作为 classical 备选，但**优先级通常低于 SGM**。

---

### 3.4 传统代价聚合路线（Census / AD-Census / Cross-based / Guided / WLS）

这不是单一模型，而是一类工程组合拳：

- 代价：AD、census、BT、NCC
- 聚合：box / cross-based / semi-global / guided filter
- 优化：左右一致性、孔洞填充、亚像素拟合、边缘保持滤波

#### 优点
- 高度可裁剪
- 算法链条可解释
- 可按芯片特点做 SIMD / DSP / FPGA 定制
- 很适合“算力固定、输入固定、场景相对固定”的产品

#### 缺点
- 需要较多人工调参
- 泛化不如学习法
- 面对极端场景上限有限

#### 适用结论
如果项目允许较强工程调优，**传统代价聚合 + SGM/局部优化** 仍然是很多板端项目最实际的解。

---

## 4. 路线二：轻量学习型 Stereo

这一类的优势在于：

- 在弱纹理/重复纹理/大视差场景下通常优于传统法
- 可借助数据驱动获得更强先验
- 更容易与后续感知网络融合

但部署时必须重点审查：

- 是否使用 heavy 3D cost volume
- 是否有大量迭代更新
- 是否依赖自定义算子
- 是否已有 ONNX/TensorRT 落地案例

下面按“板端友好度”逐个看。

---

### 4.1 AnyNet

#### 方法特点
AnyNet 属于较早一批强调**anytime / coarse-to-fine** 的双目网络：

- 先在低分辨率估计粗视差
- 再逐级 refinement
- 试图兼顾速度和精度

#### 优点
- 比早期重型 3D stereo（如 GC-Net/PSMNet 初代）轻很多
- coarse-to-fine 思路对板端友好
- 可根据时间预算提前停止，理论上适合“算力弹性”场景

#### 缺点
- 年代较早，和更新方法相比精度优势已不明显
- 生态热度一般，现成高质量部署案例不如 HitNet/RAFT-Stereo 多
- 若完整保留多阶段结构，部署图也不算特别简单

#### 部署可行性判断
- **参数量**：通常在千万级以下或接近轻中量级，明显小于早期大体积 stereo
- **显存/内存**：中等，主要取决于多阶段特征与 cost volume 规模
- **延迟**：中等偏低，适合中低分辨率实时或准实时
- **分辨率适配**：较好，适合多尺度输入
- **部署经验**：有 ONNX/TensorRT 可迁移潜力，但社区现成“板端大规模落地经验”不算最多

#### 结论
**AnyNet 是“轻量学习型 stereo 的早期代表”，可以作为板端候选，但今天通常不是第一优先。** 如果你要一个结构相对易理解、可 coarse-to-fine 裁剪的网络，它仍有价值。

---

### 4.2 MobileStereoNet

#### 方法特点
顾名思义，目标就是 mobile/edge 友好：

- backbone 更轻
- 尽量压缩 cost volume 和 3D aggregation 成本
- 面向移动端/嵌入式场景做设计

#### 优点
- 在“轻量 + 精度”之间相对均衡
- 比 PSM/Gwc 一类重模型更接近可落地
- 常被拿来与 HitNet、AnyNet 一起作为边缘部署候选

#### 缺点
- 如果内部仍保留较重的 volume 操作，在高分辨率/大视差范围下依然可能吃紧
- 真正板端表现高度依赖导出路径和 kernel 优化质量

#### 部署可行性判断
- **参数量**：通常为轻量到中轻量级（常见认知在数百万量级）
- **显存/内存**：优于传统 3D hourglass stereo，但仍需警惕 volume 内存
- **延迟**：中等，合理裁剪后有希望跑到板端实时/准实时
- **分辨率适配**：中等偏好，640×360 / 640×480 一般更现实
- **部署经验**：常见 ONNX/TensorRT 尝试；若算子标准化较好，迁移性优于含大量 custom op 的模型

#### 结论
**MobileStereoNet 属于板端“平衡型主力候选”。** 如果大哥需要比 HitNet 更追求一些精度，同时又不想上 RAFT/IGEV 这种更重路线，它很值得重点验证。

---

### 4.3 HitNet

#### 方法特点
HitNet（Hierarchical Iterative Tile Refinement）非常值得单拎出来，因为它在“速度/效果/部署友好”之间做得比较漂亮：

- 采用分层 tile/patch 思路
- 避免重型全尺寸 3D volume 堆叠
- 强调高效 refinement
- 在移动端/实时场景较受关注

#### 优点
- 通常被认为是**最贴近板端量产**的一类学习型 stereo
- 参数和算力相对友好
- 对 TensorRT/ONNX 迁移比许多大体积 stereo 更现实
- 在中等分辨率下实时性较有希望

#### 缺点
- 绝对精度上未必打得过最新重型方法
- 某些极端场景下，细节恢复和远距离精度仍受限于轻量设计
- 若要上 NPU，仍需看是否包含难支持算子

#### 部署可行性判断
- **参数量**：轻量级（社区认知通常在数百万级以内/附近）
- **显存/内存**：明显优于大 cost volume 网络
- **延迟**：低到中低，是学习型 stereo 中较容易做实时的路线
- **分辨率适配**：较好，适合 640×480、768×384、960×540 这类板端常见输入
- **部署经验**：已有较多 ONNX / TensorRT / 移动端侧讨论与实践，属于“工程可迁移性比较好”的代表

#### 结论
**如果只让我挑一个“学习型 stereo 板端优先验证对象”，HitNet 大概率排第一梯队。**

---

### 4.4 RAFT-Stereo（及轻量化变体）

#### 方法特点
RAFT-Stereo 延续 RAFT 系列的 iterative update 思路：

- 构建相关性/匹配表示
- 通过 recurrent update block 多次迭代优化视差
- 精度普遍较强，泛化也不错

#### 优点
- 精度很强
- 迭代 refinement 对复杂区域恢复能力较好
- 学术和工程关注度高

#### 缺点
- **对板端最不友好的点不一定是参数，而是迭代与相关性体**
- 迭代次数越多，延迟越难控
- 相关性查表、grid_sample、循环结构等在 ONNX/TensorRT/NPU 上都可能变成麻烦点
- 标准版本通常不算“轻”

#### 轻量化方向
可以做：

- 减 backbone 通道数
- 降 feature 分辨率
- 减迭代次数（例如从 32/24 次降到 8/6/4 次）
- 限制 disparity range
- 小型相关性窗口替代 full correlation
- 蒸馏到更小 update block

#### 部署可行性判断
- **参数量**：中等；通常不属于超大模型，但也不算极轻
- **显存/内存**：中到偏高，和相关性体、特征尺度、迭代缓存有关
- **延迟**：中到偏高；轻量化后可降，但对实时性压力仍明显
- **分辨率适配**：中等，建议从较低分辨率做起
- **部署经验**：社区有 ONNX/TensorRT 尝试，但经常需要改图、定制算子或限制结构；ncnn/NPU 迁移难度偏高

#### 结论
**RAFT-Stereo 可以作为“高精度但较重”的候选，不推荐直接拿标准版上资源紧张板端。**
如果目标平台是 Jetson Orin 这类带较强 GPU 的边缘端，经过裁剪后有现实意义；如果是 ARM CPU / 一般 NPU 板卡，优先级明显下调。

---

### 4.5 IGEV-Stereo

#### 方法特点
IGEV 强调几何先验与高质量代价表达，属于近年精度很强的一类 stereo 方法。

#### 优点
- 精度优秀
- 对复杂场景和细节恢复往往有竞争力

#### 缺点
- 本质上仍偏“高性能重模型”路线
- cost volume / geometry encoding / iterative refinement 结构一般不轻
- 部署复杂度明显高于 HitNet/MobileStereoNet

#### 部署可行性判断
- **参数量**：中到偏大（视具体变体而定）
- **显存/内存**：偏高
- **延迟**：偏高
- **分辨率适配**：更适合中低分辨率或强 GPU
- **部署经验**：通常以 PyTorch 研究复现为主，真正成熟的板端落地经验相对少；TensorRT 可尝试，但工程工作量不小

#### 结论
**IGEV 更像“精度标杆/上限参考”，不太像第一批板端部署首选。**
除非目标平台有较强 CUDA/TensorRT 环境，并且精度收益足以覆盖部署成本，否则不建议优先。

---

### 4.6 PSMNet / GwcNet 及其轻量变体

#### 方法特点
这两类都是 stereo 深度学习中的经典代表：

- **PSMNet**：stacked hourglass + 3D cost volume
- **GwcNet**：group-wise correlation + 3D aggregation

它们在研究上非常重要，但工程部署时有明显共性问题。

#### 优点
- 历史地位高
- 精度基础强
- 许多后续工作以它们为参照

#### 缺点
- **3D cost volume 太重**
- hourglass / 3D conv 对板端极不友好
- 输入分辨率和 disparity 一上去，显存/内存暴涨
- 即便参数量不离谱，运行时资源仍重

#### 轻量化变体思路
- 缩 backbone 通道
- 降 D
- 单 hourglass 替代 stacked
- group correlation 替代 full concat
- 蒸馏/剪枝/INT8

但即便如此，很多变体仍然只是“相对变轻”，不是“真板端友好”。

#### 部署可行性判断
- **参数量**：中到偏大
- **显存/内存**：高风险，尤其是 3D volume
- **延迟**：偏高
- **分辨率适配**：较差，高分辨率成本很敏感
- **部署经验**：ONNX/TensorRT 可以导，但性能常不理想；ncnn/NPU 一般不友好

#### 结论
**PSMNet/GwcNet 更适合作为精度对照基线，而不是板端首选。**
如果要上板，必须做大幅裁剪，而且通常仍不如专为轻量部署设计的模型划算。

---

## 5. Classical vs Learning：板端视角对照

### 5.1 classical stereo 的优势
- 无训练数据依赖
- 无模型更新负担
- 硬实时友好
- CPU/DSP/FPGA 易落地
- 易解释、易调试
- 对框架依赖低

### 5.2 classical stereo 的不足
- 泛化能力弱
- 在低纹理、重复纹理、反光、半透明等场景精度受限
- 需要较多工程调参

### 5.3 轻量学习型 stereo 的优势
- 对复杂纹理和遮挡一般更鲁棒
- 可通过数据适配场景
- 有机会在同等分辨率下取得更高精度/更少 bad-pixel

### 5.4 轻量学习型 stereo 的不足
- 训练与部署链条更复杂
- 模型迁移受算子支持影响很大
- 显存/延迟不只由参数量决定
- 量化后精度掉点需重新验证

### 5.5 实战经验判断
若你是做产品，不是做论文：

- **先上 SGM baseline**，因为它便于判断数据质量、标定质量、镜头/基线配置是否合理
- 再用 **HitNet / MobileStereoNet** 去判断学习法带来的精度增益值不值
- 只有当收益非常明确时，再考虑 **RAFT-Stereo / IGEV**

---

## 6. 部署可行性表（板端视角）

> 说明：以下为工程判断分级，不是绝对值。参数量用“量级/相对关系”表示，重点是部署风险排序。

| 路线/模型 | 类型 | 参数量 | 运行内存/显存压力 | 实时性潜力 | 分辨率适配 | ONNX/TensorRT 友好度 | ncnn/TVM/NPU 迁移 | 板端结论 |
|---|---|---:|---|---|---|---|---|---|
| StereoBM | classical | 无 | 低 | 高 | 中 | 不需要 | 极友好 | 仅适合最低成本 baseline |
| SGM / SGBM | classical | 无 | 低~中 | 高 | 好 | 不需要 | 极友好 | 板端首选 baseline |
| ELAS | classical | 无 | 低~中 | 中高 | 中 | 不需要 | 友好 | 可选，但通常次于 SGM |
| AD-Census + 聚合 | classical | 无 | 低~中 | 中高 | 好 | 不需要 | 极友好 | 很实用的工程路线 |
| AnyNet | learning | 轻~中 | 中 | 中高 | 较好 | 中等偏好 | 中等 | 可作为早期轻量候选 |
| MobileStereoNet | learning | 轻 | 中 | 中 | 较好 | 较好 | 中等 | 平衡型主力候选 |
| HitNet | learning | 轻 | 低~中 | 高 | 好 | 较好 | 较好 | 学习型板端优先推荐 |
| RAFT-Stereo(轻量) | learning | 中 | 中~高 | 中 | 中 | 中等 | 偏难 | 适合强 GPU 边缘端 |
| IGEV-Stereo | learning | 中~偏大 | 高 | 低~中 | 中 | 中等偏难 | 难 | 高精度备选，不宜先上板 |
| PSMNet 轻量版 | learning | 中 | 高 | 低 | 一般 | 中等 | 难 | 不推荐优先 |
| GwcNet 轻量版 | learning | 中 | 高 | 低 | 一般 | 中等 | 难 | 不推荐优先 |

---

## 7. 关于“参数量、显存、延迟”的真实判断方法

很多人选板端模型时只看参数量，这是不够的。stereo 更应该看：

### 7.1 参数量不是第一指标
比如：
- 一个参数不算很多的 3D cost volume 网络，仍可能比一个参数更多的 2D 网络更难部署。
- iterative 模型参数不大，但循环 16~32 次后，延迟依旧可能很重。

### 7.2 真正关键的是这 4 个量
1. **Feature 分辨率**（H/4, H/8, H/16）
2. **Disparity range D**
3. **是否构建 4D/5D cost volume**
4. **是否有多轮 refinement / recurrent iterations**

### 7.3 板端估算经验
对于 640×480 输入：

- **classical stereo**：通常最容易做到稳定实时
- **HitNet / 轻量 MobileStereoNet**：更有机会接近实时
- **RAFT/IGEV/PSM/Gwc 系**：很可能需要降分辨率、减迭代、减 D，甚至牺牲精度

### 7.4 最该做的 benchmark
建议在目标板上固定四组基准：

- 640×360, D=96
- 640×480, D=128
- 768×384, D=128
- 960×540, D=192

每个模型记录：
- 端到端 FPS / latency(P50/P90)
- 峰值显存/内存
- 功耗
- bad-1 / bad-2 / EPE（选定验证集）
- 导出难点（是否需要改 op）

---

## 8. 各模型的部署经验判断

### 8.1 TensorRT
最适合：
- HitNet
- MobileStereoNet
- 轻量 AnyNet
- 经过图改造的 RAFT-Stereo

相对困难：
- IGEV
- PSM/Gwc 原型或重改版

原因：
- TensorRT 对标准 conv/activation/concat/fp16/INT8 很强
- 对 recurrence、grid_sample、自定义 correlation、动态 indexing 需要额外工作

### 8.2 ONNX
几乎是第一步必经之路，但要注意：
- PyTorch 能导出，不代表 ONNX Runtime 快
- ONNX 能跑，不代表 TensorRT 能无缝吃进去
- 复杂 stereo 模型经常在导出时卡在 custom op / shape 推理 / loop

### 8.3 ncnn / MNN / TNN
更适合：
- 结构规整的轻量 2D 网络
- 尽量少用 3D conv / grid_sample / recurrence

因此：
- **HitNet / 极轻 MobileStereoNet 变体**更有希望
- **RAFT / IGEV / PSM/Gwc** 通常更难

### 8.4 TVM
TVM 理论上可以做更多编译优化，但前提是：
- 图比较干净
- 算子可 lower
- 值得投入专项工程资源

对于产品团队而言，除非量产规模很大，否则一般先用 TensorRT 或厂商工具链。

---

## 9. 面向板端的推荐梯队

下面给出我认为更实用的推荐梯队。

### 9.1 超轻量梯队（优先实时、资源敏感）

#### 推荐对象
1. **SGM / SGBM + Census/AD-Census 改进版**
2. **HitNet**
3. **AnyNet（裁剪版）**
4. **StereoBM（仅保底）**

#### 适用平台
- ARM CPU + 少量 NEON
- DSP/FPGA/低功耗 SoC
- 入门级 GPU 板卡

#### 特点
- 延迟可控
- 工程可实现性高
- 容易做产品化

#### 我的建议
如果只能做一条最稳路线：
- **classical 选 SGM**
- **learning 选 HitNet**

---

### 9.2 平衡梯队（精度/速度折中）

#### 推荐对象
1. **MobileStereoNet**
2. **HitNet 高分辨率/增强版**
3. **AnyNet 完整多阶段版**
4. **RAFT-Stereo 轻量化版（少迭代）**

#### 适用平台
- Jetson Xavier / Orin Nano / Orin NX
- 较强 NPU + CPU/GPU 混合平台
- 中功耗边缘盒子

#### 特点
- 相比 classical，复杂场景通常更鲁棒
- 相比重模型，更接近实际部署

#### 我的建议
这档是**最值得投入验证资源**的区间：
- 优先试 **MobileStereoNet vs HitNet**
- 若精度还不够，再看轻量 RAFT

---

### 9.3 高精度但较重梯队（资源允许时）

#### 推荐对象
1. **RAFT-Stereo**
2. **IGEV-Stereo**
3. **轻量化 GwcNet / PSMNet**

#### 适用平台
- Jetson Orin / 桌面 GPU / 工控机 GPU
- 对功耗不极端敏感的边缘设备

#### 特点
- 有机会拿到更好的精度上限
- 部署复杂度明显更高
- 更适合作为高精版本或后端增强模块

#### 我的建议
如果不是明确追求精度上限，**不建议一开始就押重模型**。很多时候把双目标定、曝光同步、去畸变、后处理做好，收益比从 HitNet 升到 IGEV 更划算。

---

## 10. 选型建议：按平台来选

### 10.1 极低功耗 / 无强加速器
首选：
- SGM/SGBM
- AD-Census + 后处理

不建议：
- RAFT / IGEV / PSM / Gwc

### 10.2 中等 GPU/NPU 板端
首选：
- HitNet
- MobileStereoNet
- AnyNet 裁剪版

备选：
- 轻量 RAFT-Stereo

### 10.3 强 GPU 边缘端
首选组合：
- SGM 做兜底
- HitNet/MobileStereoNet 做实时主链
- RAFT/IGEV 做高精模式或离线增强

---

## 11. 实施建议：真正落地时怎么推进

### 11.1 建议的验证顺序
1. **先做 classical baseline**：SGM/SGBM
2. **再做最轻学习型**：HitNet
3. **然后做平衡型**：MobileStereoNet
4. **最后再看重型**：RAFT/IGEV

这样做的好处：
- 快速知道硬件上限
- 快速知道数据/标定问题是否是主矛盾
- 不会一开始就陷入重模型部署泥潭

### 11.2 先把输入链路做好
很多 stereo 项目问题并不在模型，而在：
- 双目标定误差
- 左右曝光不同步
- 镜头畸变处理不一致
- baseline/focal 不匹配任务距离
- 视差范围 D 设置不合理

这些问题不解决，换再强模型收益也有限。

### 11.3 量化建议
- **classical stereo**：无所谓量化
- **轻量 CNN stereo**：建议优先 FP16，再评估 INT8
- INT8 要特别检查：
  - 细边界
  - 远距离小视差
  - 遮挡区填充
  - 亚像素精度

### 11.4 如果板端 NPU 很挑算子
优先策略：
- 选结构规整、2D conv 为主的网络
- 少用 3D conv / loop / custom correlation
- 必要时把 stereo 任务拆为：
  - 低分辨率初始匹配
  - 传统后处理/边缘细化

---

## 12. 最终推荐（可直接抄到方案里）

### A. 如果目标是“尽快在板端跑起来”
**推荐：SGM/SGBM + Census + 后处理**

理由：
- 最稳
- 最容易实时
- 最容易解释和调参
- 最适合先建立基线

### B. 如果目标是“学习型方案里最像量产候选”
**推荐：HitNet**

理由：
- 轻量
- 实时性潜力好
- 部署链路相对友好
- 通常比重型 3D volume 网络更适合板端

### C. 如果目标是“精度和部署折中”
**推荐：MobileStereoNet**

理由：
- 比 HitNet 更偏平衡型
- 比 PSM/Gwc/IGEV 更可部署
- 很适合作为中档板端主力候选

### D. 如果目标是“资源够，追求更高精度”
**推荐：轻量化 RAFT-Stereo，必要时再看 IGEV**

理由：
- 精度潜力高
- 但部署成本也高
- 更适合强 GPU 边缘平台，而不是普适板端

### E. 不建议优先的路线
**PSMNet / GwcNet 标准版，或仅做轻微裁剪的变体**

理由：
- 3D cost volume 与 hourglass 对板端不友好
- 即便能导出，也常常不够快、不够省

---

## 13. 一页版结论

### classical stereo
- **SGM/SGBM**：板端最稳、最成熟、强烈建议先做 baseline
- **BM**：最轻但精度弱，只适合保底
- **ELAS / AD-Census 系**：实用，但总体优先级通常仍低于 SGM 主线

### learning stereo
- **HitNet**：学习型板端优先推荐，适合超轻量/实时
- **MobileStereoNet**：平衡型主力候选
- **AnyNet**：早期轻量代表，可裁剪，可作为候选但一般不再是第一优先
- **RAFT-Stereo 轻量版**：适合强 GPU 板端，精度高但部署复杂
- **IGEV / PSM / Gwc 轻量版**：更像高精参考或较重备选，不建议作为第一批板端方案

### 推荐梯队
- **超轻量**：SGM / HitNet / AnyNet(裁剪)
- **平衡**：MobileStereoNet / HitNet增强 / 轻量RAFT
- **高精但较重**：RAFT-Stereo / IGEV / 轻量Gwc/PSM

---

## 14. 后续若要继续深化，建议补的内容

如果后续要把这份调研继续做成“可立项的技术选型文档”，建议下一步补三块：

1. **按具体板卡做映射**
   - Jetson Orin Nano / NX
   - RK3588
   - Horizon J5/J6
   - Ascend 310/Atlas
   - Qualcomm QRB/QCS

2. **补具体模型导出链路**
   - PyTorch -> ONNX
   - ONNX -> TensorRT
   - ONNX -> RKNN / ncnn / TVM
   - 哪些 op 会卡住

3. **补公开 benchmark 表**
   - KITTI / SceneFlow / ETH3D
   - 参数量 / FPS / EPE / D1-all
   - 输入分辨率与 disparity 配置

---

## 15. 结论性建议

如果让我给一个非常务实的项目建议：

- **第一阶段**：SGM + 完整后处理，建立板端速度/精度基线
- **第二阶段**：HitNet 与 MobileStereoNet 做 A/B 测试
- **第三阶段**：若精度仍不足，再做轻量 RAFT-Stereo
- **第四阶段**：只有在强 GPU 板端且业务真吃精度时，再评估 IGEV / 更重变体

这条路线最符合“板端落地”的现实：**先活下来，再追求 SOTA。**
