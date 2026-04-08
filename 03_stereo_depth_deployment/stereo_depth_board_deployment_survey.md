# 可板端部署的双目深度估计方案调研

> 状态：final draft
> 更新时间：2026-03-31
> 作者：虾霸

## 1. 研究目标

本文聚焦四个问题：

1. 当前有哪些适合板端部署的双目深度估计方案？
2. 如何从 `classical stereo`、轻量学习式 stereo、芯片/工具链约束三层理解这些方案？
3. Jetson、RK3588 以及常见边缘 NPU/FPGA 路线下，部署风险分别是什么？
4. 面向机器人、移动平台、嵌入式视觉系统，应该如何选型？

本文目标不是复刻论文榜单，而是给出一份更接近工程落地的判断框架。除官方硬件规格与公开开源项目外，文中关于“是否适合板端”“是否适合量产”的结论，属于基于模型结构、工具链特性与端侧部署经验的工程判断，部署前仍需在目标板卡上做实测。

## 2. 结论先行

如果目标是“近期可上线、资源可控、实时性优先”，建议按下面顺序看方案：

1. **第一选择：classical stereo**
   - 典型方案：`StereoBM`、`StereoSGBM/SGM`、`AD-Census + SGM`
   - 优点：成熟、可解释、易调参、对 CPU/DSP/FPGA 友好、硬实时更稳
   - 适合：低功耗边缘、机器人避障、第一版量产

2. **第二选择：轻量学习式 stereo**
   - 第一梯队候选：`HITNet`、`MobileStereoNet`
   - 第二梯队候选：`AnyNet`、轻量化 `RAFT-Stereo`、轻量化 `CREStereo`
   - 适合：classical 在弱纹理、重复纹理、遮挡、边界质量上不够用，但平台仍需板端可部署

3. **谨慎选择：重型 cost volume / 重迭代模型**
   - 典型代表：标准版 `RAFT-Stereo`、`IGEV-Stereo`、`PSMNet/GwcNet` 系
   - 问题不只是参数量，更是 `cost volume`、`3D conv`、迭代更新、显存和带宽压力
   - 更适合：高端 GPU 边缘盒子、离线重建、teacher model、精度上限验证

一句话概括：

- **先做 `SGM/SGBM` baseline，再决定要不要上神经双目。**
- **学习式 stereo 里，优先验证 `HITNet` 和 `MobileStereoNet`。**
- **板端真正怕的不是“模型参数大”，而是“代价体和算子链路不友好”。**

## 3. 板端部署为什么难

双目深度估计的板端瓶颈，通常不是单一 TOPS 指标，而是下面五件事共同决定的：

### 3.1 输入分辨率和视差范围

双目深度的代价往往随着 `H × W × D` 上升，其中 `D` 是最大视差范围。  
分辨率、视差范围、目标距离三者一旦同时拉高，内存和带宽就会快速恶化。

### 3.2 显存/内存与带宽

对 stereo 来说，真正容易炸的是：

- `cost volume`
- 多尺度中间特征
- 多轮 iterative refinement
- `3D conv` 或大量相关性查表

这也是为什么一些“参数不算大”的 stereo 网络，实际部署起来仍然很重。

### 3.3 算子支持

板端工具链最常见的风险不是“不能导出 ONNX”，而是：

- 导出后部分子图落回 CPU
- `3D conv` 性能差
- `correlation / cost volume / grid_sample / soft argmin / loop` 支持不好
- NPU 只支持标准 `2D conv`，导致模型需要重写

### 3.4 实时性不是平均 FPS，而是稳定时延

机器人、避障、抓取、飞控更关心：

- `P95/P99 latency`
- 稳态温度和降频
- 峰值显存/内存
- 多任务并行时是否抖动

### 3.5 功耗与热设计

实验室里“能跑”不等于上板后“能长时间稳定跑”。  
很多神经 stereo 项目最后退回更轻的模型或 `classical stereo`，不是因为精度不够，而是因为热 throttling、显存峰值或系统抖动无法接受。

## 4. 技术路线分层

### 4.1 路线 A：Classical Stereo

代表方案：

- `StereoBM`
- `StereoSGBM / SGM`
- `AD-Census + SGM`
- `ELAS`

共性特点：

- 无需训练
- 参数几乎没有
- 对 CPU/DSP/FPGA 友好
- 延迟稳定，适合实时闭环
- 对标定质量、曝光同步和后处理依赖大

适用原则：

- **如果项目目标是“先稳定交付”，这条路线一定要先做。**

### 4.2 路线 B：轻量学习式 Stereo

代表方案：

- `HITNet`
- `MobileStereoNet`
- `AnyNet`
- 轻量化 `RAFT-Stereo`
- 轻量化 `CREStereo`

共性特点：

- 在弱纹理、遮挡、重复纹理场景通常优于传统法
- 更依赖 GPU/NPU 与工具链
- 部署难度主要取决于模型结构是否规整、是否标准 `2D CNN` 友好

适用原则：

- **当 `classical stereo` 已经能用，但在 hardest 20% 场景仍有明显短板时，再引入这类模型最划算。**

### 4.3 路线 C：重型高精度 Stereo

代表方案：

- 标准版 `RAFT-Stereo`
- `IGEV-Stereo`
- `PSMNet / GwcNet` 及类似 `3D cost volume` 模型

共性特点：

- 精度上限高
- 部署复杂度显著上升
- 更适合强 GPU、离线重建、teacher model 或高端 SKU

适用原则：

- **不建议作为大多数机器人/边缘产品的首个量产方案。**

## 5. 候选方案清单

下表偏“板端可落地性”，不是论文精度榜。

| 方案 | 类型 | 开源实现 | 板端友好度 | 典型适用平台 | 主要风险 | 建议 |
|---|---|---|---|---|---|---|
| `StereoBM` | classical | OpenCV | 高 | ARM CPU / 低端 SoC | 精度有限，弱纹理差 | 仅做最低成本 baseline |
| `StereoSGBM / SGM` | classical | OpenCV / 工程实现广泛 | 很高 | CPU / DSP / FPGA / Jetson / RK3588 | 需要调参和后处理 | **首选 baseline** |
| `AD-Census + SGM` | classical | 多工程实现 | 很高 | CPU / DSP / FPGA | 工程实现细节较多 | **非常实用** |
| `ELAS` | classical | 开源实现存在 | 中高 | CPU / 中低功耗端 | 新项目生态不如 SGM 主流 | 可选备选 |
| `HITNet` | learning | Google Research | 高 | Jetson / 部分 GPU 板端 | 仍需看导出链路与算子 | **学习式第一梯队** |
| `MobileStereoNet` | learning | 官方 GitHub | 中高 | Jetson / 中高端边缘 GPU | 依旧需要关注 volume 内存 | **平衡型候选** |
| `AnyNet` | learning | 官方 GitHub | 中 | Jetson / 中等算力端 | 方案较早，现成部署案例少于前两者 | 可作为早期轻量候选 |
| 轻量化 `CREStereo` | learning | 官方 GitHub | 中 | Orin Nano / Orin NX / 高算力端 | 相关性和迭代结构仍偏重 | 适合中高端增强 |
| 轻量化 `RAFT-Stereo` | learning | 官方 GitHub | 中偏低 | Orin NX / AGX Orin | 迭代更新和相关性结构不够友好 | 作为高精备选 |
| `IGEV-Stereo` | learning | 研究实现为主 | 低 | 强 GPU / 工控机 | 显存、延迟、部署链路重 | 不建议优先 |
| `PSMNet / GwcNet` | learning | 官方/社区实现较多 | 低 | 桌面 GPU / 离线 | `3D cost volume` 太重 | 仅做精度对照 |

### 5.1 Classical 路线具体判断

#### `StereoBM`

- 优点：最简单、最容易 bring-up、CPU 即可跑
- 缺点：纹理差、边界差、抗干扰弱
- 适用：极低成本保底、教学验证、快速原型

#### `StereoSGBM / SGM`

- 优点：工业界成熟、性能/精度/可解释性平衡最好
- 缺点：低纹理、反光、重复纹理仍依赖后处理
- 推荐工程组合：
  - `Census/AD-Census`
  - 左右一致性检查
  - `speckle removal`
  - `WLS / edge-aware` 滤波
  - 亚像素拟合
- 结论：**大多数板端项目都应该先做这一版。**

### 5.2 轻量学习式路线具体判断

#### `HITNet`

- 优势：实时性潜力好、结构相对更适合端侧
- 风险：仍需确认目标工具链是否能顺畅支持
- 结论：**如果只选一个学习式 stereo 做第一批验证，优先级很高。**

#### `MobileStereoNet`

- 优势：轻量与精度之间比较均衡
- 风险：高分辨率/大视差时仍要关注 volume 成本
- 结论：**适合做“效果比 classical 更好，但又不想太重”的主力候选。**

#### `AnyNet`

- 优势：`coarse-to-fine` 思路对板端友好
- 风险：方案较早，今天通常不是第一优先
- 结论：可以做对照，但优先级通常低于 `HITNet` 和 `MobileStereoNet`

#### 轻量化 `CREStereo`

- 优势：定位偏 practical stereo，效果通常比超轻方案更强
- 风险：默认结构仍不算轻，工具链适配要提前验证
- 结论：适合 Orin 级平台做增强，不建议裸上低功耗端

#### 轻量化 `RAFT-Stereo`

- 优势：精度强、上限高
- 风险：迭代次数、相关性表示、导出和算子兼容性都更麻烦
- 结论：更适合强 GPU 平台，不适合当绝大多数板端项目的第一选择

## 6. 按硬件平台看部署现实

以下板卡判断优先依据厂商官方规格；具体是否适合 stereo，属于结合工具链和模型结构做的工程推断。

### 6.1 NVIDIA Jetson 路线

#### `Jetson Nano`

官方公开信息显示：

- `472 GFLOPS`
- `4GB LPDDR4`
- `25.6 GB/s` 内存带宽
- `5W-10W` 功耗档

工程判断：

- **传统 stereo：能做，而且很适合**
- **轻量学习式 stereo：可尝试，但要明显降分辨率和降预期**
- **重型 stereo：基本不建议**

#### `Jetson Xavier NX`

官方公开信息显示：

- `21 TOPS`
- `8GB / 16GB LPDDR4x`
- `59.7 GB/s` 带宽
- `10W / 15W / 20W` 功耗档

工程判断：

- `classical stereo` 已经比较从容
- `HITNet / MobileStereoNet` 一类开始进入可用区间
- 对较重模型，仍需严格控制输入分辨率和迭代次数

#### `Jetson Orin Nano / Orin NX / AGX Orin`

官方公开信息显示：

- `Orin Nano`：最高 `67 TOPS`，`7W-25W`
- `Orin NX`：最高 `157 TOPS`，`10W-40W`
- `AGX Orin`：最高 `275 TOPS`，`15W-60W`

工程判断：

- `Orin Nano`：是认真尝试神经双目的入门门槛
- `Orin NX`：是机器人与边缘视觉里最均衡的点
- `AGX Orin`：适合多相机、多模型并行、较高分辨率 stereo

总体结论：

- **Jetson 目前仍是神经双目最现实的板端生态。**
- 原因不只是 TOPS，更是 `CUDA + TensorRT + JetPack` 生态成熟。

### 6.2 RK3588 路线

官方公开信息显示：

- `4x Cortex-A76 + 4x Cortex-A55`
- `Mali-G610 MC4`
- `6 TOPS NPU`
- 支持 `int4/int8/int16/FP16/BF16/TF32`

工程判断：

- **非常适合传统 stereo**
- **适合 traditional + tiny neural refinement hybrid**
- **不适合直接当成 Jetson 替代品来跑重型 neural stereo**

原因不在于它没有 NPU，而在于 stereo 常见的 `cost volume`、相关性、动态结构、回归头，不一定适合 NPU 工具链。

总体结论：

- **如果追求低成本、国产化、经典 stereo 足够用，RK3588 很有竞争力。**
- **如果核心卖点是“高质量神经双目”，Jetson 更省心。**

### 6.3 其它边缘 NPU / DSP / FPGA

#### 常见 NPU SoC

如高通、地平线、爱芯、算能、寒武纪等路线，最关键不是宣传 TOPS，而是先问四个问题：

1. 支不支持 `3D conv`？
2. 支不支持 `correlation / cost volume`？
3. 支不支持 `grid_sample / warp / loop`？
4. 不支持时，fallback 到 CPU/GPU 的代价多大？

如果这四个问题里有两个以上答案不理想，基本就更适合：

- `classical stereo`
- 或 `classical + lightweight neural refinement`

#### DSP / FPGA

这类平台非常适合：

- rectification
- census transform
- `SGM`
- 左右一致性检查
- 规则后处理

对于追求低功耗、确定性时延、长生命周期的产品，`DSP/FPGA + classical stereo` 往往比“继续堆大网络”更像正确路线。

## 7. 按场景给选型建议

### 7.1 低功耗边缘设备

目标特征：

- 算法功耗预算通常在 `5W-10W` 以内
- 更多关注近距障碍感知，不追求漂亮稠密图

建议：

- 首选：`StereoSGBM / SGM`
- 次选：现成双目模组
- 升级版：`classical + 小模型做 confidence / refine`

不建议：

- 重型 neural stereo
- 高分辨率、全图稠密、同时追求高 FPS 和高质量

### 7.2 移动机器人实时避障

关注重点通常不是 `EPE`，而是：

- 近距离障碍漏检率
- 细杆、台阶、玻璃门附近稳定性
- free-space 输出是否可靠

建议：

- **量产主线：`classical stereo`**
- **增强路线：`classical + HITNet/MobileStereoNet refinement`**
- 如果一定要全学习式主链路，至少保留 fallback 和 confidence 管控

### 7.3 室内仓储 / 服务机器人

#### 仓储 AMR

- 场景相对结构化
- 对稳定性和解释性要求更高
- 建议：`SGBM/SGM + 规则层 + 补盲传感器`

#### 服务机器人

- 细小障碍、家具脚、低纹理地面更麻烦
- 建议：`classical` 为底，叠加轻量 learning 增强边界和弱纹理区域

### 7.4 高精测量 / 重建

优先级应当是：

1. 双目几何设计
2. 标定质量
3. 光照与镜头
4. 再考虑模型升级

建议：

- 首选：高质量硬件 + `SGM/AD-Census + 亚像素拟合`
- 增强：`classical + learning refinement`
- 强 GPU 平台可再考虑 `CREStereo / RAFT-Stereo`

### 7.5 远距户外

核心问题首先是硬件几何，而不是模型名字：

- 远距视差本来就小
- 更依赖基线长度、分辨率、同步和标定稳定性

建议：

- 先用更长基线、更稳结构件、更高分辨率把信号做出来
- 算法上以 `classical stereo + confidence` 为主
- 需要的话再上 learning refinement
- 不要幻想用短基线小模组加大模型解决远距问题

## 8. 工程落地建议

### 8.1 推荐验证顺序

1. 先做 `StereoSGBM / SGM` baseline
2. 再验证 `HITNet`
3. 再验证 `MobileStereoNet`
4. 最后再碰 `CREStereo / RAFT-Stereo`

这样做的价值是：

- 先知道硬件与标定的真实上限
- 避免一开始就陷入重模型部署泥潭
- 更容易区分“场景问题”还是“模型问题”

### 8.2 推荐 benchmark 维度

建议固定四组端侧 benchmark：

- `640×360, D=96`
- `640×480, D=128`
- `768×384, D=128`
- `960×540, D=192`

每组至少记录：

- `P50 / P90 / P95 latency`
- 峰值显存/内存
- 稳态功耗和温升
- 难场景失效率
- 导出链路是否需要改图或自定义 op

### 8.3 不要只看论文指标

产品侧更关键的指标往往是：

- 漏检率
- 最小可检障碍尺寸
- free-space 连续稳定性
- 低置信区域是否被正确保守处理
- 温升后性能是否退化

### 8.4 建议保留三级方案

- A 案：`classical` 生产保底
- B 案：轻量 neural 增强
- C 案：高性能高精度方案

这比单点押注某一个重模型更稳。

## 9. 推荐落地路线

### 9.1 最稳量产款

`双目相机 + StereoSGBM/SGM + 规则后处理`

适合：

- 首代量产
- 低功耗边缘
- 仓储 AMR
- 巡检避障

### 9.2 产品增强款

`classical baseline + lightweight refinement/confidence`

适合：

- 室内复杂场景
- 服务机器人
- classical 已能工作，但想补 hardest 20% 场景

### 9.3 Jetson 高端款

`Jetson Orin NX/AGX + HITNet / MobileStereoNet / 轻量 CREStereo + classical fallback`

适合：

- 中高端机器人
- 室内外混合导航
- 有 GPU 预算且能维护部署链路的产品

### 9.4 高精/研究款

`长基线双目 + 高质量标定 + RAFT-Stereo / CREStereo`

适合：

- 离线重建
- teacher model
- 高端 SKU 或研发验证

## 10. 最终建议

如果只保留一句最实用的建议：

**先把双目标定、同步、`SGM/SGBM` 和 confidence 管理做好，再决定是否引入神经 stereo。**

如果只保留一份优先级列表：

1. `classical stereo` 一定先做
2. 学习式优先验证 `HITNet`
3. 第二候选看 `MobileStereoNet`
4. `CREStereo / RAFT-Stereo` 留给更强平台或上限验证

## 11. 参考链接

### 官方硬件规格

- NVIDIA Jetson Nano  
  https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano/product-development/
- NVIDIA Jetson Xavier Series  
  https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-xavier-series/
- NVIDIA Jetson Orin Family  
  https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/
- Rockchip RK3588 官方规格页  
  https://www.rock-chips.com/a/en/products/RK35_Series/2022/0926/1660.html

### 经典 stereo / 工具链

- OpenCV `StereoBM`  
  https://docs.opencv.org/4.x/javadoc/org/opencv/calib3d/StereoBM.html
- OpenCV `StereoSGBM`  
  https://docs.opencv.org/4.x/javadoc/org/opencv/calib3d/StereoSGBM.html
- NVIDIA TensorRT Documentation  
  https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html

### 学习式 stereo 官方实现

- `HITNet`  
  https://github.com/google-research/google-research/tree/master/hitnet
- `MobileStereoNet`  
  https://github.com/cogsys-tuebingen/mobilestereonet
- `AnyNet`  
  https://github.com/mileyan/AnyNet
- `RAFT-Stereo`  
  https://github.com/princeton-vl/RAFT-Stereo
- `CREStereo`  
  https://github.com/megvii-research/CREStereo

## 12. 后续可继续补的内容

如果后面还要把这份文档继续做实，我建议补三块：

1. **按板卡细化实测预算表**
   - `RK3588 / Xavier NX / Orin Nano / Orin NX / AGX Orin`
   - 每个平台给出 `640×480 / 768×384 / 960×540` 的延迟、功耗、峰值内存预算

2. **按双目基线长度补距离建议**
   - `6cm / 9cm / 12cm / 20cm`
   - 对应近距盲区、推荐工作距离、远距可用性

3. **补一份机器人评测 checklist**
   - 玻璃门
   - 黑地毯
   - 反光地砖
   - 细杆/桌脚
   - 台阶边缘
   - 运动模糊与曝光不同步
