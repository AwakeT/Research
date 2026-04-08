# 板端可部署双目深度估计选型建议（产品 / 机器人应用视角）

> 面向“今天要上机器人、要控风险、要考虑量产”的工程选型建议，而不是纯论文榜单。
>
> 结论先行：**如果目标是近期量产/工程可控，优先把 classical stereo（BM/SGBM/SGM 系）作为主干**，再按算力余量叠加 **轻量学习式 refinement / confidence / hole-filling**。纯端到端深度 stereo 网络里，**MobileStereoNet / CREStereo-lite 一类可以作为可试产候选**；**RAFT-Stereo、重型 cost-volume 网络、需要大规模 domain adaptation 的方案更偏研究 demo 或高算力 SKU**。

---

## 1. 先给判断框架：板端双目方案到底怎么分层

从机器人产品视角，双目深度方案可以分成 4 层：

### A. 现成模组 / 专用深度相机方案
- 代表：Intel RealSense D4xx、Luxonis OAK-D、部分 FPGA/ASIC 立体模组
- 优点：
  - 软硬件一体，Bring-up 快
  - 标定、同步、ISP、驱动、点云接口成熟
  - 对产品团队最省心
- 缺点：
  - 成本、供应链、BOM 自由度受限
  - 算法可控性有限
  - 遇到特定场景（反光、弱纹理、远距）时可调空间不如自研 pipeline
- 适用：**要尽快出货 / 先做机器人产品 MVP / 小中批量工程化**

### B. 经典立体匹配（BM / SGBM / Census + SGM）
- 代表：OpenCV StereoSGBM、ADAS 常见 SGM pipeline、FPGA/SoC ISP 里的 stereo block
- 优点：
  - 可解释、可调、实时性好
  - 对 ARM / DSP / FPGA / ASIC 友好
  - 量产经验最丰富
- 缺点：
  - 弱纹理、重复纹理、反光、透明体、极低照度容易崩
  - 精度上限受限，后处理很关键
- 适用：**绝大多数“今天就能上板量产”的机器人深度需求**

### C. 轻量学习 stereo
- 代表：MobileStereoNet、CREStereo 的轻量/裁剪版、蒸馏版小模型
- 优点：
  - 在复杂纹理、边界、遮挡区域通常优于 classical
  - 可以在 Jetson / 较强 NPU 上获得不错效果
- 缺点：
  - 端侧部署链路更复杂（ONNX / TensorRT / NPU 转换 / 量化）
  - 域外泛化风险大，常需要数据闭环
  - 时延、功耗、内存波动比 classical 大
- 适用：**有一定算力预算，且 classical 在关键场景确实不够用**

### D. 重型深度 stereo / 研究型 SOTA
- 代表：RAFT-Stereo、大 cost volume Transformer / recurrent 方案、重训练依赖方案
- 优点：上限高，论文指标漂亮
- 缺点：
  - 板端实时性、功耗、内存、稳定性压力大
  - 模型/框架适配成本高
  - 线上问题定位比 classical 难很多
- 适用：**高端 SKU 验证、离线重建、研究项目，不建议直接作为主量产方案起步**

---

## 2. 对“今天能不能上板量产”先下结论

### 2.1 今天就能上板量产 / 工程可控
以下方案可认为是**工程上成熟**：

1. **双目相机模组 + 固化 stereo 深度芯片/固件**
   - 如 RealSense / OAK-D / 供应商立体模组
   - 适合快速验证与产品首版

2. **自研 classical stereo pipeline**
   - 推荐形态：
     - rectification
     - Census / SAD / SGBM / SGM
     - LR consistency check
     - speckle removal
     - confidence 过滤
     - temporal / spatial smoothing
     - occupancy / obstacle extraction
   - 这是机器人避障、仓储、AMR、清洁机器人里最稳的路线之一

3. **classical baseline + 轻量学习后处理**
   - 比如：
     - SGBM/SGM 出基础 disparity
     - 小 CNN 做 refinement / confidence / edge-aware completion
   - 这是我最推荐的**中庸但实用**路线：主链路稳定、学习模块可插拔

4. **Jetson Orin / Xavier 级平台上的轻量 stereo 网络**
   - 例如 MobileStereoNet、小分辨率 CREStereo 变体
   - 仅当：
     - 你能接受 GPU/NPU 占用
     - 有自己的场景数据集和回灌能力
     - 有人长期维护部署链路

### 2.2 仍偏研究 demo / 不建议直接作为量产第一方案
1. **RAFT-Stereo 这类较重模型直接上中低端板子**
   - 不适合作为低功耗量产主方案
   - 更适合做上限验证、teacher model、离线标注辅助

2. **完全依赖公开数据集训练、无场景微调的学习 stereo 直接量产**
   - 论文上能跑，不等于机器人现场能扛
   - 室内反光地面、仓库金属货架、户外逆光/草丛/栅栏都容易翻车

3. **把端到端深度网络当唯一感知来源**
   - 产品风险过高
   - 至少应保留 classical fallback、confidence mask、近距离安全冗余

---

## 3. 按场景给选型建议

---

## 3.1 低功耗边缘（电池设备、轻量终端、低 BOM）

### 典型约束
- 功耗严苛：最好 < 5~10W 算法侧增量
- 处理器偏 ARM / 小 NPU / DSP，内存紧
- 目标更多是**近中距障碍感知**，不是高精地图

### 优先推荐
#### 方案 A：专用深度模组 / 现成双目模组
- **推荐度：高**
- 原因：
  - 最省研发资源
  - 软硬件打包成熟
  - 对小团队友好

#### 方案 B：经典 SGBM / SGM（降分辨率 + ROI）
- **推荐度：很高**
- 推荐配置：
  - 分辨率 320×240 / 640×360 / 640×480
  - disparity range 按工作距离严格裁剪
  - 只做前向 ROI，而不是全图满配
  - 输出不是稠密深度图，而是**障碍栅格 / 最近障碍距离 / free-space**
- 为什么：
  - 很多低功耗机器人根本不需要“漂亮深度图”，只需要**稳定可用的碰撞约束**

### 不太建议
- 直接上 RAFT-Stereo / 大模型 stereo
- 大分辨率稠密深度 + 高帧率一起追求

### 产品建议组合
**推荐组合 1（最稳）：**
- 双目全局快门相机 + ARM SoC
- OpenCV StereoSGBM / 轻量 SGM
- LR check + speckle filter + obstacle clustering
- 必要时加 IMU 做时序稳定

**推荐组合 2（有小 NPU）：**
- SGBM baseline
- 小模型做 confidence / invalid region refinement
- 规则层输出栅格地图

### 量产判断
- **今天就能上板量产：是**
- **工程风险：低**

---

## 3.2 移动机器人实时避障（AMR / 配送 / 巡检 / 清洁）

### 典型约束
- 关注实时性和稳定性，通常 15~30fps 以上更有意义
- 关键不是 EPE，而是：
  - 近距离障碍漏检率
  - 细杆/台阶/玻璃门附近鲁棒性
  - 动态场景误报/漏报

### 优先推荐
#### 方案 A：classical stereo 为主干
- **推荐度：最高**
- 推荐理由：
  - 障碍检测通常只需可靠深度阈值和 free-space 提取
  - classical pipeline 更可解释、更好 debug
  - 延迟可控，故障模式相对清晰

#### 方案 B：classical + 轻量 learning refinement
- **推荐度：高**
- 使用条件：
  - classical 在低纹理地面/货架边缘/逆光区域表现不够
  - 平台有 Jetson Orin NX / Orin Nano / 中高端 NPU
- 推荐做法：
  - classical 提供 primary disparity
  - 小模型只修正困难区域，不接管全局

### 如果一定要上学习式 stereo
可优先考虑：
- **MobileStereoNet 类轻量网络**：更像可工程尝试的第一梯队
- **CREStereo 轻量化部署版**：如果算力允许，可作为效果更好的候选

但我会强调：
- 学习网络最好**不是唯一安全链路**
- 至少保留 classical fallback / safety ROI

### 产品建议组合
**推荐组合 1（主流量产解）：**
- Jetson Orin Nano / RK3588 级别
- 双目全局快门 + classical SGM/SGBM
- 时序滤波 + 占据栅格
- 遇到玻璃/黑地毯等难点，用超声/ToF/激光做补盲

**推荐组合 2（效果增强版）：**
- Jetson Orin NX
- SGBM baseline + MobileStereoNet/CREStereo-lite refinement
- 输出 confidence map，低 confidence 区域走保守避障策略

### 量产判断
- **classical 主干：今天就能上板量产**
- **轻量 learning 作为增强：可控，但需要较强算法工程团队**
- **纯学习式主链路：谨慎，偏中高端项目验证**

---

## 3.3 较高精度重建 / 测量（工业件、体积估计、局部高精地图）

### 典型约束
- 更关注亚像素精度、边缘质量、孔洞率、重复性
- 可以接受较低帧率
- 标定质量、基线长度、镜头畸变控制往往比模型名字更重要

### 优先推荐
#### 方案 A：高质量双目硬件 + classical stereo + 强后处理
- **推荐度：高**
- 条件：
  - 光照可控
  - 被测物/场景可控
  - 可以做投射纹理（主动纹理）
- 在受控场景下，classical 并不差，且结果稳定、可重复

#### 方案 B：classical + learning refinement
- **推荐度：高**
- 很适合：
  - classical 能给出 80 分结果
  - 但边缘、遮挡、低纹理区域还想再拔高
- 这是比“全量深度网络替代 classical”更现实的工业路线

#### 方案 C：较强 stereo 网络用于离线/准实时
- **推荐度：中**
- 可考虑：
  - CREStereo
  - RAFT-Stereo
- 更适合：
  - 离线重建
  - 工站级别 GPU
  - 数据生产或 teacher model

### 产品建议组合
**推荐组合 1（工程最稳）：**
- 高分辨率全球快门双目
- 精细标定 + SGM / Census-SGM + 亚像素拟合
- 左右一致性 + confidence + plane fitting / TSDF 局部融合

**推荐组合 2（精度增强）：**
- 上述 classical 结果作为初值
- 小到中等模型做 refinement
- 对关键测量区域做 ROI 高精处理，而不是全图都重模型

**推荐组合 3（高端 SKU / 研发验证）：**
- Orin AGX / 工控机 GPU
- CREStereo 或 RAFT-Stereo 做高质量 disparity
- classical 结果作为 sanity check / fallback

### 量产判断
- **受控工业场景下 classical + 光学/标定优化：今天就能量产**
- **learning enhancement：可量产，但要看数据闭环能力**
- **重模型直接量产：通常只适合高端或低速场景**

---

## 3.4 远距户外（道路、园区、农机、巡检）

### 典型约束
- 大挑战不是算子，而是物理条件：
  - 远距视差极小
  - 户外光照剧烈变化
  - 栅栏、树叶、草地、反光、水面都很难
- 远距效果高度依赖：
  - **更长基线**
  - 更高分辨率
  - 更稳的标定结构

### 核心观点
**远距户外不是“换个更强网络”就能解决，首先是硬件几何设计问题。**

### 优先推荐
#### 方案 A：长基线双目 + classical / 半全局匹配
- **推荐度：高**
- 原因：
  - 远距先靠基线和像素分辨率把信号做出来
  - 再谈算法优化

#### 方案 B：长基线 + 学习式 refinement / confidence
- **推荐度：中高**
- 适合在：
  - classical 远处噪声多
  - 但系统仍需可解释、可退化运行

### 不建议幻想
- 用普通短基线小双目模组，靠大模型获得稳定远距深度
- 户外强阳光下只靠双目，不做任何多传感器冗余

### 产品建议组合
**推荐组合 1（现实方案）：**
- 长基线双目 + 全局快门 + 偏振/遮光优化
- classical SGM 主干
- 输出中远距稀疏可信深度 + 近距高置信障碍
- 与激光雷达 / 毫米波 / 单目检测融合

**推荐组合 2（高端增强）：**
- Orin NX / AGX
- 长基线 + CREStereo-lite / 轻量 refinement
- 使用 confidence 门控，只在高置信区域参与规划

### 量产判断
- **长基线 + classical + 融合：今天可做，且是正路**
- **单靠学习 stereo 解决远距户外：偏研究 demo / 风险高**

---

## 3.5 室内仓储 / 服务机器人

### 典型场景
- 仓库货架、托盘、窄通道、地面反光
- 服务机器人室内行走、人腿/椅脚/桌边等细小障碍
- 常有大面积低纹理墙面 / 地面

### 优先推荐
#### 仓储 AMR
- **推荐：classical stereo + 几何规则 + 多传感器补盲**
- 原因：
  - 路径与障碍判断可规则化
  - 重点是稳定运行和边界条件处理

#### 服务机器人
- **推荐：classical 为底 + 轻量 learning 提升边界和弱纹理**
- 原因：
  - 室内细物体、边缘、家具脚等对 classical 比较苛刻
  - 适度引入轻量网络收益通常比仓储更明显

### 产品建议组合
**推荐组合 1（仓储标准款）：**
- 双目全局快门 + SGBM/SGM
- 地面分割 / free-space extraction
- 货架/托盘区域做 ROI 强化
- 超声 / 2D LiDAR 做安全兜底

**推荐组合 2（服务机器人增强款）：**
- Orin Nano / NX
- classical disparity + 轻量 refinement
- 结合语义分割（玻璃门、镜面、黑色障碍）做误检抑制

### 量产判断
- **仓储：今天就能量产，首选 classical 主干**
- **服务机器人：今天也能量产，但更建议加一些学习增强**

---

## 4. 方案分级：哪些成熟，哪些偏 demo

## 4.1 成熟度分级表

| 方案 | 板端量产成熟度 | 工程可控性 | 对数据依赖 | 典型问题 |
|---|---|---:|---:|---|
| 现成深度模组（RealSense/OAK-D/供应商模组） | 高 | 高 | 低 | 供应链/成本/场景边界受限 |
| StereoBM | 中 | 高 | 低 | 效果偏弱，易受纹理影响 |
| StereoSGBM / SGM / Census-SGM | 很高 | 很高 | 低 | 弱纹理、反光、重复纹理 |
| classical + refinement 小模型 | 高 | 中高 | 中 | 部署链路与数据回灌复杂 |
| MobileStereoNet | 中高 | 中 | 中高 | 需适配/量化/泛化验证 |
| CREStereo（轻量部署版） | 中 | 中 | 中高 | 算力/内存要求更高 |
| RAFT-Stereo | 低~中 | 低~中 | 高 | 更像研究/高端 GPU 方案 |

> 注：这里的“成熟度”不是论文水平，而是**产品团队能否稳定 bring-up、调优、交付、维护**。

---

## 5. 芯片 / 板卡搭配建议

## 5.1 低功耗 SoC（ARM / MCU+DSP / 小 NPU）
### 推荐
- **classical stereo only**
- 或者 classical + 极小后处理网络

### 不建议
- 直接部署 RAFT-Stereo / 大 cost volume 网络

### 适合任务
- 近距避障
- free-space detection
- 简易仓储/配送

---

## 5.2 RK3588 / 中端 NPU SoC
### 推荐
- SGBM/SGM 主干
- 有余量时加轻量 refinement / confidence 网络

### 适合任务
- 室内机器人
- 中低速 AMR
- 基础 3D 感知

### 工程提醒
- NPU 工具链、算子支持、量化精度要提前验证
- 不要先选模型再去赌芯片能不能转

---

## 5.3 Jetson Orin Nano / NX
### 推荐
- **量产主线：classical + optional learning enhancement**
- 可尝试：
  - MobileStereoNet
  - 小型 CREStereo 变体
- 如果场景复杂，可在 ROI/低分辨率下部署学习模型

### 适合任务
- 服务机器人
- 仓储增强版
- 室内外混合移动平台

### 工程提醒
- 关注：
  - GPU 占用对其它 perception stack 的挤压
  - 显存峰值
  - TensorRT 版本锁定
  - 温度降频

---

## 5.4 Jetson AGX / 工控 GPU
### 推荐
- classical + 中重型 stereo 网络并行验证
- CREStereo / RAFT-Stereo 可做高质量模式、低速高精模式、离线重建

### 适合任务
- 高端测量
- 研发验证平台
- 高精重建工站

### 工程提醒
- 这是高端 SKU 路线，不该倒推成全产品线标配

---

## 6. 我给的推荐组合（最实用版本）

## 组合 A：最稳量产款
**双目相机 + classical SGM/SGBM + 规则后处理**

适合：
- 低功耗边缘
- 仓储 AMR
- 巡检避障
- 首代量产产品

优点：
- 成熟、低风险、易解释
- 开发与维护成本最低

不足：
- 复杂场景上限一般

**结论：如果你现在就要做产品，我最推荐先从这个组合起。**

---

## 组合 B：产品增强款
**classical baseline + 轻量学习 refinement / confidence**

适合：
- 服务机器人
- 复杂室内场景
- classical 已经能用，但想减少边界错误/孔洞

优点：
- 保留 classical 的稳定性
- 学习模块只解决 hardest 20% 问题

不足：
- 需要部署和数据闭环能力

**结论：这是“效果/风险”平衡最好的升级路线。**

---

## 组合 C：Jetson 高端款
**Jetson Orin NX/AGX + MobileStereoNet / CREStereo-lite + classical fallback**

适合：
- 高端服务机器人
- 复杂环境导航
- 有 GPU 预算的产品线

优点：
- 深度质量明显更强
- 对弱纹理和遮挡往往更友好

不足：
- 算力、功耗、维护复杂度更高

**结论：可以做，但不建议把它当“低风险首版”。**

---

## 组合 D：高精/研究款
**长基线双目 + 高质量标定 + CREStereo/RAFT-Stereo（离线或低速）**

适合：
- 高精重建
- 精测量
- 算法上限验证

优点：
- 上限高

不足：
- 板端量产风险高，系统复杂

**结论：更适合研发/高端 SKU，不适合作为通用机器人量产起点。**

---

## 7. 对几个常见候选方案的判断

## 7.1 StereoSGBM / SGM
- **定位：量产主力**
- **评价：**
  - 不是最潮，但非常实用
  - 真正机器人项目里，很多时候赢在“可调、可控、可解释、实时”
- **建议：首选 baseline**

## 7.2 MobileStereoNet
- **定位：轻量学习 stereo 候选**
- 已知公开信息：该工作明确面向 lightweight stereo matching
- **评价：**
  - 比较适合拿来做板端学习方案第一候选
  - 但仍需自己验证量化、输入分辨率、域适配、极端工况
- **建议：可作为 Jetson/NPU 平台的第一批候选模型**

## 7.3 CREStereo
- **定位：偏“实用型高性能”学习 stereo**
- 已知公开信息：论文题目就叫 *Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation*
- **评价：**
  - 相比更重的纯 SOTA 方案，更接近“可以工程尝试”的一类
  - 但默认版本对板端仍不算轻，常需要裁剪/降分辨率/TensorRT 化
- **建议：适合中高算力平台做增强，不建议裸上低功耗平台**

## 7.4 RAFT-Stereo
- **定位：高性能研究/高算力方案**
- **评价：**
  - 更适合离线、高端 GPU、teacher model、上限验证
  - 不适合当大多数机器人产品的首选量产主链路
- **建议：拿来 benchmark 或离线 teacher，比拿来量产更合适**

## 7.5 RealSense / OAK-D 等模组
- **定位：最快落地的产品化入口**
- **评价：**
  - 非常适合原型验证与中小批量
  - 若业务要求高可控、低 BOM、定制结构，后面再考虑自研切换
- **建议：时间紧就先上，不要一开始就陷入“必须全栈自研”**

---

## 8. 真正决定成败的，不只是算法

很多团队把时间花在模型名词上，但在机器人里，更关键的通常是：

1. **基线长度是否合理**
   - 远距看基线，近距看盲区，别指望算法魔法

2. **是否全局快门、同步是否可靠**
   - 运动中滚快门对 stereo 非常伤

3. **标定是否长期稳定**
   - 机器人震动、温漂、结构件变形都会毁掉深度

4. **是否做 confidence / invalid 区域管理**
   - 不确定就保守，不要把坏深度直接喂给规划

5. **是否有多传感器补盲**
   - 玻璃、黑色吸光、强反射、阳光直射下，双目不是万能的

6. **是否围绕“避障/地图/测量指标”优化，而不是只盯 EPE**
   - 产品看的是漏检率、最小可检障碍、制动距离、误报率、连续稳定性

---

## 9. 最终建议（给产品负责人/机器人负责人）

### 如果你现在就要立项，并且追求低风险落地：
**首选：classical stereo（SGBM/SGM）主干。**
- 先把：
  - 双目标定
  - ROI 策略
  - confidence
  - 时序滤波
  - obstacle extraction
  - 安全冗余
  做扎实。

### 如果你已经有 Jetson/NPU 算力，且 classical 在难场景上不够：
**第二步：加轻量 learning refinement。**
- 优先考虑：**MobileStereoNet / 轻量化 CREStereo**
- 但保留 classical fallback

### 如果你要做高精重建 / 高端 SKU：
**再考虑中重型 stereo 网络。**
- RAFT-Stereo、较大 CREStereo、其它 SOTA 可用于：
  - 高质量模式
  - 低速模式
  - 离线重建
  - teacher model

### 一句话版本
- **低功耗边缘：SGBM/SGM，别折腾重模型**
- **移动机器人避障：classical 主干，学习增强做加分项**
- **高精测量：先优化硬件和标定，再加 refinement**
- **远距户外：先加基线和分辨率，再谈网络**
- **室内仓储/服务机器人：仓储偏 classical，服务机器人可适度 learning 增强**

---

## 10. 可执行的落地路线图（建议）

### Phase 1：2~4 周
- 选双目硬件（优先全局快门）
- 打通 rectification + SGBM/SGM
- 定义业务指标：
  - 最近障碍漏检率
  - 0.5m/1m/3m 深度误差
  - 玻璃/反光/弱纹理场景表现

### Phase 2：4~8 周
- 做 confidence / LR consistency / temporal filtering
- 输出 free-space / occupancy，而不是只看彩色深度图
- 补盲传感器评估（超声 / LiDAR / ToF）

### Phase 3：按需升级
- 若难场景确实不足，再上轻量 learning refinement
- 候选顺序建议：
  1. MobileStereoNet
  2. 轻量化 CREStereo
  3. 更重模型仅做 benchmark / teacher

---

## 附：本文中的“证据强弱”说明
- **较强证据**：
  - OpenCV StereoSGBM 为成熟公开实现；
  - MobileStereoNet 明确定位 lightweight stereo；
  - CREStereo 明确定位 practical stereo matching；
  - RAFT-Stereo 通常被视为较重、偏高性能研究路线。
- **工程判断/行业经验**：
  - 哪些方案“今天能量产”、哪个更适合仓储/服务机器人、哪种芯片搭配更稳，这些主要基于机器人产品常见落地路径与端侧部署经验做判断，不等同于单篇论文结论。

如果后续需要，我建议再补一版：
1. **按芯片平台（RK3588 / Orin Nano / Orin NX / AGX）细化 FPS / 功耗 / 内存预算表**；
2. **按双目相机基线（6cm / 9cm / 12cm / 20cm）给工作距离建议**；
3. **补一个“机器人避障评测 checklist”**，比只看论文指标更有用。
