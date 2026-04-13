# VLN Top 10 论文中真机部署方案详细分析

> 在Top 10论文中，共有 **3篇** 涉及真实机器人部署：SFCo-Nav、VLingNav、MA-CoNav
> EmergeNav虽使用开源VLM，但仅在仿真环境（Habitat）中验证，未涉及真机部署

---

## 一、总览对比表

| 维度 | SFCo-Nav | VLingNav | MA-CoNav |
|------|----------|----------|----------|
| **机器人平台** | 四足机器人（未披露具体型号） | Unitree Go2 四足机器人 | AgileX Limo PRO 轮式机器人 |
| **传感器** | RGB-D（未披露型号） | Intel RealSense D457 (RGB, 1280x800, 90deg FOV) | ORBBEC DaBai RGB-D (640x480, 30Hz) |
| **核心模型** | GPT-4o + Grounding-DINOv2 | LLaVA-Video-7B + SigLIP-400M | GPT-5.2 Pro + GPT-4-Turbo |
| **训练方式** | 零样本（无微调） | 三阶段训练（预训练+SFT+RL） | 零样本（无微调） |
| **推理方式** | 云端API调用 | 远程GPU推理（RTX 4090） | 云端API调用 |
| **部署环境** | 酒店套房 | 家庭/办公室/户外 | 办公/实验室/家庭/健身房 |
| **软件中间件** | 未披露 | NMPC轨迹控制器 | ROS 2 Foxy Fitzroy |

---

## 二、SFCo-Nav 真机部署详情

### 论文信息
- **全称**: SFCo-Nav: Efficient Zero-Shot Visual Language Navigation via Collaboration of Slow LLM and Fast Attributed Graph Alignment
- **arXiv**: [2603.01477](https://arxiv.org/abs/2603.01477)
- **会议**: ICRA 2026

### 模型架构

#### 慢脑（Slow Brain, Pi_slow）—— LLM推理规划
- **模型**: **GPT-4o**（OpenAI云端API）
- **备选**: 消融实验中测试了 **Deepseek-V3.1** 作为替代
- **功能**: 包含三个子模块
  - **Final Object Identifier (f_goal)**: 从自然语言指令中提取最终目标物体
  - **Policy Analyzer (f_policy)**: 根据指令、子目标历史和当前感知图，生成推理轨迹和即时子目标
  - **Subgoal Chain Generator (f_chain)**: 将推理轨迹转换为N个未来子目标的结构化链

#### 快脑（Fast Brain, Pi_fast）—— 轻量反应式控制
- **模型**: **Grounding-DINOv2**（开集目标检测）
- **关键设计**: 快脑不是VLM，而是轻量目标检测器，构建属性对象图，避免每步VLM推理
- **功能**:
  - 构建星形拓扑属性图（节点=检测到的物体，边=距离和方位）
  - 子目标执行规划（approach, through, go up, go down等技能）

#### 慢-快桥接（异步触发机制）
- 基于属性图对齐理论计算概率矩阵
- 计算导航置信度 C_t = 1 - P(A)
- C_t > 阈值: 快脑自主继续（不调用LLM）
- C_t <= 阈值: 触发慢脑重新规划

### 训练方式
- **完全零样本，无任何微调**
- 所有模型（GPT-4o、Grounding-DINOv2）均使用预训练权重
- 无需VLN数据集上的任务特定训练

### 真机部署
- **机器人**: 四足机器人（具体型号未披露）
- **传感器**: 自我中心RGB-D输入（具体型号未披露）
- **部署场景**: 配置齐全的酒店套房
- **计算**: GPT-4o通过云端API调用，Grounding-DINOv2运行位置未明确说明
- **局限**: 未披露具体硬件型号、机载计算配置、推理延迟

### 性能数据
| 指标 | 数值 |
|------|------|
| Token消耗 | 减少50%+ |
| 推理速度 | 提升3.5x |
| R2R SR | 38.2% (零样本) |
| R2R SPL | 32.5% |

---

## 三、VLingNav 真机部署详情

### 论文信息
- **全称**: VLingNav: Embodied Navigation with Adaptive Reasoning and Visual-Assisted Linguistic Memory
- **arXiv**: [2601.08665](https://arxiv.org/abs/2601.08665)

### 模型架构

#### 基座VLM
- **模型**: **LLaVA-Video-7B**（~7B参数）
- **视觉编码器**: **SigLIP-400M**（训练全程冻结）
  - 输出: N=729个图像patches, 嵌入维度C=1152
- **跨模态投影器**: 两层MLP，将SigLIP视觉特征映射到VLM隐空间
- **动作头**: MLP，将VLM隐状态转换为机器人运动轨迹 tau = {a_1, ..., a_n}, a in R^3 = (x, y, theta)
- **概率动作头**（用于RL阶段）: 多元高斯分布参数化，预测均值和对数标准差

#### 辅助机制
- **时序编码**: RoPE（旋转位置编码）处理时间戳
- **动态FPS采样**: 受艾宾浩斯遗忘曲线启发，旧帧以更低帧率采样: f_s(i) = f_s^max * exp(-DeltaT / s)

### 训练方式（三阶段）

#### 阶段1: 预训练（开放世界自适应CoT）
- **数据**: 1.6M样本（LLaVA-Video-178K + Video-R1 + ScanQA）
- **策略**: Video-R1使用CoT标注，其余为非CoT
- **训练**: 1 epoch, 标准交叉熵损失

#### 阶段2: 有监督微调（SFT）
- **数据**: Nav-AdaCoT-2.9M（具身导航） + 开放世界视频 = **4.5M总样本**
- **Nav-AdaCoT-2.9M数据集构成**:
  - HM3D ObjNav（人类演示，来自Habitat-Web）
  - MP3D ObjNav（最短路径轨迹）
  - HM3D OVON（开放词汇任务最短路径）
  - EVT-Bench（多人室内追踪）
  - HM3D Instance ImageNav（逐步动作标签最短路径）
  - 覆盖任务: ObjectNav, Visual Tracking, ImageNav
- **CoT标注**: 使用 **Qwen2.5-VL-72B** 生成~472K条推理标注
- **训练**: 20K步, batch size 512
- **损失**: MSE（轨迹预测） + CE（文本输出, CoT推理+VQA）, alpha=0.5平衡
- **更新范围**: 除视觉编码器外所有组件

#### 阶段3: 在线专家引导后训练（RL）
- **起点**: SFT检查点
- **数据**: 10轮rollout，每轮128个episode
- **环境**: HM3D OVON, HM3D Instance ImageNav, EVT-Bench DT
- **混合rollout策略**:
  - **朴素rollout**: 仅保留成功轨迹
  - **专家引导rollout**: 最短路径规划器在智能体振荡/卡住k=15步时介入提供纠正
- **损失**: PPO裁剪代理损失（REINFORCE++优势估计）+ SFT模仿损失, lambda=0.01
- **训练硬件**: **128块 NVIDIA A100 GPU**

### 真机部署

#### 硬件配置
| 组件 | 详情 |
|------|------|
| **机器人** | **Unitree Go2** 四足机器人 |
| **相机** | **Intel RealSense D457**（头部安装）, 仅RGB, 1280x800, 90deg水平FOV |
| **通信** | 便携WiFi单元（机器人背部），用于远程服务器通信 |
| **推理服务器** | **NVIDIA RTX 4090** GPU |
| **轨迹控制** | 非线性模型预测控制（NMPC），基于运动学独轮车模型 |

#### 推理性能
| 指标 | 数值 |
|------|------|
| 通信开销 | ~100ms（图像压缩传输） |
| 推理延迟 | <300ms（跨500帧视频） |
| 有效帧率 | ~2.5 FPS |
| 过去帧处理 | 视觉Token缓存，仅编码最新帧 |

#### 零样本迁移
- **部署权重与仿真完全相同，无任何真实世界微调**

#### 测试场景
| 任务类型 | 场景 | 目标 | 试次 |
|---------|------|------|------|
| **物体目标导航** | 家庭/办公室/户外 | 桌子、洗衣机、微波炉、电视、电梯、垃圾桶、自行车、路灯、树 | 每目标10次 |
| **具身视觉追踪** | 开放空间/杂乱室内/拥挤场景 | 单目标/干扰目标追踪 | 每场景10次 |
| **图像目标导航** | 家庭/办公室/户外 | 每类2个图像指定目标 | 每个10次 |

#### 关键消融发现
- 自适应CoT仅2.1%激活率即优于无CoT和密集CoT（每步都推理）
- 视觉语言记忆（VLingMem）至关重要: 移除后ObjNav SR从50.1降至15.4

---

## 四、MA-CoNav 真机部署详情

### 论文信息
- **全称**: MA-CoNav: A Master-Slave Multi-Agent Framework with Hierarchical Collaboration and Dual-Level Reflection for Long-Horizon Embodied VLN
- **arXiv**: [2603.03024](https://arxiv.org/abs/2603.03024)

### 模型架构

#### 多智能体系统使用的模型
| 智能体 | 模型 | 说明 |
|--------|------|------|
| **观察Agent** | **GPT-5.2 Pro**（多模态VLM） | 选择依据: 障碍检测79.52%, 地标识别58.62%, 优于Gemini 3 Pro Preview |
| **主Agent + 其他Agent** | **GPT-4-Turbo**（文本LLM） | 论文未明确指定每个agent的精确模型分配 |
| **备选方案** | Gemini 3 Pro Preview | 在消融实验中评估，但被弃用 |

#### 模型配合方式
- MLLM（多模态）+ LLM（纯文本）的组合至关重要
- 仅用MLLM: SR=17.60%, KPA=51.92%
- MLLM + LLM: SR=25.60%, KPA=69.89%
- 原因: 纯MLLM难以有效整合当前观察图像、导航文本和历史信息

#### 架构: 1个主Agent + 4个从属Agent
- **主Agent（Master）**: 基于LLM的状态机控制器，循环: PLANNING -> PERCEPTION -> ACTION -> EVALUATION
- **观察Agent**: 多模态感知，捕获4个基方向图像 -> 提取感知元组 P = (Obstacles, Landmarks, Traversability, Metadata)
- **规划Agent**: 任务分解为子任务序列 S = {s_1, ..., s_k}，子任务完成验证
- **执行Agent**: 更新几何地图G_t和拓扑地图T_t，生成离散动作（MoveForward, TurnRight 90deg, TurnLeft 90deg, Stop）
- **记忆Agent**: 存储所有状态-动作对 H = {(t, s_t, a_t, O_t, M_t)}

#### 双层反思机制
- **局部反思（在线，每步）**: 18.7%的步骤被标记，反思准确率85.2%, 回退成功率72.3%
- **全局反思（离线，任务后）**: 根因分析失败案例，编码为结构化经验存入记忆Agent

### 训练方式
- **完全零样本，无任何场景特定微调**
- 所有模型通过Prompt工程和架构设计使用
- 无参数更新

### 真机部署

#### 硬件配置
| 组件 | 详情 |
|------|------|
| **机器人** | **AgileX Limo PRO** 轮式移动机器人 |
| **相机** | **ORBBEC DaBai** RGB-D相机, 640x480, 30Hz |
| **多视角采集** | 每时间步捕获4个基方向图像（可能通过原地旋转实现） |
| **LiDAR** | 未使用，完全依赖RGB-D |
| **软件** | **ROS 2 Foxy Fitzroy** |
| **计算** | LLM通过云端API调用，其他模块在机器人板载系统运行 |

#### 测试设置
- **场景**: 4个真实室内场景（办公区、实验室、家庭环境、健身房）
- **指令**: 50条复杂自然语言指令（每条含3+目标物体）
- **重复**: 每条指令在相同初始条件下测试5次

#### 性能结果
| 方法 | SR | KPA | NE(m) |
|------|-----|------|-------|
| NavGPT | 0% | 14.14% | 6.78 |
| MapGPT | 0% | 21.82% | 5.97 |
| CoELA | 2.80% | 32.01% | 5.71 |
| RCTAMP | 8.40% | 34.54% | 4.27 |
| **MA-CoNav** | **25.60%** | **69.89%** | **2.93** |

---

## 五、三者方案核心差异分析

### 1. 模型选择策略

| 方案 | 策略 | 优劣势 |
|------|------|--------|
| **SFCo-Nav** | 闭源API（GPT-4o）+ 轻量开源检测器（Grounding-DINOv2） | 推理能力强但依赖网络，成本高；快脑轻量高效 |
| **VLingNav** | 开源VLM（LLaVA-Video-7B）端到端训练 | 可本地部署，延迟可控；但训练成本极高（128x A100） |
| **MA-CoNav** | 闭源API（GPT-5.2 Pro + GPT-4-Turbo）多模型协同 | 利用最强模型能力，但完全依赖云端，延迟和成本不可控 |

### 2. 训练范式

| 方案 | 范式 | 训练资源 |
|------|------|----------|
| **SFCo-Nav** | 零样本（无训练） | 无需GPU训练 |
| **VLingNav** | 三阶段（预训练 -> SFT 4.5M样本 -> RL 128 A100） | 极高：128x A100 GPU |
| **MA-CoNav** | 零样本（无训练） | 无需GPU训练 |

### 3. 部署架构

| 方案 | 机载计算 | 云端依赖 | 推理延迟 |
|------|---------|---------|---------|
| **SFCo-Nav** | 部分（Grounding-DINOv2可能本地） | GPT-4o云端API | 未披露 |
| **VLingNav** | 无（纯远程推理） | RTX 4090远程GPU | <300ms + 100ms通信 |
| **MA-CoNav** | 除LLM外全部本地 | GPT-5.2 Pro + GPT-4-Turbo云端API | 未披露 |

### 4. 关键启示

- **零样本方案**（SFCo-Nav, MA-CoNav）部署门槛低但严重依赖闭源云端API，延迟和成本不可预测
- **训练方案**（VLingNav）训练成本极高但部署后模型自主可控，支持本地推理，延迟稳定
- **VLingNav是唯一实现真正端到端训练的方案**，且训练后权重直接零样本迁移至真机
- **SFCo-Nav的慢-快分离架构**有效降低了LLM调用频率（50%+ Token减少），是平衡性能与效率的可行策略
- **MA-CoNav的多智能体方案**在复杂长程任务上优势明显，但多模型API调用的累积延迟是实际部署的瓶颈
