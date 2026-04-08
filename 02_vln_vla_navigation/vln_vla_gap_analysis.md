# VLN 方向机会差距调研（含银河通用、VLA/导航前沿）

> 状态：final draft
> 更新时间：2026-03-31
> 作者：虾霸

---

## 1. 研究问题

本文聚焦 4 个问题：

1. 当前 VLN（Vision-Language Navigation）在学术界和产业界已经突破到什么程度？
2. 重点关注银河通用：其在导航、VLA、具身智能上的公开技术路线与已验证突破是什么？
3. 哪些问题已经基本解决，哪些问题还没有解决？
4. 为什么 VLN 相对 VLA 不再是最热点的叙事？当前还有哪些值得投入的机会差距？

---

## 2. 执行摘要

**核心判断：VLN 不是没价值，而是“作为独立赛道”正在被更大范式吸收。**

行业关注点已经从“单点导航能力”逐渐迁移到：

- VLA（Vision-Language-Action）
- embodied agent
- open-vocabulary mobile manipulation
- 长时程任务执行（task completion）
- 导航 + 抓取 + 操作 + 持续运行的一体化系统

因此，今天真正有价值的突破，不再只是“导航 benchmark 分数更高”，而是：

1. 能否把导航嵌入任务闭环；
2. 能否在真实动态环境中持续稳定运行；
3. 能否用合成数据和大模型降低真机数据成本；
4. 能否跨场景复制，而不是只在单一 demo 场景中成功。

对银河通用而言，更合理的判断不是“它是不是一家纯 VLN 公司”，而是：

> **它代表的是把 VLN 吸收到具身大模型、合成数据、机器人本体和零售/工业闭环中的产业化路径。**

---

## 3. VLN 已经突破到什么程度

### 3.1 纯学术 VLN：经典问题已经被打透一大半

传统 VLN 的标准设定是：

- 输入：自然语言指令 + 第一视角视觉观察；
- 输出：导航动作序列；
- 目标：按语言指令到达目标位置或按路径完成导航。

近几年，这条线的主要进展包括：

#### （1）基础视觉-语言对齐能力明显成熟

早年难点在于诸如：

- “在沙发处左转”
- “经过厨房后继续前进”
- “在桌边停下”

这类局部视觉 landmark 与语言描述的精确对齐。

随着更强的：

- 视觉编码器
- 历史记忆建模（Transformer / memory module）
- VLM / CLIP / LLM grounding
- 多步推理与外部记忆

基础 instruction grounding 已经比 2018–2021 年显著增强。

#### （2）长指令拆解与 landmark-based navigation 已具备较强可行性

代表工作如 **LM-Nav**（CoRL 2023）：

- LLM 从语言中抽取 landmarks；
- 图像-语言模型完成 landmark grounding；
- 视觉导航模型负责逐步到达。

这说明 VLN 并不一定依赖大量语言标注轨迹，也可以通过预训练模块组合构造出可用系统。

#### （3）传统 benchmark 上的 VLN 已经有明显“刷题化”趋势

在经典室内 VLN benchmark 上，近几年大量工作持续刷新指标。这意味着：

- 在既定仿真环境内，纯导航跟随语言的能力显著增强；
- 新工作越来越容易沦为 recipe 改进；
- 学术热度开始向更大、更接近真实闭环的问题迁移。

---

### 3.2 真实世界 VLN：从“会做题”进入“能上机”

这两年的关键变化，是 VLN 已不再只停留在仿真。

已被证明可行的方向包括：

#### （1）真实机器人长程语言导航可做

例如 **LM-Nav** 已证明：

- 真实机器人可以执行长程自然语言导航；
- 可以利用大模型、VLM 和视觉导航模块组合完成任务；
- 不必完全依赖昂贵的语言导航标注数据。

#### （2）VLN 正在向开放词汇语义导航融合

今天系统不再只是“走到固定点”，而越来越多涉及：

- 到某类物体附近；
- 到某个语义区域；
- 在未见过的物体类别或描述下执行导航。

这让 VLN 自然过渡到：

- semantic navigation
- open-vocabulary navigation
- task-oriented embodied navigation

#### （3）导航被更大任务吸收

以 **HomeRobot / OVMM** 为代表，研究重点已不再是单独导航，而是：

- 在未知环境中探索；
- 找物体；
- 抓取；
- 搬运与放置。

在这类任务里，导航只是必要子模块，而不再是最终目标本身。

---

## 4. 2024–2026 年前沿：VLN / embodied navigation / VLA 正在怎么演化

### 4.1 先给一个三层分类

为了避免把不同问题混在一起，可以把近两年前沿大致分为三层：

1. **纯 VLN**：输入仍以“语言导航指令 + 视觉观测”为核心，目标仍然是路线跟随、目标搜索或语义导航；
2. **embodied navigation / agentic navigation system**：仍以导航为主，但开始引入主动探索、3D 场景图、鲁棒性评测、跨平台部署和真实机器人闭环；
3. **VLA / embodied agent / mobile manipulation**：语言不只决定“往哪走”，还决定“做什么动作序列”；导航只是更长时程具身任务中的一个环节。

---

### 4.2 纯 VLN 仍在演化，但方向变了

#### AgentVLN（2026）
- **定位**：偏纯 VLN，但明显带有 agentic/navigation system 色彩。
- **核心点**：把 VLN 建模为 POSMDP，用 **VLM-as-Brain + skill library** 的方式，将高层语义推理与底层感知/规划解耦，并加入 2D-3D 表征桥接、自纠错和主动探索。
- **意义**：代表 VLN 从“端到端跟路”走向“在不确定环境里像 agent 一样执行路线”。

#### OmniVLN（2026）
- **定位**：VLN / embodied navigation system。
- **核心点**：面向空地平台（air + ground）的零样本视觉语言导航框架，引入 **全向感知 + 3D Dynamic Scene Graph + token-efficient LLM reasoning**。
- **意义**：显示 VLN 已经从窄视场 indoor setting，扩展到全向感知、跨平台、分层空间推理。

#### DyGeoVLN（2026）
- **定位**：纯 VLN。
- **核心点**：引入 dynamic geometry foundation model，专门针对动态真实场景中的几何变化；并提出 pose-free、自适应分辨率 token pruning。
- **意义**：代表 VLN 从“静态环境刷分”走向“动态环境建模”。

#### SOL-Nav（Structured Observation Language for Navigation, 2026）
- **定位**：纯 VLN。
- **核心点**：把 RGB-D 视觉观测转换为结构化文本，再与导航指令拼接后输入语言模型，避免重型视觉 token 融合。
- **意义**：代表 VLN 的另一条路线——不是继续堆更大的视觉 backbone，而是把观测语言化/符号化，借助 PLM 推理。

#### SignNav（2026）
- **定位**：纯导航任务，但更贴近真实大型室内环境。
- **核心点**：显式引入 signage（标识牌）语义导航，处理商场、医院、机场等高度依赖环境文字线索的场景。
- **意义**：把“看懂环境中的文字/指示牌”正式拉进了具身导航主线。

---

### 4.3 评测和系统正在逼近真实世界

#### NavTrust（2026）
- **定位**：embodied navigation benchmark。
- **核心点**：系统性地在 RGB、depth、instruction 上引入现实腐蚀与扰动，统一评估导航系统的 trustworthiness / robustness。
- **意义**：研究重点开始从“谁在干净测试集上分数更高”，转向“谁在脏环境里更可靠”。

#### CeRLP（2026）
- **定位**：visual navigation / embodied navigation system。
- **核心点**：解决 cross-embodiment 局部规划问题，显式建模机器人几何差异、相机差异和单目深度歧义。
- **意义**：提醒我们，真实导航泛化不只是换场景，还包括换机器人本体的泛化。

#### HUGE-Bench（2026）
- **定位**：高层 UAV Vision-Language-Action benchmark。
- **核心点**：针对高层语言动作、多阶段行为与安全约束，强调 collision-aware evaluation。
- **意义**：说明 VLN 与 VLA 的边界开始模糊，任务不再只是“飞到哪里”，而是“按高层语义完成一段安全且过程正确的行动”。

#### CityWalker（CVPR 2025）
- **定位**：embodied urban navigation。
- **核心点**：从 web-scale videos 学习城市导航，将研究从室内模拟器进一步推向开放城市环境。
- **意义**：探索“能否从海量人类城市视频中蒸馏真实世界导航常识”。

---

### 4.4 VLA / embodied agent / mobile manipulation：导航不再是终点

#### OpenVLA（2024）
- **定位**：典型 VLA，不是纯 VLN。
- **核心点**：7B 开源 VLA，基于大规模视觉语言预训练和 97 万条真实机器人 demonstrations，支持高效微调和跨任务泛化。
- **意义**：把“视觉-语言-动作统一建模”做成了可复现、可微调、可比较的开源基座。

#### Octo（2024）
- **定位**：generalist robot policy / VLA 前身路线。
- **核心点**：基于 Open X-Embodiment 的 80 万轨迹训练通用机器人策略，可快速适配新观察空间和动作空间。
- **意义**：为后续 VLA/agent 工作提供了开源通用策略底座。

#### π0（2024）
- **定位**：generalist robot policy，偏 mobile manipulation 与长时程操作。
- **核心点**：支持高频连续控制，展示了 folding laundry、table bussing、build box 等复杂长时程任务。
- **意义**：表明具身前沿已明显超出“导航到目标点”，而转向多阶段规划、操作恢复和移动-操作一体化。

---

## 5. 银河通用：其突破的本质是什么

### 5.1 基本判断

银河通用的主线，不是做一个纯 VLN 模型，而是在做一条更完整的链路：

- 合成仿真数据基础设施；
- 抓取 VLA / 跟踪-导航 VLA；
- 机器人本体与行业场景结合；
- 再通过零售、工业等方向做真实世界落地尝试。

因此，对银河通用的分析不能停留在“导航准不准”，而更应该看：

> **它是否把导航变成了产品系统中的稳定子能力，并与抓取、跟踪、操作共同形成任务闭环。**

---

### 5.2 可核验公开事实：当前证据最强的技术路线是什么

#### （1）公司与组织信息

根据官网公开信息：

- 公司主体为 **北京银河通用机器人股份有限公司**；
- 成立时间为 **2023 年 5 月**；
- 在 **北京、深圳、苏州、香港**设有研发中心；
- 官网上写有与 **北京大学、北京智源人工智能研究院、宣武医院、北京中关村学院** 的联合实验室/研究中心信息。

这些属于较可靠公开事实。

#### （2）前序能力：Open6DOR / NaVid / DexGraspNet

官网“技术研究”页公开列出若干方向：

- 材质无关的泛化抓取；
- **Open6DOR**：开放指令、可控制物体朝向的取放系统；
- **NaVid**：场景泛化的具身导航模型；
- 双臂协同柔性操作；
- **DexGraspNet**：百万级灵巧手数据集。

这说明银河通用的公开技术路线并不是 2025 年突然出现，而是沿着：
**抓取/操作、导航、数据集与基础模型** 逐步推进。

#### （3）GraspVLA：证据最强的抓取 VLA 主线

这是目前公开证据最强的一条路线。

- arXiv 论文：**GraspVLA: a Grasping Foundation Model Pre-trained on Billion-scale Synthetic Action Data**（arXiv:2505.03233）；
- 项目页明确写到：构建了 **SynGrasp-1B**（十亿帧规模仿真抓取数据集）；
- 目标是探索**完全基于大规模合成动作数据**训练 VLA 模型；
- 方法上结合视觉-语言骨干与 flow-matching 动作生成；
- 强调通过互联网语义数据与合成动作数据结合，增强 open-vocabulary 抓取泛化；
- 项目页列出工业、零售、家庭等应用外延。

**一句话判断**：银河通用已经明确公开了一条“合成数据大规模预训练 → 通用抓取基础模型 → 少样本后训练迁移到行业场景”的技术路线。

#### （4）TrackVLA：导航/跟随方向的证据也较强

- arXiv 论文：**TrackVLA: Embodied Visual Tracking in the Wild**（arXiv:2505.23189）；
- 任务是具身视觉跟踪，核心是机器人在动态环境中依靠第一视角视觉持续跟踪目标；
- 把目标识别与轨迹规划通过共享 LLM backbone 联合建模；
- 构建 **EVT-Bench**，并收集 **170 万样本**；
- 论文声称在仿真和真实环境都有较强泛化，并在真实环境达到 **10 FPS** 推理速度。

**一句话判断**：如果说 GraspVLA 对应“看懂+抓到”，那么 TrackVLA 更接近“看懂+跟住+走到”。

#### （5）官网可见产品与硬件线索

官网和前端静态内容显示至少存在：

- **G1**
- **S1**

并可提取到一些硬件参数线索，例如：

- 4 舵轮全向移动底盘；
- 最大速度 1.5 m/s；
- 双臂 7×2 自由度；
- 工作时长 8 小时；
- 快速换电；
- 腕部六维力传感器、3D 雷达等。

这些更像“官网前端静态可见公开参数”，可信度高于媒体转述，但仍建议后续结合产品页或产品手册复核型号对应关系。

---

### 5.3 哪些属于宣传口径，不能直接当事实写死

下面这些内容目前证据闭环不够强，应在正式结论中标注 **待独立验证**：

#### （1）AstraBrain / AstraSynth / DeepVLA

本次能核验到的公开原始材料中，**没有形成足够强的官方证据闭环**。因此：

- 可以提到外部材料中存在这些命名；
- 但**不能把它们与 GraspVLA / TrackVLA 这种有 arXiv + 项目页支持的内容等量齐观**；
- 正式写法应为：**待独立验证**。

#### （2）“全球首个 / 世界第一 / 市场领先”类表述

包括但不限于：

- “全球首个全面泛化端到端具身抓取基础大模型”；
- “全球首个产品级端到端具身 FSD 大模型”；
- “世界第一个场景泛化具身导航大模型”；
- “市场领先”“引领行业前沿”。

这些说法可能有技术依据，但如果没有系统竞品对比和时间线校验，**不能直接当作客观事实**。

#### （3）“常态化运营”“广泛落地”“重载能力”

当前可以看到的，是：

- 若干公开演示；
- 媒体报道；
- 项目页应用外延；
- 官网新闻标题线索。

但目前缺少足够多的：

- 客户名单；
- 部署点位；
- 连续运行时长；
- 人工接管率；
- 故障率；
- 复购与扩点信息。

因此关于“常态化运营”“广泛落地”“重载能力”，当前最稳妥的写法是：

> **有公开案例线索和验证叙事，但大规模、长期、稳定的商业化证据仍然不足。**

---

### 5.4 银河通用这条路线对 VLN / VLA 的启示

#### （1）更像“按技能拆解的具身模型栈”

从可核验材料看，银河通用并不是一步追求一个包打天下的单体通用模型，而是至少沿着几条高价值技能链推进：

- **GraspVLA**：抓取 / 操作；
- **TrackVLA**：跟踪 / 导航；
- **NaVid / Open6DOR / DexGraspNet**：为后续模型准备任务、数据和先验。

这说明：**通用具身智能的商业化落地，短期更可能依赖“按高价值技能纵深突破”，而不是一步到位的大一统模型。**

#### （2）合成数据不是辅助，而是主训练范式

GraspVLA 的价值不只是“用了仿真”，而是明确提出：

- 用十亿帧级别合成动作数据做预训练主干；
- 再通过互联网语义和少量真实后训练补齐语义与偏好。

这对 VLN / VLA 很关键，因为：

- 导航、跟随、抓取这类任务若要覆盖长尾变化，真实采集成本极高；
- 高质量合成数据 + 任务结构设计，很可能是把 VLA 从 demo 推向产品化的现实路径。

#### （3）真正的产业指标不是“会不会”，而是“能不能稳定、便宜、持续地干活”

银河通用公开材料有一个很明显的特征：

- 技术论文与演示很多；
- 长期运营指标很少。

这反而给行业判断一个提醒：

- 具身/VLA 公司是否进入商业拐点，不能只看模型名字和视频；
- 更要看连续运行时长、故障率、人工接管率、部署密度、客户复购和单位经济性。

---

## 6. 哪些问题已经解决，哪些还没解决

### 6.1 已基本解决 / 阶段性解决

#### 在学术 VLN 中

- 基础视觉-语言指令对齐；
- landmark-based navigation；
- 长指令分段执行；
- 仿真 benchmark 上较高成功率；
- 用预训练 VLM / LLM 降低语言标注轨迹依赖。

#### 在产业早期落地中

- 真实机器人执行一定长度的语言导航；
- 导航与语义感知融合；
- 导航嵌入更大任务系统；
- 在受控 / 半受控环境中实现相对稳定运行。

#### 在银河通用这类路线中

- 导航不再是单点能力，而是任务闭环一部分；
- 合成数据成为关键训练资源；
- 导航与抓取 / 作业开始串联；
- 从演示迈向商业化运行的阶段性验证。

---

### 6.2 仍未解决 / 远未解决

- 完全开放环境下的鲁棒 VLN；
- 动态环境、多主体干扰下的稳定导航；
- 真正通用的语言理解与空间 grounding；
- 长期空间记忆与可恢复的任务执行；
- 导航和 manipulation 一体化后的 credit assignment；
- 低成本跨场景复制；
- 产业级安全、可审计、可解释；
- 从 demo 成功到长期无人值守运行。

---

## 7. 为什么 VLN 相对 VLA 不是热点

### 7.1 VLN 的任务边界太窄

VLN 关注的是：

- 看；
- 听懂语言；
- 走到目标。

但产业需要的是：

- 走过去；
- 找到目标；
- 操作物体；
- 处理异常；
- 最后完成完整任务。

因此从价值链上讲，VLA / embodied agent 自然更受关注。

---

### 7.2 导航越来越像基础设施能力，而不是主舞台

导航很重要，但越来越像：

- 自动驾驶里的“车道保持”；
- 大模型 agent 里的“工具调用”。

即：

- 没它不行；
- 但只做它不够；
- 商业价值在更完整的闭环里。

---

### 7.3 VLA 更贴合 foundation model 叙事

VLA 天然更适合当前 AI 叙事：

- 大规模预训练；
- 统一多模态表征；
- 统一 action interface；
- 端到端 / 通用机器人基础模型。

而 VLN 更像一个经典 embodied 子任务，难以独立承接最大关注度。

---

### 7.4 纯 VLN benchmark 有明显“刷题化”倾向

热度下降的典型信号是：

- benchmark 上不断变强；
- 但真实世界 gap 仍然明显；
- 新工作大量停留在局部改进。

于是资源自然向更接近真实任务闭环的问题迁移。

---

## 8. 当前机会差距在哪里

### 8.1 从“语言导航”升级到“任务语义导航”

高价值问题不再只是：

- “左转，直走，在门口停下”

而是：

- “去拿离收银台最近的矿泉水”；
- “去看看床头柜上的药盒还在不在”；
- “把新到货样品从仓库取来”。

这里导航只是语义任务执行链中的一环。

---

### 8.2 navigation-to-action handoff

当前的大痛点，不是能否到目标附近，而是：

- 何时停止导航；
- 何时切换到抓取 / 交互；
- 如何从“靠近目标”变成“处于可操作位姿”；
- 失败后如何恢复与重规划。

这是兼具研究价值与产业价值的接口层问题。

---

### 8.3 强空间记忆与长期任务执行

真实具身系统需要：

- landmark graph；
- object-centric memory；
- instruction progress tracking；
- 错误恢复与回退策略；
- 跨时间任务状态管理。

当前多模态大模型在这些方面仍明显不够稳定。

---

### 8.4 动态环境中的鲁棒导航

谁能把 VLN 从静态环境刷题，提升为动态开放环境中的稳定运行，谁就会有明显优势。典型场景包括：

- 人流密集零售环境；
- 产线临时堆放环境；
- 家庭宠物/儿童干扰；
- 多机器人协同空间。

---

### 8.5 sim2real 的可验证闭环

银河通用正在押这条路线，但整个行业都还没完全做透：

- 合成数据如何做到足够“语义真实”；
- 哪些能力可以迁移，哪些必须真机校准；
- 最小真机校准集如何设计；
- 如何量化 sim2real gap。

---

## 9. 研究机会地图

### 一级机会：高价值、高缺口、值得长期投入

#### 9.1 行动就绪导航（action-ready navigation）
不是到目标点，而是到达能执行下一步动作的最佳观察位姿/交互位姿。

**关键词：**
- affordance-aware waypointing
- task-conditioned viewpoint selection
- navigation-manipulation co-training

#### 9.2 开放世界与动态变化下的可恢复导航
当地标失效、路被挡、目标变化、指令模糊时，agent 如何感知不确定性、主动澄清、恢复任务。

**关键词：**
- recovery-aware VLN
- uncertainty estimation
- interactive grounding
- dynamic scene language navigation

#### 9.3 长时记忆驱动的空间智能体
机器人如何积累对环境、任务和偏好的长期记忆，并用于未来导航与执行。

**关键词：**
- episodic spatial memory
- lifelong navigation
- environment operational memory

---

### 二级机会：价值明确，适合切入，但需选对场景

- 低成本、弱地图、快速部署导航；
- 人机协同导航与可解释接管；
- 多模态合成与真实日志驱动的数据引擎；
- 面向真实部署的 VLN 评测升级。

---

### 三级机会：容易做，但信息增量偏低

- 仅在经典 VLN 数据集上继续刷 SOTA；
- 仅把更大 backbone 套进传统 VLN pipeline；
- 只做单轮导航语言理解增强。

---

## 10. 产业切入地图

### 10.1 最值得切入的不是“卖 VLN”，而是卖“任务完成能力中的导航层”

#### 切入点 1：移动操作机器人中的 pre-action navigation
**场景**：仓储拣选、机房巡检、零售补货、楼宇服务。  
**价值**：把“到点成功率”转成“任务闭环成功率”。

#### 切入点 2：弱地图 / 免建图快速部署方案
**场景**：中小商场、酒店、办公区、临时展馆、轻工业现场。  
**价值**：缩短部署周期，降低售前和实施成本。

#### 切入点 3：动态环境中的自主恢复与远程接管协同
**场景**：医院、园区、写字楼、配送机器人。  
**价值**：降低卡死率、人工值守压力和安全事故率。

#### 切入点 4：空间记忆与多站点运营 intelligence
**场景**：连锁零售、连锁酒店、园区运维、仓储网络。  
**价值**：把一次性算法能力变成长期数据资产。

#### 切入点 5：安全合规导航中间件
**场景**：医院、机场、商场、工厂、公共空间。  
**价值**：帮助机器人从“能演示”走向“能商用”。

### 10.2 国内外相关企业补充（截至 2026-03-31）

如果按本文口径筛选，真正值得关注的不是“纯 VLN 公司名录”，而是那些把 VLN / 导航能力吸收到 **VLA、具身导航系统、移动操作和服务闭环** 中的企业。大致可以分成两类：

- **一类是 VLA / embodied navigation 证据较强的公司**；
- **另一类是导航商业化落地较强、但语言-动作闭环相对弱一些的公司**。

#### 国内

- **银河通用（Galbot）**：与本文主题最贴近。根据前文已核验公开事实，其技术路线同时覆盖 **NaVid、TrackVLA、GraspVLA**，说明它并不是单做路线跟随，而是在把导航、跟踪、抓取和合成数据训练整合进一条具身模型栈中。若只选一家最值得持续跟踪的国内 VLN 相关公司，银河通用仍是优先级最高的样本。
- **智元机器人（AGIBOT）**：更偏通用 embodied AI / VLA，而不是纯导航公司。官网公开其“**Embodiment + AI**”全栈平台；研究页给出 **GO-1 ViLLA foundation model、1M+ trajectories、100+ robots、1000+ task scenarios** 等信息。它的意义在于：把语言-视觉-动作能力和大规模机器人数据工厂、量产机器人平台结合起来，代表“VLN 被吸收到通用具身平台”的路线。
- **普渡机器人（PUDU）**：更偏“导航商业化 + 具身服务机器人”。官网写明其核心技术覆盖 **mobility、manipulation、AI** 三项；FlashBot Arm 页面则显示其具备 **embodied AI、自主任务执行、环境识别、任务理解、VSLAM + LiDAR 导航、多机器人协同** 等能力。它不是学术意义上的 VLN 公司，但很适合作为“把导航变成服务机器人基础设施”的产业样本。
- **擎朗智能（KEENON）**：从自主配送导航起家，向具身服务 / VLA 升级。官网 about 页显示其 **2010 年成立**，并在 **2016 年** 推出 **world’s first autonomous delivery robot**；官方新闻页在 **2025-09-26** 发布 **KOM 2.0**，将其定义为面向服务业的自研 VLA 模型。它的代表性在于：从商用配送 / 酒店 / 清洁机器人积累的导航部署经验，向“感知-理解-行动”一体化模型演进。

#### 国外

- **Figure**：典型的“VLA 吸收 VLN”路线。官方 **Helix** 页面把它定义为 **generalist VLA**；Figure 03 / 官网又明确强调其在 home environment 中依赖 Helix 处理 household tasks，并能适应楼梯、转角和动态家庭布局。它不是“纯导航公司”，但很适合作为“语言理解 + 空间移动 + 操作一体化”的头部样本。
- **1X**：其 **Redwood AI** 明确强调 home setting 下的 **mobile manipulation**。官方写到它可执行 **retrieving objects、opening doors、navigating around the home**，并称其是较早把 **locomotion 和 manipulation jointly control** 的 VLA 之一。它非常接近本文所说的“action-ready navigation”。
- **Physical Intelligence（PI）**：其 **π0 / π0.5** 是当前最典型的 VLA / physical intelligence 路线之一。官方写到 π0 是 **general-purpose robot foundation model**，π0.5 则强调对 **entirely new environments** 的 generalization，可在全新家居环境中完成 kitchen / bedroom cleanup 一类任务。它的重要性在于：把语言、空间、操作、恢复统一到一个通用策略里。
- **Skild AI**：明确走“**omni-bodied brain**”路线。官网直接写到其目标是用一个统一大脑控制任意机器人完成任意任务，并列出 **Security/Inspection Robot Platform** 与 **Mobile Manipulation Platform**，其中明确包含 **navigation** 技能。它代表的是“跨本体导航 / 操作基础模型”路径。
- **ANYbotics**：更偏“导航商业化落地”而非 VLA 叙事。ANYmal 官方页面强调 **AI-based mobility and autonomy**、复杂多层工业设施导航、360° LiDAR 本地化与自动 docking。虽然它不属于纯 VLN，但在“真实复杂环境中的鲁棒导航和持续运营”上，非常值得作为海外产业对照样本。

#### 一句话筛选建议

- 如果你要找**和本文口径最一致**的企业，国内优先看 **银河通用**，国外优先看 **Figure、1X、Physical Intelligence、Skild AI**。
- 如果你要找**导航先行、商业闭环更强**的企业，国内可重点看 **PUDU、KEENON**，国外可重点看 **ANYbotics**。

---

## 11. 对银河通用的总判断

一句话总结：

> **银河通用代表的不是“VLN 赛道突破”，而是“VLN 被吸收到具身大模型与场景闭环里”的产业化路径。**

### 它已经证明的

- 导航必须和抓取 / 作业一起看；
- 合成数据可以成为具身训练的重要杠杆；
- 在零售、工业等方向，具身系统可以从 demo 走向真实场景验证；
- 导航必须工程化，而不能只停留在 benchmark。

### 它还没证明完的

- 是否具备低成本跨场景泛化；
- 公开效果中模型能力与场景工程各自贡献有多大；
- 是否能在开放长尾环境中保持稳定；
- 是否能真正形成通用 embodied intelligence，而非高质量场景特化。

---

## 12. 结论

**VLN 相对 VLA 不再是热点，不是因为导航没用了，而是因为：**

1. 它的主问题定义太窄，越来越像系统组件；
2. 其 benchmark 红利与语言 grounding 红利被消耗，并部分被大模型吸收；
3. 产业买单的是任务闭环、部署效率和安全可靠，而不是单点导航能力；
4. VLA 更能承接“通用具身智能平台”这一时代叙事。

但与此同时，**VLN 最有价值的部分并未消失，而是转移到了几个新的高价值接口上：**

- 导航—操作衔接；
- 动态开放环境恢复；
- 长期空间记忆；
- 弱地图低成本部署；
- 安全可控与人机协同。

如果一句话总结机会：

> **不要再把 VLN 当作一个孤立 benchmark 问题去追；要把它当作“真实世界具身系统中的空间智能底座”，在闭环任务、长期记忆、动态恢复和低成本部署上重新定义价值。**

---

## 13. 参考资料与链接

### 综述与背景
- Wang et al., *Vision-Language Navigation with Embodied Intelligence: A Survey* (2024): https://arxiv.org/html/2402.14304v1
- Shah et al., *LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action* (CoRL 2023): https://proceedings.mlr.press/v205/shah23b.html
- HomeRobot / OVMM: https://ovmm.github.io/
- EmbodiedBench: https://embodiedbench.github.io/

### 纯 VLN / 导航核心方向
- AgentVLN: https://arxiv.org/abs/2603.17670
- OmniVLN: https://arxiv.org/abs/2603.17351
- DyGeoVLN: https://arxiv.org/abs/2603.21269
- SOL-Nav: https://arxiv.org/abs/2603.27577
- SignNav: https://arxiv.org/abs/2603.16166

### benchmark / system / embodied navigation
- NavTrust: https://arxiv.org/abs/2603.19229
- NavTrust project: https://navtrust.github.io/
- CeRLP: https://arxiv.org/abs/2603.19602
- HUGE-Bench: https://arxiv.org/abs/2603.19822
- CityWalker: https://ai4ce.github.io/CityWalker/

### VLA / embodied agent / mobile manipulation
- OpenVLA paper: https://arxiv.org/abs/2406.09246
- OpenVLA project: https://openvla.github.io/
- Octo paper: https://arxiv.org/abs/2405.12213
- Octo project: https://octo-models.github.io/
- π0: https://www.physicalintelligence.company/blog/pi0

### 国内外企业补充
- AGIBOT about: https://www.agibot.com/about/210.html
- AGIBOT Research: https://www.agibot.com/research/
- PUDU about: https://www.pudurobotics.com/company
- PUDU FlashBot Arm: https://www.pudurobotics.com/en/products/flashbot-arm
- KEENON about: https://www.keenon.com/en/about/index.html
- KEENON news: https://www.keenon.com/en/news/index.html
- Figure company: https://www.figure.ai/company
- Figure 03: https://www.figure.ai/figure
- Figure Helix: https://www.figure.ai/news/helix
- 1X AI: https://www.1x.tech/ai
- 1X Redwood AI: https://www.1x.tech/ja_jp/discover/redwood-ai
- Physical Intelligence home: https://www.pi.website/
- π0.5: https://www.pi.website/blog/pi05
- Skild AI: https://www.skild.ai/
- ANYmal: https://www.anybotics.com/robotics/anymal/

### 银河通用相关
- 银河通用官网: https://www.galbot.com/
- 银河通用新闻页: https://www.galbot.com/news
- GraspVLA paper: https://arxiv.org/abs/2505.03233
- GraspVLA project: https://pku-epic.github.io/GraspVLA-web/
- TrackVLA paper: https://arxiv.org/abs/2505.23189
- TrackVLA project: https://pku-epic.github.io/TrackVLA-web
- IT之家（GraspVLA）: https://www.ithome.com/0/823/777.htm
- IT之家（TrackVLA）: https://www.ithome.com/0/857/593.htm
- IT之家（智慧药房）: https://www.ithome.com/0/795/720.htm
