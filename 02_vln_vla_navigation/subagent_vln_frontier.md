## 前沿进展补充

下面这部分聚焦 **2024–2026 年间值得补进综述的 VLN / VLA / embodied navigation 前沿内容**。为了避免“只列论文名”，我按 **任务属性** 做了拆分：哪些属于**纯 VLN**，哪些已经明显跨到 **VLA / embodied agent / mobile manipulation**，并尽量给出一句可核验的贡献判断与公开链接。

### 一、先给一个分类判断

- **纯 VLN（Vision-Language Navigation）**：输入核心仍是“自然语言导航指令 + 视觉观测”，目标是完成路线跟随、目标搜索或语义导航，本质仍是导航问题。
- **VLN 扩展到 embodied navigation / agentic navigation**：仍以导航为主，但开始引入更强的 3D 表征、主动探索、鲁棒性评测、跨平台部署、真实机器人闭环等，已经不只是传统 R2R/RxR 范式。
- **VLA / embodied agent / mobile manipulation**：语言不仅决定“往哪走”，还决定“做什么动作序列”；任务经常包含抓取、整理、开关、搬运、桌面操作、移动操作臂等，导航只是更长时程具身任务中的一个子环节。

---

### 二、纯 VLN / 以导航为核心的代表工作

#### 1. AgentVLN（2026）
- **类型**：偏纯 VLN，但明显带有 agentic/navigation system 色彩。
- **核心内容**：把 VLN 建模为 POSMDP，用 **VLM-as-Brain + skill library** 的方式，把高层语义推理和底层感知/规划解耦；同时做了 **2D-3D 表征桥接**、**自纠错** 和 **主动探索**。
- **为什么值得写进综述**：它代表了 VLN 从“端到端跟路”走向“有策略地探索、纠错、查询几何信息”的趋势，说明 2026 年前沿已经不满足于单纯 instruction following，而是开始强调 **agentic decision-making**。
- **一句判断**：如果说早期 VLN 解决的是“听懂路线”，那么 AgentVLN 解决的是“在不确定环境里怎么像智能体一样把路线执行下去”。

#### 2. OmniVLN（2026）
- **类型**：VLN / embodied navigation system。
- **核心内容**：面向空地平台（air + ground）的零样本视觉语言导航框架，强调 **全向感知 + 3D Dynamic Scene Graph + token-efficient LLM reasoning**。
- **为什么值得写进综述**：它说明 VLN 已经从常见的单目/窄视场 indoor setting，扩展到 **跨平台、全向感知、分层空间推理**；尤其“**DSG + LLM**”这一组合，是近年具身导航很值得跟踪的方向。
- **一句判断**：OmniVLN 的价值不只是“导航成功率更高”，而是把大模型推理真正接到了可压缩的 3D 场景图上，缓解了上下文爆炸问题。

#### 3. DyGeoVLN（2026）
- **类型**：纯 VLN。
- **核心内容**：把 **dynamic geometry foundation model** 注入 VLN 框架，解决传统 VLN 假设静态场景、难适应动态真实环境的问题；还提出了 **pose-free、自适应分辨率 token pruning**。
- **为什么值得写进综述**：这个工作很能代表“**从静态 VLN 走向动态 VLN**”的趋势。过去很多 VLN benchmark 默认环境稳定，但真实世界的人、门、家具、遮挡都在变。
- **一句判断**：DyGeoVLN 的代表性在于，它把“几何变化”从噪声变成建模对象，向真实场景 VLN 更近了一步。

#### 4. SOL-Nav / Structured Observation Language for Navigation（2026）
- **类型**：纯 VLN。
- **核心内容**：把 RGB-D 视觉观测转换为 **结构化文本描述**，再和导航指令拼接后喂给纯语言模型，尽量避免重型视觉 token 融合。
- **为什么值得写进综述**：这类工作代表了另一条路线：不是继续堆更大的视觉编码器，而是把导航观测“语言化 / 符号化”，借助 PLM 的推理能力获得更轻量、可泛化的 VLN。
- **一句判断**：SOL-Nav 很值得在综述里作为“**VLN 的语言化接口趋势**”举例，它说明并非所有 VLN 都必须依赖大规模多模态端到端融合。

#### 5. SignNav（2026）
- **类型**：纯导航任务，但比传统 VLN 更贴近真实大型室内环境。
- **核心内容**：引入 **基于标识牌（signage）的语义导航**；构建 LSI-Dataset，并提出 START 模型处理空间 grounding 与时间依赖。
- **为什么值得写进综述**：医院、机场、商场这类大场景里，人类导航强依赖标识牌，而传统 VLN 很少显式建模这类“环境提供的外部语言线索”。
- **一句判断**：SignNav 的贡献不是单纯多一个 benchmark，而是把“**看懂环境里的文字/指示牌**”正式拉进了具身导航主线。

---

### 三、benchmark / system：把 VLN 推向“更真实、更鲁棒、更可部署”

#### 6. NavTrust（2026）
- **类型**：embodied navigation benchmark（覆盖 VLN 与 OGN，不是纯 VLN 算法论文）。
- **核心内容**：系统性地在 RGB、depth、instruction 上引入现实腐蚀与扰动，统一评估 embodied navigation 的 **trustworthiness / robustness**。
- **为什么值得写进综述**：过去很多导航论文只在 nominal setting 下比 success rate，但真实部署真正会遇到传感器噪声、模糊、错误指令、缺词、遮挡等问题。NavTrust 把这些问题 benchmark 化了。
- **一句判断**：NavTrust 代表着评价标准的变化——研究重点开始从“谁在干净测试集上分数更高”转向“谁在脏环境里更可靠”。

#### 7. CeRLP（2026）
- **类型**：visual navigation / embodied navigation system，可服务 VLN 场景。
- **核心内容**：面向 **cross-embodiment** 机器人局部规划，显式建模机器人几何差异、相机参数差异、单目深度尺度歧义，并把视觉输入统一抽象为 height-adaptive laser scans。
- **为什么值得写进综述**：这是“导航系统工程化”很典型的一篇。很多导航方法在单一机器人上有效，但换底盘、换相机就掉。CeRLP 直接把“跨本体迁移”当作核心问题。
- **一句判断**：CeRLP 的价值在于提醒我们，真实导航泛化不只是换场景泛化，还包括 **换机器人本体的泛化**。

#### 8. HUGE-Bench（2026）
- **类型**：高层 UAV Vision-Language-Action benchmark。
- **核心内容**：面向无人机高层语言动作任务，强调 **brief high-level commands**、多阶段行为、安全约束、collision-aware evaluation；采用 **3DGS-Mesh** 数字孪生表征。
- **为什么值得写进综述**：它很适合拿来说明 VLN 与 VLA 的边界开始模糊：任务不再只是“飞到哪里”，而是“按高层语义要求完成一段安全、过程正确的空中行动”。
- **一句判断**：HUGE-Bench 不是传统 UAV-VLN 的延长线，而更像是“高层具身飞行智能”的诊断基准。

#### 9. CityWalker（CVPR 2025）
- **类型**：embodied urban navigation，偏导航，不是 manipulation。
- **核心内容**：从 **web-scale videos** 学习城市导航，目标是把具身导航从室内模拟器进一步推向开放城市环境。
- **为什么值得写进综述**：VLN/embodied navigation 长期过度依赖室内、离散视点、模拟环境；CityWalker 代表“**开放世界、城市级、从互联网视频学导航**”这条很新的路线。
- **一句判断**：如果室内 VLN 解决的是“室内看图走路”，CityWalker 更像在探索“能不能从海量人类城市视频里蒸馏真实世界导航常识”。

---

### 四、VLA / embodied agent / mobile manipulation：导航不再是终点，而是长时程具身任务的一部分

#### 10. OpenVLA（2024）
- **类型**：典型 **VLA（Vision-Language-Action）**，不是纯 VLN。
- **核心内容**：7B 开源 VLA，基于大规模视觉语言预训练和 **97 万条真实机器人 demonstrations**，支持高效微调、跨任务泛化与消费级 GPU 微调。
- **为什么值得写进综述**：OpenVLA 是 2024 年之后讨论 VLA 几乎绕不过去的开源基线。虽然主打 manipulation，但它对“语言驱动具身行动”的范式影响很大，也会反过来影响导航研究的建模方式。
- **一句判断**：OpenVLA 的代表性不在“是不是导航模型”，而在于它把“视觉-语言-动作统一建模”真正做成了可复现、可微调、可比较的开放基座。

#### 11. π0（Physical Intelligence, 2024）
- **类型**：VLA / generalist robot policy，偏 mobile manipulation 与长时程操作，不是纯 VLN。
- **核心内容**：将大规模多机器人、多任务数据与新架构结合，支持高频连续控制；展示了 **folding laundry、table bussing、build box** 等复杂长时程任务。
- **为什么值得写进综述**：π0 说明具身智能前沿已经明显超出“导航到目标点”——模型需要处理多阶段规划、操作恢复、策略切换、移动与操作结合。
- **一句判断**：π0 更像“机器人版 foundation policy”的雏形，其意义在于把语言理解、视觉感知、连续控制和长时程任务放进了一个统一叙事里。

#### 12. Octo（2024）
- **类型**：generalist robot policy / VLA 前身路线，偏 manipulation。
- **核心内容**：基于 Open X-Embodiment 的 **80 万轨迹** 训练的开源通用机器人策略，可用语言或 goal image 指令，并能快速适配新观察空间和动作空间。
- **为什么值得写进综述**：虽然 Octo 更偏 manipulation，但它与 OpenVLA 一起构成了 2024 年“开源通用机器人基础策略”的核心背景，很多后续 VLA/agent 工作都在沿着这条线扩展。
- **一句判断**：Octo 的关键贡献是把“多本体、多任务、可微调”的机器人通用策略做成了真正可用的开放基座。

---

### 五、写综述时建议怎么组织这部分

如果主文档主线是 **VLN**，我建议把这批工作分成三层写，而不是混成一个列表：

1. **VLN 本体仍在演化**：
   - 从静态到动态：DyGeoVLN
   - 从端到端到 agentic：AgentVLN
   - 从视觉 token 堆叠到结构化语言接口：SOL-Nav
   - 从室内目标跟随到语义线索导航：SignNav

2. **评测和系统正在逼近真实世界**：
   - 鲁棒性/可信性：NavTrust
   - 跨本体真实部署：CeRLP
   - 高层飞行任务与安全：HUGE-Bench
   - 城市开放环境：CityWalker

3. **VLN 与 VLA/具身智能边界正在模糊**：
   - OpenVLA / Octo / π0 表明，导航越来越像长时程具身任务中的子能力；
   - 未来很多“导航论文”未必只评估 route following，而会评估 **搜索、交互、操作、恢复、任务完成度**。

一个可以直接塞进综述的概括句是：

> 2024–2026 年间，VLN 的前沿趋势不再只是提高 R2R/RxR 上的 success rate，而是明显转向动态场景、主动探索、结构化 3D 认知、真实部署鲁棒性，以及与 VLA/通用具身智能的逐步融合；导航正在从独立任务，转变为长时程具身任务中的基础能力。

## 参考链接

### 纯 VLN / 导航核心方向
- AgentVLN (arXiv 2603.17670): https://arxiv.org/abs/2603.17670
- OmniVLN (arXiv 2603.17351): https://arxiv.org/abs/2603.17351
- DyGeoVLN (arXiv 2603.21269): https://arxiv.org/abs/2603.21269
- Structured Observation Language for Navigation / SOL-Nav (arXiv 2603.27577): https://arxiv.org/abs/2603.27577
- SignNav (arXiv 2603.16166): https://arxiv.org/abs/2603.16166

### Benchmark / system / real-world embodied navigation
- NavTrust (arXiv 2603.19229): https://arxiv.org/abs/2603.19229
- NavTrust project: https://navtrust.github.io/
- CeRLP (arXiv 2603.19602): https://arxiv.org/abs/2603.19602
- HUGE-Bench (arXiv 2603.19822): https://arxiv.org/abs/2603.19822
- CityWalker project: https://ai4ce.github.io/CityWalker/

### VLA / embodied agent / mobile manipulation
- OpenVLA paper (arXiv 2406.09246): https://arxiv.org/abs/2406.09246
- OpenVLA project: https://openvla.github.io/
- Octo paper (arXiv 2405.12213): https://arxiv.org/abs/2405.12213
- Octo project: https://octo-models.github.io/
- π0 / Our First Generalist Policy: https://www.physicalintelligence.company/blog/pi0

### 可作为背景的早期导航系统脉络
- VALAN (Google Research): https://google-research.github.io/valan/
