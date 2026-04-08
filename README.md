# Research Archive

本目录汇总家庭语义记忆、VLN/VLA 导航、板端双目深度部署，以及 VLM 模型对比与评测相关研究文档。各子目录使用编号前缀保证排序稳定，后续可继续按 `07_...`、`08_...` 扩展。

本文档基于当前目录内现有材料整理，目标是让根目录 `README` 更像一页研究导航，而不只是文件清单。

## 当前主线速览

- **家庭空间表示主线**：方向已经从“寻找几何地图替代品”转向“构建分层、异构、可检索、可纠错的家庭语义记忆系统”；拓扑结构仍然重要，但更适合作为中间骨架而不是最终目标。
- **VLN / VLA 主线**：VLN 没有失去价值，但更像基础设施能力；真正的机会在把导航嵌回 agent loop、长期任务、识别闭环和真实部署链路。
- **端侧视觉主线**：板端双目深度估计优先走 classical stereo 基线，神经网络方案作为增强项；VLM 选型则需要分开看“通用 benchmark 表现”和“导航目标检测任务表现”。
- **当前模型选型结论**：从现有文档看，Qwen3-VL 更适合高吞吐通用视觉理解，Qwen3.5 更适合复杂多步推理；在当前 `homeobjects` 检测任务和单 4B 部署约束下，文档结论偏向 `Qwen3.5-4B (no-thinking)` 作为部署候选，`Qwen3.5-27B` 作为 teacher 路线。

## 按问题快速导航

- 如果你想看“家庭场景为什么不该再把几何地图当主表示”，先读 `01_home_semantic_memory/home-memory-representation-final.md`。
- 如果你想看“家庭语义记忆系统到底怎么分层设计”，先读 `01_home_semantic_memory/home-semantic-memory-architecture.md`。
- 如果你想看“显式记忆如何支持查询、训练和模型注入”，先读 `01_home_semantic_memory/memory-query-and-model-ingestion.md`。
- 如果你想看“VLN 为什么降温、VLA 为什么更热”，先读 `02_vln_vla_navigation/vln_vla_gap_analysis.md` 和 `02_vln_vla_navigation/subagent_gap_map.md`。
- 如果你想看“ABot-N0 对通用具身导航意味着什么”，先读 `02_vln_vla_navigation/abot_n0_vla_navigation_technical_report.md`，再对照 `02_vln_vla_navigation/2602.11598v1.pdf`。
- 如果你想看“导航目标识别如何做 teacher-student 微调和蒸馏”，先读 `02_vln_vla_navigation/vlm_detection_finetune_distillation_plan.md`，再看 `06_model_compare/vlm_candidate_evaluation_for_detection.md`。
- 如果你想看“板端双目到底该怎么选”，先读 `03_stereo_depth_deployment/stereo_depth_board_deployment_survey.md` 和 `03_stereo_depth_deployment/subagent_stereo_selection.md`。
- 如果你想看“Qwen3.5 和 Qwen3-VL 架构差异是什么”，先读 `04_model_architecture_compare/qwen35_vs_qwen3vl_architecture_diff.md`。
- 如果你想看“当前 4B/8B 模型在通用 VLM 评测和 homeobjects 导航任务上的表现”，直接看 `06_model_compare/`。

## 目录索引

### 01_home_semantic_memory

主题：家庭场景非几何地图、语义记忆、拓扑结构与用户可解释展示。

这一组文档已经形成比较清晰的方向收敛：

- 不是做“一张更好的地图”，而是做一套面向记忆、检索、推理与用户纠错的空间表示系统。
- 拓扑结构不应被完全拿掉，而应作为房间级组织、跨房间连接、对象落位和草图生成的中间骨架。
- 显式记忆库需要同时服务在线推理、历史查询、teacher 数据生产和 student 蒸馏。

推荐入口：

- `home-memory-representation-final.md`：最终版方向判断，明确从“地图问题”转向“语义记忆系统问题”。
- `home-semantic-memory-architecture.md`：分层架构设计，补齐对象身份、版本一致性、用户纠错和展示可信边界。
- `memory-query-and-model-ingestion.md`：围绕查询效率和模型摄入方式，衔接 `27B teacher + 4B student` 的长期路线。

文件说明：

- `home-semantic-map-directions-draft.md`：最初版本的问题定义与约束整理。
- `map-partition-semantic-analysis.md`：从地图分区 PDF 出发，分析几何分区路线与语义地图演进方向。
- `non-geometric-representation-research.md`：非传统几何空间表示调研，重点在家居记忆与用户可视化。
- `non-geometric-map-review-checklist.md`：用于评审“非传统几何地图”方案是否讲清问题、边界和价值。
- `topology-algorithm-gap-check.md`：纠偏文档，强调拓扑算法内容不应在转向语义记忆后被过度弱化。
- `home-semantic-memory-architecture.drawio`：家庭语义记忆系统架构图源文件。
- `地图分区及房间语义地图演进.pdf`：相关原始资料。

### 02_vln_vla_navigation

主题：VLN / VLA / embodied navigation 机会差距、产业路线、论文拆解，以及导航目标识别微调方案。

这一组文档的核心共识是：

- VLN 更像导航基础设施，而不是最强平台叙事本身。
- 更值得投入的方向是 agentic navigation、长期任务闭环、识别与导航联合建模、低成本部署与真实运营证据。
- ABot-N0 和后续方案文档把“统一具身导航模型”进一步落到 teacher-student 检测微调和蒸馏设计上。

推荐入口：

- `vln_vla_gap_analysis.md`：整组导航研究的总览文档，覆盖 VLN 进展、银河通用、VLA 前沿与机会差距。
- `abot_n0_vla_navigation_technical_report.md`：ABot-N0 技术报告拆解。
- `vlm_detection_finetune_distillation_plan.md`：面向导航目标识别的微调与蒸馏方案设计稿。

文件说明：

- `subagent_gap_map.md`：从研究与产业结合角度解释“VLN 为何不再是热点”。
- `subagent_vln_frontier.md`：补充 2024-2026 年值得写进综述的 VLN / VLA / embodied navigation 前沿工作。
- `subagent_galaxybot.md`：银河通用公开资料整理，区分可验证事实与宣传口径。
- `2602.11598v1.pdf`：ABot-N0 原始论文 PDF。

### 03_stereo_depth_deployment

主题：双目深度估计在板端 / 边缘设备的部署约束、模型路线与工程选型。

这部分文档的结论比较一致：

- 如果目标是稳定可量产、时延可控、功耗可控，优先把 `classical stereo` 当成基线。
- 只有在低纹理、重复纹理、弱光、反光、跨域鲁棒性等问题成为明确瓶颈时，再考虑神经双目。
- 神经双目路线里，优先关注轻量模型与工具链可支持性，而不是只看论文榜单。

推荐入口：

- `stereo_depth_board_deployment_survey.md`：总览版调研，覆盖技术路线、平台约束与落地建议。
- `subagent_stereo_selection.md`：从产品 / 机器人视角给出的选型建议。

文件说明：

- `subagent_stereo_hardware.md`：从输入链路、功耗、带宽、算子支持等角度梳理板端约束。
- `subagent_stereo_models.md`：按算法与部署可行性对主流 stereo 模型做系统比较。

### 04_model_architecture_compare

主题：多模态模型架构差异与适用场景分析。

文件说明：

- `qwen35_vs_qwen3vl_architecture_diff.md`：结合本机适配代码与评测结果，对比 Qwen3.5 与 Qwen3-VL 在模型定位、输入处理、生成行为和适用任务上的差异。当前文档结论是：Qwen3.5 更偏原生多模态生成与复杂推理，Qwen3-VL 更偏高吞吐视觉理解与 benchmark 型任务。

### 05_history_search_misc

主题：历史检索或一次性排查记录。

文件说明：

- `claude-history-search.md`：对 Claude Code `history.jsonl` 的关键词检索记录，目前结果为 0 命中。

### 06_model_compare

主题：VLM 候选模型评测、benchmark 汇总，以及面向导航目标检测的模型选型结论。

这一目录主要分成三类材料：

- **任务型评测**：`test_report_*.md`，基于 `homeobjects_mini_v1`，覆盖开集目标检测、指代目标定位、几何关系、房间类型分类。
- **通用 benchmark 评测**：`vlm_eval_*.md`，主要是 MMBench-EN (Dev) 结果。
- **选型结论文档**：`vlm_candidate_evaluation_for_detection.md`，把通用 benchmark、任务实测和部署约束汇总为最终建议。

推荐入口：

- `vlm_candidate_evaluation_for_detection.md`：如果只看一篇，先看这一篇。文中把 teacher / student 角色、单 4B 部署约束、thinking 与 no-thinking 差异都讲清楚了。
- `test_report_nothinking.md`：当前最关键的部署实测之一，显示 `Qwen3.5-4B` 在 `no-thinking` 模式下对检测任务更有优势。
- `vlm_eval_report_qwen3-vl-4b.md`、`vlm_eval_8b_report.md`、`vlm_eval_qwen35_4b_report.md`：用于对比通用视觉 benchmark 表现。

文件说明：

- `test_report_qwen_4b.md`：Qwen3-VL-4B 的 `homeobjects_mini_v1` 测试报告。
- `test_report_qwen3_8b.md`：Qwen3-VL-8B 的 `homeobjects_mini_v1` 测试报告。
- `test_report_qwen35_2b.md`：Qwen3.5-2B 的 `homeobjects_mini_v1` 测试报告。
- `test_report_qwen35_4b.md`：Qwen3.5-4B thinking 模式测试报告。
- `test_report_nothinking.md`：Qwen3.5-4B no-thinking 模式测试报告。
- `vlm_eval_qwen35_2b_report.md`：Qwen3.5-2B 的 MMBench 报告。
- `vlm_eval_qwen35_4b_report.md`：Qwen3.5-4B 的 MMBench 报告。
- `vlm_eval_qwen35_4b_nothinking_report.md`：Qwen3.5-4B no-thinking 的 MMBench 报告。
- `vlm_eval_report_qwen3-vl-4b.md`：Qwen3-VL-4B 的 MMBench 报告。
- `vlm_eval_8b_report.md`：Qwen3-VL-8B 的 MMBench 报告。

## 维护建议

- 新增文档时，优先保持“一个目录对应一条研究主线”的组织方式。
- 如果某条主线已经从调研转向方案设计，建议在对应目录内明确标注 `draft`、`final`、`evaluation report` 等状态。
- 若后续继续扩展 `06_model_compare`，建议把“任务型评测”和“通用 benchmark”继续区分命名，便于快速定位。
