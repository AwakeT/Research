# Qwen3.5 与 Qwen3-VL 架构差异整理

**日期：** 2026-03-28  
**目的：** 结合本机已有评测与适配代码，整理 Qwen3.5 与 Qwen3-VL 在模型定位、推理链路、输入处理、生成行为与适用场景上的核心差异。

---

## 1. 一句话结论

- **Qwen3.5** 更像是一个 **原生多模态生成模型**，强调通用生成能力与显式 thinking 推理，适合复杂视觉推理、结构化生成和长输出场景。
- **Qwen3-VL** 更像是一个 **针对视觉理解任务优化的专用视觉语言模型（VL）**，强调短答、稳定、高吞吐和 benchmark/在线推理效率。

所以：
- **Qwen3.5 的强项** 在“需要想一想”的多步视觉推理；
- **Qwen3-VL 的强项** 在“看完直接答”的视觉理解与评测效率。

---

## 2. 本机可直接引用的证据来源

### 2.1 模型适配代码
- `/home/zktian3/lmms-eval/lmms_eval/models/simple/qwen3_5.py`
- `/home/zktian3/lmms-eval/lmms_eval/models/simple/qwen3_vl.py`

### 2.2 本机评测汇总
- `/home/zktian3/lmms-eval/vlm_eval_final_report.md`
- `/home/zktian3/lmms-eval/vlm_eval_mmbench_overall_report.md`
- `/home/zktian3/lmms-eval/vlm_eval_qwen35_4b_report.md`
- `/home/zktian3/lmms-eval/vlm_eval_qwen35_4b_nothinking_report.md`
- `/home/zktian3/lmms-eval/vlm_eval_4b_report.md`
- `/home/zktian3/lmms-eval/vlm_eval_8b_report.md`

### 2.3 原始结果文件
- `/home/zktian3/lmms-eval/results_qwen3_vl_8b/Qwen3-VL-8B/20260320_145138_results.json`
- `/home/zktian3/lmms-eval/results_qwen3_vl_4b/Qwen3-VL-4B/20260320_140728_results.json`
- `/home/zktian3/lmms-eval/results_qwen35_4b/Qwen3.5-4B/20260320_154215_results.json`
- `/home/zktian3/lmms-eval/results_qwen35_4b_no_thinking/Qwen3.5-4B/20260326_100956_results.json`
- `/home/zktian3/lmms-eval/results_qwen35_2b/Qwen3.5-2B/20260320_153226_results.json`

---

## 3. 架构定位差异

## 3.1 Qwen3.5：原生多模态生成 + thinking 驱动

从本机适配代码可见，Qwen3.5 通过以下类加载：

- `Qwen3_5ForConditionalGeneration`
- `Qwen3_5MoeForConditionalGeneration`

这说明在当前本机推理链路里，它被当作 **conditional generation（条件生成）模型** 使用，而不是一个只为短答视觉问答做窄优化的模型。

更关键的是，它显式支持：

- `enable_thinking`

并且在生成后，会对回答内容进行 `</think>` 分割，抽取真正输出答案。

这说明它的推理过程是：

1. 接收图像/视频 + 文本输入；
2. 允许模型先生成内部思考内容；
3. 再从最终输出中剥离思考部分，保留对外答案。

这种机制带来的直接结果是：
- 在复杂视觉推理题上更容易“想明白”；
- 在空间关系、属性比较、跨实例理解等题上更可能拿到更高分；
- 但会显著增加 token 开销和时延。

---

## 3.2 Qwen3-VL：面向视觉任务优化的专用 VL 架构

Qwen3-VL 在本机适配代码中通过以下类加载：

- `Qwen3VLForConditionalGeneration`
- `Qwen3VLMoeForConditionalGeneration`

虽然它也属于生成式接口，但从适配参数和测试行为上看，它明显更偏向：

- 图像理解
- OCR / 场景识别
- 多选题 / 短回答
- benchmark 型任务

它在当前本机代码里没有像 Qwen3.5 那样直接暴露 `enable_thinking` 这种显式思考开关；默认生成长度也明显更短。这说明它更像是一个 **为视觉理解效率和稳定输出优化过的 VL 模型分支**。

---

## 4. 输入处理链路差异

两者都使用：
- `AutoProcessor`
- `qwen_vl_utils.process_vision_info`

来处理图像/视频输入。

但它们的视觉预算和处理策略不同。

### 4.1 Qwen3.5 的视觉预算更大

在本机 `qwen3_5.py` 中可以看到：

- `min_pixels = 64 * 32 * 32`
- `max_pixels = 128 * 32 * 32`
- `total_pixels = 224 * 1024 * 32 * 32`
- `max_frames = 768`

这说明 Qwen3.5 在当前适配里更愿意保留较多视觉信息，尤其在视频/多帧输入下给了更高预算。

这类策略的优点是：
- 有利于复杂场景理解；
- 更适合需要上下文整合和长链推理的输入。

代价是：
- 显存压力更高；
- 推理成本更高；
- 速度更慢。

### 4.2 Qwen3-VL 的视觉预算更偏保守和实用

在本机 `qwen3_vl.py` 中可以看到：

- `min_pixels = 256 * 28 * 28`
- `max_pixels = 1605632`
- `max_num_frames = 32`

尤其是视频帧数上限更紧，说明它在本机适配里更强调：

- 吞吐
- 稳定性
- 统一的视觉输入规模控制

这更符合一个主力视觉模型在线部署时的需求。

---

## 5. 生成行为差异

## 5.1 Qwen3.5：长输出、强推理、token 消耗大

本机默认生成参数：

- `max_new_tokens = 1024`
- `temperature = 0.7`
- `top_p = 0.8`
- `top_k = 20`
- 支持 `enable_thinking = True`

这套参数组合意味着：

- 输出会更长；
- 更容易生成中间推理文本；
- 更适合自由生成与复杂解释；
- 更不适合极简 benchmark 短答。

从本机评测结果也能印证这一点：

- `Qwen3.5-4B` 总输出 tokens：`5,315,540`
- 评测耗时：`1742s`

对应文件：
- `/home/zktian3/lmms-eval/results_qwen35_4b/Qwen3.5-4B/20260320_154215_results.json`

## 5.2 Qwen3-VL：短输出、低温度、追求直接答案

本机默认生成参数：

- `max_new_tokens = 128`
- `temperature = 0.0`
- `num_beams = 1`

这套参数意味着：

- 更像标准答案输出器；
- 更适合 benchmark、分类、多选和问答；
- 更稳定；
- 更省 token。

本机结果：

- `Qwen3-VL-8B` 总输出 tokens：`31,390`
- `Qwen3-VL-4B` 总输出 tokens：`37,421`

对应文件：
- `/home/zktian3/lmms-eval/results_qwen3_vl_8b/Qwen3-VL-8B/20260320_145138_results.json`
- `/home/zktian3/lmms-eval/results_qwen3_vl_4b/Qwen3-VL-4B/20260320_140728_results.json`

---

## 6. 为什么 Qwen3.5 在部分复杂题上更强

### 6.1 空间关系能力更强

本机报告显示：

- `Qwen3.5-4B` 空间关系：`71.11%`
- `Qwen3-VL-8B` 空间关系：`51.11%`
- `Qwen3-VL-4B` 空间关系：`48.89%`

这说明在需要多步视觉推断、目标之间相对位置判断时，Qwen3.5 更容易占优。

### 6.2 原因并不是“Qwen3.5 全面更强”，而是它更愿意付出更高推理成本

Qwen3.5 的优势主要来自：

- 更长的生成链路
- thinking 机制
- 更高的输出 token 预算
- 更适合复杂结构化生成

本质上，它是在用更高成本换更强的复杂推理能力。

---

## 7. 为什么 Qwen3-VL 在综合评测里更适合当主模型

本机 MMBench 综合结果：

- `Qwen3-VL-8B`: `84.97`
- `Qwen3-VL-4B`: `83.68`
- `Qwen3.5-4B thinking`: `81.79`
- `Qwen3.5-4B no-thinking`: `73.20`

说明：

1. **Qwen3-VL 在综合视觉理解上更均衡**；
2. **Qwen3-VL 的输出成本和时延低很多**；
3. **Qwen3.5 的表现高度依赖 thinking**。

这意味着一旦线上为了加速而关掉 thinking，Qwen3.5 的优势会迅速缩水。

对应证据：
- `/home/zktian3/lmms-eval/vlm_eval_qwen35_4b_report.md`
- `/home/zktian3/lmms-eval/vlm_eval_qwen35_4b_nothinking_report.md`

其中可见：
- Qwen3.5-4B thinking：`81.79`
- Qwen3.5-4B no-thinking：`73.20`

说明 Qwen3.5 的不少能力是“依赖推理链激活”的，不是天然低成本就能保留。

---

## 8. 对当前项目的架构选型建议

### 8.1 如果目标是主力视觉模型
**优先：Qwen3-VL-8B**

原因：
- 综合分最高；
- 吞吐最好；
- 更适合 benchmark、在线推理、批量测试；
- 更符合视觉主模型定位。

### 8.2 如果目标是轻量方案
**可选：Qwen3-VL-4B**

原因：
- 性价比高；
- 分数与 8B 接近；
- 仍明显优于 Qwen3.5 no-thinking。

### 8.3 如果目标是复杂空间推理 / 难例复核
**保留：Qwen3.5-4B（thinking）**

适合用法：
- 第一阶段：Qwen3-VL 快速全量推理；
- 第二阶段：对空间关系难题、定位难题、低置信样本，用 Qwen3.5 thinking 复核。

这种“双阶段架构”比直接让 Qwen3.5 当主模型更合理。

---

## 9. 最终总结

Qwen3.5 和 Qwen3-VL 的本质差别，不只是“一个分高一个分低”，而是 **模型角色不同**：

- **Qwen3.5**：偏生成式、偏推理型、偏长输出，适合复杂视觉理解与结构化生成；
- **Qwen3-VL**：偏视觉任务优化、偏短答、偏高吞吐，适合综合 benchmark 和主力部署。

在本机已有测试里：

- **Qwen3-VL 更适合当主模型**；
- **Qwen3.5 更适合在复杂空间推理上当专项增强或复核模型**。

这也是为什么你会看到：
- `Qwen3-VL` 综合表现更好；
- 但 `Qwen3.5` 在某些“需要想”的题上更强。

本质原因不是谁绝对更强，而是 **两者的架构目标和推理成本分配方式不同**。
