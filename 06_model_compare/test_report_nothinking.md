# Qwen3.5-4B No-Thinking 测试报告

## 1. 测试概况

- 模型名称: `Qwen3.5-4B`
- 运行模式: `no-thinking`
- 服务地址: `http://localhost:9004/v1`
- 结果目录: `/home/zktian3/benchmarks/homeobjects_mini_v1/reports/run_20260325_183848_qwen3_5_4b`
- Benchmark: `homeobjects_mini_v1`
- 测试任务总数: `352`
- 平均推理耗时: `0.529s / task`
- 总推理耗时: `186.091s`

本轮使用专用脚本进行测试，默认配置为关闭 thinking，并通过 `chat_template_kwargs.enable_thinking = false` 发送到本地服务。

本次 benchmark 包含 4 类任务:

- 开集目标检测 `open_vocab_detection`: 130 个任务
- 指代目标定位 `referring_grounding`: 94 个任务
- 几何关系判断 `geometry_relation`: 94 个任务
- 房间类型分类 `room_classification`: 34 个任务

其中数据来源为:

- `sofa` COCO 子集: 66 张图
- `table` COCO 子集: 64 张图
- `room_context` 子集: 34 张图

## 2. 总体结果

| 任务 | 数量 | 指标 |
| --- | ---: | --- |
| 开集目标检测 | 130 | Precision `0.8054`, Recall `0.8278`, F1 `0.8164`, Presence Acc `0.8846`, Mean Matched IoU `0.9294` |
| 指代目标定位 | 94 | Mean IoU `0.6610`, Acc@0.5 `0.7021` |
| 几何关系判断 | 94 | Accuracy `0.8298` |
| 房间类型分类 | 34 | Accuracy `0.7059` |

整体来看，这一轮 `Qwen3.5-4B no-thinking` 的表现相当稳：

- 开集检测 F1 达到 `0.8164`，是当前非常强的一档。
- 指代定位表现优秀，Mean IoU `0.6610`。
- 几何关系判断达到 `0.8298`，是这类任务中的高水平结果。
- 房间类型分类恢复正常，不再出现上一轮几乎全部输出 `unknown` 的异常情况。
- 推理速度也明显恢复，`0.529s / task`，远优于上一轮异常慢的 3.5-4B 表现。

## 3. 分项分析

### 3.1 开集检测

| 子集 | 数量 | TP | FP | FN | Precision | Recall | Presence Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sofa | 66 | 47 | 23 | 1 | `0.6714` | `0.9792` | `0.8030` |
| table | 64 | 102 | 13 | 30 | `0.8870` | `0.7727` | `0.9688` |

结论:

- `sofa` 检测依然是高召回策略，几乎不漏掉沙发，但误报偏多。
- `table` 检测相比很多前面的结果更好，召回已经提升到 `0.7727`，说明关闭 thinking 后，多桌场景的枚举完整性明显改善。
- 整体 `mean_matched_iou = 0.9294`，说明一旦找到了目标，框本身仍然比较准。

额外观察:

- `sofa` 负样本共 23 张，其中 13 张出现误报，负样本误报仍是一个需要单独优化的问题。
- `table` 任务中，虽然仍有部分多桌场景漏检，但已经能覆盖更多实例，不再像上一轮那样频繁整图漏光。

### 3.2 指代目标定位

| 子集 | 方向 | 数量 | Mean IoU | Acc@0.5 |
| --- | --- | ---: | ---: | ---: |
| sofa | left | 5 | `0.9611` | `1.0000` |
| sofa | right | 5 | `0.9820` | `1.0000` |
| table | left | 42 | `0.5732` | `0.6190` |
| table | right | 42 | `0.6748` | `0.7143` |

结论:

- `sofa` 指代定位几乎可以视为非常稳定，左右两类任务都达到 `1.0` 的 Acc@0.5。
- `table-right` 表现很好，已经达到 `0.7143` 的 Acc@0.5。
- 整体上，关闭 thinking 之后，这一轮的指代任务表现非常强，说明该模型在 no-thinking 模式下更适合做“明确目标选择”类任务。

### 3.3 几何关系判断

| 子集 | 方向 | 数量 | Accuracy | 预测分布 |
| --- | --- | ---: | ---: | --- |
| sofa | left | 5 | `1.0000` | `left_of: 5` |
| sofa | right | 5 | `1.0000` | `right_of: 5` |
| table | left | 42 | `0.9762` | `left_of: 41`, `overlapping: 1` |
| table | right | 42 | `0.6429` | `right_of: 27`, `left_of: 14`, `overlapping: 1` |

结论:

- `sofa` 几何关系判断已经达到非常理想的水平，左右方向都稳定正确。
- `table-left` 很强，`table-right` 也显著优于很多前面的模型版本。
- 尽管 `table-right` 仍存在一定 `left_of` 偏置，但整体几何关系任务已经达到可用且较强的状态。

### 3.4 房间类型分类

- Accuracy: `0.7059`
- 预测分布:
  - `living_room`: 24
  - `bedroom`: 3
  - `kitchen`: 2
  - `office`: 2
  - `dining_room`: 1
  - `library`: 1
  - `unknown`: 1

结论:

- 房间分类恢复到正常水平，和其他稳定模型的结果处于同一档。
- 错误样例中，仍然有一部分可能来自 room 子集自身的弱标签噪声，而不完全是模型理解错误。
- 至少可以确认：no-thinking 配置下，Qwen3.5-4B 不再出现系统性 `unknown` 退化。

## 4. 典型失败样例

### 4.1 检测失败样例

| 任务 | 图像 | 现象 |
| --- | --- | --- |
| `det_table_living_room_1p__31` | `images/table_coco/living_room_1p (31).jpg` | `tp=2, fp=3, fn=3`，多桌场景中仍有漏检与误检 |
| `det_table_living_room_1p__40` | `images/table_coco/living_room_1p (40).jpg` | `tp=2, fp=0, fn=2`，部分桌子没有被列出 |
| `det_table_living_room_1p__57` | `images/table_coco/living_room_1p (57).jpg` | `tp=2, fp=0, fn=2`，多目标枚举仍不完整 |
| `det_table_living_room_1p__78` | `images/table_coco/living_room_1p (78).jpg` | `tp=3, fp=0, fn=2`，仍存在漏检 |

这些样例说明 no-thinking 虽然改善了 table 检测，但多桌场景仍然不是完全解决的问题。

### 4.2 指代失败样例

| 任务 | 图像 | IoU |
| --- | --- | ---: |
| `ref_left_table_living_room_1p__16` | `images/table_coco/living_room_1p (16).jpg` | `0.0000` |
| `ref_right_table_living_room_1p__20` | `images/table_coco/living_room_1p (20).jpg` | `0.0000` |
| `ref_right_table_living_room_1p__29` | `images/table_coco/living_room_1p (29).jpg` | `0.0000` |
| `ref_left_table_living_room_1p__31` | `images/table_coco/living_room_1p (31).jpg` | `0.0000` |
| `ref_left_table_living_room_1p__34` | `images/table_coco/living_room_1p (34).jpg` | `0.0000` |

这些失败样例表明：在某些拥挤多桌场景中，模型仍会选错具体目标实例。

### 4.3 几何关系失败样例

- `geom_right_table_living_room_1p__11`
- `geom_right_table_living_room_1p__12`
- `geom_right_table_living_room_1p__13`
- `geom_right_table_living_room_1p__21`
- `geom_right_table_living_room_1p__29`
- `geom_right_table_living_room_1p__34`

这些失败样例的共性是：正确答案应为 `right_of`，但模型仍有一定概率输出 `left_of`。

### 4.4 房间类型失败样例

- `room_room_living_room_1p__453` -> `dining_room`
- `room_room_living_room_461` -> `kitchen`
- `room_room_living_room_1p__464` -> `bedroom`
- `room_room_living_room_1024` -> `library`
- `room_room_living_room_987` -> `office`
- `room_room_living_room_1p__177` -> `bedroom`

这些样例中，有一部分误差可能来自图像内容本身就接近非客厅场景，而不仅仅是模型判断失误。

## 5. 主要结论

本轮 `Qwen3.5-4B no-thinking` 测试可以概括为:

1. 这是一个整体非常强、非常均衡的结果版本。
2. 开集检测尤其是 `table` 检测明显改善，说明 no-thinking 对多目标枚举有帮助。
3. 指代定位是强项，`sofa` 左右目标几乎完全稳定，`table-right` 也表现很好。
4. 几何关系判断达到高水平，尤其 `sofa` 方向关系已经很稳定。
5. 房间类型分类恢复正常，说明 no-thinking 配置避免了上一轮 thinking 版的系统性退化。
6. 速度也恢复到合理区间，明显优于之前异常偏慢的 3.5-4B 结果。

## 6. 后续优化建议

1. 可以把 `Qwen3.5-4B no-thinking` 作为新的主候选模型之一，重点和 `Qwen3-VL-8B` 做正面对比。
2. 对 `sofa` 负样本误报问题，可以继续通过 stricter detection prompt 或后处理置信过滤来优化。
3. 对 `table` 多目标场景，仍建议强化“列出所有实例”的 prompt，进一步补召回。
4. 对 `table-right` 几何关系可加入更多 `right_of` few-shot，继续降低残留的方向偏置。
5. 建议保留一套 `enable_thinking=true/false` 的成对实验，进一步确认该模型在 no-thinking 模式下是否普遍更优。

## 7. 相关文件

- 指标汇总: `metrics.json`
- 原始预测: `predictions.jsonl`
- 简版摘要: `summary.md`
- 本报告: `test_report_nothinking.md`
