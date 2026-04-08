# Qwen3.5-2B 测试报告

## 1. 测试概况

- 模型名称: `Qwen3.5-2B`
- 服务地址: `http://localhost:9003/v1`
- 结果目录: `/home/zktian3/benchmarks/homeobjects_mini_v1/reports/run_20260325_172649_qwen3_5_2b`
- Benchmark: `homeobjects_mini_v1`
- 测试任务总数: `352`
- 平均推理耗时: `0.339s / task`
- 总推理耗时: `119.178s`

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
| 开集目标检测 | 130 | Precision `0.8693`, Recall `0.7389`, F1 `0.7988`, Presence Acc `0.8846`, Mean Matched IoU `0.9461` |
| 指代目标定位 | 94 | Mean IoU `0.5922`, Acc@0.5 `0.6170` |
| 几何关系判断 | 94 | Accuracy `0.5532` |
| 房间类型分类 | 34 | Accuracy `0.7059` |

整体来看，Qwen3.5-2B 这一轮表现有一个比较鲜明的特征：

- 检测和指代任务表现并不弱，尤其 bbox 精度非常高。
- 主要短板集中在几何关系判断，特别是 `table-right` 这类方向性任务。
- 也就是说，这个模型更像“能找到目标”，但在“判断相对方向”上稳定性明显不足。

## 3. 分项分析

### 3.1 开集检测

| 子集 | 数量 | TP | FP | FN | Precision | Recall | Presence Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sofa | 66 | 47 | 15 | 1 | `0.7581` | `0.9792` | `0.8182` |
| table | 64 | 86 | 5 | 46 | `0.9451` | `0.6515` | `0.9531` |

结论:

- `sofa` 检测非常偏向高召回，几乎不漏掉沙发，但仍有一定误报。
- `table` 检测依旧是高精度、低召回，主要问题仍然是多桌场景下“只报出部分桌子”。
- `mean_matched_iou = 0.9461` 非常高，说明一旦找到目标，框通常是准的。

额外观察:

- `sofa` 负样本共 23 张，其中 12 张出现误报，负样本误报仍然比较明显。
- `sofa` 正样本 43 张中，仅 1 张完全漏检，说明 `sofa` 检测是典型的“高召回优先”模式。
- `table` 任务的主要问题依旧不是框偏，而是实例枚举不完整。

### 3.2 指代目标定位

| 子集 | 方向 | 数量 | Mean IoU | Acc@0.5 |
| --- | --- | ---: | ---: | ---: |
| sofa | left | 5 | `0.8000` | `0.8000` |
| sofa | right | 5 | `0.9809` | `1.0000` |
| table | left | 42 | `0.6113` | `0.6667` |
| table | right | 42 | `0.5020` | `0.5000` |

结论:

- `sofa` 指代定位表现很好，尤其 `sofa-right` 已经非常稳定。
- `table-left` 和 `table-right` 都还可以，但右侧桌子的选择仍不稳定。
- 说明 2B 在“少量候选实例”的目标选择上有能力，但在多桌图像中依旧容易选错实例。

### 3.3 几何关系判断

| 子集 | 方向 | 数量 | Accuracy | 预测分布 |
| --- | --- | ---: | ---: | --- |
| sofa | left | 5 | `1.0000` | `left_of: 5` |
| sofa | right | 5 | `0.4000` | `left_of: 3`, `right_of: 2` |
| table | left | 42 | `1.0000` | `left_of: 42` |
| table | right | 42 | `0.0714` | `left_of: 39`, `right_of: 3` |

结论:

- 几何关系是这轮最弱的一项任务。
- `left` 类关系判断几乎全对，但 `right` 类关系任务严重失衡。
- `table-right` 基本处于失效状态，模型对这类任务几乎固定输出 `left_of`。
- 这说明 Qwen3.5-2B 存在非常强的方向偏置，尤其是在复杂多桌场景下几乎无法稳定区分左右关系。

### 3.4 房间类型分类

- Accuracy: `0.7059`
- 预测分布:
  - `living_room`: 24
  - `dining_room`: 2
  - `kitchen`: 2
  - `study`: 2
  - `bedroom`: 2
  - `office`: 1
  - `storage`: 1

结论:

- 房间分类结果总体可用，但主预测仍偏向 `living_room`。
- 错误样例里不少图像本身确实可能更接近餐厅、厨房、卧室或书房。
- 因为 room 子集是弱标签构建，这里的准确率仍应被看作趋势性指标，而不是严格上限。

## 4. 典型失败样例

### 4.1 检测失败样例

| 任务 | 图像 | 现象 |
| --- | --- | --- |
| `det_table_living_room_1p__73` | `images/table_coco/living_room_1p (73).jpg` | `tp=2, fp=0, fn=3`，多桌场景明显漏检 |
| `det_table_living_room_1p__31` | `images/table_coco/living_room_1p (31).jpg` | `tp=3, fp=1, fn=2`，存在误检且漏掉部分实例 |
| `det_table_living_room_1p__36` | `images/table_coco/living_room_1p (36).jpg` | `tp=0, fp=1, fn=2`，目标没有对上 |
| `det_table_living_room_1p__21` | `images/table_coco/living_room_1p (21).jpg` | `tp=0, fp=0, fn=2`，完全漏检 |

这些样例说明 2B 在多桌场景中的主要问题仍然是“找不全”。

### 4.2 指代失败样例

| 任务 | 图像 | IoU |
| --- | --- | ---: |
| `ref_right_table_living_room_1` | `images/table_coco/living_room_1.jpg` | `0.0000` |
| `ref_right_table_living_room_1p__20` | `images/table_coco/living_room_1p (20).jpg` | `0.0000` |
| `ref_left_table_living_room_1p__21` | `images/table_coco/living_room_1p (21).jpg` | `0.0000` |
| `ref_right_table_living_room_1p__29` | `images/table_coco/living_room_1p (29).jpg` | `0.0000` |
| `ref_left_table_living_room_1p__34` | `images/table_coco/living_room_1p (34).jpg` | `0.0000` |

这些失败样例的共性是: 多个桌子同时出现时，模型能找到某个桌子，但很难稳定选中被指代的那个目标。

### 4.3 几何关系失败样例

- `geom_right_sofa_living_room_1p__13`
- `geom_right_sofa_living_room_1p__70`
- `geom_right_sofa_living_room_1p__83`
- `geom_right_table_living_room_1`
- `geom_right_table_living_room_1p__4`
- `geom_right_table_living_room_1p__11`

这些失败样例的共性是: 期望答案是 `right_of`，但模型大概率输出 `left_of`。

### 4.4 房间类型失败样例

- `room_room_living_room_1p__453` -> `dining_room`
- `room_room_living_room_461` -> `kitchen`
- `room_room_living_room_278` -> `dining_room`
- `room_room_living_room_1024` -> `study`
- `room_room_living_room_987` -> `study`
- `room_room_living_room_1p__177` -> `bedroom`

这些样例中，一部分误差可能来源于图像语义本身与文件名弱标签不完全一致。

## 5. 主要结论

本轮 Qwen3.5-2B 测试可以概括为:

1. 检测任务表现不差，尤其 bbox 精度很高。
2. `sofa` 检测召回很强，但仍有负样本误报。
3. `table` 检测的核心问题仍然是多目标漏检。
4. 指代任务总体可用，尤其 `sofa` 表现不错。
5. 几何关系任务是最大短板，尤其 `table-right` 几乎失效，说明方向性偏置非常明显。
6. 房间分类结果总体可用，但仍受到弱标签噪声影响。

## 6. 后续优化建议

1. 对 `table` 检测 prompt 增强“枚举所有桌子”的约束，尤其强调边桌、小桌和遮挡桌子也要输出。
2. 保留当前指代定位任务，但建议改成两阶段流程: 先检测全部候选框，再从候选中选择目标实例。
3. 对几何关系任务单独重写 prompt，并加入明确的 `right_of` few-shot，否则 2B 很容易退化成固定输出 `left_of`。
4. 若几何关系是重点能力，2B 目前不太适合作为主模型；更适合作为轻量检测/指代基线。
5. 房间分类若要严谨评估，建议先人工清洗 room 子集的弱标签。

## 7. 相关文件

- 指标汇总: `metrics.json`
- 原始预测: `predictions.jsonl`
- 简版摘要: `summary.md`
- 本报告: `test_report.md`
