# Qwen3-VL-8B 测试报告

## 1. 测试概况

- 模型名称: `Qwen3-VL-8B`
- 服务地址: `http://localhost:9002/v1`
- 结果目录: `/home/zktian3/benchmarks/homeobjects_mini_v1/reports/run_20260325_170920_qwen3_vl_8b`
- Benchmark: `homeobjects_mini_v1`
- 测试任务总数: `352`
- 平均推理耗时: `0.664s / task`
- 总推理耗时: `233.561s`

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
| 开集目标检测 | 130 | Precision `0.8645`, Recall `0.7444`, F1 `0.8000`, Presence Acc `0.8923`, Mean Matched IoU `0.9002` |
| 指代目标定位 | 94 | Mean IoU `0.5323`, Acc@0.5 `0.6064` |
| 几何关系判断 | 94 | Accuracy `0.7660` |
| 房间类型分类 | 34 | Accuracy `0.7353` |

整体来看，8B 的表现呈现出一个比较清晰的特点：

- 检测框精度依然较高，说明模型已经具备较稳定的目标定位能力。
- 主要瓶颈仍然是多同类目标场景下的枚举完整性和目标选择能力。
- 相比简单的单目标存在性判断，模型在“多个候选实例中选中正确实例”这一类任务上仍然更容易出错。

## 3. 分项分析

### 3.1 开集检测

| 子集 | 数量 | TP | FP | FN | Precision | Recall | Presence Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sofa | 66 | 45 | 15 | 3 | `0.7500` | `0.9375` | `0.8182` |
| table | 64 | 89 | 6 | 43 | `0.9368` | `0.6742` | `0.9688` |

结论:

- `sofa` 检测表现为高召回、适中精度。模型大多数情况下能发现沙发，但在负样本图像中仍有一定误报。
- `table` 检测表现为高精度、较低召回。模型通常能识别出桌子，但在一张图里有多张桌子时，仍容易只返回部分实例。
- 一旦模型检测成功，框通常较准，整体 `mean_matched_iou = 0.9002`，说明错误主要集中在“没找全”而不是“框偏得很离谱”。

额外观察:

- `sofa` 负样本共 23 张，其中 11 张出现误报，负样本误报率约 `47.83%`。
- `sofa` 正样本 43 张中，仅 1 张出现完全漏检，说明 `sofa` 检测更偏向“宁可多报也尽量别漏”。
- `table` 检测的主要误差模式仍然是多桌场景下的系统性漏检。

### 3.2 指代目标定位

| 子集 | 方向 | 数量 | Mean IoU | Acc@0.5 |
| --- | --- | ---: | ---: | ---: |
| sofa | left | 5 | `0.5766` | `0.6000` |
| sofa | right | 5 | `0.9411` | `1.0000` |
| table | left | 42 | `0.5593` | `0.6429` |
| table | right | 42 | `0.4513` | `0.5238` |

结论:

- `sofa-right` 的指代定位效果很好，说明 8B 在少量双沙发场景中已经具备较强的右侧目标识别能力。
- `table-right` 仍然是难点，平均 IoU 和 Acc@0.5 都明显低于 `table-left`。
- 多桌场景里，模型经常会返回一个“看起来合理”的桌子框，但不是被指代的那个实例。

### 3.3 几何关系判断

| 子集 | 方向 | 数量 | Accuracy | 预测分布 |
| --- | --- | ---: | ---: | --- |
| sofa | left | 5 | `1.0000` | `left_of: 5` |
| sofa | right | 5 | `0.6000` | `right_of: 3`, `left_of: 2` |
| table | left | 42 | `0.9762` | `left_of: 41`, `right_of: 1` |
| table | right | 42 | `0.5476` | `left_of: 19`, `right_of: 23` |

结论:

- `left` 类几何关系判断整体已经很稳。
- `right` 类任务相比 4B 已经明显改善，但依旧存在一定的 `left_of` 偏置。
- 8B 在几何关系任务上总体可用，但对于“右侧实例”仍然不够稳定。

### 3.4 房间类型分类

- Accuracy: `0.7353`
- 预测分布:
  - `living_room`: 25
  - `bedroom`: 3
  - `kitchen`: 2
  - `study`: 2
  - `dining_room`: 1
  - `library`: 1

结论:

- 房间分类总体可用，主预测仍然集中在 `living_room`。
- 一部分“错误分类”实际上可能对应图像内容本身更接近卧室、书房、厨房或餐厅。
- 因为 room 子集是弱标签构建，这里的准确率更适合作为“趋势参考”，不宜作为严格上限结论。

## 4. 典型失败样例

### 4.1 检测失败样例

| 任务 | 图像 | 现象 |
| --- | --- | --- |
| `det_table_living_room_1p__31` | `images/table_coco/living_room_1p (31).jpg` | `tp=2, fp=1, fn=3`，多桌场景明显漏检 |
| `det_table_living_room_1p__73` | `images/table_coco/living_room_1p (73).jpg` | `tp=2, fp=0, fn=3`，多目标未完整枚举 |
| `det_table_living_room_1p__36` | `images/table_coco/living_room_1p (36).jpg` | `tp=0, fp=2, fn=2`，存在误检且目标漏掉 |
| `det_table_living_room_1p__40` | `images/table_coco/living_room_1p (40).jpg` | `tp=2, fp=0, fn=2`，部分桌子漏检 |

这些样例说明 8B 在 `table` 检测上的主要问题仍是“多目标枚举不全”。

### 4.2 指代失败样例

| 任务 | 图像 | IoU |
| --- | --- | ---: |
| `ref_left_sofa_living_room_1p__83` | `images/sofa_coco/living_room_1p (83).jpg` | `0.0000` |
| `ref_left_table_living_room_1p__11` | `images/table_coco/living_room_1p (11).jpg` | `0.0000` |
| `ref_right_table_living_room_1p__11` | `images/table_coco/living_room_1p (11).jpg` | `0.0000` |
| `ref_left_table_living_room_1p__16` | `images/table_coco/living_room_1p (16).jpg` | `0.0000` |
| `ref_right_table_living_room_1p__20` | `images/table_coco/living_room_1p (20).jpg` | `0.0000` |

这些失败样例的共性是: 图中存在多个同类实例时，模型会返回“某一个桌子/沙发”，但不一定是被指代的目标实例。

### 4.3 几何关系失败样例

- `geom_right_sofa_living_room_1p__31`
- `geom_right_sofa_living_room_1p__83`
- `geom_right_table_living_room_1`
- `geom_right_table_living_room_1p__11`
- `geom_right_table_living_room_1p__12`
- `geom_right_table_living_room_1p__13`

这些失败样例的共性是: 期望答案是 `right_of`，但模型仍有一定概率输出 `left_of`。

### 4.4 房间类型失败样例

- `room_room_living_room_1p__453` -> `dining_room`
- `room_room_living_room_461` -> `kitchen`
- `room_room_living_room_987` -> `study`
- `room_room_living_room_1p__177` -> `bedroom`
- `room_room_living_room_1p__56` -> `study`
- `room_room_living_room_1p__89` -> `bedroom`

这些样例中，不少从图像语义上看并不完全像“纯客厅”，因此误差里夹杂了一部分弱标签噪声。

## 5. 主要结论

本轮 Qwen3-VL-8B 测试可以概括为:

1. 检测框质量依然较高，说明模型具备稳定的视觉定位能力。
2. `table` 多目标检测仍是主要难点，错误集中在漏检而不是框偏移。
3. 指代定位与几何关系任务相比检测更难，尤其是在多个同类实例并存时。
4. `right` 类目标和关系任务仍然明显弱于 `left` 类任务，说明方向性偏置尚未完全消除。
5. 房间分类结果总体可用，但仍受到 room 子集弱标签属性的影响。

## 6. 后续优化建议

1. 对 `table` 检测增强 prompt，明确要求“列出所有可见桌子，不要遗漏小桌、边桌和部分遮挡桌子”。
2. 将指代定位改成两阶段流程: 先检测全部候选框，再从候选框中选择 `left/right` 目标。
3. 为几何关系任务加入更多 `right_of` few-shot 示例，进一步抑制 `left_of` 偏置。
4. 若要更严谨评估房间分类，建议先人工清洗 `room_context` 子集中的弱标签。

## 7. 相关文件

- 指标汇总: `metrics.json`
- 原始预测: `predictions.jsonl`
- 简版摘要: `summary.md`
- 本报告: `test_report.md`
