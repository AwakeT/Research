# Qwen3-VL-4B 测试报告

## 1. 测试概况

- 模型名称: `Qwen3-VL-4B`
- 服务地址: `http://localhost:9001/v1`
- 结果目录: `/home/zktian3/benchmarks/homeobjects_mini_v1/reports/run_20260325_163117_qwen3_vl_4b`
- Benchmark: `homeobjects_mini_v1`
- 测试任务总数: `352`
- 平均推理耗时: `0.361s / task`
- 总推理耗时: `126.977s`

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
| 开集目标检测 | 130 | Precision `0.8544`, Recall `0.7500`, F1 `0.7988`, Presence Acc `0.8846`, Mean Matched IoU `0.9328` |
| 指代目标定位 | 94 | Mean IoU `0.5188`, Acc@0.5 `0.5745` |
| 几何关系判断 | 94 | Accuracy `0.6915` |
| 房间类型分类 | 34 | Accuracy `0.7059` |

## 3. 分项分析

### 3.1 开集检测

| 子集 | 数量 | TP | FP | FN | Precision | Recall | Presence Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sofa | 66 | 46 | 16 | 2 | `0.7419` | `0.9583` | `0.8030` |
| table | 64 | 89 | 7 | 43 | `0.9271` | `0.6742` | `0.9688` |

结论:

- `sofa` 检测的主要特点是召回高、误检偏多。模型基本不会漏掉沙发，但在负样本上有较明显误报。
- `table` 检测的主要特点是精度高、召回偏低。模型通常能找到桌子，但在多桌场景下经常“找不全”。
- 当模型检测成功时，框质量很好，整体 `mean_matched_iou = 0.9328`，说明问题主要不在定位精度，而在枚举完整性。

额外观察:

- `sofa` 负样本共 23 张，其中 12 张出现误报，负样本误报率约 `52.17%`。
- `sofa` 正样本 43 张中，仅 1 张出现完全漏检，说明 `sofa` 任务更多是“过检”而不是“漏检”。
- `table` 在单桌图像上表现较稳，但在多桌图像中明显漏检。

### 3.2 指代目标定位

| 子集 | 方向 | 数量 | Mean IoU | Acc@0.5 |
| --- | --- | ---: | ---: | ---: |
| sofa | left | 5 | `0.6077` | `0.6000` |
| sofa | right | 5 | `0.7870` | `0.8000` |
| table | left | 42 | `0.6099` | `0.6905` |
| table | right | 42 | `0.3852` | `0.4286` |

结论:

- `table-right` 是最明显短板，模型在多张桌子的场景中经常把“右边桌子”选成左边或中间的桌子。
- `sofa` 指代表现整体好于 `table`，但样本量较小。
- 当前 prompt 下，模型更擅长“找到一个同类物体”，不够擅长“在多个同类物体中选中正确目标”。

### 3.3 几何关系判断

| 子集 | 方向 | 数量 | Accuracy | 预测分布 |
| --- | --- | ---: | ---: | --- |
| sofa | left | 5 | `1.0000` | `left_of: 5` |
| sofa | right | 5 | `0.0000` | `left_of: 5` |
| table | left | 42 | `0.9524` | `left_of: 40`, `right_of: 2` |
| table | right | 42 | `0.4762` | `left_of: 22`, `right_of: 20` |

结论:

- 模型存在明显的 `left_of` 方向偏置。
- `left` 类任务整体表现很好，`right` 类任务则明显下降。
- `sofa-right` 几乎全部失败，属于非常稳定的系统性错误，而不是随机波动。

### 3.4 房间类型分类

- Accuracy: `0.7059`
- 预测分布:
  - `living_room`: 24
  - `dining_room`: 2
  - `kitchen`: 2
  - `study`: 2
  - `bedroom`: 2
  - `office`: 1
  - `unknown`: 1

结论:

- 房间分类结果中，有一部分“错误”其实更像是弱标签噪声。
- 当前 room 子集主要依据 `living_room_*` 文件名构建，但部分图像本身确实更像餐厅、厨房、卧室或书房。
- 因此 `70.59%` 的准确率应谨慎解释，它很可能低估了模型真实的场景理解能力。

## 4. 典型失败样例

### 4.1 检测失败样例

| 任务 | 图像 | 现象 |
| --- | --- | --- |
| `det_table_living_room_1p__73` | `images/table_coco/living_room_1p (73).jpg` | `tp=2, fp=1, fn=3`，多桌场景明显漏检 |
| `det_table_living_room_1p__36` | `images/table_coco/living_room_1p (36).jpg` | `tp=0, fp=1, fn=2`，检测完全错位 |
| `det_table_living_room_1p__38` | `images/table_coco/living_room_1p (38).jpg` | `tp=2, fp=1, fn=2`，部分桌子漏掉 |
| `det_table_living_room_1p__40` | `images/table_coco/living_room_1p (40).jpg` | `tp=2, fp=1, fn=2`，多目标未完整枚举 |

### 4.2 指代失败样例

| 任务 | 图像 | IoU |
| --- | --- | ---: |
| `ref_right_sofa_living_room_1p__31` | `images/sofa_coco/living_room_1p (31).jpg` | `0.0000` |
| `ref_right_table_living_room_1` | `images/table_coco/living_room_1.jpg` | `0.0000` |
| `ref_left_table_living_room_1p__11` | `images/table_coco/living_room_1p (11).jpg` | `0.0000` |
| `ref_right_table_living_room_1p__13` | `images/table_coco/living_room_1p (13).jpg` | `0.0000` |

这些样例的共性是: 图中存在多个同类物体时，模型会返回“某个合理框”，但不一定是被指代的那个框。

### 4.3 几何关系失败样例

- `geom_right_sofa_living_room_1p__13`
- `geom_right_sofa_living_room_1p__31`
- `geom_right_sofa_living_room_1p__45`
- `geom_right_table_living_room_1`
- `geom_right_table_living_room_1p__11`

这些失败样例的共性是: 期望答案是 `right_of`，但模型经常统一输出 `left_of`。

## 5. 主要结论

本轮 Qwen3-VL-4B 测试可以概括为:

1. 检测框质量很高，说明模型具备较好的目标定位能力。
2. 真正的主要问题不是框不准，而是多目标场景中的漏检与枚举不完整。
3. 指代定位与几何关系任务都表现出明显的“右侧目标偏弱”问题。
4. 房间分类的数值结果受弱标签影响较大，建议不要仅凭 accuracy 做结论。

## 6. 后续优化建议

1. 对检测任务增加强调“不要遗漏所有同类目标”的 prompt，尤其针对 `table` 多目标场景。
2. 将指代定位改成两阶段流程: 先检测全部候选框，再从候选框中选择 `left/right` 目标。
3. 为几何关系加入 `right_of` 的 few-shot 示例，减少 `left_of` 偏置。
4. 对房间类型子集做一次人工复核，过滤掉明显不属于 `living_room` 的样本，以获得更可信的分类指标。

## 7. 相关文件

- 指标汇总: `metrics.json`
- 原始预测: `predictions.jsonl`
- 简版摘要: `summary.md`
- 本报告: `test_report.md`
