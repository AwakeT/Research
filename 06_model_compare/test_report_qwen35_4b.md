# Qwen3.5-4B 测试报告

## 1. 测试概况

- 模型名称: `Qwen3.5-4B`
- 服务地址: `http://localhost:9004/v1`
- 结果目录: `/home/zktian3/benchmarks/homeobjects_mini_v1/reports/run_20260325_173653_qwen3_5_4b`
- Benchmark: `homeobjects_mini_v1`
- 测试任务总数: `352`
- 平均推理耗时: `2.422s / task`
- 总推理耗时: `852.676s`

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
| 开集目标检测 | 130 | Precision `0.8477`, Recall `0.7111`, F1 `0.7734`, Presence Acc `0.8231`, Mean Matched IoU `0.9316` |
| 指代目标定位 | 94 | Mean IoU `0.6352`, Acc@0.5 `0.7021` |
| 几何关系判断 | 94 | Accuracy `0.7021` |
| 房间类型分类 | 34 | Accuracy `0.0000` |

整体来看，Qwen3.5-4B 这一轮结果非常有特点：

- 指代定位能力较强，尤其 `sofa` 指代表现非常好。
- 开集检测和几何关系属于可用但不算突出。
- 房间类型分类几乎完全失效，是这轮最大的异常点。
- 同时推理耗时显著偏高，说明当前部署配置下的运行效率也需要关注。

## 3. 分项分析

### 3.1 开集检测

| 子集 | 数量 | TP | FP | FN | Precision | Recall | Presence Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sofa | 66 | 45 | 18 | 3 | `0.7143` | `0.9375` | `0.8030` |
| table | 64 | 83 | 5 | 49 | `0.9432` | `0.6288` | `0.8438` |

结论:

- `sofa` 检测依然偏向高召回，但误报略多。
- `table` 检测依旧是高精度、低召回，且 presence accuracy 明显下降，说明不仅多桌枚举不完整，连图像级存在性判断也更容易失误。
- `mean_matched_iou = 0.9316` 依然说明框本身比较准，主要问题仍在“漏检”和“没列全”。

额外观察:

- `sofa` 负样本共 23 张，其中 12 张出现误报，负样本误报仍然明显。
- `table` 检测出现了多张图完全漏检的情况，比如部分 5 桌图像直接 `tp=0, fn=5`。

### 3.2 指代目标定位

| 子集 | 方向 | 数量 | Mean IoU | Acc@0.5 |
| --- | --- | ---: | ---: | ---: |
| sofa | left | 5 | `0.8967` | `1.0000` |
| sofa | right | 5 | `0.9440` | `1.0000` |
| table | left | 42 | `0.5732` | `0.6190` |
| table | right | 42 | `0.6293` | `0.7143` |

结论:

- `sofa` 指代定位几乎可以视为非常稳定，左右两类都达到 `1.0` 的 Acc@0.5。
- `table` 指代任务整体也比很多前面模型更强，尤其 `table-right` 表现并不差。
- 这说明 Qwen3.5-4B 在“多个候选实例中选中目标”的能力上是它的强项之一。

### 3.3 几何关系判断

| 子集 | 方向 | 数量 | Accuracy | 预测分布 |
| --- | --- | ---: | ---: | --- |
| sofa | left | 5 | `1.0000` | `left_of: 5` |
| sofa | right | 5 | `0.8000` | `right_of: 4`, `left_of: 1` |
| table | left | 42 | `1.0000` | `left_of: 42` |
| table | right | 42 | `0.3571` | `left_of: 27`, `right_of: 15` |

结论:

- `left` 类几何关系判断很稳定。
- `sofa-right` 也已经明显可用。
- 真正的短板集中在 `table-right`，模型仍然存在较强的 `left_of` 偏置。
- 也就是说，Qwen3.5-4B 能较好完成单实例指代，但在“多个桌子之间的方向关系判断”上仍然不够稳。

### 3.4 房间类型分类

- Accuracy: `0.0000`
- 预测分布:
  - `unknown`: 33
  - `bedroom`: 1

结论:

- 这一轮 room classification 可以视为异常失败。
- 模型几乎把所有房间图都判成了 `unknown`，说明不是普通误差，而更像是 prompt 对齐、输出约束、或 thinking/模板设置导致的系统性退化。
- 这一项结果不建议直接与其它模型横向比较，应先排查部署和 prompt 配置问题。

## 4. 典型失败样例

### 4.1 检测失败样例

| 任务 | 图像 | 现象 |
| --- | --- | --- |
| `det_table_living_room_1p__31` | `images/table_coco/living_room_1p (31).jpg` | `tp=0, fp=0, fn=5`，整张图 5 张桌子全部漏检 |
| `det_table_living_room_1p__73` | `images/table_coco/living_room_1p (73).jpg` | `tp=0, fp=0, fn=5`，多桌场景完全漏检 |
| `det_table_living_room_1p__86` | `images/table_coco/living_room_1p (86).jpg` | `tp=0, fp=0, fn=4`，整图桌子未被识别 |
| `det_table_living_room_1p__71` | `images/table_coco/living_room_1p (71).jpg` | `tp=0, fp=0, fn=3`，存在明显漏检 |

这些样例说明 Qwen3.5-4B 在一部分多桌图像上会发生“整图漏掉所有桌子”的现象。

### 4.2 指代失败样例

| 任务 | 图像 | IoU |
| --- | --- | ---: |
| `ref_left_table_living_room_1p__11` | `images/table_coco/living_room_1p (11).jpg` | `0.0000` |
| `ref_left_table_living_room_1p__16` | `images/table_coco/living_room_1p (16).jpg` | `0.0000` |
| `ref_right_table_living_room_1p__20` | `images/table_coco/living_room_1p (20).jpg` | `0.0000` |
| `ref_right_table_living_room_1p__29` | `images/table_coco/living_room_1p (29).jpg` | `0.0000` |
| `ref_right_table_living_room_1p__31` | `images/table_coco/living_room_1p (31).jpg` | `0.0000` |

这些失败样例说明：尽管整体指代表现较强，但在部分多桌图像中仍会出现完全选错实例的情况。

### 4.3 几何关系失败样例

- `geom_right_sofa_living_room_1p__31`
- `geom_right_table_living_room_1`
- `geom_right_table_living_room_1p__4`
- `geom_right_table_living_room_1p__11`
- `geom_right_table_living_room_1p__12`
- `geom_right_table_living_room_1p__16`

这些失败样例的共性是: 正确答案应为 `right_of`，但模型仍偏向输出 `left_of`。

### 4.4 房间类型失败样例

- `room_room_living_room_1p__453` -> `unknown`
- `room_room_living_room_1p__374` -> `unknown`
- `room_room_living_room_1p__22` -> `unknown`
- `room_room_living_room_461` -> `unknown`
- `room_room_living_room_14` -> `unknown`
- `room_room_living_room_1p__541` -> `unknown`

这些失败样例说明 room 分类在当前配置下几乎完全退化成“默认 unknown”。

## 5. 主要结论

本轮 Qwen3.5-4B 测试可以概括为:

1. 指代定位是最强项，尤其 `sofa` 左右目标几乎全部正确。
2. 开集检测整体可用，但 `table` 多目标漏检问题依然明显，且存在部分整图级漏检。
3. 几何关系任务中，`table-right` 仍是短板，说明方向偏置尚未消除。
4. 房间类型分类在本轮配置下完全异常，不应直接作为模型能力结论。
5. 平均耗时 `2.422s / task` 明显偏高，需要关注部署参数和推理链路效率。

## 6. 后续优化建议

1. 优先排查 room classification 的 prompt、输出格式约束和 thinking 配置，因为当前结果明显像系统性异常而不是普通误差。
2. 对 `table` 检测增加“枚举所有桌子”的强约束，减少多桌场景整图漏检。
3. 保留当前指代任务设置，因为这是 Qwen3.5-4B 的优势项，可作为后续优化的稳定基线。
4. 对几何关系任务加入更多 `right_of` few-shot 示例，继续压制 `left_of` 偏置。
5. 检查服务端配置与吞吐限制，分析为什么当前 4B 版本耗时远高于 2B/8B。

## 7. 相关文件

- 指标汇总: `metrics.json`
- 原始预测: `predictions.jsonl`
- 简版摘要: `summary.md`
- 本报告: `test_report.md`
