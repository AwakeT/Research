# Qwen3.5-4B Thinking 模式深度技术解析

---

文档版本：v1.0
创建日期：2026-04-08
文档性质：技术原理分析
适用范围：理解 Qwen3.5-4B 开启/关闭 thinking 模式在模型处理层面的具体差异

---

## 1. Thinking 模式的控制机制

### 1.1 这不是架构开关，而是模板控制

`enable_thinking` 是一个 **Jinja2 模板变量**，通过 `tokenizer.apply_chat_template()` 传入。它不修改模型权重、不改变网络结构、不切换推理路径——它只改变输入给模型的 prompt 内容。

**开启 thinking（`enable_thinking=True`）**：

模板仅生成标准的 assistant turn header，不做任何预填充：

```
<|im_start|>assistant
```

模型在此基础上**自主决定**是否生成 `<think>` 标签。这与 QwQ（Qwen 早期推理模型）不同——QwQ 会在每轮强制预填充 `<think>\n`，而 Qwen3.5 让模型自己选择。

**关闭 thinking（`enable_thinking=False`）**：

模板**预填充一个空的 think 块**：

```jinja2
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\n\n</think>\n\n' }}
{%- endif %}
```

实际送入模型的 prompt 变为：

```
<|im_start|>assistant
<think>

</think>

```

模型"看到"思考已经完成（空 think 块已闭合），因此直接生成最终答案。

### 1.2 Qwen3.5-4B 的默认行为

对于 Qwen3.5 小模型（0.8B、2B、4B、9B），**thinking 默认关闭**。需要显式传入 `enable_thinking=True` 才会启用。

对于大模型（27B、35B-A3B、122B-A10B、397B-A17B），thinking 默认开启。

**已知问题**：通过 `processor.apply_chat_template()` 传入 `enable_thinking=False` 会产生警告："is not a valid argument for this processor and will be ignored"。正确做法是通过 tokenizer 而非 processor 传入该参数。

### 1.3 特殊 Token 与普通 Token 的区别

| Token 类型 | 示例 | 本质 | Token ID |
|-----------|------|------|---------|
| **真正的特殊 token** | `<\|im_start\|>`、`<\|im_end\|>` | 词表中的单一 token，有独立 ID | 固定 |
| **XML 风格标签** | `<think>`、`</think>` | **不是特殊 token**，被分词为多个普通 token 序列 | `</think>` 对应 token ID 151668 |

`<think>` 和 `</think>` 的语义完全是**训练过程中学到的**，不是架构级的控制信号。解析代码通过搜索 token ID 151668 来分割 thinking 内容和最终答案。

---

## 2. 生成流程的差异

### 2.1 逐 Token 生成流程对比

两种模式都是标准的**自回归生成**（逐 token 预测下一个 token），没有独立的"推理引擎"或并行计算路径。

**Thinking 开启时的生成流程**：

```
输入 prompt:
  <|im_start|>system\n{系统提示}<|im_end|>\n
  <|im_start|>user\n{用户输入}<|im_end|>\n
  <|im_start|>assistant\n

自回归生成:
  → <think>\n                          ← 模型自主决定开始思考
  → 当前视角看到客厅，左侧有沙发...    ← 推理 token（逐个生成）
  → 目标应该在卧室方向...              ← 继续推理
  → </think>\n\n                       ← 模型自主决定结束思考（token ID 151668）
  → {"action": "turn_left", ...}       ← 最终输出
  → <|im_end|>                         ← 生成结束
```

**Thinking 关闭时的生成流程**：

```
输入 prompt:
  <|im_start|>system\n{系统提示}<|im_end|>\n
  <|im_start|>user\n{用户输入}<|im_end|>\n
  <|im_start|>assistant\n
  <think>\n\n</think>\n\n              ← 模板预填充的空 think 块

自回归生成:
  → {"action": "turn_left", ...}       ← 直接输出最终结果
  → <|im_end|>                         ← 生成结束
```

### 2.2 模型如何决定"停止思考"

没有硬编码规则。模型通过训练（四阶段后训练流程）学会在推理"足够"时生成 `</think>` token。这是一个概率预测行为——当模型认为思考完成时，`</think>` 的生成概率自然上升，与预测任何其他 token 的机制完全相同。

**这带来一个风险**：模型可能在推理不充分时过早关闭 think，或在简单问题上冗长推理后才关闭。Think 链的长度不可靠地反映推理质量。

### 2.3 采样参数差异

Qwen3.5-4B 官方推荐对两种模式使用**不同的采样参数**：

| 场景 | Temperature | top_p | top_k | min_p | presence_penalty |
|------|-----------|-------|-------|-------|-----------------|
| **Thinking — 通用任务** | 1.0 | 0.95 | 20 | 0.0 | 1.5 |
| **Thinking — 精确编码** | 0.6 | 0.95 | 20 | 0.0 | 0.0 |
| **No-Thinking — 通用任务** | 0.7 | 0.8 | 20 | 0.0 | 1.5 |
| **No-Thinking — 推理任务** | 1.0 | 1.0 | 40 | 0.0 | 2.0 |

关键差异：
- Thinking 模式使用**更高的 temperature（1.0）**，鼓励推理链中的多样化探索
- No-Thinking 模式使用**更低的 temperature（0.7）和 top_p（0.8）**，输出更加确定性
- 官方**明确警告不要使用 greedy decoding（temperature=0）**——会导致性能退化和无限重复

**这意味着**：两种模式不仅在"是否生成推理链"上不同，**整个概率分布的采样策略都不同**。Thinking 模式的生成行为本质上更"发散"，no-thinking 模式更"收敛"。

---

## 3. 注意力机制与上下文窗口的影响

### 3.1 Qwen3.5-4B 的混合注意力架构

Qwen3.5-4B 采用了与 Qwen3 完全不同的**混合注意力架构**：

```
Qwen3-4B:   36 层 × 全 Softmax Attention（标准 Transformer）
Qwen3.5-4B: 32 层 × 混合架构：

  8 个重复单元 × (
    3 × (Gated DeltaNet → FFN)   ← 线性注意力层（24 层）
    1 × (Gated Attention → FFN)   ← 标准 Softmax 注意力层（8 层）
  )
```

| 架构参数 | Qwen3-4B | Qwen3.5-4B |
|---------|----------|-------------|
| 总层数 | 36 | 32 |
| 全注意力层数 | 36（100%） | 8（25%） |
| 线性注意力层数 | 0 | 24（75%） |
| Q-heads（全注意力） | 32 | 16 |
| KV-heads（全注意力） | 8 | 4 |
| GDN V-heads | — | 32 |
| GDN QK-heads | — | 16 |

### 3.2 Thinking Token 对 KV Cache 的影响

这是两种架构的关键区别：

**全注意力层（Gated Attention，8 层）**：
- 每生成一个 token，都会在 KV cache 中添加一对 key-value 向量
- Thinking token **与普通 token 完全等价**——占用相同的 KV cache 空间
- 如果 thinking 链生成了 500 个 token，这 8 层的 KV cache 就增长了 500 个条目
- 使用标准 causal attention mask——每个 token 可以 attend 到所有之前的 token

**线性注意力层（Gated DeltaNet，24 层）**：
- 维护一个**固定大小的状态矩阵**（~128×128 per head），不随序列长度增长
- Thinking token 会**更新**这个状态矩阵，但**不增加其内存占用**
- 无论 thinking 链有 10 个 token 还是 1000 个 token，这 24 层的内存开销不变

**实际影响**：

```
假设 thinking 链生成 500 个 token：

Qwen3-4B（全注意力）:
  KV cache 增量 = 500 tokens × 36 layers × (K + V) × head_dim
  = 500 × 36 × 2 × 128 bytes × num_kv_heads
  → 所有 36 层都线性增长

Qwen3.5-4B（混合注意力）:
  全注意力层 KV cache 增量 = 500 tokens × 8 layers × (K + V) × head_dim
  线性注意力层 KV cache 增量 = 0（固定状态矩阵）
  → 仅 8 层线性增长，KV cache 压力降至 ~22%（8/36）
```

**结论**：Qwen3.5-4B 的混合架构使 thinking token 的 KV cache 成本大幅低于 Qwen3-4B。但 thinking token 仍然消耗上下文窗口长度配额。

### 3.3 Thinking Token 消耗上下文窗口

**Thinking token 与普通 token 完全等价地消耗上下文窗口**。没有任何特殊的注意力掩码或优先级机制来区分它们。

```
Qwen3.5-4B 原生上下文窗口: 262,144 tokens

示例：多步导航任务中的上下文消耗

Step 1:  系统提示（200 tokens）+ 图像（~1000 tokens）+ thinking（300 tokens）+ 输出（30 tokens）
Step 2:  历史（1530）+ 新图像（1000）+ thinking（300）+ 输出（30）
Step 3:  历史（2860）+ 新图像（1000）+ thinking（300）+ 输出（30）
...
Step N:  历史上下文被 thinking token 快速膨胀

每步 thinking 链额外消耗 ~300 tokens
10 步后：thinking 链累计占用 ~3000 tokens
100 步后：thinking 链累计占用 ~30,000 tokens
```

### 3.4 多轮对话中的 Rolling Checkpoint 机制

Qwen3.5 的 Jinja2 聊天模板实现了一个**滚动清理机制**来缓解历史 thinking token 的上下文污染：

1. 从消息列表末尾反向查找最新的用户消息
2. 该消息之后的 assistant 回复：**保留完整 `<think>` 块**
3. 该消息之前的所有历史：**剥除 `<think>` 块内容**

```
多轮对话示例：

Turn 1 (历史):
  User: "找到客厅的遥控器"
  Assistant: <think>██████████</think> → 被剥除 → Assistant: {"action": "turn_left"}

Turn 2 (历史):
  User: "继续前进"
  Assistant: <think>██████████</think> → 被剥除 → Assistant: {"action": "move_forward"}

Turn 3 (当前):
  User: "停下来"
  Assistant: <think>当前位置接近目标...</think>{"action": "stop"} → 保留完整
```

**但这个机制有局限**：
- 仅在多轮对话模板中生效，**单次长序列生成中不适用**
- VLN 导航如果实现为单次长序列（而非多轮对话），thinking token 无法被清理
- 每轮的输出 token（含 thinking）在当轮仍然完整消耗上下文

---

## 4. 对视觉 Token 处理的影响

### 4.1 Qwen3.5 的 Early Fusion 架构

Qwen3.5 是**原生多模态模型**，视觉 token 和文本 token 从第一层开始就在同一个 Transformer 中混合处理：

```
输入序列（thinking 开启时）:

[系统提示 text tokens] [图像 vision tokens] [用户指令 text tokens]
      ↓                      ↓                     ↓
  ┌──────────────────────────────────────────────────────┐
  │           Unified Transformer (32 layers)             │
  │                                                      │
  │  Layer 1-3:  GDN → FFN (视觉和文本 token 在同一空间交互) │
  │  Layer 4:    Gated Attention → FFN                   │
  │  Layer 5-7:  GDN → FFN                               │
  │  Layer 8:    Gated Attention → FFN                   │
  │  ...                                                 │
  └──────────────────────────────────────────────────────┘
      ↓
[<think> thinking tokens </think> output tokens]
```

### 4.2 Thinking Token 能直接 Attend 到 Vision Token

在 Early Fusion 架构中，thinking token 生成时可以**通过所有层的注意力机制直接访问原始视觉表示**：

- **全注意力层（8 层）**：thinking token 的 Q 向量与 vision token 的 K/V 向量做标准 softmax attention，获得精确的视觉-推理交互
- **GDN 层（24 层）**：thinking token 通过线性注意力更新状态矩阵，vision token 的信息已被编码在状态中

**对比 Qwen3-VL**：Qwen3-VL 的 thinking token 只能访问经过 ViT → MLP 投影后的压缩视觉特征，而非原始视觉 token。

```
Qwen3-VL 中 thinking token 的视觉访问路径：
  Image → ViT编码 → MLP投影 → [压缩视觉token] → LLM → thinking token attend to 压缩视觉token
                                    ↑
                              信息瓶颈在此

Qwen3.5 中 thinking token 的视觉访问路径：
  Image → [原始视觉token] + [文本token] → 统一Transformer → thinking token attend to 原始视觉token
                                                              ↑
                                                         无信息瓶颈
```

### 4.3 Thinking 模式对视觉推理的理论优势

这解释了为什么 Qwen3.5-4B thinking 模式在 MMBench 空间关系上大幅领先（71.11% vs no-thinking 46.67%）：

1. Thinking token 在生成过程中可以**反复 attend 到 vision token**，相当于多次"回看"图像
2. 推理链中的中间结论可以引导后续 thinking token 关注图像中的不同区域
3. 这种"边想边看"的机制在 Early Fusion 架构中比 Encoder-Decoder 架构更高效

**但在实际部署中**：
- 这个优势被 thinking 模式的其他问题（输出溢出、延迟、上下文污染）抵消
- No-thinking 模式下模型仍然能通过 Early Fusion 架构获得比 Qwen3-VL 更好的基础视觉-语言交互能力

---

## 5. Token 预算与截断行为

### 5.1 Thinking Token 与 Output Token 共享预算

Thinking token 和最终输出 token 共享同一个 `max_new_tokens` 预算。没有独立的 "thinking 预算" 和 "output 预算"。

```
max_new_tokens = 1024 的场景：

Thinking 开启:
  <think> ←────────── 800 tokens ──────────→ </think> ← 224 tokens → <|im_end|>
  │              thinking 消耗               │    剩余给输出   │

如果 thinking 链超长:
  <think> ←─────────── 1024 tokens ──────────────→ （生成中止）
  │              thinking 消耗完全部预算                        │
  │              没有生成 </think>，没有最终输出                  │
  │              API 返回 content = None                       │
```

### 5.2 截断时的行为

如果 `max_new_tokens` 在 thinking 阶段就耗尽：
- 生成**直接中止**——不会自动补上 `</think>` 和最终答案
- API 返回的 `content` 字段为 `None`（因为没有 `</think>` 后的内容）
- 只有不完整的 `reasoning_content`

**官方推荐的截断处理方法**：

```
当 thinking token 数接近预算时：
1. 停止生成
2. 注入过渡消息：
   "Considering the limited time by the user, I have to give the solution
    based on the thinking directly now.\n</think>\n\n"
3. 让模型从此处继续生成最终答案
```

Qwen3 技术报告指出，模型具有**从不完整推理中生成合理答案的涌现能力**——这不是显式训练的，而是 Thinking Mode Fusion 阶段训练中自然涌现的。

### 5.3 Thinking Budget 控制方案

由于 thinking token 没有原生的独立预算机制，社区和厂商开发了多种控制方案：

| 方案 | 实现方式 | 状态 |
|------|---------|------|
| **自定义 LogitsProcessor**（Zach Mueller） | HuggingFace `transformers` 自定义处理器，计数 think 内 token 数，超限后强制生成 `</think>` | 已可用 |
| **NVIDIA NIM BudgetControlLogitsProcessor** | 通过 `nvext.max_thinking_tokens` 参数控制 | 生产可用 |
| **vLLM reasoning_budget** | PR #37112，追踪 think 嵌套深度，超限后强制注入 `</think>` | 开发中 |
| **HuggingFace 原生 max_thinking_tokens** | Issue #42111，计划内置支持 | 规划中 |
| **阿里云 DashScope API thinking_budget** | 通过 API 参数直接控制 CoT 最大 token 数 | 生产可用 |

### 5.4 VLN 导航场景中的 Token 预算实测

来自 Habitat VLN 实测数据（model_analysis.md）：

```
Qwen3.5-4B（thinking 默认开启）:
  每步输出: thinking 链（~200-500 tokens） + 动作指令（~10-30 tokens）
  50 个 episode × 平均 59.9 步 = ~2995 步
  615 次解析错误，绝大多数由 thinking 链溢出导致
  97.6% 的步骤包含 </think> 标签

对比 Qwen3-VL-4B（无 thinking）:
  每步输出: 动作指令（~10-30 tokens）
  50 个 episode × 平均 296.7 步 = ~14835 步
  0 次解析错误
```

---

## 6. Soft-Switch 与 Hard-Switch

### 6.1 Hard Switch：`enable_thinking` 参数

这是**模板级控制**，效果是确定性的：

| 设置 | 行为 |
|------|------|
| `enable_thinking=True` | 模型可以自由决定是否生成 `<think>` |
| `enable_thinking=False` | 模板预填充空 think 块，模型**不可能**生成推理链 |

当 `enable_thinking=False` 时，即使在用户消息中写 `/think`，也会被完全忽略。

### 6.2 Soft Switch：`/think` 和 `/no_think`（仅 Qwen3）

在 **Qwen3**（非 Qwen3.5）中，当 `enable_thinking=True` 时，用户可以在消息末尾添加 `/think` 或 `/no_think` 来动态切换：

```
用户消息: "找到客厅的遥控器 /no_think"
→ 模型收到指令后跳过推理链，直接输出

用户消息: "分析一下当前环境 /think"
→ 模型生成推理链后输出
```

**Qwen3.5 不支持 soft switch**。`/think` 和 `/no_think` 内联标签在 Qwen3.5 中无效。思考模式完全通过 `enable_thinking` API 参数控制。

### 6.3 Qwen3.5 API 的三种命名模式

在阿里云 Model Studio API 中，Qwen3.5 支持三种模式：

| 模式名称 | 行为 | 对应设置 |
|---------|------|---------|
| **auto** | 模型自适应决定是否 thinking | `enable_thinking=True`（默认大模型） |
| **thinking** | 强制每次都 thinking | `enable_thinking=True` + 预填充 `<think>` |
| **fast** | 完全禁止 thinking | `enable_thinking=False` |

### 6.4 模型的"自主决定"能力

当 `enable_thinking=True` 时，Qwen3.5 模型可以**自主决定**是否生成推理链。这意味着：
- 简单问题：模型可能跳过 `<think>` 直接回答
- 复杂问题：模型生成完整的 `<think>...</think>` 推理链

**但实测表明这个"自主决定"并不可靠**：
- 在 Habitat VLN 测试中，97.6% 的步骤都生成了推理链，包括很多不需要推理的简单动作步
- 这说明 4B 模型的"自主判断是否需要思考"的能力较弱，倾向于默认开启思考

---

## 7. 两种模式下的完整处理流水线对比

### 7.1 Thinking 开启：完整流水线

```
┌──────────────────────────────────────────────────────────────────────┐
│  1. 输入构造                                                         │
│     system prompt + image tokens + user query                        │
│     + "<|im_start|>assistant\n"                                      │
│     → 不预填充 <think>，让模型自主决定                                  │
├──────────────────────────────────────────────────────────────────────┤
│  2. 采样配置                                                         │
│     temperature=1.0, top_p=0.95, top_k=20                           │
│     → 高温度鼓励推理链中的多样探索                                     │
├──────────────────────────────────────────────────────────────────────┤
│  3. 自回归生成 — Phase 1: Thinking                                   │
│     模型生成 <think> token                                           │
│     逐 token 生成推理链:                                              │
│       - 每个 thinking token 在 8 个全注意力层中扩展 KV cache           │
│       - 每个 thinking token 在 24 个 GDN 层中更新固定状态矩阵          │
│       - 每个 thinking token 可以 attend 到所有之前的 token             │
│         (包括 vision token、system prompt、之前的 thinking token)      │
│     直到模型预测出 </think> (token ID 151668)                         │
│     典型长度: 200-800 tokens                                         │
├──────────────────────────────────────────────────────────────────────┤
│  4. 自回归生成 — Phase 2: Output                                     │
│     模型生成最终答案                                                   │
│       - 此时注意力可以访问: prompt + 完整 thinking 链 + 已生成的输出     │
│       - thinking 链中的推理结论"残留"在注意力中，影响输出               │
│     直到模型预测出 <|im_end|>                                         │
│     典型长度: 10-50 tokens（结构化输出）                                │
├──────────────────────────────────────────────────────────────────────┤
│  5. 输出解析                                                         │
│     搜索 token ID 151668 (</think>)                                  │
│     → 之前的内容 = reasoning_content                                  │
│     → 之后的内容 = content (最终答案)                                  │
│     如果未找到 151668 → content = None（thinking 被截断）              │
├──────────────────────────────────────────────────────────────────────┤
│  6. 总 Token 消耗                                                    │
│     = thinking tokens + output tokens                                │
│     = (200-800) + (10-50) = 210-850 tokens                          │
│     延迟 = thinking 生成时间 + output 生成时间                         │
│     实测: 2.422s / task (homeobjects)                                │
└──────────────────────────────────────────────────────────────────────┘
```

### 7.2 Thinking 关闭：完整流水线

```
┌──────────────────────────────────────────────────────────────────────┐
│  1. 输入构造                                                         │
│     system prompt + image tokens + user query                        │
│     + "<|im_start|>assistant\n<think>\n\n</think>\n\n"               │
│     → 预填充空 think 块，思考"已完成"                                  │
├──────────────────────────────────────────────────────────────────────┤
│  2. 采样配置                                                         │
│     temperature=0.7, top_p=0.8, top_k=20                            │
│     → 低温度确保输出更确定性                                           │
├──────────────────────────────────────────────────────────────────────┤
│  3. 自回归生成 — 仅 Output Phase                                     │
│     模型直接生成最终答案（无 thinking phase）                           │
│       - 注意力访问: prompt + 空 think 块 + 已生成的输出                 │
│       - 无推理链残留，输出完全基于 prompt 中的信息                      │
│     直到模型预测出 <|im_end|>                                         │
│     典型长度: 10-50 tokens                                           │
├──────────────────────────────────────────────────────────────────────┤
│  4. 输出解析                                                         │
│     content = 全部生成内容                                            │
│     reasoning_content = ""（空）                                      │
├──────────────────────────────────────────────────────────────────────┤
│  5. 总 Token 消耗                                                    │
│     = output tokens only                                             │
│     = 10-50 tokens                                                   │
│     延迟 = output 生成时间                                            │
│     实测: 0.529s / task (homeobjects)                                │
└──────────────────────────────────────────────────────────────────────┘
```

### 7.3 关键差异汇总

| 维度 | Thinking 开启 | Thinking 关闭 |
|------|-------------|-------------|
| **Prompt 构造** | 不预填充，模型自主决定 | 预填充空 `<think></think>` 块 |
| **采样策略** | 高温度（1.0），高 top_p（0.95） | 低温度（0.7），低 top_p（0.8） |
| **生成阶段** | 两阶段：thinking → output | 单阶段：output only |
| **新增 Token 数** | 210-850 tokens（thinking + output） | 10-50 tokens（output only） |
| **KV Cache 增长**（全注意力层） | 大幅增长（含 thinking tokens） | 最小增长（仅 output tokens） |
| **KV Cache 增长**（GDN 层） | 固定（状态矩阵大小不变） | 固定（状态矩阵大小不变） |
| **Vision Token 访问** | Thinking token 可多次 attend to vision token | Output token 直接 attend to vision token |
| **上下文消耗** | 快速膨胀（多步累积） | 线性增长（仅 output 累积） |
| **输出格式稳定性** | 低（推理链可能溢出到输出区域） | 高（无推理链干扰） |
| **多步场景** | 推理链累积污染上下文 | 干净的上下文管理 |

---

## 8. 对不同任务类型的处理差异

### 8.1 感知任务（检测、定位）

```
任务: 检测图像中的所有桌子

Thinking 开启:
  <think>
  我看到图像中有几个桌子状的物体。左下角有一个木质餐桌，上面放着餐具。
  中间偏右有一个小茶几，颜色较深。右上角可能有一个边桌，但被沙发部分遮挡。
  让我仔细确认每个物体的位置和边界...
  餐桌: 左下方，约占图像 1/4 面积
  茶几: 中间偏右，较小
  边桌: 右上角，部分遮挡，不确定是否算桌子...
  综合判断，我认为有 2-3 个桌子。
  </think>
  [{"label":"table","bbox":[0.1,0.5,0.4,0.9]},{"label":"table","bbox":[0.5,0.3,0.7,0.6]}]

  问题:
  - 推理链中的犹豫（"不确定是否算桌子"）导致漏检第三个桌子
  - 推理过程消耗 ~400 tokens，远超 10 token 的检测输出
  - 推理链中出现"2-3个"的模糊结论，最终只输出 2 个

Thinking 关闭:
  [{"label":"table","bbox":[0.1,0.5,0.4,0.9]},{"label":"table","bbox":[0.5,0.3,0.7,0.6]},{"label":"table","bbox":[0.7,0.1,0.85,0.35]}]

  优势:
  - 直接从视觉特征生成检测结果，无推理链犹豫
  - 输出 ~30 tokens，格式干净
  - 三个桌子全部检出（包括被遮挡的边桌）
```

**实测验证**（homeobjects）：
- Table 召回: no-thinking 77.27% vs thinking 62.88%（+22.9%）
- 整体 F1: no-thinking 0.8164 vs thinking 0.7734（+5.6%）

### 8.2 空间推理任务（几何关系、方向判断）

```
任务: 判断 table_A 在 table_B 的哪个方向

Thinking 开启（MMBench 风格的描述性推理）:
  <think>
  table_A 在图像左侧约 30% 位置，table_B 在图像右侧约 70% 位置。
  从观察者视角看，table_A 在 table_B 的左边。
  但需要考虑深度关系：table_A 距离相机更近，table_B 更远。
  在三维空间中，table_A 仍然在 table_B 的左侧。
  </think>
  table_A is to the left of table_B.

  → MMBench 空间关系: 71.11%（thinking 开启）

Thinking 关闭（直接感知判断）:
  left_of

  → homeobjects 几何关系: 82.98%（no-thinking）
```

**矛盾解释**：
- MMBench 的空间关系题目是**描述性推理**——需要用文字解释空间关系，thinking 链帮助组织答案
- homeobjects 的几何关系是**直接感知判断**——只需输出 `left_of` 或 `right_of`，thinking 链中的多步推理反而引入犹豫和错误

### 8.3 多步闭环导航任务

```
任务: 导航到目标物体（Habitat VLN）

Thinking 开启:
  Step 1: <think>当前在走廊，目标椅子可能在卧室。应该先找到卧室入口。
          左侧有一个门洞...因此左转是合理的第一步。</think>
          右转30度  ← 推理说左转，输出说右转（自相矛盾）

  Step 2: <think>转了右之后看到另一个房间...ภาษาไทย...   ← 语言混乱
          可能是客厅。目标不在这里。</think>
          直行0.25

  Step 3: <think>已经走了几步，前方似乎看到了类似椅子的物体。
          距离看起来不远，可能已经接近目标。</think>
          停止  ← 实际距离目标 12.3m，过早停止

  → 成功率: 16%（最低），平均仅 59.9 步

Thinking 关闭（Qwen3-VL-4B, 无 thinking 能力）:
  Step 1: 左转30度
  Step 2: 直行0.25
  Step 3: 左转15度
  ...
  Step 297: 停止

  → 成功率: 24%（最高），平均 296.7 步，0 解析错误
```

**多步场景的特有问题链**：

```
Step 1-10:   thinking 链正常工作，推理质量可接受
Step 11-30:  历史 thinking 文本开始占据大量上下文
             → 模型对自身导航状态的追踪能力下降
Step 31-50:  上下文被 thinking 文本严重污染
             → 推理链质量下降，出现语言混乱、自相矛盾
Step 51-60:  模型"自信"判断已到达目标
             → 触发错误停止，episode 结束
             → 平均步数仅 59.9 步
```

---

## 9. 对后续任务流程的影响总结

### 9.1 训练阶段影响

| 训练环节 | Thinking 影响 | 建议 |
|---------|-------------|------|
| **Teacher SFT 数据生成** | Thinking 可提升推理密集型任务的标注质量 | 感知数据用 no-thinking，认知推理数据用 thinking |
| **Teacher SFT 训练** | 混合 thinking/no-thinking 数据需确保模式切换可靠 | 分开训练两种模式的 adapter，或确保 template 标记清晰 |
| **蒸馏数据准备** | Teacher thinking 结论可作为高质量监督信号 | 提取 `</think>` 后的结论部分，丢弃推理链 |
| **Student SFT** | 4B 模型的 thinking 链质量不可靠 | **全程 no-thinking** |
| **RL 训练** | Thinking 降低 rollout 采样效率 4.6x | **全程 no-thinking** |
| **VLN 闭环测试** | Thinking 导致上下文污染和过早停止 | **全程 no-thinking** |

### 9.2 部署阶段影响

| 部署场景 | Thinking 影响 | 建议 |
|---------|-------------|------|
| **端侧实时推理** | 延迟 4.6x，token 预算超标 50-100% | 锁定 `enable_thinking=false` |
| **高频轻量层**（4-5 Hz） | Thinking TPS 需求 3200+，完全不可行 | 锁定 `enable_thinking=false` |
| **离线难例分析** | Thinking 可提升推理质量 | 可选 `enable_thinking=true` |
| **Teacher 在线推理**（如果需要） | Thinking 推理质量更好，但延迟更高 | 按任务类型切换 |

### 9.3 核心原则

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Thinking 模式 = 更高的推理质量 + 更差的执行稳定性                  │
│                                                                  │
│  对于需要"想清楚"的离线单次任务 → 开启 thinking                    │
│  对于需要"做准确"的在线实时任务 → 关闭 thinking                    │
│  对于多步闭环执行任务           → 必须关闭 thinking                │
│                                                                  │
│  Thinking 的价值在于: 提升 Teacher 数据质量                        │
│  Thinking 的风险在于: 破坏 Student 执行稳定性                      │
│                                                                  │
│  正确用法: Teacher (thinking) 生成高质量结论                       │
│           → 蒸馏为 Student (no-thinking) 的直觉判断                │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 参考资料

- [Qwen3.5-4B Model Card (HuggingFace)](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Qwen3.5-27B Model Card (HuggingFace)](https://huggingface.co/Qwen/Qwen3.5-27B)
- [The 4 Things Qwen-3's Chat Template Teaches Us (HuggingFace Blog)](https://huggingface.co/blog/qwen-3-chat-template-deep-dive)
- [Qwen3: Think Deeper, Act Faster (Official Blog)](https://qwenlm.github.io/blog/qwen3/)
- [Qwen3.5: Towards Native Multimodal Agents (Official Blog)](https://qwen.ai/blog?id=qwen3.5)
- [Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/abs/2505.09388)
- [Deep Dive: Qwen 3.5 Brings Native Multimodality (Trilogy AI)](https://trilogyai.substack.com/p/deep-dive-qwen-35-brings-native-multimodality)
- [Qwen3.5: Nobody Agrees on Attention Anymore (HuggingFace Blog)](https://huggingface.co/blog/mlabonne/qwen35)
- [Limiting Qwen 3's Thinking (Zach Mueller)](https://muellerzr.github.io/til/end_thinking.html)
- [NVIDIA NIM Thinking Budget Control](https://docs.nvidia.com/nim/large-language-models/latest/thinking-budget-control.html)
- [vLLM PR #37112: Reasoning Budget](https://github.com/vllm-project/vllm/pull/37112)
- [HuggingFace Transformers Issue #42111](https://github.com/huggingface/transformers/issues/42111)
- [阿里云 Deep Thinking 文档](https://www.alibabacloud.com/help/en/model-studio/deep-thinking)
- [Qwen3.5 Hybrid Attention Explained](https://ai.tekin.cn/en/blog/qwen3-5-hybrid-attention-gated-deltanet-moe-deployment)
- [QwenLM/Qwen3.5 Issue #97](https://github.com/QwenLM/Qwen3.5/issues/97)
- 本项目 homeobjects 实测报告: `test_report_qwen35_4b.md`, `test_report_nothinking.md`
- 本项目 VLN 导航实测报告: `model_analysis.md`
