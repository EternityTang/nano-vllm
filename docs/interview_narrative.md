# Memory-Aware Optimizer Interview Narrative

## 一句话说明

这个项目把 Nano-vLLM 的 KV cache 从单一 FULL 存储，扩展成默认关闭、可回滚、可观测的 FULL/QUANT/EVICT 分级系统，并通过 scheduler/admission、VisibleBlockTable 和 mixed-KV kernel 让节省下来的 full blocks 真正回到 serving 循环。

## 为什么这样做

长上下文推理的核心瓶颈不是模型权重，而是 KV cache。原始 Nano-vLLM 的 full-only KV path 简洁但缺少以下能力：

- scheduler 不知道未来 decode reserve。
- admission 不能利用可回收 KV。
- KV block 没有独立的 logical / physical / visible 视图。
- FULL->QUANT 没有 rollback-safe commit。
- attention 不能安全读取 FULL/QUANT 混合上下文。

因此优化目标不是单点 kernel，而是先让内存回收进入真实 serving loop，再逐步减少 fallback 开销。

## 关键设计

1. P-1/P0 先校准仓库能力和 benchmark schema，不把不成立的研究假设写进结果。
2. P1 只做同质 batch scheduler/admission，避免 runner 语义和调度语义同时变化。
3. P2 拆分 `PhysicalBlockMeta`、`SequenceKVRef`、`VisibleBlockTable`，让 write view 和 read view 分离。
4. P3 用统一 `total_kv_budget_bytes` 切分 full/quant/scale/scratch/metadata，并实现 FULL->QUANT 两阶段提交。
5. P4a 实现 decode-only mixed-KV fallback，让 QUANT block 能被真实读取，commit 后旧 FULL 才能释放。
6. P5 把 EVICT 锁在 quality-gated optional path，避免污染 QUANT-only headline。
7. P6a/P6b 分别加速 gather/dequant 和 decode-only mixed-KV attention。
8. P6c 用 profile 解释端到端差距，并把 metadata packing 从 per-layer hot path 移到 decode preparation。

## 如何保证正确性

- 所有 optimizer flags 默认关闭。
- `slot_mapping` 仍只写 FULL。
- `VisibleBlockTable` 是 mixed-KV attention 唯一读视图。
- FULL->QUANT 失败时 rollback，commit 成功后才释放旧 FULL。
- Triton path 都有 torch reference、parity test 和 fallback。
- EVICT 只在 P5 之后且 quality gate 通过时作为 optional result。

## 当前结果怎么解释

P6c profile 是最清晰的端到端解释：

- B3 QUANT-only fallback：`slo_goodput_tokens_per_s=23.04`，`fallback_count=15512`。
- B4 Triton gather/dequant：`slo_goodput_tokens_per_s=25.16`，说明 materialization 有改善但不是唯一瓶颈。
- B5 fused decode kernel：`slo_goodput_tokens_per_s=38.64`，`fused_kernel_calls=15512`，`fused_kernel_fallbacks=0`，说明 fused path 已真实生效。

旧 B5 只有 `10.02 tok/s`，P6c 后提升到 `38.64 tok/s`，主要原因是消除了每层 attention 重复 packing visible metadata 的 Python/CUDA scalar write 开销。剩余差距主要来自 eager model-forward overhead 和大量小 kernel launch。

## 面试中应强调的取舍

- 先做 correctness 和 serving activation，再做 kernel speedup。
- QUANT 和 EVICT 是不同风险等级：QUANT 保留 token，EVICT 改变上下文语义。
- 指标不能只看 raw peak VRAM，必须看 effective KV memory、free full blocks、OOM rate、stable concurrency 和 SLO-goodput。
- B4/B5 不依赖 EVICT，因为 kernel 加速要和容量删除收益隔离。
- fallback 不是临时 hack，而是 release 安全边界。

## 下一步

P7 之后的安全方向是发布前消融复跑、文档校准和小范围 hardening。不要在 P7 新增 optimizer 行为；新的 policy 或 kernel fusion 应作为 P8/P9 独立阶段重新定义风险门控。
