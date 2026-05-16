from __future__ import annotations

# 中文说明：
# P0 及后续阶段的长上下文压力 workload，用较长 prompt 和中等输出长度模拟 KV cache 容量压力。
# 该 workload 用来观察 raw/effective KV memory、OOM 风险和后续 quant/reclaim 策略在长上下文场景下的效果。

from random import Random


def generate(concurrency: int, max_requests: int, seed: int = 0) -> list[dict]:
    rng = Random(seed)
    requests = []
    for i in range(max_requests):
        prompt_len = rng.randint(1024, 4096)
        output_len = rng.randint(64, 256)
        requests.append(
            {
                "request_id": f"long_context_pressure-{i}",
                "prompt_token_ids": [rng.randint(1, 10000) for _ in range(prompt_len)],
                "output_tokens": output_len,
                "arrival_ts": i / max(concurrency, 1),
            }
        )
    return requests
