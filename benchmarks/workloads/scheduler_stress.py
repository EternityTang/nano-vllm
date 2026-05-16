from __future__ import annotations

# 中文说明：
# P0/P1 scheduler 压力 workload，混合短中 prompt 和不同输出长度，用于观察调度队列、TTFT、TPOT 和 admission 行为。
# P1 memory-aware scheduler 的 decode-first、chunked prefill 和 starvation guard 回归主要依赖该请求分布。

from random import Random


def generate(concurrency: int, max_requests: int, seed: int = 0) -> list[dict]:
    rng = Random(seed)
    requests = []
    for i in range(max_requests):
        prompt_len = rng.randint(16, 256)
        output_len = rng.randint(16, 128)
        requests.append(
            {
                "request_id": f"scheduler_stress-{i}",
                "prompt_token_ids": [rng.randint(1, 10000) for _ in range(prompt_len)],
                "output_tokens": output_len,
                "arrival_ts": i / max(concurrency, 1),
            }
        )
    return requests
