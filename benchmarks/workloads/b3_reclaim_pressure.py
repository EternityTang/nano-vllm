from __future__ import annotations

# 中文说明：
# P4a/B3 reclaim 压力 workload，构造超过两个 KV block 的 prompt 和持续 decode，让 memory-aware scheduler/admission
# 与 ARKV Q8 runtime 在不削弱 sink/recent/inflight 保护的前提下产生旧 FULL block reclaim。

from random import Random


def generate(concurrency: int, max_requests: int, seed: int = 0) -> list[dict]:
    rng = Random(seed)
    requests = []
    for i in range(max_requests):
        prompt_len = rng.randint(768, 1280)
        output_len = rng.randint(32, 96)
        requests.append(
            {
                "request_id": f"b3_reclaim_pressure-{i}",
                "prompt_token_ids": [rng.randint(1, 10000) for _ in range(prompt_len)],
                "output_tokens": output_len,
                "arrival_ts": i / max(concurrency, 1),
            }
        )
    return requests
