from __future__ import annotations

# 中文说明：
# P0 预留的质量门控 workload，在长 filler 中插入 passkey，供后续 P5 EVICT quality gate 检查语义保真度。
# 当前阶段主要生成确定性请求和 expected_passkey 元数据，不直接判分；后续质量测试会读取这些字段。

from random import Random


def generate(concurrency: int, max_requests: int, seed: int = 0) -> list[dict]:
    rng = Random(seed)
    requests = []
    for i in range(max_requests):
        passkey = 100000 + i
        filler = [rng.randint(1, 10000) for _ in range(1024)]
        requests.append(
            {
                "request_id": f"quality_passkey-{i}",
                "prompt_token_ids": filler[:512] + [passkey] + filler[512:],
                "output_tokens": 32,
                "arrival_ts": i / max(concurrency, 1),
                "expected_passkey": passkey,
            }
        )
    return requests
