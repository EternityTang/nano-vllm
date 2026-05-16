from __future__ import annotations

# 中文说明：
# P1/P2 shared-prefix workload，生成带公共前缀的多请求集合，用于触发 prefix cache 共享、owner refs 和 metadata policy 保护逻辑。
# P2 metadata dry-run 使用它验证 shared-prefix ref_count、保护比例和 conservative reclaimable blocks 是否能被报告出来。

from random import Random


def generate(concurrency: int, max_requests: int, seed: int = 0) -> list[dict]:
    rng = Random(seed)
    shared_prefix = [rng.randint(1, 10000) for _ in range(512)]
    requests = []
    for i in range(max_requests):
        suffix_len = rng.randint(16, 128)
        output_len = rng.randint(32, 128)
        requests.append(
            {
                "request_id": f"shared_prefix-{i}",
                "prompt_token_ids": shared_prefix + [rng.randint(1, 10000) for _ in range(suffix_len)],
                "output_tokens": output_len,
                "arrival_ts": i / max(concurrency, 1),
            }
        )
    return requests
