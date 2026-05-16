from __future__ import annotations

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
