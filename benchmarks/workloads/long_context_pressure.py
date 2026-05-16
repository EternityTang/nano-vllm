from __future__ import annotations

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
