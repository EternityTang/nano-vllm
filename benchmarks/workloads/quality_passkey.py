from __future__ import annotations

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
