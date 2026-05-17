"""Microbenchmark P6b decode-only mixed-KV attention."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanovllm.kernels.mixed_kv_decode_attention import (
    FULL_STATE,
    QUANT_STATE,
    mixed_kv_decode_attention,
    mixed_kv_decode_attention_reference,
    mixed_kv_decode_attention_supported,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark mixed FULL/Q8 decode attention.")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--kv-mix-ratio", type=float, default=0.5)
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    torch.manual_seed(44)
    dtype = torch.float16
    q = torch.randn(args.batch_size, args.q_heads, args.head_dim, dtype=dtype, device=device)
    full_k = torch.randn(args.blocks, args.block_size, args.kv_heads, args.head_dim, dtype=dtype, device=device)
    full_v = torch.randn_like(full_k)
    quant_k = torch.randint(-64, 63, (args.blocks, args.block_size, args.kv_heads, args.head_dim), dtype=torch.int8, device=device)
    quant_v = torch.randint(-64, 63, quant_k.shape, dtype=torch.int8, device=device)
    k_scale = torch.rand(args.blocks, args.block_size, args.kv_heads, 1, dtype=torch.float32, device=device) * 0.02
    v_scale = torch.rand_like(k_scale) * 0.02
    visible, counts = make_visible_entries(
        batch_size=args.batch_size,
        blocks=args.blocks,
        block_size=args.block_size,
        kv_mix_ratio=args.kv_mix_ratio,
        device=device,
    )
    scale = args.head_dim**-0.5

    reference = mixed_kv_decode_attention_reference(
        q,
        full_k,
        full_v,
        quant_k,
        quant_v,
        k_scale,
        v_scale,
        visible,
        counts,
        softmax_scale=scale,
    )

    triton_supported = mixed_kv_decode_attention_supported(
        head_dim=args.head_dim,
        dtype=dtype,
        block_size=args.block_size,
        device=device,
        num_q_heads=args.q_heads,
        num_kv_heads=args.kv_heads,
    )
    triton_ms = None
    max_abs_diff = None
    if triton_supported:
        actual = mixed_kv_decode_attention(
            q,
            full_k,
            full_v,
            quant_k,
            quant_v,
            k_scale,
            v_scale,
            visible,
            counts,
            softmax_scale=scale,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        max_abs_diff = float((actual - reference).abs().max().item())
        triton_ms = bench(
            lambda: mixed_kv_decode_attention(
                q,
                full_k,
                full_v,
                quant_k,
                quant_v,
                k_scale,
                v_scale,
                visible,
                counts,
                softmax_scale=scale,
            ),
            args.warmup,
            args.iters,
            device,
        )

    reference_ms = bench(
        lambda: mixed_kv_decode_attention_reference(
            q,
            full_k,
            full_v,
            quant_k,
            quant_v,
            k_scale,
            v_scale,
            visible,
            counts,
            softmax_scale=scale,
        ),
        max(1, args.warmup // 2),
        max(1, args.iters // 5),
        device,
    )
    tokens = args.batch_size * args.blocks * args.block_size
    print(
        json.dumps(
            {
                "device": str(device),
                "head_dim": args.head_dim,
                "block_size": args.block_size,
                "blocks": args.blocks,
                "batch_size": args.batch_size,
                "q_heads": args.q_heads,
                "kv_heads": args.kv_heads,
                "kv_mix_ratio": args.kv_mix_ratio,
                "triton_supported": triton_supported,
                "reference_ms": reference_ms,
                "triton_ms": triton_ms,
                "speedup": None if triton_ms is None else reference_ms / triton_ms,
                "tokens": tokens,
                "max_abs_diff": max_abs_diff,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def make_visible_entries(
    *,
    batch_size: int,
    blocks: int,
    block_size: int,
    kv_mix_ratio: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    quant_blocks = int(round(blocks * kv_mix_ratio))
    quant_start = blocks - quant_blocks
    visible = torch.empty(batch_size, blocks, 5, dtype=torch.int64, device=device)
    for batch_idx in range(batch_size):
        for block_idx in range(blocks):
            is_quant = block_idx >= quant_start
            visible[batch_idx, block_idx, 0] = QUANT_STATE if is_quant else FULL_STATE
            visible[batch_idx, block_idx, 1] = -1 if is_quant else block_idx
            visible[batch_idx, block_idx, 2] = block_idx if is_quant else -1
            visible[batch_idx, block_idx, 3] = block_idx * block_size
            visible[batch_idx, block_idx, 4] = (block_idx + 1) * block_size
    counts = torch.full((batch_size,), blocks, dtype=torch.int32, device=device)
    return visible, counts


def bench(fn, warmup: int, iters: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


if __name__ == "__main__":
    raise SystemExit(main())
