#!/usr/bin/env python3
"""Microbenchmark for the torch reference Q8 KV path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanovllm.kernels.q8_kv import dequantize_q8_reference, quantize_q8_reference


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Torch reference Q8 KV microbenchmark.")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--iters", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full = torch.randn(2, args.layers, args.block_size, args.kv_heads, args.head_dim, device=device, dtype=dtype)

    if device == "cuda":
        torch.cuda.synchronize()
    started = perf_counter()
    quantized = scales = restored = None
    for _ in range(args.iters):
        quantized, scales = quantize_q8_reference(full)
        restored = dequantize_q8_reference(quantized, scales, dtype)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = perf_counter() - started
    max_abs_error = (full.float() - restored.float()).abs().max().item()
    print(
        json.dumps(
            {
                "status": "ok",
                "device": device,
                "dtype": args.dtype,
                "head_dim": args.head_dim,
                "block_size": args.block_size,
                "iters": args.iters,
                "seconds": elapsed,
                "iter_per_s": args.iters / elapsed if elapsed > 0 else 0.0,
                "max_abs_error": max_abs_error,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
