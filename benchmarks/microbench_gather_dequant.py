#!/usr/bin/env python3
"""Microbenchmark P6a Q8 gather/dequant materialization."""

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

from nanovllm.engine.kv_meta import KVBlockState
from nanovllm.engine.quant_cache import QuantCache, QuantCacheSpec
from nanovllm.engine.visible_tables import VisibleBlockEntry
from nanovllm.kernels.triton_gather_dequant import (
    gather_dequant_reference,
    gather_dequant_triton,
    triton_gather_dequant_supported,
)


def make_entry(index: int, block_size: int) -> VisibleBlockEntry:
    return VisibleBlockEntry(
        seq_id=0,
        logical_block_id=index,
        storage_id=index,
        state=KVBlockState.QUANT,
        full_block_id=None,
        quant_block_id=index,
        logical_start=index * block_size,
        logical_end=(index + 1) * block_size,
        visible_start=index * block_size,
        visible_end=(index + 1) * block_size,
    )


def build_cache(args) -> tuple[QuantCache, list[VisibleBlockEntry], torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    spec = QuantCacheSpec(
        num_quant_blocks=args.quant_blocks,
        num_layers=1,
        block_size=args.block_size,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        dtype=torch.float16,
        device=device,
    )
    cache = QuantCache(spec)
    torch.manual_seed(args.seed)
    full = torch.randn(spec.block_shape, dtype=torch.float16, device=device)
    entries = []
    for index in range(args.quant_blocks):
        quant_id = cache.allocate()
        cache.write_from_full(quant_id, full)
        entries.append(make_entry(index, args.block_size))
    return cache, entries, device


def time_cuda_or_cpu(fn, iters: int, warmup: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = perf_counter()
    for _ in range(iters):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (perf_counter() - started) / iters


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Q8 gather/dequant materialization.")
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--quant-blocks", type=int, required=True)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    cache, entries, device = build_cache(args)
    output = torch.empty(
        args.quant_blocks,
        2,
        args.block_size,
        args.num_kv_heads,
        args.head_dim,
        dtype=torch.float16,
        device=device,
    )
    ref_latency = time_cuda_or_cpu(
        lambda: gather_dequant_reference(cache, entries, output, layer_id=0),
        args.iters,
        args.warmup,
        device,
    )

    triton_supported = triton_gather_dequant_supported(args.head_dim, torch.float16, args.block_size, device)
    triton_latency = None
    max_abs_diff = None
    if triton_supported:
        ids = torch.arange(args.quant_blocks, dtype=torch.int64, device=device)
        triton_output = torch.empty_like(output)
        triton_latency = time_cuda_or_cpu(
            lambda: gather_dequant_triton(
                cache.q_cache[:, 0, 0],
                cache.q_cache[:, 1, 0],
                cache.scales[:, 0, 0],
                cache.scales[:, 1, 0],
                ids,
                triton_output[:, 0],
                triton_output[:, 1],
                block_size=args.block_size,
                head_dim=args.head_dim,
            ),
            args.iters,
            args.warmup,
            device,
        )
        reference = gather_dequant_reference(cache, entries, torch.empty_like(output), layer_id=0)
        gather_dequant_triton(
            cache.q_cache[:, 0, 0],
            cache.q_cache[:, 1, 0],
            cache.scales[:, 0, 0],
            cache.scales[:, 1, 0],
            ids,
            triton_output[:, 0],
            triton_output[:, 1],
            block_size=args.block_size,
            head_dim=args.head_dim,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        max_abs_diff = float((reference.float() - triton_output.float()).abs().max().item())

    element_bytes = torch.tensor([], dtype=torch.float16).element_size()
    bytes_per_iter = args.quant_blocks * 2 * args.block_size * args.num_kv_heads * (
        args.head_dim * (1 + element_bytes) + 4
    )
    result = {
        "device": str(device),
        "head_dim": args.head_dim,
        "block_size": args.block_size,
        "quant_blocks": args.quant_blocks,
        "num_kv_heads": args.num_kv_heads,
        "reference_latency_ms": ref_latency * 1000,
        "reference_bandwidth_gb_s": bytes_per_iter / ref_latency / 1e9,
        "triton_supported": triton_supported,
        "triton_latency_ms": triton_latency * 1000 if triton_latency is not None else None,
        "triton_bandwidth_gb_s": bytes_per_iter / triton_latency / 1e9 if triton_latency else None,
        "max_abs_diff": max_abs_diff,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
