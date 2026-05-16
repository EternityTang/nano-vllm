#!/usr/bin/env python3
# 中文说明：
# P-1 阶段能力探针入口，用于在不改变推理运行时行为的前提下记录仓库当前能支持的模型、KV block size、依赖环境、GPU 可见性和 mixed-KV fallback 参考实现状态。
# 该脚本默认 dry-run，不加载大模型权重；输出的 JSON 是后续 P0/P1/P2 benchmark 选择正式模型、block size 和验证前置条件的事实依据。
from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "nanovllm" / "config.py"
MODEL_RUNNER_PATH = REPO_ROOT / "nanovllm" / "engine" / "model_runner.py"
ATTENTION_PATH = REPO_ROOT / "nanovllm" / "layers" / "attention.py"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "p_minus_1_capability.json"
GIB = 1024**3


@dataclass(slots=True)
class ProbeOptions:
    dry_run: bool = True
    output_json: Path | None = None
    model_roots: tuple[Path, ...] = ()
    target_gpu_memory_gb: float = 12.0
    run_import_probe: bool = True
    run_gpu_probe: bool = True


def default_model_roots() -> tuple[Path, ...]:
    home = Path.home()
    env_roots = tuple(
        Path(root).expanduser()
        for root in os.environ.get("NANO_VLLM_MODEL_ROOTS", "").split(os.pathsep)
        if root
    )
    return (
        *env_roots,
        REPO_ROOT / "weight",
        home / "huggingface",
        home / "models",
        home / ".cache" / "huggingface" / "hub",
    )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run_command(command: list[str], timeout: float = 10.0) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - host-specific failure path
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _find_config_default_and_alignment() -> dict[str, Any]:
    source = _read_text(CONFIG_PATH)
    tree = ast.parse(source)
    default_block_size = None
    alignment = None

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Config":
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.target, ast.Name)
                    and stmt.target.id == "kvcache_block_size"
                    and isinstance(stmt.value, ast.Constant)
                ):
                    default_block_size = stmt.value.value
        if (
            isinstance(node, ast.Assert)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.BinOp)
            and isinstance(node.test.left.op, ast.Mod)
            and isinstance(node.test.left.right, ast.Constant)
        ):
            left = node.test.left
            if isinstance(left.left, ast.Attribute) and left.left.attr == "kvcache_block_size":
                alignment = left.right.value

    candidates = (16, 32, 64, 128, 256, 512)
    supported = {
        str(candidate): (
            candidate % alignment == 0 if isinstance(alignment, int) and alignment > 0 else candidate == default_block_size
        )
        for candidate in candidates
    }
    formal_block_size = int(default_block_size or 256)
    if not supported.get(str(formal_block_size), False):
        formal_block_size = 256

    return {
        "config_default_block_size": default_block_size,
        "config_alignment_requirement": alignment,
        "candidate_support": supported,
        "formal_block_size": formal_block_size,
        "smaller_than_256_supported": any(
            supported[str(candidate)] for candidate in candidates if candidate < 256
        ),
        "reason": "Config.__post_init__ requires kvcache_block_size % 256 == 0.",
    }


def _model_runner_probe() -> dict[str, Any]:
    source = _read_text(MODEL_RUNNER_PATH)
    tree = ast.parse(source)
    constructs_qwen3 = False
    imported_qwen3 = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "nanovllm.models.qwen3":
            imported_qwen3 = any(alias.name == "Qwen3ForCausalLM" for alias in node.names)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            constructs_qwen3 = constructs_qwen3 or node.func.id == "Qwen3ForCausalLM"
    return {
        "repo_model_family": "qwen3" if imported_qwen3 and constructs_qwen3 else "unknown",
        "qwen3_first": imported_qwen3 and constructs_qwen3,
        "model_runner_constructs": "Qwen3ForCausalLM" if constructs_qwen3 else None,
        "non_qwen3_models": {
            "formal_ablation": "optional",
            "reason": "ModelRunner directly constructs Qwen3ForCausalLM; model adapters are not present.",
        },
    }


def _attention_backend_probe() -> dict[str, Any]:
    source = _read_text(ATTENTION_PATH)
    return {
        "prefill_backend": "flash_attn_varlen_func" if "flash_attn_varlen_func" in source else "unknown",
        "decode_backend": "flash_attn_with_kvcache" if "flash_attn_with_kvcache" in source else "unknown",
        "uses_block_tables": "block_table" in source,
    }


def _iter_config_paths(roots: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in roots:
        root = root.expanduser()
        if not root.exists():
            continue
        if root.is_file() and root.name == "config.json":
            paths = [root]
        elif (root / "config.json").is_file():
            paths = [root / "config.json"]
        else:
            paths = root.rglob("config.json")
        for path in paths:
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            yield resolved


def _model_size_from_name(path: Path) -> float | None:
    name = path.name.lower()
    match = re.search(r"qwen3-(\d+(?:\.\d+)?)b", name)
    if match:
        return float(match.group(1))
    return None


def _model_weight_bytes(path: Path) -> int:
    total = 0
    for weight in path.glob("*.safetensors"):
        try:
            total += weight.stat().st_size
        except OSError:
            pass
    return total


def discover_qwen3_models(roots: Iterable[Path]) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for config_path in _iter_config_paths(roots):
        try:
            config = _read_json(config_path)
        except (OSError, json.JSONDecodeError):
            continue
        architectures = config.get("architectures") or []
        is_qwen3 = config.get("model_type") == "qwen3" or "Qwen3ForCausalLM" in architectures
        if not is_qwen3:
            continue
        model_path = config_path.parent
        quantization = config.get("quantization_config")
        weight_bytes = _model_weight_bytes(model_path)
        supported_by_repo = not quantization and "Qwen3ForCausalLM" in architectures
        models.append(
            {
                "path": str(model_path),
                "name": model_path.name,
                "architectures": architectures,
                "model_type": config.get("model_type"),
                "torch_dtype": config.get("torch_dtype"),
                "max_position_embeddings": config.get("max_position_embeddings"),
                "num_hidden_layers": config.get("num_hidden_layers"),
                "num_attention_heads": config.get("num_attention_heads"),
                "num_key_value_heads": config.get("num_key_value_heads"),
                "hidden_size": config.get("hidden_size"),
                "quantized": bool(quantization),
                "weight_bytes": weight_bytes,
                "size_b_from_name": _model_size_from_name(model_path),
                "supported_by_current_loader": supported_by_repo,
                "support_note": (
                    "supported Qwen3 safetensors path"
                    if supported_by_repo
                    else "quantized or non-standard Qwen3 path; current loader has no quant adapter"
                ),
            }
        )
    return sorted(
        models,
        key=lambda item: (
            item["size_b_from_name"] is None,
            item["size_b_from_name"] or 0,
            item["path"],
        ),
    )


def select_formal_model(models: list[dict[str, Any]], target_gpu_memory_gb: float) -> dict[str, Any]:
    supported = [model for model in models if model["supported_by_current_loader"]]
    if not supported:
        return {
            "status": "blocked_no_supported_qwen3_model",
            "path": None,
            "reason": "No local non-quantized Qwen3ForCausalLM config with safetensors was found.",
        }

    repo_weight_root = REPO_ROOT / "weight"
    repo_weight_models = [
        model
        for model in supported
        if Path(model["path"]).resolve().is_relative_to(repo_weight_root.resolve())
    ]
    selection_scope = repo_weight_models or supported
    dry_run_weight_budget = int(target_gpu_memory_gb * 0.75 * GIB)
    fitting = [
        model
        for model in selection_scope
        if model["weight_bytes"] == 0 or model["weight_bytes"] <= dry_run_weight_budget
    ]
    pool = fitting or selection_scope
    selected = max(
        pool,
        key=lambda item: (
            item["size_b_from_name"] or 0,
            item["weight_bytes"],
            item["path"],
        ),
    )
    status = "dry_run_selected"
    reason = (
        "Selected largest repo-local weight/ Qwen3 path within a conservative "
        f"{target_gpu_memory_gb:.1f}GB dry-run weight budget."
    )
    if selected not in fitting:
        status = "requires_real_gpu_probe"
        reason = "No supported Qwen3 path in the selected scope fits the dry-run weight budget; real GPU probe required."

    return {
        "status": status,
        "path": selected["path"],
        "name": selected["name"],
        "weight_bytes": selected["weight_bytes"],
        "size_b_from_name": selected["size_b_from_name"],
        "reason": reason,
    }


def materialize_visible_kv(
    k_cache: Any,
    v_cache: Any,
    block_table: Iterable[int],
    context_len: int,
    block_size: int,
) -> tuple[Any, Any]:
    import torch

    k_pieces = []
    v_pieces = []
    remaining = context_len
    for block_id in block_table:
        if remaining <= 0:
            break
        if block_id < 0:
            break
        take = min(block_size, remaining)
        k_pieces.append(k_cache[block_id, :take])
        v_pieces.append(v_cache[block_id, :take])
        remaining -= take
    if remaining != 0:
        raise ValueError("block_table does not cover context_len")
    return torch.cat(k_pieces, dim=0), torch.cat(v_pieces, dim=0)


def _expand_kv_heads(visible: Any, num_q_heads: int) -> Any:
    num_kv_heads = visible.shape[1]
    if num_kv_heads == num_q_heads:
        return visible
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be a multiple of num_kv_heads")
    return visible.repeat_interleave(num_q_heads // num_kv_heads, dim=1)


def decode_attention_reference(q: Any, k_visible: Any, v_visible: Any, scale: float) -> Any:
    import torch

    k_for_q = _expand_kv_heads(k_visible, q.shape[0])
    v_for_q = _expand_kv_heads(v_visible, q.shape[0])
    scores = torch.einsum("hd,shd->hs", q.float(), k_for_q.float()) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hs,shd->hd", probs, v_for_q.float()).to(q.dtype)


def run_mixed_kv_reference_check() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - dependency-specific
        return {
            "status": "blocked",
            "reason": f"torch import failed: {type(exc).__name__}: {exc}",
        }

    torch.manual_seed(0)
    block_size = 4
    context_len = 9
    num_blocks = 3
    num_kv_heads = 2
    num_q_heads = 4
    head_dim = 8
    block_table = [2, 0, 1]
    scale = head_dim**-0.5

    k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
    v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
    q = torch.randn(num_q_heads, head_dim)

    k_scratch, v_scratch = materialize_visible_kv(
        k_cache,
        v_cache,
        block_table=block_table,
        context_len=context_len,
        block_size=block_size,
    )
    k_full = torch.cat([k_cache[block_id] for block_id in block_table], dim=0)[:context_len]
    v_full = torch.cat([v_cache[block_id] for block_id in block_table], dim=0)[:context_len]
    full_output = decode_attention_reference(q, k_full, v_full, scale)
    scratch_output = decode_attention_reference(q, k_scratch, v_scratch, scale)

    return {
        "status": "passed",
        "reference": "cpu_torch_full_cache_to_visible_scratch_decode_attention",
        "context_len": context_len,
        "block_size": block_size,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "materialization_allclose": bool(torch.allclose(k_scratch, k_full) and torch.allclose(v_scratch, v_full)),
        "decode_output_allclose": bool(torch.allclose(scratch_output, full_output, atol=1e-6, rtol=1e-6)),
        "atol": 1e-6,
        "rtol": 1e-6,
    }


def _probe_environment(run_import_probe: bool, run_gpu_probe: bool) -> dict[str, Any]:
    pyproject = _read_text(PYPROJECT_PATH)
    packages = {
        "torch": _package_version("torch"),
        "triton": _package_version("triton"),
        "transformers": _package_version("transformers"),
        "huggingface-hub": _package_version("huggingface-hub"),
        "flash-attn": _package_version("flash-attn"),
        "pytest": _package_version("pytest"),
    }
    import_smoke = None
    if run_import_probe:
        import_smoke = _run_command(
            [sys.executable, "-c", "import nanovllm; print(nanovllm.__all__)"],
            timeout=20.0,
        )
    gpu_probe = None
    if run_gpu_probe:
        gpu_probe = _run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            timeout=10.0,
        )
    return {
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "project_requires": ">=3.10,<3.13" if 'requires-python = ">=3.10,<3.13"' in pyproject else None,
            "matches_project_requires": sys.version_info >= (3, 10) and sys.version_info < (3, 13),
        },
        "packages": packages,
        "import_smoke": import_smoke,
        "gpu_probe": gpu_probe,
    }


def _validation_blockers(environment: dict[str, Any], formal_model: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    python_info = environment["python"]
    if not python_info["matches_project_requires"]:
        blockers.append(
            f"Python {python_info['version']} does not satisfy project requires-python {python_info['project_requires']}."
        )
    packages = environment["packages"]
    if packages.get("flash-attn") is None:
        blockers.append("flash-attn is not installed in the active interpreter.")
    if packages.get("pytest") is None:
        blockers.append("pytest is not installed in the active interpreter.")
    import_smoke = environment.get("import_smoke")
    if import_smoke and not import_smoke.get("ok"):
        blockers.append("nanovllm import smoke failed in the active interpreter.")
    gpu_probe = environment.get("gpu_probe")
    if gpu_probe and not gpu_probe.get("ok"):
        blockers.append("nvidia-smi GPU probe failed; real B0 smoke is blocked on GPU access.")
    if formal_model["status"].startswith("blocked"):
        blockers.append(formal_model["reason"])
    return blockers


def build_capability_report(options: ProbeOptions | None = None) -> dict[str, Any]:
    options = options or ProbeOptions()
    roots = options.model_roots or default_model_roots()
    kv_policy = _find_config_default_and_alignment()
    model_runner = _model_runner_probe()
    attention_backend = _attention_backend_probe()
    available_models = discover_qwen3_models(roots)
    formal_model = select_formal_model(available_models, options.target_gpu_memory_gb)
    mixed_reference = run_mixed_kv_reference_check()
    environment = _probe_environment(options.run_import_probe, options.run_gpu_probe)
    formal_block_size = kv_policy["formal_block_size"]

    max_smoke_config = {
        "status": "requires_real_gpu_probe",
        "candidate": {
            "model": formal_model.get("path"),
            "max_model_len": 4096,
            "max_num_seqs": 2,
            "max_num_batched_tokens": 4096,
            "kvcache_block_size": formal_block_size,
            "enforce_eager": True,
        },
        "reason": "Dry-run records a conservative candidate; actual maximum requires GPU/model-load smoke.",
    }
    if formal_model["status"].startswith("blocked"):
        max_smoke_config["status"] = "blocked_no_formal_model"

    report = {
        "schema_version": 1,
        "phase": "P-1 repo capability calibration / smoke spike",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": options.dry_run,
        "optimizer_behavior_enabled": False,
        "repo": {
            "root": str(REPO_ROOT),
            "source_constraints": {
                "config": str(CONFIG_PATH.relative_to(REPO_ROOT)),
                "model_runner": str(MODEL_RUNNER_PATH.relative_to(REPO_ROOT)),
                "attention": str(ATTENTION_PATH.relative_to(REPO_ROOT)),
            },
        },
        "environment": environment,
        "model_support": {
            **model_runner,
            "available_qwen3_models": available_models,
            "formal_benchmark_model": formal_model,
        },
        "attention_backend": attention_backend,
        "kv_cache": kv_policy,
        "mixed_kv_fallback_reference": mixed_reference,
        "cuda_graph_policy": {
            "enable_mixed_kv_fallback_default": False,
            "requires_enforce_eager_when_enabled": True,
            "policy": "enable_mixed_kv_fallback=True implies enforce_eager=True until P6b graph safety is proven.",
            "reason": "Current decode path replays CUDA graphs when enforce_eager=False; dynamic mixed visible tables are not graph-safe yet.",
        },
        "max_smoke_config": max_smoke_config,
        "validation_commands": [
            'python -c "import nanovllm; print(nanovllm.__all__)"',
            "python benchmarks/capability_probe.py --dry-run --output-json results/p_minus_1_capability.json",
            "python -m pytest tests/bench/test_capability_smoke.py -v",
        ],
    }
    report["blockers"] = _validation_blockers(environment, formal_model)
    return report


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run Nano-VLLM P-1 capability calibration.")
    parser.add_argument("--dry-run", action="store_true", help="Do not load model weights or run generation.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model-root", action="append", type=Path, default=[])
    parser.add_argument("--target-gpu-memory-gb", type=float, default=12.0)
    parser.add_argument("--skip-import-probe", action="store_true")
    parser.add_argument("--skip-gpu-probe", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    options = ProbeOptions(
        dry_run=args.dry_run,
        output_json=args.output_json,
        model_roots=tuple(args.model_root) if args.model_root else default_model_roots(),
        target_gpu_memory_gb=args.target_gpu_memory_gb,
        run_import_probe=not args.skip_import_probe,
        run_gpu_probe=not args.skip_gpu_probe,
    )
    report = build_capability_report(options)
    write_report(report, args.output_json)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
