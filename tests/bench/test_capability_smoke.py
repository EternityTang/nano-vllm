from __future__ import annotations

# 中文说明：
# P-1 能力探针的 smoke/regression 测试，验证 capability report 的 schema、默认约束、Qwen3 模型选择逻辑和 mixed-KV CPU 参考一致性。
# 这些测试不依赖真实 GPU 推理，用来保证后续 P0/P1/P2 阶段读取的 capability JSON 具有稳定字段和可信默认假设。

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_PATH = REPO_ROOT / "benchmarks" / "capability_probe.py"


def load_probe_module():
    spec = importlib.util.spec_from_file_location("capability_probe", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_qwen3_config(model_dir: Path, *, quantized: bool = False) -> None:
    model_dir.mkdir(parents=True)
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "torch_dtype": "bfloat16",
        "max_position_embeddings": 40960,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 2560,
    }
    if quantized:
        config["quantization_config"] = {"quant_method": "awq"}
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"fake")


class CapabilityProbeSmokeTest(unittest.TestCase):
    def test_probe_schema_and_default_constraints(self):
        probe = load_probe_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_qwen3_config(root / "Qwen3-4B")
            write_qwen3_config(root / "Qwen3-4B-AWQ", quantized=True)

            options = probe.ProbeOptions(
                dry_run=True,
                model_roots=(root,),
                run_import_probe=False,
                run_gpu_probe=False,
            )
            report = probe.build_capability_report(options)

        self.assertEqual(report["schema_version"], 1)
        self.assertFalse(report["optimizer_behavior_enabled"])
        self.assertTrue(report["model_support"]["qwen3_first"])
        self.assertEqual(report["model_support"]["formal_benchmark_model"]["name"], "Qwen3-4B")
        self.assertEqual(report["kv_cache"]["formal_block_size"], 256)
        self.assertFalse(report["kv_cache"]["candidate_support"]["128"])
        self.assertTrue(report["kv_cache"]["candidate_support"]["256"])
        self.assertTrue(report["cuda_graph_policy"]["requires_enforce_eager_when_enabled"])

    def test_mixed_kv_materialization_reference_passes(self):
        probe = load_probe_module()
        result = probe.run_mixed_kv_reference_check()
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["materialization_allclose"])
        self.assertTrue(result["decode_output_allclose"])

    def test_write_report_creates_json(self):
        probe = load_probe_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_root = root / "models"
            output_path = root / "result.json"
            write_qwen3_config(model_root / "Qwen3-4B")
            options = probe.ProbeOptions(
                dry_run=True,
                output_json=output_path,
                model_roots=(model_root,),
                run_import_probe=False,
                run_gpu_probe=False,
            )
            report = probe.build_capability_report(options)
            probe.write_report(report, output_path)
            loaded = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(loaded["phase"], "P-1 repo capability calibration / smoke spike")
        self.assertEqual(loaded["model_support"]["formal_benchmark_model"]["name"], "Qwen3-4B")


if __name__ == "__main__":
    unittest.main()
