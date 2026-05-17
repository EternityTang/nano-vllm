<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |

## Memory-Aware Optimizer

This fork contains an experimental memory-aware optimizer for long-context KV pressure. It is release-hardened as default-off: the original full-only path remains the default unless flags are explicitly enabled.

Key documents:

- `docs/memory_aware_optimizer.md`: architecture, flags, commands, rollback, limitations.
- `docs/benchmark_ablation.md`: B0-B5 ablation summary generated from benchmark JSON artifacts.
- `docs/risk_gates.md`: release risk gates and validation commands.
- `docs/interview_narrative.md`: design rationale and performance interpretation.

Reproduce the B5 QUANT-only fused decode profile:

```bash
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 16 --output-json results/b5_profile.json --enable-memory-aware-scheduler --enable-admission-controller --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --enable-mixed-kv-decode-kernel --reclaim-policy arkv_q8
```

Generate the release ablation report:

```bash
python benchmarks/report.py --results-dir results --output-markdown docs/benchmark_ablation.md --output-csv results/ablation_summary.csv
```

B2c EVICT is optional and quality-gated. B4/B5 default to B3 QUANT-only and must keep `evicted_blocks=0`.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
