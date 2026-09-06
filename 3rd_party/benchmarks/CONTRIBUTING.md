# Contributing to MOSAIC Benchmarks

This guide covers the two benchmark suites under `3rd_party/benchmarks/` and
how to contribute to each. For general contribution guidelines (dev setup,
lint/test), see the [top-level CONTRIBUTING.md](../../CONTRIBUTING.md).

## The two benchmark suites

### workers_benchmark

Measures throughput, overhead, and training speed across RL worker frameworks.
Currently covers: CleanRL, Stable-Baselines3, Tianshou, TorchRL, XuanCe, Ray
RLlib, RL-Tools, SBX, and Jumanji. Each framework has its own subdirectory
under `benchmarks/`.

### workers_llm_benchmark

Benchmarks LLM agents (currently Qwen2.5-3B and Qwen3.5-4B) on MOSAIC
evaluation tasks. Each model has its own directory under `models/`.

## What you can contribute

- **New framework benchmark**: add a new RL framework to `workers_benchmark`
  so it appears in the throughput comparison.
- **New LLM/VLM model**: add a new model configuration to `workers_llm_benchmark`.
- **New benchmark scenario**: add a new environment or task configuration to
  an existing framework's benchmark directory.
- **Result submission**: run the existing suite on your hardware and open a PR
  with your results appended to the results files.
- **Bug fix**: fix a benchmark script, config schema, or result parser.

## Adding a new framework to workers_benchmark

1. Create a subdirectory under `3rd_party/benchmarks/workers_benchmark/benchmarks/<framework>/`.
2. Implement the benchmark entry point. Follow the structure of an existing
   framework directory (e.g., `benchmarks/cleanrl/` or `benchmarks/sb3/`) as
   a reference.
3. Add a configuration section for the new framework in
   `structured_configs.py` so the Hydra sweep picks it up.
4. Add a test under `tests/` that runs the benchmark for one iteration and
   checks the result schema.
5. Update the `workers_benchmark/README.md` to list the new framework.

## Adding a new model to workers_llm_benchmark

1. Create a directory under `3rd_party/benchmarks/workers_llm_benchmark/models/<model_name>/`.
2. Add the model configuration and any required adapter files.
3. Verify the model runs end-to-end with the existing `benchmark.py` entry
   point.
4. Update `workers_llm_benchmark/README.md` with the new model entry.

## Running the benchmarks locally

```bash
# RL worker throughput benchmark
cd 3rd_party/benchmarks/workers_benchmark
python -m workers_benchmark --config-name <framework>

# LLM benchmark
cd 3rd_party/benchmarks/workers_llm_benchmark
python benchmark.py --model <model_name>
```

See each suite's `README.md` for full configuration options and Hydra sweep
instructions.

## Pull request checklist (benchmark-specific)

- [ ] New benchmark directory follows the structure of an existing framework
      directory.
- [ ] Configuration is registered in `structured_configs.py` (workers) or the
      equivalent config entry point (LLM).
- [ ] A one-iteration smoke test passes under `tests/`.
- [ ] The relevant `README.md` is updated to list the new framework or model.
- [ ] Result files use the established schema; no schema-breaking changes
      without a version bump.
