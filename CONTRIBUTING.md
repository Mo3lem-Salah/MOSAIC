# Contributing to MOSAIC

Thanks for your interest in improving MOSAIC! Contributions of all kinds are
welcome.

## Ways to contribute

- **New worker**: integrate a new RL framework, LLM API, or VLM API.
  See [3rd_party/workers/CONTRIBUTING.md](3rd_party/workers/CONTRIBUTING.md).
- **New environment family**: wrap a new Gymnasium or PettingZoo suite.
  See [3rd_party/environments/CONTRIBUTING.md](3rd_party/environments/CONTRIBUTING.md).
- **Benchmark**: add a new framework or model to the benchmark suites, or
  submit results from your hardware.
  See [3rd_party/benchmarks/CONTRIBUTING.md](3rd_party/benchmarks/CONTRIBUTING.md).
- **Bug fix**: fix issues in the GUI, FastLane, workers, or environment wrappers.
- **Documentation**: improve RST docs, docstrings, or this README.

Report bugs and request features via
[GitHub Issues](https://github.com/Abdulhamid97Mousa/mosaic/issues).

## Development setup

```bash
git clone https://github.com/Abdulhamid97Mousa/mosaic.git
cd mosaic
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[full,dev]"
```

Install only what you need by combining individual extras:

```bash
pip install -e ".[cleanrl,minigrid,dev]"         # CleanRL + MiniGrid
pip install -e ".[xuance,mosaic_multigrid,dev]"  # XuanCe + competitive envs
```

## Before you open a PR

CI runs lint, format checks, and the test suite on Python 3.10, 3.11, and 3.12.
Run the same checks locally first:

```bash
ruff check .           # lint
ruff format --check .  # formatting
pytest                 # full test suite
```

Auto-fix before committing:

```bash
ruff check --fix .
ruff format .
```

## PR checklist

- [ ] Change is focused; the PR description links any related issue.
- [ ] `ruff check .` and `ruff format --check .` pass.
- [ ] `pytest` passes; new behaviour has a test where practical.
- [ ] Public API changes are reflected in docstrings and the Sphinx docs.

For worker-specific and environment-specific checklist items, see the relevant
sub-guide linked above.

## Code of conduct

By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE) that covers this project.
