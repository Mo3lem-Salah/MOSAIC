# Contributing a Worker to MOSAIC

This guide covers what a worker is, how contributions are structured, and the
PR checklist for worker contributions. For the step-by-step implementation
walkthrough, see the full developer documentation:

- **[Developing a Worker (overview)](../../docs/source/documents/architecture/workers/development/index.rst)**:
  architecture diagram, telemetry paths (Slow Lane vs FastLane), and how the
  backend and frontend fit together.
- **[Backend guide](../../docs/source/documents/architecture/workers/development/backend.rst)**:
  step-by-step implementation covering `config.py`, `runtime.py`, `telemetry.py`,
  `analytics.py`, `pyproject.toml`, and the Daemon entry point.
- **[Frontend guide](../../docs/source/documents/architecture/workers/development/frontend.rst)**:
  step-by-step implementation covering the Qt6 training form, worker catalog
  entry, presenter, and how the form config reaches the backend via gRPC.

For general contribution guidelines (dev setup, lint/test), see the
[top-level CONTRIBUTING.md](../../CONTRIBUTING.md).

## What is a worker?

A worker is a standalone Python package under `3rd_party/workers/` that MOSAIC
launches as a managed subprocess. It has two sides:

- **Backend** (`3rd_party/workers/<name>_worker/`): training logic, telemetry,
  and the CLI entry point that the Daemon spawns.
- **Frontend** (`gym_gui/ui/`): the Qt6 training form dialog, worker catalog
  entry, and presenter that routes user configuration to the backend via gRPC.

Both sides are required for a fully integrated worker. If you are contributing
only a backend proof-of-concept, make that clear in your PR so the maintainers
can schedule the frontend work separately.

## What you can contribute

- **New worker integration**: wrap a new RL framework, LLM API, or VLM API.
- **New algorithm**: add an algorithm to an existing worker (e.g., a new
  CleanRL or XuanCe algorithm variant).
- **Bug fix**: fix issues in the runtime loop, telemetry, FastLane integration,
  or the frontend form.
- **Test coverage**: add tests to an existing worker's `tests/` directory.

## Pull request checklist (worker-specific)

- [ ] Backend `config.py` and `runtime.py` implement the full
      step/reset/close contract documented in the backend guide.
- [ ] Worker `pyproject.toml` declares all dependencies; none bleed into
      the root `pyproject.toml`.
- [ ] A dry-run smoke test passes: worker starts, steps once, exits cleanly.
- [ ] FastLane integration is present if the worker renders frames (see
      backend guide for shared-memory cleanup requirements).
- [ ] Frontend training form, catalog entry, and presenter are implemented
      (or the PR description explains why only the backend is included).
- [ ] Supported Workers table in `README.md` is updated.
- [ ] A documentation page is added under
      `docs/source/documents/architecture/workers/integrated_workers/<name>_worker/`.
