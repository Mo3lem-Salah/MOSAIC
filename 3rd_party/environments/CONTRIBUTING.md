# Contributing an Environment to MOSAIC

This guide covers what an environment family is, how contributions are
structured, and the PR checklist for environment contributions.

For the full catalogue of existing environment families and their documentation,
see:

- **[Environment Families index](../../docs/source/documents/environments/index.rst)**:
  all supported families organised by category (Gymnasium, Grid Worlds,
  First-Person, Procedural, JAX, Board Games, Multi-Agent Benchmarks).

For general contribution guidelines (dev setup, lint/test), see the
[top-level CONTRIBUTING.md](../../CONTRIBUTING.md).

## What is an environment family?

An environment family is an optional Python package under
`3rd_party/environments/` that wraps a Gymnasium or PettingZoo suite and
registers it with the MOSAIC platform. Each family is installed independently:

```bash
pip install -e ".[minigrid]"          # MiniGrid only
pip install -e ".[smac,mosaic_multigrid]"  # SMAC + MOSAIC MultiGrid
```

Environment families and workers are independent; install any combination.

## Capability flags

Every environment row in `README.md` and `docs/source/index.rst` must carry
three capability flags, set honestly:

| Flag | Meaning |
|---|---|
| `Human-Control` | A human can play via keyboard through the MOSAIC GUI. |
| `Single-Agent` | A single-agent RL worker (e.g., CleanRL PPO) can be assigned. |
| `Multi-Agent` | A multi-agent worker (e.g., XuanCe MAPPO) can be assigned. |

Use ✅ for fully tested end-to-end in the platform, ❌ for not supported, and
🚧 for experimental. Do not mark a flag ✅ unless it has been verified in the
platform, not just in isolation.

## What you can contribute

- **New environment family**: wrap a new Gymnasium or PettingZoo suite.
- **Capability upgrade**: promote a flag from 🚧 or ❌ to ✅ once support is
  implemented and tested.
- **Bug fix**: fix a wrapper, observation transform, or registration issue.
- **Test coverage**: add tests to an existing family's `tests/` directory.

## Steps to add a new environment family

1. Create the package directory under `3rd_party/environments/<family>/`.
2. Implement the wrapper exposing a standard Gymnasium (single-agent) or
   PettingZoo AEC/parallel API (multi-agent).
3. Register all environment IDs in `__init__.py` via `gymnasium.register()` or
   the PettingZoo equivalent.
4. Write `pyproject.toml` with all dependencies declared independently of the
   root `pyproject.toml`.
5. Place a representative looping GIF (200 px wide) at
   `docs/source/images/envs/<family>/<family>.gif`.
6. Add a row to the Supported Environment Families table in `README.md`.
7. Add a matching row to the equivalent table in `docs/source/index.rst`.
8. Add the family as a named optional extra in the root `pyproject.toml` and
   include it in the `full` extra.
9. Add a documentation page under `docs/source/documents/environments/<family>/`
   and link it from the environments index.

## Versioning and reproducibility

Any change to environment dynamics, observations, reward signals, or termination
conditions is a reproducibility-breaking change. Call it out explicitly in your
PR description so it can be released under an appropriate version bump.

## Pull request checklist (environment-specific)

- [ ] All registered environments instantiate cleanly and pass a one-step smoke
      test.
- [ ] Capability flags are set honestly and verified end-to-end in the platform.
- [ ] GIF placed at `docs/source/images/envs/<family>/` (200 px wide, looping).
- [ ] Row added to the Supported Environment Families table in `README.md`.
- [ ] Matching row added to `docs/source/index.rst`.
- [ ] Optional extra added to root `pyproject.toml` and included in `full`.
- [ ] Documentation page added and linked from the environments index.
- [ ] Environment dynamics changes are called out in the PR description.
