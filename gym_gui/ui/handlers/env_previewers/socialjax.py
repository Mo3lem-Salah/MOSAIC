"""SocialJax environment previewer.

Extracted from `MainWindow._on_initialize_operator` (backup lines 5173-5222).
SocialJax is a pure JAX multi-agent social-dilemma environment. Uses
`_ensure_socialjax_path` for path setup, then `socialjax.make(env_id, ...)`.

Reads `num_agents`, `num_inner_steps`, and `shared_rewards` from the first
worker's settings (mirrors the original inline behaviour). All exceptions are
caught and rethrown as EnvPreviewError (matching the original branch's
single-except pattern that includes traceback logging).
"""

from __future__ import annotations

import logging
import traceback
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from gym_gui.ui.handlers.env_previewers.base import EnvPreviewError

if TYPE_CHECKING:
    from gym_gui.services.operator import OperatorConfig


# Original branch used _LOGGER (main_window's __name__ logger), not _OP_LOGGER.
# Preserve that: SocialJax preview errors route to the UI logger, not to
# operators.log.
_LOGGER = logging.getLogger("gym_gui.ui.main_window")


class SocialjaxEnvPreview:
    """Preview SocialJax multi-agent social-dilemma environments."""

    def preview(
        self,
        config: "OperatorConfig",
        seed: int,
        operator_id: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[dict], Optional[Tuple[str, int]]]:
        task = config.task
        try:
            from gym_gui.core.adapters.socialjax import _ensure_socialjax_path
            _ensure_socialjax_path()
            import socialjax

            # task is "socialjax/coop_mining"; strip prefix
            env_id_sj = task.split("/", 1)[-1] if "/" in task else task

            # Read settings from first worker (matches original inline branch)
            sj_num_agents = None
            sj_num_inner_steps = None
            sj_shared_rewards = False
            if config.workers:
                first_w = next(iter(config.workers.values()))
                sj_num_agents = first_w.settings.get("num_agents")
                sj_num_inner_steps = first_w.settings.get("num_inner_steps")
                sj_shared_rewards = bool(first_w.settings.get("shared_rewards", False))

            sj_kwargs: dict = {}
            if sj_num_agents is not None:
                sj_kwargs["num_agents"] = int(sj_num_agents)
            if sj_num_inner_steps is not None:
                sj_kwargs["num_inner_steps"] = int(sj_num_inner_steps)
            sj_kwargs["shared_rewards"] = sj_shared_rewards

            sj_env = socialjax.make(env_id_sj, **sj_kwargs)
            import jax
            rng = jax.random.PRNGKey(seed if seed is not None else 42)
            rng, rng_reset = jax.random.split(rng)
            _, sj_state = sj_env.reset(rng_reset)

            img = sj_env.render(sj_state)
            arr = np.array(img)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

            return arr, None, None

        except Exception as e:
            _LOGGER.error(
                "SocialJax preview failed for %s: %s\n%s",
                task, e, traceback.format_exc(),
            )
            raise EnvPreviewError(f"Cannot preview SocialJax {task}: {e}") from e


__all__ = ["SocialjaxEnvPreview"]
