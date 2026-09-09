"""MOSAIC MultiGrid environment previewer (competitive team sports).

Extracted from `MainWindow._on_initialize_operator` (backup lines 4989-5022).
Passes `view_size` kwarg when the operator config carries one. The env's
`render(highlight=True)` is preferred but falls back to `render()` when the
signature does not accept `highlight`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, cast

import numpy as np

from gym_gui.ui.handlers.env_previewers.base import EnvPreviewError, EnvPreviewImportError

if TYPE_CHECKING:
    from gym_gui.services.operator import OperatorConfig


_OP_LOGGER = logging.getLogger("gym_gui.operators.main_window")


class MosaicMultigridEnvPreview:
    """Preview MOSAIC MultiGrid competitive team sports envs."""

    def preview(
        self,
        config: "OperatorConfig",
        seed: int,
        operator_id: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[dict], Optional[Tuple[str, int]]]:
        try:
            import gymnasium
            import mosaic_multigrid.envs  # noqa: F401 - triggers gymnasium.register() calls
        except ImportError as import_err:
            raise EnvPreviewImportError(
                f"mosaic_multigrid not installed - cannot preview: {import_err}"
            ) from import_err

        task = config.task
        try:
            extra_kwargs: Dict[str, Any] = {}
            if config.view_size is not None:
                extra_kwargs["view_size"] = config.view_size
                _OP_LOGGER.info(
                    "Preview: view_size=%d applied to %s",
                    config.view_size, task,
                )
            env = gymnasium.make(task, render_mode="rgb_array", **extra_kwargs)
            env.reset(seed=seed)
            # mosaic_multigrid render() does not accept tile_size, but some
            # variants accept highlight=True.
            try:
                rgb_frame = cast(Optional[np.ndarray], env.render(highlight=True))
            except TypeError:
                rgb_frame = cast(Optional[np.ndarray], env.render())
            env.close()

            return rgb_frame, None, None
        except Exception as e:
            raise EnvPreviewError(f"Cannot preview mosaic_multigrid {task}: {e}") from e


__all__ = ["MosaicMultigridEnvPreview"]
