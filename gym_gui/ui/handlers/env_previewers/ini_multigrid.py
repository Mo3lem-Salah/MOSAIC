"""INI MultiGrid environment previewer (cooperative exploration).

Extracted from `MainWindow._on_initialize_operator` (backup lines 5024-5074).
INI multigrid lives in 3rd_party/multigrid-ini and requires a sys.path prepend
to import. Uses INI_CONFIGURATIONS lookup when available; otherwise falls back
to gymnasium.make.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING, Optional, Tuple, cast

import numpy as np

from gym_gui.ui.handlers.env_previewers.base import EnvPreviewError, EnvPreviewImportError

if TYPE_CHECKING:
    from gym_gui.services.operator import OperatorConfig


_OP_LOGGER = logging.getLogger("gym_gui.operators.main_window")


class IniMultigridEnvPreview:
    """Preview INI MultiGrid cooperative exploration envs."""

    def preview(
        self,
        config: "OperatorConfig",
        seed: int,
        operator_id: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[dict], Optional[Tuple[str, int]]]:
        try:
            import gymnasium

            # Prepend the 3rd_party path to sys.path so the local multigrid
            # package is importable. Path is anchored to this file's location:
            # env_previewers/ini_multigrid.py -> handlers/ -> ui/ -> gym_gui/ -> repo
            # The original inline branch used a similar relative resolution
            # from main_window.py; we preserve that intent.
            ini_multigrid_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "..", "3rd_party", "multigrid-ini"
            )
            if os.path.exists(ini_multigrid_path) and ini_multigrid_path not in sys.path:
                sys.path.insert(0, ini_multigrid_path)

            try:
                from multigrid.envs import CONFIGURATIONS as INI_CONFIGURATIONS
            except ImportError:
                INI_CONFIGURATIONS = {}
        except ImportError as import_err:
            raise EnvPreviewImportError(
                f"ini_multigrid not available - cannot preview: {import_err}"
            ) from import_err

        task = config.task
        try:
            num_agents = len(config.workers) if config.workers else 1
            tile_size = config.settings.get("square_size") or 32

            if task in INI_CONFIGURATIONS:
                env_cls, config_kwargs = INI_CONFIGURATIONS[task]
                config_kwargs = {
                    **config_kwargs,
                    "agents": num_agents,
                    "render_mode": "rgb_array",
                    "tile_size": tile_size,
                }
                env = env_cls(**config_kwargs)
            else:
                env = gymnasium.make(task, render_mode="rgb_array", tile_size=tile_size)

            env.reset(seed=seed)
            rgb_frame = cast(Optional[np.ndarray], env.render())
            env.close()

            return rgb_frame, None, None
        except Exception as e:
            raise EnvPreviewError(f"Cannot preview ini_multigrid {task}: {e}") from e


__all__ = ["IniMultigridEnvPreview"]
