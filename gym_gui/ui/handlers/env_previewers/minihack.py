"""MiniHack environment previewer.

Extracted from `MainWindow._on_initialize_operator` (backup lines 4750-4764).
MiniHack supports `rgb_array` mode directly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple, cast

import numpy as np

from gym_gui.ui.handlers.env_previewers.base import EnvPreviewImportError

if TYPE_CHECKING:
    from gym_gui.services.operator import OperatorConfig


_OP_LOGGER = logging.getLogger("gym_gui.operators.main_window")


class MinihackEnvPreview:
    """Preview MiniHack roguelike environments."""

    def preview(
        self,
        config: "OperatorConfig",
        seed: int,
        operator_id: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[dict], Optional[Tuple[str, int]]]:
        try:
            import gymnasium as gym
            import minihack  # noqa: F401
        except ImportError as e:
            raise EnvPreviewImportError("MiniHack not installed - cannot preview") from e

        env = gym.make(config.task, render_mode="rgb_array")
        env.reset(seed=seed)
        rgb_frame = cast(Optional[np.ndarray], env.render())
        env.close()

        return rgb_frame, None, None


__all__ = ["MinihackEnvPreview"]
