"""Crafter environment previewer.

Extracted from `MainWindow._on_initialize_operator` (backup lines 4700-4715).
Crafter passes seed to `__init__`, not to `reset()`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple, cast

import numpy as np

from gym_gui.core.ui.game_config import game_configs
from gym_gui.ui.handlers.env_previewers.base import EnvPreviewImportError

if TYPE_CHECKING:
    from gym_gui.services.operator import OperatorConfig


_OP_LOGGER = logging.getLogger("gym_gui.operators.main_window")


class CrafterEnvPreview:
    """Preview the Crafter open-world survival environment."""

    def preview(
        self,
        config: "OperatorConfig",
        seed: int,
        operator_id: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[dict], Optional[Tuple[str, int]]]:
        try:
            import crafter
        except ImportError as e:
            raise EnvPreviewImportError("Crafter not installed - cannot preview") from e

        cfg = game_configs.CrafterConfig()
        env = crafter.Env(size=cfg.size, seed=seed)
        env.reset()
        rgb_frame = cast(Optional[np.ndarray], env.render())
        env.close()

        return rgb_frame, None, None


__all__ = ["CrafterEnvPreview"]
