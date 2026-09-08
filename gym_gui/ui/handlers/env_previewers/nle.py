"""NLE (NetHack Learning Environment) previewer.

Extracted from `MainWindow._on_initialize_operator` (backup lines 4717-4748).
NLE does not support `rgb_array` render mode natively; the preview converts
TTY chars/colors to RGB via the project's `nle_render` adapter, then scales
3x for readability.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from gym_gui.ui.handlers.env_previewers.base import EnvPreviewError, EnvPreviewImportError

if TYPE_CHECKING:
    from gym_gui.services.operator import OperatorConfig


_OP_LOGGER = logging.getLogger("gym_gui.operators.main_window")


class NLEEnvPreview:
    """Preview NLE (NetHack) environments via TTY-to-RGB conversion."""

    def preview(
        self,
        config: "OperatorConfig",
        seed: int,
        operator_id: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[dict], Optional[Tuple[str, int]]]:
        try:
            import gymnasium as gym
            import nle  # noqa: F401

            from gym_gui.core.adapters.nle_render import render_tty_to_rgb
        except ImportError as e:
            raise EnvPreviewImportError("NLE not installed - cannot preview") from e

        try:
            # NLE does not support rgb_array mode; use default and get tty_chars
            env = gym.make(
                config.task,
                observation_keys=("tty_chars", "tty_colors", "blstats"),
            )
            obs, _ = env.reset(seed=seed)
            tty_chars = obs.get("tty_chars")
            tty_colors = obs.get("tty_colors")
            env.close()

            rgb_frame: Optional[np.ndarray] = None
            if tty_chars is not None:
                rgb_frame = render_tty_to_rgb(tty_chars, tty_colors)
                # Scale up for better visibility (3x)
                rgb_frame = np.repeat(np.repeat(rgb_frame, 3, axis=0), 3, axis=1)

            return rgb_frame, None, None
        except Exception as e:
            raise EnvPreviewError(f"Cannot preview NLE: {e}") from e


__all__ = ["NLEEnvPreview"]
