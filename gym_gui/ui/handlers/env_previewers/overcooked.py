"""Overcooked-AI environment previewer.

Extracted from `MainWindow._on_initialize_operator` (backup lines 5125-5171).
Overcooked-AI uses its own API (not gymnasium) and renders via pygame Surface;
the preview converts to numpy and transposes to (height, width, 3).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from gym_gui.ui.handlers.env_previewers.base import EnvPreviewError, EnvPreviewImportError

if TYPE_CHECKING:
    from gym_gui.services.operator import OperatorConfig


_OP_LOGGER = logging.getLogger("gym_gui.operators.main_window")


class OvercookedEnvPreview:
    """Preview Overcooked-AI cooperative cooking envs."""

    def preview(
        self,
        config: "OperatorConfig",
        seed: int,
        operator_id: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[dict], Optional[Tuple[str, int]]]:
        try:
            import pygame
            from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
            from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Recipe
            from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
        except ImportError as e:
            raise EnvPreviewImportError("Overcooked-AI not installed - cannot preview") from e

        task = config.task
        try:
            # Extract layout name from task (format: overcooked/layout_name)
            layout_name = task.split("/", 1)[-1] if "/" in task else task

            # Configure Recipe class
            Recipe.configure({})

            # Create MDP from layout
            mdp = OvercookedGridworld.from_layout_name(layout_name)

            # Create environment
            env = OvercookedEnv.from_mdp(mdp, horizon=400)
            env.reset()

            # Render state to RGB using StateVisualizer with higher resolution.
            # tile_size controls native render resolution:
            #   75 (default) = 375x300, 100 = 500x400, 150 = 750x600
            tile_size = 100  # High quality native rendering
            visualizer = StateVisualizer(tile_size=tile_size)
            surface = visualizer.render_state(env.state, grid=mdp.terrain_mtx)

            # Convert pygame Surface to numpy array
            rgb_array = pygame.surfarray.array3d(surface)
            # surfarray returns (width, height, 3), transpose to (height, width, 3)
            rgb_frame = np.transpose(rgb_array, (1, 0, 2))

            return rgb_frame, None, None
        except Exception as e:
            raise EnvPreviewError(f"Cannot preview Overcooked {task}: {e}") from e


__all__ = ["OvercookedEnvPreview"]
