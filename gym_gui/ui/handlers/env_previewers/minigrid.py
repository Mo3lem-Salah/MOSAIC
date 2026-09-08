"""MiniGrid and BabyAI environment previewer.

Extracted from `MainWindow._on_initialize_operator` (lines 4655-4698 of
main_window.backup.py) and `MainWindow._apply_minigrid_custom_state`
(lines 1165-1244 of main_window.backup.py). Behaviour is preserved verbatim:
same env creation, same custom-state application logic, same log lines and
levels, same status-bar messages with their original timeouts.

Uses the literal string "gym_gui.operators.main_window" for the operator
logger so log routing to operators.log is preserved (see refactor plan
section 7.1). Do not change to __name__ without updating
`gym_gui/logging_config/config.py`.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Optional, Tuple, cast

import numpy as np

from gym_gui.ui.handlers.env_previewers.base import EnvPreviewImportError

if TYPE_CHECKING:
    from gym_gui.services.operator import OperatorConfig


_OP_LOGGER = logging.getLogger("gym_gui.operators.main_window")


class MinigridEnvPreview:
    """Preview MiniGrid and BabyAI environments.

    Both env families share the same preview flow: create via `gymnasium.make`
    with the requested tile size, reset with the given seed, optionally apply
    a custom initial grid state from the first worker's settings, render one
    RGB frame, and close.
    """

    def preview(
        self,
        config: "OperatorConfig",
        seed: int,
        operator_id: Optional[str] = None,  # accepted for protocol uniformity; unused
    ) -> Tuple[Optional["np.ndarray"], Optional[dict], Optional[Tuple[str, int]]]:
        try:
            import gymnasium as gym
            import minigrid
        except ImportError as e:
            raise EnvPreviewImportError(
                f"MiniGrid not installed - cannot preview {config.task}"
            ) from e

        # Only register if not already in registry
        if "MiniGrid-Empty-5x5-v0" not in gym.envs.registry:
            minigrid.register_minigrid_envs()

        tile_size = config.settings.get("square_size") or 32
        env = gym.make(config.task, render_mode="rgb_array", tile_size=tile_size)
        env.reset(seed=seed)

        # Extract custom initial state from the first worker's settings
        initial_state = None
        if config.workers:
            first_worker_id = next(iter(config.workers.keys()))
            initial_state = config.workers[first_worker_id].settings.get("initial_state")
            _OP_LOGGER.debug(
                f"MiniGrid preview: worker={first_worker_id}, "
                f"settings_keys={list(config.workers[first_worker_id].settings.keys())}, "
                f"initial_state={'SET' if initial_state else 'NOT SET'}"
            )

        status_message: Optional[Tuple[str, int]] = None
        if initial_state:
            if self._apply_custom_state(env, initial_state):
                status_message = ("Custom MiniGrid configuration applied!", 3000)
            else:
                status_message = ("Failed to apply custom MiniGrid configuration", 5000)

        # env.render() is typed loosely by gymnasium (list | None) but returns
        # a numpy array at runtime for rgb_array mode. Cast to preserve the
        # protocol's return type.
        rgb_frame = cast(Optional[np.ndarray], env.render())
        env.close()

        return rgb_frame, None, status_message

    def _apply_custom_state(self, env: Any, state_json: str) -> bool:
        """Apply a custom grid state to a MiniGrid environment.

        Verbatim port of `MainWindow._apply_minigrid_custom_state`
        (backup lines 1165-1244).

        Args:
            env: The MiniGrid gymnasium environment (any wrapper depth).
            state_json: JSON string describing the custom grid.

        Returns:
            True if the state was applied successfully. False on any error
            (invalid JSON, missing minigrid object types, malformed state).
            Errors are logged; the caller keeps the environment in its
            default reset state.
        """
        try:
            state_dict = json.loads(state_json)
        except json.JSONDecodeError as e:
            _OP_LOGGER.warning(f"Invalid MiniGrid state JSON: {e}")
            return False

        try:
            from minigrid.core.world_object import (
                Ball,
                Box,
                Door,
                Goal,
                Key,
                Lava,
                Wall,
            )

            # Colors available: red, green, blue, purple, yellow, grey
            obj_type_map = {
                "wall": lambda color: Wall(),
                "goal": lambda color: Goal(),
                "lava": lambda color: Lava(),
                "key": lambda color: Key(color=color if color != "none" else "yellow"),
                "door": lambda color: Door(color=color if color != "none" else "yellow"),
                "ball": lambda color: Ball(color=color if color != "none" else "blue"),
                "box": lambda color: Box(color=color if color != "none" else "red"),
            }

            unwrapped = env.unwrapped
            grid = unwrapped.grid
            rows = state_dict.get("rows", unwrapped.height)
            cols = state_dict.get("cols", unwrapped.width)

            # Clear the interior of the grid (keep border walls if present)
            for x in range(1, cols - 1):
                for y in range(1, rows - 1):
                    grid.set(x, y, None)

            # Place objects from state
            for cell_data in state_dict.get("cells", []):
                row = cell_data.get("row", 0)
                col = cell_data.get("col", 0)

                # Convert our (row, col) to MiniGrid's (x, y) where x=col, y=row
                x, y = col, row

                for obj_data in cell_data.get("objects", []):
                    obj_type = obj_data.get("type", "empty")
                    color = obj_data.get("color", "none")

                    if obj_type in obj_type_map:
                        obj = obj_type_map[obj_type](color)
                        grid.set(x, y, obj)

            # Set agent position and direction
            agent_pos = state_dict.get("agent_pos")
            if agent_pos:
                # Convert (row, col) to (x, y)
                agent_row, agent_col = agent_pos
                unwrapped.agent_pos = (agent_col, agent_row)

            agent_dir = state_dict.get("agent_dir", 0)
            unwrapped.agent_dir = agent_dir

            _OP_LOGGER.info(
                f"Applied custom MiniGrid state: {rows}x{cols} grid, "
                f"agent at ({agent_pos}), dir={agent_dir}"
            )
            return True

        except Exception as e:
            _OP_LOGGER.error(f"Failed to apply MiniGrid custom state: {e}")
            return False


__all__ = ["MinigridEnvPreview"]
