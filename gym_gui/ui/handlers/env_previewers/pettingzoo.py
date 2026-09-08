"""PettingZoo classic games previewer.

Extracted from `MainWindow._on_initialize_operator` (backup lines 4828-4987).

Handles chess_v6, connect_four_v3, go_v5, tictactoe_v3. For chess, applies
custom FEN when the operator config carries one. Builds board-game-specific
render payloads (chess FEN + legal moves, connect_four board, tictactoe board)
that the `BoardGameRendererStrategy` in the render container consumes directly.

This previewer inherits `LogConstantMixin` because it emits several
`self.log_constant(...)` calls internally with structured `extra=` fields
(LOG_OPERATOR_ENV_PREVIEW_STARTED for informational checks, and
LOG_UI_BOARD_CONFIG_ENV_INIT_CUSTOM when a custom chess position is applied).
Other previewers stay pure; this is the one exception (see plan section 7).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, cast

import numpy as np

from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_OPERATOR_ENV_PREVIEW_ERROR,
    LOG_OPERATOR_ENV_PREVIEW_STARTED,
    LOG_UI_BOARD_CONFIG_ENV_INIT_CUSTOM,
)
from gym_gui.ui.handlers.env_previewers.base import EnvPreviewError, EnvPreviewImportError

if TYPE_CHECKING:
    from gym_gui.services.operator import OperatorConfig


_OP_LOGGER = logging.getLogger("gym_gui.operators.main_window")


class PettingzooEnvPreview(LogConstantMixin):
    """Preview PettingZoo classic board games."""

    def __init__(self) -> None:
        # Required by LogConstantMixin.log_constant which reads self._logger.
        self._logger = _OP_LOGGER

    def preview(
        self,
        config: "OperatorConfig",
        seed: int,
        operator_id: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[dict], Optional[Tuple[str, int]]]:
        try:
            from pettingzoo.classic import (
                chess_v6,
                connect_four_v3,
                go_v5,
                tictactoe_v3,
            )
        except ImportError as e:
            raise EnvPreviewImportError(
                f"PettingZoo classic games not installed: {e}"
            ) from e

        # Map task names to environment factories
        pz_env_factories = {
            "chess_v6": chess_v6.env,
            "connect_four_v3": connect_four_v3.env,
            "go_v5": go_v5.env,
            "tictactoe_v3": tictactoe_v3.env,
        }

        task = config.task
        if task not in pz_env_factories:
            raise EnvPreviewError(f"Unknown PettingZoo game: {task}")

        try:
            # Create environment with rgb_array rendering
            env = pz_env_factories[task](render_mode="rgb_array")
            env.reset(seed=seed)

            # Apply custom initial state if configured (for board games)
            initial_state = None
            status_message: Optional[Tuple[str, int]] = None
            if config.workers:
                first_worker_id = next(iter(config.workers.keys()))
                initial_state = config.workers[first_worker_id].settings.get("initial_state")
                self.log_constant(
                    LOG_OPERATOR_ENV_PREVIEW_STARTED,
                    message=f"Checking for custom initial state in worker '{first_worker_id}'",
                    extra={
                        "operator_id": operator_id,
                        "worker_id": first_worker_id,
                        "worker_settings": str(config.workers[first_worker_id].settings),
                        "initial_state_found": initial_state is not None,
                        "initial_state": initial_state[:50] if initial_state else None,
                    },
                )

            if initial_state and task == "chess_v6" and hasattr(env, "board"):
                # Use python-chess to set custom FEN position
                try:
                    self.log_constant(
                        LOG_OPERATOR_ENV_PREVIEW_STARTED,
                        message="Applying custom chess position",
                        extra={
                            "operator_id": operator_id,
                            "custom_fen": initial_state,
                        },
                    )
                    env.board.set_fen(initial_state)
                    self.log_constant(
                        LOG_UI_BOARD_CONFIG_ENV_INIT_CUSTOM,
                        message="Custom chess position applied successfully",
                        extra={
                            "operator_id": operator_id,
                            "game_id": task,
                            "custom_fen": initial_state,
                            "applied_fen": env.board.fen(),
                        },
                    )
                    status_message = ("Custom chess position applied!", 3000)
                except Exception as e:
                    self.log_constant(
                        LOG_OPERATOR_ENV_PREVIEW_ERROR,
                        message=f"Failed to apply custom chess position: {e}",
                        extra={
                            "operator_id": operator_id,
                            "custom_fen": initial_state,
                            "error": str(e),
                        },
                    )
                    status_message = (f"Failed to apply custom position: {e}", 5000)
            else:
                self.log_constant(
                    LOG_OPERATOR_ENV_PREVIEW_STARTED,
                    message="Using standard starting position (no custom state configured)",
                    extra={
                        "operator_id": operator_id,
                        "task": task,
                        "has_initial_state": initial_state is not None,
                        "is_chess": task == "chess_v6",
                        "has_board_attr": hasattr(env, "board"),
                    },
                )

            # PettingZoo AEC envs render() returns the board
            rgb_frame = cast(Optional[np.ndarray], env.render())

            # Build game-specific payload for BoardGameRendererStrategy
            board_game_payload: Optional[Dict[str, Any]] = None
            if task == "chess_v6" and hasattr(env, "board"):
                import chess
                board: chess.Board = env.board
                legal_moves = [move.uci() for move in board.legal_moves]
                current_player = "white" if board.turn == chess.WHITE else "black"
                board_game_payload = {
                    "chess": {
                        "fen": board.fen(),
                        "legal_moves": legal_moves,
                        "current_player": current_player,
                        "is_check": board.is_check(),
                    },
                    "game_id": "chess",
                }
            elif task == "connect_four_v3" and hasattr(env, "board"):
                board_game_payload = {
                    "connect_four": {
                        "board": env.board.tolist() if hasattr(env.board, "tolist") else list(env.board),
                        "current_player": getattr(env, "agent_selection", "player_0"),
                    },
                    "game_id": "connect_four",
                }
            elif task == "tictactoe_v3" and hasattr(env, "board"):
                board_game_payload = {
                    "board": env.board.tolist() if hasattr(env.board, "tolist") else list(env.board),
                    "current_player": getattr(env, "agent_selection", "player_1"),
                    "game_id": "tictactoe",
                }

            env.close()

            return rgb_frame, board_game_payload, status_message

        except EnvPreviewError:
            raise
        except Exception as e:
            raise EnvPreviewError(f"Cannot preview PettingZoo {task}: {e}") from e


__all__ = ["PettingzooEnvPreview"]
