"""PettingZoo multi-agent handler: shared env + turn-based coordination.

Extracted from `MainWindow` (backup: `_is_pettingzoo_multiagent`,
`_create_pettingzoo_env`, `_get_chess_legal_moves`, `_convert_uci_to_action_index`,
`_on_reset_pettingzoo_multiagent`, `_render_pettingzoo_frame`,
`_on_step_pettingzoo_multiagent`, `_poll_pettingzoo_action`,
`_execute_pettingzoo_action`, `_submit_pettingzoo_human_move`, `_on_step_player`)
as part of refactor plan step 7.

Owns the shared PettingZoo environment for multi-agent games (chess, go,
connect_four, tictactoe). Workers are initialized in action_selector mode
(they do not own an env); the GUI holds ONE shared env and coordinates
turn-based action selection.

Cross-handler state exposed:
- `is_active()` -> bool: True if a shared env is running (used by
  OperatorLifecycleHandler dispatch and _on_stop_operators cleanup).
- `shutdown()`: closes env, stops player handles, resets mode flag.

Uses `LogConstantMixin` because it emits many LOG constants
(`LOG_UI_MAINWINDOW_ERROR/INFO/WARNING/TRACE`, `LOG_OPERATOR_RESET_ALL_STARTED`,
`LOG_OPERATOR_STEP_ALL_COMPLETED`, `LOG_UI_BOARD_CONFIG_ENV_INIT_CUSTOM`).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from qtpy import QtCore

from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_OPERATOR_RESET_ALL_STARTED,
    LOG_OPERATOR_STEP_ALL_COMPLETED,
    LOG_UI_MAINWINDOW_ERROR,
    LOG_UI_MAINWINDOW_INFO,
    LOG_UI_MAINWINDOW_TRACE,
    LOG_UI_MAINWINDOW_WARNING,
)
from gym_gui.services.operator import OperatorConfig
from gym_gui.services.operator_launcher import OperatorLaunchError

if TYPE_CHECKING:
    from qtpy import QtWidgets  # noqa: F401

    from gym_gui.services.operator import MultiOperatorService
    from gym_gui.services.operator_launcher import OperatorLauncher
    from gym_gui.ui.widgets.control_panel import ControlPanelWidget
    from gym_gui.ui.widgets.render_tabs import RenderTabs


_OP_LOGGER = logging.getLogger("gym_gui.operators.main_window")


class PettingzooHandler(LogConstantMixin):
    """Shared PettingZoo env + turn-based action coordination.

    Owned state:
        _shared_pettingzoo_env: the single env instance (chess, go, ...)
        _pettingzoo_multiagent_mode: True when running
        _pettingzoo_player_handles: dict player_id -> worker handle
        _pettingzoo_current_seed: last seed used
        _pettingzoo_step_index: step counter within current episode

    Signal connections (owner: MainWindow._connect_signals):
        control_panel.step_player_requested -> on_step_player
    """

    def __init__(
        self,
        *,
        parent: Any,  # MainWindow at runtime
        render_tabs: "RenderTabs",
        control_panel: "ControlPanelWidget",
        status_bar: "QtWidgets.QStatusBar",
        operator_launcher: "OperatorLauncher",
        multi_operator_service: "MultiOperatorService",
    ) -> None:
        self._logger = _OP_LOGGER
        self._parent = parent
        self._render_tabs = render_tabs
        self._control_panel = control_panel
        self._status_bar = status_bar
        self._operator_launcher = operator_launcher
        self._multi_operator_service = multi_operator_service
        # Owned state (per plan section 8)
        self._shared_pettingzoo_env: Any = None
        self._pettingzoo_multiagent_mode: bool = False
        self._pettingzoo_player_handles: Dict[str, Any] = {}
        self._pettingzoo_current_seed: int = 42
        self._pettingzoo_step_index: int = 0

    def is_active(self) -> bool:
        """True if a shared PettingZoo env is running.

        Called by OperatorLifecycleHandler to decide whether "Step All" and
        "Stop" should dispatch to the PettingZoo flow.
        """
        return self._pettingzoo_multiagent_mode and self._shared_pettingzoo_env is not None

    def shutdown(self) -> None:
        """Close the env, stop player handles, reset mode. Idempotent."""
        if self._shared_pettingzoo_env is not None:
            try:
                self._shared_pettingzoo_env.close()
            except Exception:
                pass
            self._shared_pettingzoo_env = None
        for _handle in self._pettingzoo_player_handles.values():
            try:
                _handle.stop(timeout=2.0)
            except Exception:
                pass
        self._pettingzoo_player_handles.clear()
        self._pettingzoo_multiagent_mode = False
        self._control_panel.set_pettingzoo_mode(False)
        self._control_panel.set_turn_indicator("", visible=False)

    def is_pettingzoo_multiagent(self) -> tuple[bool, Optional["OperatorConfig"]]:
        """Check if we're in PettingZoo multi-agent mode.

        Returns:
            Tuple of (is_multiagent, first_config) where first_config is used
            to get env_name and task for creating the shared environment.
        """
        active_operators = self._multi_operator_service.get_active_operators()
        if not active_operators:
            return False, None

        # Get first operator config to check env_name
        first_id = next(iter(active_operators.keys()))
        first_config = self._multi_operator_service.get_operator(first_id)
        if first_config is None:
            return False, None

        # PettingZoo multi-agent: env_name in ("pettingzoo", "pettingzoo_classic") and multiple workers per operator
        # or multiple operators assigned to different players
        if first_config.env_name in ("pettingzoo", "pettingzoo_classic"):
            # Check if we have multiple workers (player assignments)
            if len(first_config.workers) > 1:
                return True, first_config
            # Or check if multiple operators exist (each controlling one player)
            if len(active_operators) > 1:
                return True, first_config

        return False, None

    def _create_pettingzoo_env(
        self, task: str, seed: int, initial_state: Optional[str] = None
    ) -> Any:
        """Create a PettingZoo environment for multi-agent games.

        Args:
            task: The specific game (e.g., "chess_v6", "connect_four_v3").
            seed: Random seed for initialization.
            initial_state: Optional custom initial state (FEN for chess).

        Returns:
            The PettingZoo AEC environment.
        """
        from pettingzoo.classic import (
            chess_v6,
            connect_four_v3,
            go_v5,
            tictactoe_v3,
        )

        env_factories = {
            "chess_v6": chess_v6.env,
            "connect_four_v3": connect_four_v3.env,
            "go_v5": go_v5.env,
            "tictactoe_v3": tictactoe_v3.env,
        }

        if task not in env_factories:
            raise ValueError(f"Unknown PettingZoo task: {task}")

        env = env_factories[task](render_mode="rgb_array")
        env.reset(seed=seed)

        # Apply custom initial state if provided (for board games like chess)
        if initial_state and task == "chess_v6" and hasattr(env, "board"):
            try:
                env.board.set_fen(initial_state)
                _OP_LOGGER.info(
                    f"Applied custom FEN position: {initial_state[:50]}..."
                )
            except Exception as e:
                _OP_LOGGER.warning(f"Failed to apply custom FEN: {e}")

        return env

    def _get_chess_legal_moves(self, env: Any) -> list[str]:
        """Get legal moves for the current player in chess.

        Returns:
            List of UCI move strings (e.g., ["e2e4", "g1f3", ...]).
        """
        try:
            # PettingZoo chess uses python-chess internally
            board = env.board
            return [move.uci() for move in board.legal_moves]
        except Exception as e:
            self.log_constant(
                LOG_UI_MAINWINDOW_WARNING,
                message=f"Failed to get legal moves: {e}",
            )
            return []

    def _convert_uci_to_action_index(self, env: Any, uci_move: str) -> Optional[int]:
        """Convert UCI move string to PettingZoo action index.

        Uses PettingZoo's chess_utils module for correct AlphaZero-style action encoding.

        Args:
            env: The PettingZoo chess environment.
            uci_move: UCI move string (e.g., "e2e4").

        Returns:
            Action index for env.step(), or None if invalid.
        """
        try:
            import chess
            from pettingzoo.classic.chess import chess_utils

            board = env.board
            move = chess.Move.from_uci(uci_move)

            if move not in board.legal_moves:
                _OP_LOGGER.debug("_convert_uci_to_action_index: %s not in legal_moves", uci_move)
                return None

            # Determine current player (0 = white, 1 = black)
            current_agent = env.agent_selection
            current_player = 0 if current_agent == "player_0" else 1

            # For black, we need to mirror the move since PettingZoo encodes from white's perspective
            if current_player == 1:
                # Mirror the move for black's encoding
                move_for_encoding = chess_utils.mirror_move(move)
            else:
                move_for_encoding = move

            # Get the UCI string for the (possibly mirrored) move
            move_for_encoding.uci()

            # Use PettingZoo's encoding: action = (col * 8 + row) * 73 + plane
            source = move_for_encoding.from_square
            coord = chess_utils.square_to_coord(source)
            panel = chess_utils.get_move_plane(move_for_encoding)
            action = (coord[0] * 8 + coord[1]) * 73 + panel

            _OP_LOGGER.debug(
                "_convert_uci_to_action_index: %s -> action=%s, current_player=%s, coord=%s, panel=%s",
                uci_move, action, current_player, coord, panel,
            )

            # Verify this action is legal
            obs = env.observe(current_agent)
            if isinstance(obs, dict) and "action_mask" in obs:
                if obs["action_mask"][action] == 1:
                    return action
                else:
                    _OP_LOGGER.debug("_convert_uci_to_action_index: action %s not in action_mask", action)
                    return None
            else:
                # No action mask, return the computed action
                return action

        except Exception as e:
            _OP_LOGGER.debug("_convert_uci_to_action_index EXCEPTION: %s", e, exc_info=True)
            self.log_constant(
                LOG_UI_MAINWINDOW_WARNING,
                message=f"Failed to convert UCI move '{uci_move}': {e}",
            )
            return None

    def on_reset_pettingzoo_multiagent(self, seed: int, config: "OperatorConfig") -> None:
        """Reset for PettingZoo multi-agent mode.

        In this mode:
        1. GUI creates ONE shared PettingZoo environment
        2. Each worker is initialized in action_selector mode (no env ownership)
        3. Workers provide actions when asked, GUI executes them

        Args:
            seed: Random seed for environment.
            config: The operator config with task and worker assignments.
        """
        task = config.task  # e.g., "chess_v6"

        # Extract custom initial state from worker settings (e.g., custom FEN for chess)
        initial_state = None
        if config.workers:
            first_worker_id = next(iter(config.workers.keys()))
            initial_state = config.workers[first_worker_id].settings.get("initial_state")
            if initial_state:
                _OP_LOGGER.debug(
                    f"Found custom initial_state in worker '{first_worker_id}': {initial_state[:50]}..."
                )

        # Close existing shared environment if any
        if self._shared_pettingzoo_env is not None:
            try:
                self._shared_pettingzoo_env.close()
            except Exception:
                pass

        # Create the shared environment in the GUI
        try:
            self._shared_pettingzoo_env = self._create_pettingzoo_env(task, seed, initial_state)
            self._pettingzoo_multiagent_mode = True
            self._pettingzoo_current_seed = seed
            self._pettingzoo_step_index = 0  # reset step counter on new episode

            # Stop any existing worker handles before relaunching (handles re-reset)
            for existing_handle in self._pettingzoo_player_handles.values():
                try:
                    existing_handle.stop(timeout=2.0)
                except Exception:
                    pass
            self._pettingzoo_player_handles.clear()

            self.log_constant(
                LOG_UI_MAINWINDOW_INFO,
                message=f"Created shared PettingZoo environment: {task}",
                extra={"task": task, "seed": seed},
            )
        except Exception as e:
            self.log_constant(
                LOG_UI_MAINWINDOW_ERROR,
                message=f"Failed to create PettingZoo environment: {e}",
                extra={"task": task, "seed": seed},
            )
            self._status_bar.showMessage(f"Failed to create {task}: {e}", 5000)
            return

        # Launch workers in action_selector mode
        # Each worker controls one player (e.g., player_0 = White, player_1 = Black)
        started_ids = []
        failed_ids = []

        # Reset operator states so start_all() returns them even on re-reset
        active_operators = self._multi_operator_service.get_active_operators()
        _OP_LOGGER.debug("active_operators count=%d", len(active_operators))
        for operator_id in active_operators:
            self._multi_operator_service.set_operator_state(operator_id, "pending")
        pending_ids = self._multi_operator_service.start_all()
        _OP_LOGGER.debug("pending_ids=%s", pending_ids)

        for operator_id in pending_ids:
            _OP_LOGGER.debug("Processing operator_id=%s", operator_id)
            op_config = self._multi_operator_service.get_operator(operator_id)
            if op_config is None:
                _OP_LOGGER.debug("op_config is None for %s", operator_id)
                continue

            _OP_LOGGER.debug("op_config.workers=%s", list(op_config.workers.keys()))
            # Determine which player this operator controls
            # If multiple workers in one operator, each controls a player
            # If single worker per operator, operator controls one player
            for player_id, worker_assignment in op_config.workers.items():
                _OP_LOGGER.debug(
                    "Launching worker for player_id=%s, worker_id=%s",
                    player_id, worker_assignment.worker_id,
                )
                try:
                    # Create single-agent config for this player's worker
                    # (multiagent OperatorConfig has operator_type="multiagent" which
                    #  the launcher doesn't handle directly)
                    player_config = OperatorConfig.single_agent(
                        operator_id=f"{operator_id}_{player_id}",
                        display_name=f"{op_config.display_name} - {player_id}",
                        worker_id=worker_assignment.worker_id,
                        worker_type=worker_assignment.worker_type,
                        env_name="pettingzoo",  # Action-selector mode
                        task=task,
                        settings=worker_assignment.settings,
                    )

                    # Launch subprocess
                    handle = self._operator_launcher.launch_operator(
                        player_config,
                        interactive=True,
                    )

                    # Initialize in action_selector mode (not reset with env ownership)
                    handle.send_init_agent(
                        game_name=task,
                        player_id=player_id,
                    )

                    # Store mapping for step coordination
                    self._pettingzoo_player_handles[player_id] = handle
                    _OP_LOGGER.debug(
                        "Stored handle for %s, total handles=%d",
                        player_id, len(self._pettingzoo_player_handles),
                    )

                    # Assign run_id
                    self._multi_operator_service.assign_run_id(operator_id, handle.run_id)
                    self._multi_operator_service.set_operator_state(operator_id, "running")

                    started_ids.append(operator_id)

                    self.log_constant(
                        LOG_UI_MAINWINDOW_INFO,
                        message="Initialized worker in action_selector mode",
                        extra={
                            "operator_id": operator_id,
                            "player_id": player_id,
                            "game": task,
                            "run_id": handle.run_id,
                        },
                    )
                except OperatorLaunchError as e:
                    self._multi_operator_service.set_operator_state(operator_id, "error")
                    failed_ids.append(operator_id)
                    self.log_constant(
                        LOG_UI_MAINWINDOW_ERROR,
                        message=f"Failed to launch operator: {e}",
                        extra={"operator_id": operator_id, "player_id": player_id},
                    )

        # Update status indicators
        for operator_id in started_ids:
            self._render_tabs.set_operator_status(operator_id, "running")
        for operator_id in failed_ids:
            self._render_tabs.set_operator_status(operator_id, "error")

        # Set the container display size for ALL active operators
        # (In PettingZoo mode, all operators share the same environment rendering)
        # Use active_operators instead of started_ids because the render uses active_ops
        container_size = config.settings.get("container_size", 0)
        if container_size and container_size > 0:
            for op_id in active_operators.keys():
                self._render_tabs.set_operator_display_size(
                    op_id, container_size, container_size
                )

        # Enable interactive mode for human operators BEFORE rendering
        # (so that legal moves panel is shown during render)
        for op_id, op_config in active_operators.items():
            for player_id, worker_assignment in op_config.workers.items():
                if worker_assignment.worker_type == "human" or worker_assignment.worker_id == "human_worker":
                    self._render_tabs.set_interactive(op_id, True)
                    _OP_LOGGER.debug("Enabled interactive mode for operator %s", op_id)
                    break  # One human worker is enough to enable interactive mode

        # Render initial board state (after enabling interactive mode)
        self.render_pettingzoo_frame()

        # Show turn indicator for first player
        current_player = self._shared_pettingzoo_env.agent_selection
        self._control_panel.set_turn_indicator(current_player, visible=True)

        # Enable PettingZoo mode: show player step buttons instead of Step All
        _OP_LOGGER.debug("Enabling PettingZoo mode, current_player=%s", current_player)
        self._control_panel.set_pettingzoo_mode(True)
        self._control_panel.set_current_player(current_player)
        _OP_LOGGER.debug("PettingZoo mode enabled")

        # Switch to Multi-Operator tab
        self._render_tabs.switch_to_multi_operator_tab()

        player_count = len(self._pettingzoo_player_handles)
        self._status_bar.showMessage(
            f"PettingZoo {task} ready: {player_count} players (seed={seed})",
            3000
        )
        self.log_constant(
            LOG_OPERATOR_RESET_ALL_STARTED,
            message="PettingZoo multi-agent reset complete",
            extra={
                "task": task,
                "seed": seed,
                "players": list(self._pettingzoo_player_handles.keys()),
            },
        )

    def render_pettingzoo_frame(self) -> None:
        """Render the current state of the shared PettingZoo environment."""
        if self._shared_pettingzoo_env is None:
            _OP_LOGGER.debug("_render_pettingzoo_frame: No environment")
            return

        try:
            env = self._shared_pettingzoo_env
            active_ops = self._multi_operator_service.get_active_operators()
            if not active_ops:
                return

            first_id = next(iter(active_ops.keys()))
            first_config = active_ops[first_id]
            task = first_config.task

            # For chess, use board game renderer with FEN data
            if task == "chess_v6" and hasattr(env, "board"):
                import chess
                board: chess.Board = env.board
                _OP_LOGGER.debug("_render_pettingzoo_frame: Chess board FEN=%s", board.fen())

                legal_moves = [move.uci() for move in board.legal_moves]
                current_player = "white" if board.turn == chess.WHITE else "black"

                # Build chess-specific payload for BoardGameRendererStrategy
                # Note: "render_payload" key is required for _extract_render_payload
                payload = {
                    "step_index": self._pettingzoo_step_index,
                    "episode_index": 0,
                    "render_payload": {
                        "chess": {
                            "fen": board.fen(),
                            "legal_moves": legal_moves,
                            "current_player": current_player,
                            "is_check": board.is_check(),
                        },
                        "game_id": "chess",
                    },
                }
                _OP_LOGGER.debug(
                    "_render_pettingzoo_frame: Sending payload to operator_id=%s, keys=%s",
                    first_id, list(payload.keys()),
                )
                self._render_tabs.display_operator_payload(first_id, payload)
                _OP_LOGGER.debug("_render_pettingzoo_frame: Payload sent")
            else:
                # Fallback to RGB rendering for other games
                rgb_frame = env.render()
                if rgb_frame is not None and isinstance(rgb_frame, np.ndarray):
                    payload = {
                        "step_index": self._pettingzoo_step_index,
                        "episode_index": 0,
                        "render_payload": {
                            "mode": "rgb",
                            "rgb": rgb_frame.tolist(),
                            "width": rgb_frame.shape[1],
                            "height": rgb_frame.shape[0],
                        },
                    }
                    self._render_tabs.display_operator_payload(first_id, payload)
        except Exception as e:
            _OP_LOGGER.debug("_render_pettingzoo_frame EXCEPTION: %s", e, exc_info=True)
            self.log_constant(
                LOG_UI_MAINWINDOW_WARNING,
                message=f"Failed to render PettingZoo frame: {e}",
            )

    def submit_pettingzoo_human_move(self, operator_id: str, uci_move: str) -> None:
        """Submit a Human move to the shared PettingZoo environment.

        Converts UCI move to action index and executes it.

        Args:
            operator_id: The operator ID.
            uci_move: Move in UCI notation (e.g., "e2e4").
        """
        env = self._shared_pettingzoo_env
        if env is None:
            _OP_LOGGER.warning("No shared PettingZoo env for human move")
            return

        try:
            # Use the existing conversion function
            action_index = self._convert_uci_to_action_index(env, uci_move)

            if action_index is None:
                _OP_LOGGER.warning(f"Move {uci_move} not found in legal moves")
                self._status_bar.showMessage(f"Illegal move: {uci_move}", 3000)
                return

            _OP_LOGGER.info(
                f"Submitting human chess move: {uci_move} -> action {action_index}"
            )

            # Execute the action in the environment
            env.step(action_index)

            # Render the updated state
            self.render_pettingzoo_frame()

            # Update turn indicator
            next_player = env.agent_selection
            self._control_panel.operators_tab.set_current_player(next_player)

            self._status_bar.showMessage(
                f"Human move: {uci_move}, next: {next_player}", 2000
            )

        except Exception as e:
            _OP_LOGGER.error(f"Error submitting human chess move: {e}", exc_info=True)
            self._status_bar.showMessage(f"Error: {e}", 3000)

    def on_step_player(self, player_id: str, seed: int) -> None:
        """Handle step request for a specific player (PettingZoo mode).

        Called when user clicks one of the player-specific step buttons.

        Args:
            player_id: Which player to step (e.g., "player_0", "player_1").
            seed: Random seed (currently unused for PettingZoo steps).
        """
        _OP_LOGGER.debug("_on_step_player: player_id=%s, seed=%s", player_id, seed)
        env = self._shared_pettingzoo_env
        if env is None:
            _OP_LOGGER.debug("_on_step_player: No environment")
            self._status_bar.showMessage("No PettingZoo environment active", 3000)
            return

        # Validate it's actually this player's turn
        current_player = env.agent_selection
        _OP_LOGGER.debug("_on_step_player: current_player from env=%s", current_player)
        if current_player != player_id:
            self.log_constant(
                LOG_UI_MAINWINDOW_WARNING,
                message="Attempted to step wrong player",
                extra={"requested": player_id, "current": current_player},
            )
            self._status_bar.showMessage(
                f"Not {player_id}'s turn! Current: {current_player}",
                3000
            )
            # Fix button states
            self._control_panel.set_current_player(current_player)
            return

        # Delegate to existing step logic
        self.on_step_pettingzoo_multiagent()

    def on_step_pettingzoo_multiagent(self) -> None:
        """Step the shared PettingZoo environment with turn-based coordination.

        Flow:
        1. Get current player from env.agent_selection
        2. Get observation and legal moves for that player
        3. Send select_action to that player's worker
        4. Wait for action response
        5. Execute action on shared environment
        6. Render updated board
        """
        _OP_LOGGER.debug("_on_step_pettingzoo_multiagent: Starting")
        env = self._shared_pettingzoo_env
        if env is None:
            _OP_LOGGER.debug("_on_step_pettingzoo_multiagent: No environment")
            return

        # Check if game is over
        if env.terminations.get(env.agent_selection, False) or \
           env.truncations.get(env.agent_selection, False):
            _OP_LOGGER.debug("_on_step_pettingzoo_multiagent: Game over")
            self._status_bar.showMessage("Game over! Use Reset to start new game.", 3000)
            return

        # Get current player
        current_player = env.agent_selection  # e.g., "player_0" or "player_1"
        _OP_LOGGER.debug("_on_step_pettingzoo_multiagent: current_player=%s", current_player)

        # Get handle for this player's worker
        _OP_LOGGER.debug(
            "_on_step_pettingzoo_multiagent: Available handles=%s",
            list(self._pettingzoo_player_handles.keys()),
        )
        handle = self._pettingzoo_player_handles.get(current_player)
        if handle is None:
            _OP_LOGGER.debug("_on_step_pettingzoo_multiagent: NO HANDLE for %s", current_player)
            self.log_constant(
                LOG_UI_MAINWINDOW_ERROR,
                message=f"No worker handle for player: {current_player}",
                extra={"player": current_player, "available": list(self._pettingzoo_player_handles.keys())},
            )
            self._status_bar.showMessage(f"No worker for {current_player}", 3000)
            self._control_panel.set_current_player(current_player)
            return

        _OP_LOGGER.debug("_on_step_pettingzoo_multiagent: Got handle, is_running=%s", handle.is_running)
        if not handle.is_running:
            _OP_LOGGER.debug("_on_step_pettingzoo_multiagent: Handle not running")
            self.log_constant(
                LOG_UI_MAINWINDOW_WARNING,
                message=f"Worker not running for player: {current_player}",
            )
            self._status_bar.showMessage(f"Worker stopped for {current_player}", 3000)
            self._control_panel.set_current_player(current_player)
            return

        # Get observation for current player
        obs = env.observe(current_player)

        # Extract action_mask (e.g. chess_v6 returns {"observation": ..., "action_mask": ...})
        action_mask: Optional[List[bool]] = None
        if isinstance(obs, dict) and "action_mask" in obs:
            action_mask = obs["action_mask"].tolist()
            _OP_LOGGER.debug(
                "_on_step_pettingzoo_multiagent: action_mask extracted, legal=%d/%d",
                sum(action_mask), len(action_mask),
            )

        # Get legal moves (for chess, convert to UCI strings)
        legal_moves = self._get_chess_legal_moves(env)
        _OP_LOGGER.debug("_on_step_pettingzoo_multiagent: legal_moves count=%d", len(legal_moves))

        # Build observation string for LLM
        obs_str = f"Current player: {current_player}\n"
        obs_str += f"Board state:\n{env.board}\n" if hasattr(env, "board") else str(obs)

        # Send select_action command to worker; pass action_mask so workers that
        # sample randomly (e.g. Random Worker) stay within the legal action set.
        info = {"legal_moves": legal_moves}
        _OP_LOGGER.debug("_on_step_pettingzoo_multiagent: Sending select_action to worker...")
        if handle.send_select_action(obs_str, current_player, info, action_mask=action_mask):
            _OP_LOGGER.debug("_on_step_pettingzoo_multiagent: select_action sent successfully")
            self.log_constant(
                LOG_UI_MAINWINDOW_TRACE,
                message=f"Sent select_action to {current_player}",
                extra={"player": current_player, "legal_moves_count": len(legal_moves)},
            )

            # Poll for response with timeout
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(100, lambda: self._poll_pettingzoo_action(handle, current_player))
        else:
            _OP_LOGGER.debug("_on_step_pettingzoo_multiagent: Failed to send select_action")
            self.log_constant(
                LOG_UI_MAINWINDOW_ERROR,
                message=f"Failed to send select_action to {current_player}",
            )
            # Re-enable the current player's button so the user can retry
            self._control_panel.set_current_player(current_player)

    def _poll_pettingzoo_action(
        self,
        handle: Any,
        player_id: str,
        attempts: int = 0,
        max_attempts: int = 300,  # 30 seconds at 100ms intervals
    ) -> None:
        """Poll for action response from worker and execute on shared env.

        Args:
            handle: The worker process handle.
            player_id: Which player we're waiting for.
            attempts: Current attempt number.
            max_attempts: Maximum polling attempts before timeout.
        """
        from PyQt6.QtCore import QTimer

        if attempts >= max_attempts:
            _OP_LOGGER.debug("_poll_pettingzoo_action: TIMEOUT for %s after %d attempts", player_id, max_attempts)
            self.log_constant(
                LOG_UI_MAINWINDOW_ERROR,
                message=f"Timeout waiting for action from {player_id}",
            )
            self._status_bar.showMessage(f"Timeout: {player_id} didn't respond", 5000)
            # Re-enable the current player's button after timeout
            self._control_panel.set_current_player(player_id)
            return

        # Try to read response
        response = handle.try_read_response(timeout=0.1)

        if response is None:
            # No response yet, poll again
            QTimer.singleShot(100, lambda: self._poll_pettingzoo_action(
                handle, player_id, attempts + 1, max_attempts
            ))
            return

        response_type = response.get("type", "")

        if response_type == "action_selected":
            # Got the action!
            action_str = response.get("action_str", "")
            action_index = response.get("action")
            _OP_LOGGER.debug(
                "_poll_pettingzoo_action: Got action_selected from %s: %s (index=%s)",
                player_id, action_str, action_index,
            )

            self.log_constant(
                LOG_UI_MAINWINDOW_INFO,
                message=f"Received action from {player_id}: {action_str}",
                extra={"player": player_id, "action": action_str, "index": action_index},
            )

            # Execute action on shared environment
            self._execute_pettingzoo_action(player_id, action_str, action_index)

        elif response_type == "agent_ready":
            # Worker just initialized, poll again for the actual action
            QTimer.singleShot(100, lambda: self._poll_pettingzoo_action(
                handle, player_id, attempts + 1, max_attempts
            ))

        elif response_type == "error":
            error_msg = response.get("message", "Unknown error")
            _OP_LOGGER.debug("_poll_pettingzoo_action: Got error from %s: %s", player_id, error_msg)
            self.log_constant(
                LOG_UI_MAINWINDOW_ERROR,
                message=f"Worker error for {player_id}: {error_msg}",
            )
            self._status_bar.showMessage(f"Error: {error_msg}", 5000)
            # Re-enable the current player's button after error
            self._control_panel.set_current_player(player_id)

        else:
            # Unknown response, log and poll again
            self.log_constant(
                LOG_UI_MAINWINDOW_WARNING,
                message=f"Unexpected response type from {player_id}: {response_type}",
            )
            QTimer.singleShot(100, lambda: self._poll_pettingzoo_action(
                handle, player_id, attempts + 1, max_attempts
            ))

    def _execute_pettingzoo_action(
        self,
        player_id: str,
        action_str: str,
        action_index: Optional[int] = None,
    ) -> None:
        """Execute an action on the shared PettingZoo environment.

        Args:
            player_id: Which player made the move.
            action_str: The action as a string (e.g., UCI move "e2e4").
            action_index: Optional pre-computed action index.
        """
        _OP_LOGGER.debug(
            "_execute_pettingzoo_action: player_id=%s, action_str=%s, action_index=%s",
            player_id, action_str, action_index,
        )
        env = self._shared_pettingzoo_env
        if env is None:
            _OP_LOGGER.debug("_execute_pettingzoo_action: No environment")
            return

        # Convert action string to index if needed
        if action_index is None or not isinstance(action_index, int):
            _OP_LOGGER.debug("action_index is not int (%s), converting UCI to index", type(action_index))
            action_index = self._convert_uci_to_action_index(env, action_str)
            _OP_LOGGER.debug("Converted action_index = %s", action_index)

        if action_index is None:
            # Invalid move - try to pick a random legal move using action mask
            _OP_LOGGER.debug("Invalid move '%s', picking random legal move from action_mask", action_str)
            self.log_constant(
                LOG_UI_MAINWINDOW_WARNING,
                message=f"Invalid move '{action_str}' from {player_id}, selecting random",
            )
            # Get legal actions from the environment's action mask
            try:
                # PettingZoo AEC environments provide action_mask
                obs = env.observe(player_id)
                if isinstance(obs, dict) and "action_mask" in obs:
                    action_mask = obs["action_mask"]
                else:
                    # Try getting mask directly
                    action_mask = env.action_mask(player_id) if hasattr(env, "action_mask") else None

                if action_mask is not None:
                    legal_action_indices = np.where(action_mask == 1)[0]
                    _OP_LOGGER.debug("legal_action_indices count = %d", len(legal_action_indices))
                    if len(legal_action_indices) > 0:
                        import random
                        action_index = int(random.choice(legal_action_indices))
                        _OP_LOGGER.debug("Random legal action_index = %s", action_index)
                    else:
                        _OP_LOGGER.debug("No legal actions in action_mask")
                        self._status_bar.showMessage("No legal moves available", 3000)
                        self._control_panel.set_current_player(player_id)
                        return
                else:
                    _OP_LOGGER.debug("No action_mask available")
                    self._status_bar.showMessage("Cannot determine legal moves", 3000)
                    self._control_panel.set_current_player(player_id)
                    return
            except Exception as mask_err:
                _OP_LOGGER.debug("Error getting action_mask: %s", mask_err)
                self._status_bar.showMessage(f"Error: {mask_err}", 3000)
                self._control_panel.set_current_player(player_id)
                return

        # Execute the action
        try:
            _OP_LOGGER.debug("Executing env.step(%s)", action_index)
            env.step(action_index)
            _OP_LOGGER.debug("env.step completed successfully")

            # Track step count and sync widget
            self._pettingzoo_step_index += 1
            self._control_panel.set_step_count(self._pettingzoo_step_index)

            # Check for game end - in PettingZoo AEC, check if ALL agents are terminated
            # or if there are no more agents to act
            next_player = env.agent_selection
            _OP_LOGGER.debug("After step, agent_selection=%s", next_player)

            # Check if game is truly over (all agents terminated or no agents left)
            all_terminated = all(env.terminations.values())
            all_truncated = all(env.truncations.values())
            no_agents_left = len(env.agents) == 0
            game_over = all_terminated or all_truncated or no_agents_left
            _OP_LOGGER.debug(
                "all_terminated=%s, all_truncated=%s, no_agents=%s, game_over=%s",
                all_terminated, all_truncated, no_agents_left, game_over,
            )

            # For backward compat with the rest of the code
            terminated = game_over
            truncated = False

            # Render updated board
            _OP_LOGGER.debug("Rendering frame...")
            self.render_pettingzoo_frame()
            _OP_LOGGER.debug("Frame rendered")

            if terminated or truncated:
                _OP_LOGGER.debug("Game over path")
                # Game over
                rewards = env.rewards
                winner = "Draw"
                for p, r in rewards.items():
                    if r > 0:
                        winner = p
                        break
                self._status_bar.showMessage(f"Game over! Winner: {winner}", 5000)
                self.log_constant(
                    LOG_UI_MAINWINDOW_INFO,
                    message="PettingZoo game ended",
                    extra={"winner": winner, "rewards": rewards},
                )
                # Disable both player step buttons when game is over
                self._control_panel.set_current_player("")  # Empty disables both
            else:
                _OP_LOGGER.debug("Game continues path")
                _OP_LOGGER.debug(
                    "_execute_pettingzoo_action: player_id=%s, action=%s, next_player=%s",
                    player_id, action_str, next_player,
                )
                # Update turn indicator for next player
                self._control_panel.set_turn_indicator(next_player, visible=True)
                # Toggle player step buttons for next turn
                _OP_LOGGER.debug("Calling set_current_player(%s)", next_player)
                self._control_panel.set_current_player(next_player)
                self._status_bar.showMessage(
                    f"{player_id} played {action_str}. Next: {next_player}",
                    2000
                )

            self.log_constant(
                LOG_OPERATOR_STEP_ALL_COMPLETED,
                message="PettingZoo step completed",
                extra={
                    "player": player_id,
                    "action": action_str,
                    "next_player": env.agent_selection,
                },
            )
            _OP_LOGGER.debug("_execute_pettingzoo_action completed successfully")

        except Exception as e:
            _OP_LOGGER.warning("_execute_pettingzoo_action EXCEPTION: %s", e, exc_info=True)
            self.log_constant(
                LOG_UI_MAINWINDOW_ERROR,
                message=f"Failed to execute action: {e}",
                extra={"player": player_id, "action": action_str, "index": action_index},
            )
            self._status_bar.showMessage(f"Move failed: {e}", 5000)
            # Re-enable the current player's button after error
            self._control_panel.set_current_player(player_id)


__all__ = ["PettingzooHandler"]
