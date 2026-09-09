"""Standard multi-operator lifecycle: reset, step, poll, response, stop.

Extracted from `MainWindow` (backup: `_on_reset_all_operators`,
`_on_step_all_operators`, `_poll_operator_responses`, `_handle_operator_response`,
`_on_stop_operators`) as part of refactor plan step 5. This is the foundation
the mode-specific handlers depend on. It dispatches from the generic
"reset all" / "step all" / "stop" buttons into either the standard
single-agent flow (each worker owns its own env) or the mode-specific flows
(PettingZoo AEC, Parallel multi-agent, GAR ghost replacement).

Cross-handler state still on MainWindow (will migrate in steps 7 and 8):
- `_pettingzoo_multiagent_mode`, `_shared_pettingzoo_env`,
  `_pettingzoo_player_handles` (owned by future PettingzooHandler).
- `_parallel_multiagent_mode`, `_parallel_multiagent_env`,
  `_parallel_multiagent_step_state`, `_parallel_player_handles`,
  `_linkgroup_handles`, `_multigrid_aec_mode`,
  `_clear_parallel_action_panel()` (owned by future ParallelMultiAgentHandler).
- `_is_pettingzoo_multiagent()`, `_is_parallel_multiagent()`,
  `_on_reset_pettingzoo_multiagent()`, `_on_step_pettingzoo_multiagent()`,
  `_on_reset_parallel_multiagent()`, `_on_step_parallel_multiagent()`,
  `_on_step_multigrid_aec()` — dispatched via `self._parent` for now.

Uses `LogConstantMixin` because it emits many LOG constants
(`LOG_OPERATOR_RESET_ALL_STARTED`, `LOG_OPERATOR_STEP_ALL_COMPLETED`,
`LOG_OPERATOR_STOP_ALL_COMPLETED`, `LOG_UI_MAINWINDOW_*`,
`LOG_OPERATOR_ENV_PREVIEW_ERROR`, `LOG_GAR_GHOST_ACTION_DISCARDED`,
`LOG_GAR_REPLACEMENT_LAUNCHED`).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict

from qtpy import QtCore

from gym_gui.core.enums import GameId
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_OPERATOR_RESET_ALL_STARTED,
    LOG_OPERATOR_STEP_ALL_COMPLETED,
    LOG_OPERATOR_STOP_ALL_COMPLETED,
    LOG_UI_MAINWINDOW_ERROR,
    LOG_UI_MAINWINDOW_INFO,
    LOG_UI_MAINWINDOW_TRACE,
    LOG_UI_MAINWINDOW_WARNING,
)
from gym_gui.services.operator_launcher import OperatorLaunchError

if TYPE_CHECKING:
    from qtpy import QtWidgets  # noqa: F401

    from gym_gui.services.auto_step_manager import AutoStepManager
    from gym_gui.services.operator import MultiOperatorService, OperatorConfig  # noqa: F401
    from gym_gui.services.operator_launcher import OperatorLauncher
    from gym_gui.ui.widgets.control_panel import ControlPanelWidget
    from gym_gui.ui.widgets.render_tabs import RenderTabs


_OP_LOGGER = logging.getLogger("gym_gui.operators.main_window")


class OperatorLifecycleHandler(LogConstantMixin):
    """Standard multi-operator lifecycle.

    Dispatches reset / step / stop from the Multi-Operator UI into either the
    standard single-agent flow (each worker owns its own env, IPC via
    interactive subprocess) or the mode-specific flows (PettingZoo AEC,
    Parallel multi-agent). Owns the shared `_handle_operator_response`
    response router that AutoStepHandler and ScriptModeHandler both feed into.

    Signal connections (owner: MainWindow._connect_signals):
        control_panel.reset_all_requested       -> on_reset_all_operators
        control_panel.step_all_requested        -> on_step_all_operators
        control_panel.stop_operators_requested  -> on_stop_operators
    """

    def __init__(
        self,
        *,
        parent: Any,  # MainWindow at runtime; typed Any because this handler
                     # dispatches to MainWindow-specific mode methods
                     # (_is_pettingzoo_multiagent, _on_reset_parallel_multiagent,
                     # etc.) that live on MainWindow until steps 7 and 8.
        render_tabs: "RenderTabs",
        control_panel: "ControlPanelWidget",
        status_bar: "QtWidgets.QStatusBar",
        operator_launcher: "OperatorLauncher",
        multi_operator_service: "MultiOperatorService",
        auto_step_mgr: "AutoStepManager",
    ) -> None:
        self._logger = _OP_LOGGER
        self._parent = parent
        self._render_tabs = render_tabs
        self._control_panel = control_panel
        self._status_bar = status_bar
        self._operator_launcher = operator_launcher
        self._multi_operator_service = multi_operator_service
        self._auto_step_mgr = auto_step_mgr

    def on_reset_all_operators(self, seed: int | None) -> None:
        """Reset all configured operators with shared seed.

        Scientific Execution Model (inspired by BALROG):
        - All environments reset with identical seed for fair comparison
        - Launches worker subprocesses in interactive mode if not already running
        - Sends reset command with seed to each subprocess
        - Switches to Multi-Operator tab for viewing

        For PettingZoo multi-agent games (chess, etc.):
        - GUI creates ONE shared environment
        - Workers are initialized in action_selector mode
        - Turn-based coordination handled by _on_step_all_operators
        """
        active_operators = self._multi_operator_service.get_active_operators()
        if not active_operators:
            self._status_bar.showMessage("No operators configured to reset", 3000)
            return

        # Check if this is PettingZoo multi-agent mode
        is_multiagent, first_config = self._parent._pettingzoo_handler.is_pettingzoo_multiagent()
        _OP_LOGGER.debug(
            "_on_reset_all_operators: is_multiagent=%s, first_config=%s",
            is_multiagent, first_config,
        )
        if first_config:
            _OP_LOGGER.debug(
                "_on_reset_all_operators: env_name=%s, workers=%s",
                first_config.env_name, list(first_config.workers.keys()),
            )

        if is_multiagent and first_config is not None:
            # PettingZoo multi-agent: GUI owns the shared environment
            _OP_LOGGER.debug("_on_reset_all_operators: Taking PettingZoo path")
            self._parent._pettingzoo_handler.on_reset_pettingzoo_multiagent(seed, first_config)
            return

        # Check if this is parallel multi-agent mode (MultiGrid, MeltingPot, Overcooked)
        is_parallel, parallel_config = self._parent._parallel_multiagent_handler.is_parallel_multiagent()
        _OP_LOGGER.debug(
            "_on_reset_all_operators: is_parallel=%s, parallel_config=%s",
            is_parallel, parallel_config,
        )
        if is_parallel and parallel_config is not None:
            # Parallel multi-agent: GUI owns the shared environment
            _OP_LOGGER.debug("_on_reset_all_operators: Taking Parallel multi-agent path")
            self._parent._parallel_multiagent_handler.on_reset_parallel_multiagent(seed, parallel_config)
            return

        # Standard single-agent flow: each worker owns its own environment
        # Get operators that are pending (not yet started)
        pending_ids = self._multi_operator_service.start_all()

        # Launch subprocess workers for each operator that needs to be started
        # All operators (human, LLM, RL) use subprocess workers for consistency
        started_ids = []
        failed_ids = []

        for operator_id in pending_ids:
            config = self._multi_operator_service.get_operator(operator_id)
            if config is None:
                continue

            # Human operators: launch subprocess (same pattern as LLM/RL workers)
            # The human_worker subprocess owns the gymnasium environment
            if config.worker_id == "human_worker":
                try:
                    # Launch human_worker subprocess in interactive mode
                    handle = self._operator_launcher.launch_operator(
                        config,
                        interactive=True,
                    )

                    # Read the "init" message that the worker emits on startup
                    init_response = handle.read_response(timeout=5.0)
                    if init_response is None:
                        raise RuntimeError("Timeout waiting for human_worker init")
                    if init_response.get("type") != "init":
                        self.log_constant(
                            LOG_UI_MAINWINDOW_WARNING,
                            message=f"Unexpected first response from human_worker: {init_response.get('type')}",
                        )

                    # Send reset command with seed and env configuration
                    # Include initial_state if configured (for custom board/grid positions)
                    reset_cmd: Dict[str, Any] = {
                        "cmd": "reset",
                        "seed": seed,
                        "env_name": config.env_name,
                        "task": config.task,
                    }
                    # Pass settings including initial_state for custom configurations
                    if config.settings:
                        reset_cmd["settings"] = config.settings
                    handle.send_command(reset_cmd)

                    # Wait for "ready" response with action labels and initial render
                    response = handle.read_response(timeout=10.0)
                    if response is None:
                        raise RuntimeError("Timeout waiting for human_worker ready response")

                    if response.get("type") == "error":
                        raise RuntimeError(response.get("message", "Unknown error from human_worker"))

                    if response.get("type") != "ready":
                        raise RuntimeError(f"Unexpected response type: {response.get('type')}")

                    # Extract action labels and render payload from ready response
                    action_labels = response.get("action_labels", [])
                    action_space_n = response.get("action_space", len(action_labels))
                    render_payload = response.get("render_payload")

                    # Update render container with initial state
                    if render_payload:
                        wrapped_payload = {
                            "render_payload": render_payload,
                            "episode_index": 0,
                            "step_index": 0,
                            "reward": 0.0,
                            "episode_reward": 0.0,
                        }
                        self._render_tabs.display_operator_payload(operator_id, wrapped_payload)

                    # Enable interactive mode for human operators
                    self._render_tabs.set_interactive(operator_id, True)

                    # Set game-specific keyboard mappings
                    try:
                        game_id = GameId(config.env_name)
                        self._render_tabs.set_game_id(operator_id, game_id)
                    except ValueError:
                        pass

                    # Set available actions from the worker's response
                    actions = list(range(action_space_n))
                    self._render_tabs.set_available_actions(operator_id, actions, action_labels)

                    # Set "Your Turn" indicator
                    self._render_tabs.set_human_turn(operator_id, True)

                    # Assign run_id and set state
                    self._multi_operator_service.assign_run_id(operator_id, handle.run_id)
                    self._multi_operator_service.set_operator_state(operator_id, "running")
                    started_ids.append(operator_id)

                    self.log_constant(
                        LOG_UI_MAINWINDOW_INFO,
                        message="Launched human_worker subprocess for operator",
                        extra={
                            "operator_id": operator_id,
                            "seed": seed,
                            "env_name": config.env_name,
                            "task": config.task,
                            "action_count": action_space_n,
                            "run_id": handle.run_id,
                            "pid": handle.pid,
                        },
                    )
                except Exception as e:
                    self._multi_operator_service.set_operator_state(operator_id, "error")
                    failed_ids.append(operator_id)
                    self.log_constant(
                        LOG_UI_MAINWINDOW_ERROR,
                        message=f"Failed to launch human_worker: {e}",
                        extra={"operator_id": operator_id, "seed": seed},
                    )
                continue

            # Non-human operators: launch subprocess
            try:
                # Launch the subprocess in interactive mode for step-by-step control
                handle = self._operator_launcher.launch_operator(
                    config,
                    interactive=True,  # Enable step-by-step control
                )

                # Send reset command with seed
                handle.send_reset(seed)

                # Assign run_id to the service for telemetry routing
                self._multi_operator_service.assign_run_id(operator_id, handle.run_id)
                self._multi_operator_service.set_operator_state(operator_id, "running")

                started_ids.append(operator_id)

                self.log_constant(
                    LOG_UI_MAINWINDOW_INFO,
                    message="Launched interactive operator subprocess with seed",
                    extra={
                        "operator_id": operator_id,
                        "seed": seed,
                        "run_id": handle.run_id,
                        "pid": handle.pid,
                        "log_path": str(handle.log_path),
                        "interactive": True,
                    },
                )
            except OperatorLaunchError as e:
                self._multi_operator_service.set_operator_state(operator_id, "error")
                failed_ids.append(operator_id)
                self.log_constant(
                    LOG_UI_MAINWINDOW_ERROR,
                    message=f"Failed to launch operator: {e}",
                    extra={"operator_id": operator_id, "seed": seed},
                )

        # Update status indicators and set container display sizes
        for operator_id in started_ids:
            self._render_tabs.set_operator_status(operator_id, "running")
            # Ensure container display size is set from config
            op_config = self._multi_operator_service.get_operator(operator_id)
            if op_config:
                container_size = op_config.settings.get("container_size", 0)
                if container_size and container_size > 0:
                    self._render_tabs.set_operator_display_size(
                        operator_id, container_size, container_size
                    )
        for operator_id in failed_ids:
            self._render_tabs.set_operator_status(operator_id, "error")

        # Switch to Multi-Operator tab
        self._render_tabs.switch_to_multi_operator_tab()

        # Show "Your Turn" indicator for human operators
        human_operator_ids = self._render_tabs.get_human_operator_ids()
        for human_op_id in human_operator_ids:
            self._render_tabs.set_human_turn(human_op_id, True)

        # Send reset commands to ALREADY-RUNNING operators (for subsequent episodes)
        already_running_ids = [op_id for op_id in active_operators if op_id not in started_ids and op_id not in failed_ids]
        reset_count = 0
        for operator_id in already_running_ids:
            handle = self._operator_launcher.get_handle(operator_id)
            if handle and handle.is_running:
                if handle.send_reset(seed):
                    reset_count += 1
                    self.log_constant(
                        LOG_UI_MAINWINDOW_TRACE,
                        message="Sent reset command to already-running operator",
                        extra={"operator_id": operator_id, "seed": seed},
                    )

        count = len(started_ids) + reset_count
        if failed_ids:
            self._status_bar.showMessage(
                f"Reset {count} operator{'s' if count != 1 else ''} (seed={seed}), {len(failed_ids)} failed",
                5000
            )
        else:
            self._status_bar.showMessage(
                f"Reset all operators with seed={seed}",
                3000
            )
        self.log_constant(
            LOG_OPERATOR_RESET_ALL_STARTED,
            message=f"Reset {count} operators with shared seed",
            extra={"operator_ids": started_ids + already_running_ids, "failed_ids": failed_ids, "seed": seed, "newly_started": len(started_ids), "already_running_reset": reset_count},
        )

    def on_step_all_operators(self, seed: int) -> None:
        """Step all running operators by exactly one step.

        Scientific Execution Model (inspired by BALROG):
        - Lock-step execution: each operator's agent selects one action
        - This ensures scientifically fair side-by-side comparison
        - No arbitrary timing delays between operators

        For PettingZoo multi-agent games:
        - GUI owns the shared environment
        - Gets current player from env.agent_selection
        - Sends observation to that player's worker via select_action
        - Executes returned action on shared environment
        """
        active_operators = self._multi_operator_service.get_active_operators()
        if not active_operators:
            self._status_bar.showMessage("No operators to step", 3000)
            return

        # Check if we're in PettingZoo multi-agent mode
        if self._parent._pettingzoo_handler.is_active():
            self._parent._pettingzoo_handler.on_step_pettingzoo_multiagent()
            return

        # Check if we're in parallel multi-agent mode (MultiGrid, MeltingPot, Overcooked)
        if self._parent._parallel_multiagent_handler._parallel_multiagent_mode and self._parent._parallel_multiagent_handler._parallel_multiagent_env is not None:
            # AEC sub-mode: each agent acts one at a time (true sequential stepping)
            if self._parent._parallel_multiagent_handler._multigrid_aec_mode:
                self._parent._parallel_multiagent_handler.on_step_multigrid_aec()
            else:
                self._parent._parallel_multiagent_handler.on_step_parallel_multiagent()
            return

        # Standard single-agent flow: each worker owns its own environment
        # Send step command to each running operator subprocess
        # NOTE: Human operators are EXCLUDED - they only respond to action button clicks
        stepped_count = 0
        skipped_human_count = 0
        stepped_handles = []  # Track handles that received step command
        for operator_id in active_operators:
            # Skip Human operators - they are controlled by action buttons, not Step All
            config = self._multi_operator_service.get_operator(operator_id)
            if config and config.worker_id == "human_worker":
                skipped_human_count += 1
                self.log_constant(
                    LOG_UI_MAINWINDOW_TRACE,
                    message="Skipping human operator in Step All (use action buttons)",
                    extra={"operator_id": operator_id},
                )
                continue

            handle = self._operator_launcher.get_handle(operator_id)
            if handle is None:
                self.log_constant(
                    LOG_UI_MAINWINDOW_WARNING,
                    message="No process handle for operator",
                    extra={"operator_id": operator_id},
                )
                continue

            if not handle.is_running:
                self.log_constant(
                    LOG_UI_MAINWINDOW_WARNING,
                    message="Operator process not running",
                    extra={"operator_id": operator_id, "return_code": handle.return_code},
                )
                self._multi_operator_service.set_operator_state(operator_id, "stopped")
                continue

            # Send step command (only for non-human operators)
            if handle.send_step():
                stepped_count += 1
                stepped_handles.append((operator_id, handle))
                self.log_constant(
                    LOG_UI_MAINWINDOW_TRACE,
                    message="Sent step command to operator",
                    extra={"operator_id": operator_id},
                )
            else:
                self.log_constant(
                    LOG_UI_MAINWINDOW_WARNING,
                    message="Failed to send step command to operator",
                    extra={"operator_id": operator_id},
                )

        # Read responses from operators and update render view
        # Use a short delay to allow LLM inference to complete
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, lambda: self.poll_operator_responses(stepped_handles))

        self.log_constant(
            LOG_OPERATOR_STEP_ALL_COMPLETED,
            message="Step all operators completed",
            extra={
                "stepped_count": stepped_count,
                "skipped_human": skipped_human_count,
                "total_active": len(active_operators),
            },
        )
        # Build status message
        status_parts = []
        if stepped_count > 0:
            status_parts.append(f"Stepped {stepped_count} AI operator{'s' if stepped_count != 1 else ''}")
        if skipped_human_count > 0:
            status_parts.append(f"{skipped_human_count} Human (use action buttons)")
        if status_parts:
            self._status_bar.showMessage(" | ".join(status_parts), 3000)
        else:
            self._status_bar.showMessage("No operators to step", 2000)

    def poll_operator_responses(self, handles: list, max_wait_ms: int = 30000) -> None:
        """Poll for responses from operator subprocesses and update render view.

        Args:
            handles: List of (operator_id, handle) tuples to poll.
            max_wait_ms: Maximum time to wait for responses in milliseconds.
        """
        from PyQt6.QtCore import QTimer

        pending = list(handles)
        start_time = datetime.now()

        def poll_once():
            nonlocal pending
            if not pending:
                return

            # Check if we've exceeded max wait time
            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            if elapsed > max_wait_ms:
                self.log_constant(
                    LOG_UI_MAINWINDOW_WARNING,
                    message="Timeout waiting for operator responses",
                    extra={"pending_count": len(pending)},
                )
                return

            still_pending = []
            for operator_id, handle in pending:
                # Try to read response (non-blocking)
                response = handle.try_read_response(timeout=0.1)
                if response is not None:
                    response_type = response.get("type", "unknown")
                    self.handle_operator_response(operator_id, response)
                    # If we got a non-step response (e.g., "ready" from reset), keep polling
                    if response_type != "step" and handle.is_running:
                        still_pending.append((operator_id, handle))
                else:
                    # Check if process is still running
                    if handle.is_running:
                        still_pending.append((operator_id, handle))
                    else:
                        self.log_constant(
                            LOG_UI_MAINWINDOW_WARNING,
                            message="Operator process terminated while waiting for response",
                            extra={"operator_id": operator_id},
                        )
                        self._multi_operator_service.set_operator_state(operator_id, "stopped")
                        self._render_tabs.set_operator_status(operator_id, "stopped")

            pending = still_pending
            if pending:
                # Schedule another poll
                QTimer.singleShot(200, poll_once)

        poll_once()

    def handle_operator_response(self, operator_id: str, response: dict) -> None:
        """Handle a response from an operator subprocess.

        Args:
            operator_id: The operator that sent the response.
            response: The parsed JSON response dict.
        """
        response_type = response.get("type", "unknown")

        if response_type == "step":
            # Build render payload from step response
            payload = {
                "step_index": response.get("step_index", 0),
                "episode_index": response.get("episode_index", 0),
                "reward": response.get("reward", 0.0),
                "total_reward": response.get("total_reward", 0.0),
                "terminated": response.get("terminated", False),
                "truncated": response.get("truncated", False),
                "action": response.get("action", ""),
                "observation": response.get("observation", ""),
                # Include render payload if available (format: {"mode": "rgb", "rgb": [...], "width": N, "height": N})
                "render_payload": response.get("render_payload"),
            }
            # Route to auto-step manager during collection phase
            if self._auto_step_mgr.is_active_for(operator_id):
                episode_done = payload.get("terminated", False) or payload.get("truncated", False)
                self._auto_step_mgr.on_step_collected(operator_id, payload, episode_done)
                return
            self._render_tabs.display_operator_payload(operator_id, payload)
            self.log_constant(
                LOG_UI_MAINWINDOW_TRACE,
                message="Received step response from operator",
                extra={
                    "operator_id": operator_id,
                    "step_index": payload["step_index"],
                    "reward": payload["reward"],
                },
            )

            # Notify script execution manager for automatic stepping
            script_mgr = self._control_panel.operators_tab.script_execution_manager
            script_mgr.on_step_received(operator_id)

        elif response_type == "ready":
            # Build payload to reset stats and display initial render
            # Include observation for conversation tracker (system prompt/initial context)
            payload = {
                "step_index": response.get("step_index", 0),
                "episode_index": response.get("episode_index", 0),
                "reward": 0.0,
                "episode_reward": response.get("episode_reward", 0.0),
                "render_payload": response.get("render_payload"),
                "observation": response.get("observation", ""),  # For conversation tracking
                "system_prompt": response.get("system_prompt", ""),  # Env-family instruction
            }

            # Route to auto-step manager during collection phase
            if self._auto_step_mgr.is_active_for(operator_id):
                self._render_tabs.display_operator_payload(operator_id, payload)
                self._render_tabs.set_operator_status(operator_id, "running")
                self._auto_step_mgr.on_ready_received(operator_id)
                return

            self._render_tabs.display_operator_payload(operator_id, payload)

            # Update status to "running" after operator is ready
            self._render_tabs.set_operator_status(operator_id, "running")

            self.log_constant(
                LOG_UI_MAINWINDOW_INFO,
                message="Operator ready",
                extra={"operator_id": operator_id, "seed": response.get("seed")},
            )

            # Notify script execution manager (start stepping after reset).
            # For RL workers that are NOT part of a script experiment, start a
            # free-run loop so agents visibly move without needing Script Experiments.
            script_mgr = self._control_panel.operators_tab.script_execution_manager
            if not script_mgr.is_running:
                handle = self._operator_launcher.get_handle(operator_id)
                if handle is not None and handle.config.operator_type == "rl":
                    script_mgr.start_free_run(operator_id)
                    return
            script_mgr.on_ready_received(operator_id)

        elif response_type == "error":
            self.log_constant(
                LOG_UI_MAINWINDOW_ERROR,
                message=f"Operator error: {response.get('message', 'Unknown error')}",
                extra={"operator_id": operator_id},
            )
            self._render_tabs.set_operator_status(operator_id, "error")

        elif response_type == "episode_done":
            # Episode completed - workers send "episode_done" with fields:
            # total_reward, episode_length (or num_steps), episode_number (or episode_index)
            payload = {
                "step_index": response.get("num_steps", response.get("episode_length", 0)),
                "episode_index": response.get("episode_number", response.get("episode_index", 0)),
                "reward": response.get("reward", 0.0),
                "total_reward": response.get("total_reward", 0.0),
                "terminated": response.get("terminated", False),
                "truncated": response.get("truncated", False),
                "render_payload": response.get("render_payload"),
            }
            # Route to auto-step manager -- episode_done signals end of collection
            if self._auto_step_mgr.is_active_for(operator_id):
                self._auto_step_mgr.on_step_collected(operator_id, payload, episode_done=True)
                return
            self._render_tabs.display_operator_payload(operator_id, payload)
            self.log_constant(
                LOG_UI_MAINWINDOW_INFO,
                message="Episode ended",
                extra={
                    "operator_id": operator_id,
                    "return": payload["total_reward"],
                    "steps": payload["step_index"],
                    "terminated": payload["terminated"],
                },
            )

            # Notify script execution manager for automatic execution
            script_mgr = self._control_panel.operators_tab.script_execution_manager
            script_mgr.on_episode_ended(
                operator_id,
                response.get("terminated", False),
                response.get("truncated", False)
            )

            # Auto-reset human_worker for next episode (RL workers auto-reset internally)
            # Skip auto-reset if script mode is active (script manager controls resets)
            # Only auto-reset for "interactive" mode (worker owns env), not "board-game" mode (GUI owns env)
            if not script_mgr.is_running:
                operator = self._multi_operator_service.get_operator(operator_id)
                if operator and operator.workers:
                    # Get the first worker to check if it's human_worker
                    first_worker = next(iter(operator.workers.values()), None)
                    if first_worker and first_worker.worker_id == "human_worker":
                        # Check if this is interactive mode (not PettingZoo board-game mode)
                        is_interactive_mode = operator.env_name != "pettingzoo"
                        if is_interactive_mode:
                            self.log_constant(
                                LOG_UI_MAINWINDOW_INFO,
                                message="Auto-resetting human_worker for next episode",
                                extra={"operator_id": operator_id, "env_name": operator.env_name},
                            )
                            # Get the operator handle and send reset command
                            handle = self._operator_launcher.get_handle(operator_id)
                            if handle:
                                handle.send_command({"cmd": "reset"})

        elif response_type == "stopped":
            self._multi_operator_service.set_operator_state(operator_id, "stopped")
            self._render_tabs.set_operator_status(operator_id, "stopped")

        elif response_type == "init":
            # Worker startup message — logged for diagnostics but no UI action needed
            self.log_constant(
                LOG_UI_MAINWINDOW_TRACE,
                message="Operator init received",
                extra={"operator_id": operator_id, "run_id": response.get("run_id")},
            )

        else:
            self.log_constant(
                LOG_UI_MAINWINDOW_WARNING,
                message="Unknown response type from operator",
                extra={"operator_id": operator_id, "type": response_type},
            )

    def on_stop_operators(self) -> None:
        """Stop all running operators.

        Terminates all worker subprocesses and updates status indicators.
        """
        # Stop any active free-run loop before killing subprocesses
        script_mgr = self._control_panel.operators_tab.script_execution_manager
        script_mgr.stop_free_run()

        # First stop via the launcher (actually terminates subprocesses)
        stopped_launcher_ids = self._operator_launcher.stop_all()

        # Then update the service state
        stopped_service_ids = self._multi_operator_service.stop_all()

        # Combine stopped IDs (may differ if launcher had extras)
        all_stopped = set(stopped_launcher_ids) | set(stopped_service_ids)

        if not all_stopped:
            self._status_bar.showMessage("No operators running", 3000)
            return

        # Update status indicators
        for operator_id in all_stopped:
            self._render_tabs.set_operator_status(operator_id, "stopped")

        count = len(all_stopped)
        self._status_bar.showMessage(
            f"Stopped {count} operator{'s' if count != 1 else ''}",
            3000
        )
        self.log_constant(
            LOG_OPERATOR_STOP_ALL_COMPLETED,
            message=f"Stopped {count} operators",
            extra={"operator_ids": list(all_stopped)},
        )

        # Disable PettingZoo mode if it was active
        if self._parent._pettingzoo_handler.is_active():
            self._parent._pettingzoo_handler.shutdown()

        # Disable parallel multi-agent mode if it was active
        if self._parent._parallel_multiagent_handler._parallel_multiagent_mode:
            self._control_panel.operators_tab.set_parallel_mode(False)
            self._parent._parallel_multiagent_handler.clear_parallel_action_panel()
            self._parent._parallel_multiagent_handler._parallel_multiagent_mode = False
            self._parent._parallel_multiagent_handler._parallel_multiagent_step_state = None
            if self._parent._parallel_multiagent_handler._parallel_multiagent_env is not None:
                try:
                    self._parent._parallel_multiagent_handler._parallel_multiagent_env.close()
                except Exception:
                    pass
                self._parent._parallel_multiagent_handler._parallel_multiagent_env = None
            self._parent._parallel_multiagent_handler._parallel_player_handles.clear()
            for _gid, _handle in self._parent._parallel_multiagent_handler._linkgroup_handles.items():
                try:
                    _handle.stop()
                except Exception:
                    pass
            self._parent._parallel_multiagent_handler._linkgroup_handles.clear()


__all__ = ["OperatorLifecycleHandler"]
