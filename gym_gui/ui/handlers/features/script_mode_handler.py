"""Script Mode handler: automated batch evaluation independent from Manual Mode.

Extracted from `MainWindow` (backup lines 5981-6167) as part of refactor plan
step 3. Script Mode manages its own operator lifecycle separately from the
Manual Mode multi_operator_service so it can drive long-running deterministic
runs (fixed seeds, procedural seeds, configurable episode counts) without
triggering Manual Mode UI updates.

Owns the `_script_parallel_operators` dict per plan section 8. Exposes
`owns_operator(op_id) -> bool` so `_execute_parallel_multiagent_step` in
MainWindow can check membership without reaching into the dict directly.

Parallel-multi-agent state (`_parallel_multiagent_env`, `_parallel_player_handles`,
`_linkgroup_handles`, etc.) is currently still owned by MainWindow and is
accessed through the injected parent reference. When the ParallelMultiAgentHandler
extraction lands in step 8, those accesses migrate to that handler in one place.

Uses `LogConstantMixin` because it emits `LOG_UI_MAINWINDOW_INFO`,
`LOG_UI_MAINWINDOW_ERROR`, and `LOG_UI_MAINWINDOW_WARNING` constants.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict

from qtpy import QtCore

from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_UI_MAINWINDOW_ERROR,
    LOG_UI_MAINWINDOW_INFO,
    LOG_UI_MAINWINDOW_WARNING,
)

if TYPE_CHECKING:
    from qtpy import QtWidgets  # noqa: F401 - kept for status_bar typing

    from gym_gui.services.operator import OperatorConfig
    from gym_gui.services.operator_launcher import OperatorLauncher
    from gym_gui.ui.widgets.control_panel import ControlPanelWidget
    from gym_gui.ui.widgets.render_tabs import RenderTabs


_OP_LOGGER = logging.getLogger("gym_gui.operators.main_window")


class ScriptModeHandler(LogConstantMixin):
    """Handle Script Mode operator lifecycle (launch, reset, step, stop).

    Script Mode is deliberately independent from Manual Mode:
    - Does NOT add operators to `multi_operator_service` (avoids Manual Mode
      UI updates like operator badges in the Multi-Operator tab).
    - Owns its own `_script_parallel_operators` dict for parallel-multi-agent
      env routing (mosaic_multigrid, meltingpot, overcooked, ini_multigrid).
    - Delegates parallel-mode reset/step to callbacks into MainWindow, because
      the parallel env and its per-agent handles still live on MainWindow
      until step 8 of the refactor.

    Signal connections (owner: MainWindow._connect_signals):
        script_execution_manager.launch_operator -> on_launch_operator
        script_execution_manager.reset_operator  -> on_reset_operator
        script_execution_manager.step_operator   -> on_step_operator
        script_execution_manager.stop_operator   -> on_stop_operator
    """

    def __init__(
        self,
        *,
        parent: Any,  # MainWindow at runtime; typed Any because this handler
                     # reaches into MainWindow-specific parallel-mode state
                     # (`_parallel_multiagent_env`, `_parallel_player_handles`,
                     # `_linkgroup_handles`, etc.) that lives on MainWindow
                     # until step 8 extracts ParallelMultiAgentHandler.
        render_tabs: "RenderTabs",
        operator_launcher: "OperatorLauncher",
        control_panel: "ControlPanelWidget",
        status_bar: "QtWidgets.QStatusBar",
        reset_parallel_multiagent: Callable[[int, "OperatorConfig"], None],
        step_parallel_multiagent: Callable[[], None],
        poll_operator_responses: Callable[[list], None],
    ) -> None:
        self._logger = _OP_LOGGER
        self._parent = parent
        self._render_tabs = render_tabs
        self._operator_launcher = operator_launcher
        self._control_panel = control_panel
        self._status_bar = status_bar
        self._reset_parallel_multiagent = reset_parallel_multiagent
        self._step_parallel_multiagent = step_parallel_multiagent
        self._poll_operator_responses = poll_operator_responses
        # Own the script-parallel operators registry (per plan section 8).
        # ParallelMultiAgentHandler (future step 8) reads this via
        # owns_operator() when episodes end.
        self._script_parallel_operators: Dict[str, "OperatorConfig"] = {}

    def owns_operator(self, operator_id: str) -> bool:
        """Return True if this handler is driving `operator_id` in parallel mode.

        Called by `_execute_parallel_multiagent_step` (still on MainWindow)
        to decide whether an episode-end should be handed back to the script
        manager or should auto-reset for manual continuous play.
        """
        return operator_id in self._script_parallel_operators

    def on_launch_operator(
        self,
        operator_id: str,
        config: "OperatorConfig",
        seed: int,
    ) -> None:
        """Handle launch request from script execution manager.

        This is separate from Manual Mode's initialize_operator signal.
        Launches operator and sends reset - responses handled asynchronously.

        Args:
            operator_id: Operator ID to launch.
            config: Operator configuration.
            seed: Initial seed for the operator.
        """
        self.log_constant(
            LOG_UI_MAINWINDOW_INFO,
            message=f"Script Mode: Launching operator {operator_id} with seed {seed}",
            extra={"operator_id": operator_id, "seed": seed},
        )

        # DON'T add to multi_operator_service - that triggers Manual Mode UI updates!
        # Script Mode manages its own operators independently

        # Parallel multi-agent envs (mosaic_multigrid, meltingpot, overcooked) need
        # the GUI to own the environment and route observations/actions itself.
        # Route these through the parallel reset callback instead of a single subprocess.
        _PARALLEL_ENV_NAMES = ("mosaic_multigrid", "meltingpot", "overcooked", "ini_multigrid")
        if len(config.workers) > 1 and config.env_name in _PARALLEL_ENV_NAMES:
            self._script_parallel_operators[operator_id] = config
            self._render_tabs.add_operator_view(config)
            self._reset_parallel_multiagent(seed, config)
            # reset_parallel_multiagent blocks via processEvents until all
            # LinkGroup workers are loaded, so workers are ready when it returns.
            script_mgr = self._control_panel.operators_tab.script_execution_manager
            script_mgr.on_ready_received(operator_id)
            return

        # Add operator view to render tabs
        self._render_tabs.add_operator_view(config)

        # Launch operator subprocess in INTERACTIVE mode (required for stdin commands!)
        try:
            handle = self._operator_launcher.launch_operator(config, interactive=True)
            if handle is None:
                self.log_constant(
                    LOG_UI_MAINWINDOW_ERROR,
                    message=f"Failed to launch operator {operator_id}",
                    extra={"operator_id": operator_id},
                )
                self._render_tabs.set_operator_status(operator_id, "error")
                return

            # Set operator status to running in UI
            self._render_tabs.set_operator_status(operator_id, "running")

            # Send reset command with full environment configuration
            # (same pattern as Manual Mode - need env_name, task, settings)
            reset_cmd: Dict[str, Any] = {
                "cmd": "reset",
                "seed": seed,
                "env_name": config.env_name,
                "task": config.task,
            }
            if config.settings:
                reset_cmd["settings"] = config.settings

            if handle.send_command(reset_cmd):
                self.log_constant(
                    LOG_UI_MAINWINDOW_INFO,
                    message=f"Script Mode: Sent reset to {operator_id}",
                    extra={"operator_id": operator_id, "seed": seed},
                )

                # Start polling for responses (same pattern as Manual Mode)
                QtCore.QTimer.singleShot(
                    100, lambda: self._poll_operator_responses([(operator_id, handle)])
                )

        except Exception as e:
            self.log_constant(
                LOG_UI_MAINWINDOW_ERROR,
                message=f"Exception launching operator {operator_id}: {e}",
                extra={"operator_id": operator_id, "error": str(e)},
            )

    def on_reset_operator(self, operator_id: str, seed: int) -> None:
        """Handle reset request from script execution manager.

        Sends reset - response handled asynchronously via existing polling.

        Args:
            operator_id: Operator ID to reset.
            seed: Seed for the new episode.
        """
        if operator_id in self._script_parallel_operators:
            # Parallel mode: workers stay alive; just reset the env with the new seed.
            # Parallel-mode state still lives on MainWindow until step 8; access
            # through the parent reference.
            env = getattr(self._parent, "_parallel_multiagent_env", None)
            if env is not None:
                try:
                    result = env.reset(seed=seed)
                    if result is not None:
                        obs, _ = result
                        if isinstance(obs, dict):
                            self._parent._parallel_multiagent_handler._parallel_multiagent_obs = obs
                except Exception as exc:
                    _OP_LOGGER.error("Parallel env reset failed: %s", exc)
            self._parent._parallel_multiagent_handler._parallel_step_index = 0
            self._parent._parallel_multiagent_handler._parallel_episode_reward = 0.0
            self._parent._parallel_multiagent_handler._parallel_multiagent_step_state = None
            script_mgr = self._control_panel.operators_tab.script_execution_manager
            script_mgr.on_ready_received(operator_id)
            return

        handle = self._operator_launcher.get_handle(operator_id)
        if handle is None or not handle.is_running:
            self.log_constant(
                LOG_UI_MAINWINDOW_WARNING,
                message=f"Operator {operator_id} not running, cannot reset",
                extra={"operator_id": operator_id},
            )
            return

        # Send reset command with seed only (env already initialized from first reset)
        if handle.send_reset(seed):
            # Start polling for "ready" response
            QtCore.QTimer.singleShot(
                100, lambda: self._poll_operator_responses([(operator_id, handle)])
            )

    def on_step_operator(self, operator_id: str) -> None:
        """Handle step request from script execution manager.

        Sends step - response handled asynchronously via existing polling.

        Args:
            operator_id: Operator ID to step.
        """
        if operator_id in self._script_parallel_operators:
            # Parallel multi-agent: GUI owns the env; trigger the shared step mechanism.
            self._step_parallel_multiagent()
            return

        handle = self._operator_launcher.get_handle(operator_id)
        if handle is None or not handle.is_running:
            return

        # Send step command (don't block!)
        if handle.send_step():
            # Start polling for "step" response
            QtCore.QTimer.singleShot(
                100, lambda: self._poll_operator_responses([(operator_id, handle)])
            )

    def on_stop_operator(self, operator_id: str) -> None:
        """Handle stop request from script execution manager.

        Args:
            operator_id: Operator ID to stop.
        """
        self.log_constant(
            LOG_UI_MAINWINDOW_INFO,
            message=f"Script Mode: Stopping operator {operator_id}",
            extra={"operator_id": operator_id},
        )

        if operator_id in self._script_parallel_operators:
            del self._script_parallel_operators[operator_id]
            # Stop all per-agent worker subprocesses. Access parallel-mode
            # handles through parent until step 8 moves them into the
            # ParallelMultiAgentHandler.
            for handle in set(self._parent._parallel_multiagent_handler._parallel_player_handles.values()):
                try:
                    handle.stop()
                except Exception:
                    pass
            self._parent._parallel_multiagent_handler._parallel_player_handles.clear()
            # Stop shared link-group subprocesses
            for handle in self._parent._parallel_multiagent_handler._linkgroup_handles.values():
                try:
                    handle.stop()
                except Exception:
                    pass
            self._parent._parallel_multiagent_handler._linkgroup_handles.clear()
            self._render_tabs.remove_operator_view(operator_id)
            return

        handle = self._operator_launcher.get_handle(operator_id)
        if handle and handle.is_running:
            handle.stop()

        # Remove the operator view from render tabs so it can be relaunched
        self._render_tabs.remove_operator_view(operator_id)


__all__ = ["ScriptModeHandler"]
