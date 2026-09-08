"""Auto-Step handler: pre-cache one episode of frames, replay at user interval.

Extracted from `MainWindow` (backup lines 4408-4623 and the class constant at
line 4500) as part of the refactor plan step 2. The `AutoStepManager` itself
(from `gym_gui.services.auto_step_manager`) stays as a MainWindow attribute
because it is shared across multiple handlers via cross-handler reads
(currently `_handle_operator_response` in main_window still calls
`_auto_step_mgr.is_active_for(op_id)` / `on_step_collected(...)`; when the
OperatorLifecycleHandler extraction lands in step 5, that cross-handler
coupling can be revisited).

Uses `LogConstantMixin` because it emits several `LOG_AUTO_STEP_*` constants
with structured `extra=` fields. Preserves the exact log-line shape, the
`_MAX_AUTO_STEP_POLL_ATTEMPTS = 300` timeout budget, and the specific
`QTimer.singleShot` delays (300ms for reset, 50ms for step, 100ms for
subsequent polls).

The `_poll_auto_step_response` method delegates operator responses back to
main_window's `_handle_operator_response` via an injected callback. When
that method moves in step 5, the callback wiring updates in one place.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from qtpy import QtCore, QtWidgets

from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_AUTO_STEP_COLLECTION_DONE,
    LOG_AUTO_STEP_COLLECTION_STARTED,
    LOG_AUTO_STEP_LLM_SKIPPED,
    LOG_AUTO_STEP_NO_RL_OPERATORS,
    LOG_AUTO_STEP_OPERATOR_ERROR,
    LOG_AUTO_STEP_OPERATOR_NOT_RUNNING,
    LOG_AUTO_STEP_REPLAY_STARTED,
    LOG_AUTO_STEP_REPLAY_STOPPED,
    LOG_AUTO_STEP_STEP_TIMEOUT,
)

if TYPE_CHECKING:
    from gym_gui.services.auto_step_manager import AutoStepManager
    from gym_gui.services.operator import MultiOperatorService
    from gym_gui.services.operator_launcher import OperatorLauncher
    from gym_gui.ui.widgets.control_panel import ControlPanelWidget
    from gym_gui.ui.widgets.render_tabs import RenderTabs


_OP_LOGGER = logging.getLogger("gym_gui.operators.main_window")


class AutoStepHandler(LogConstantMixin):
    """Coordinate Auto-Step collection and replay.

    Auto-Step pre-caches up to 1000 steps of episode frames from all RL
    operators using the shared seed, then replays them at the user-selected
    interval so LLM/human operators (which cannot cache) still see paced
    playback. RL steps run as fast as JAX can compute; replay is decoupled.

    Signal connections (owner: MainWindow._connect_signals):
        control_panel.auto_step_requested       -> on_auto_step_requested
        control_panel.auto_step_stop_requested  -> on_auto_step_stop
        auto_step_mgr.reset_requested           -> on_auto_step_reset_operator
        auto_step_mgr.step_requested            -> on_auto_step_step_operator
        auto_step_mgr.frame_collected           -> on_auto_step_frame_collected
        auto_step_mgr.collection_done           -> on_auto_step_collection_done
        auto_step_mgr.display_frame             -> on_auto_step_display_frame
    """

    # How many 100ms polls before declaring a timeout. 300 * 100ms = 30 seconds.
    # JAX IPPO/MAPPO steps take <10ms each in practice; this budget is hit
    # only on crashes.
    _MAX_AUTO_STEP_POLL_ATTEMPTS = 300

    def __init__(
        self,
        *,
        parent: "QtWidgets.QWidget",
        auto_step_mgr: "AutoStepManager",
        operator_launcher: "OperatorLauncher",
        multi_operator_service: "MultiOperatorService",
        control_panel: "ControlPanelWidget",
        render_tabs: "RenderTabs",
        status_bar: "QtWidgets.QStatusBar",
        handle_operator_response: Callable[[str, dict], None],
    ) -> None:
        # LogConstantMixin has no __init__; explicit super() is unnecessary
        # but safe. Keeping parent as a QObject reference so Qt doesn't
        # garbage-collect the handler while signal connections are live.
        self._logger = _OP_LOGGER
        self._parent = parent
        self._auto_step_mgr = auto_step_mgr
        self._operator_launcher = operator_launcher
        self._multi_operator_service = multi_operator_service
        self._control_panel = control_panel
        self._render_tabs = render_tabs
        self._status_bar = status_bar
        self._handle_operator_response = handle_operator_response
        # Default matches the original getattr fallback in _on_auto_step_collection_done
        self._auto_step_interval_ms: int = 500

    def on_auto_step_requested(self, seed: int, interval_ms: int) -> None:
        """Start Auto-Step: reset all RL operators then cache one episode."""
        script_mgr = self._control_panel.operators_tab.script_execution_manager
        script_mgr.stop_free_run()
        self._auto_step_mgr.stop()

        active_operators = self._multi_operator_service.get_active_operators()
        rl_ids = []
        for op_id in active_operators:
            config = self._multi_operator_service.get_operator(op_id)
            if config is None:
                continue
            # operator_type == "multiagent" for any config with >1 worker, regardless of
            # worker_type.  Check all workers to determine if this operator is RL-capable.
            has_rl_worker = any(
                w.worker_type == "rl"
                for w in config.workers.values()
            )
            if not has_rl_worker:
                self.log_constant(
                    LOG_AUTO_STEP_LLM_SKIPPED,
                    message="Operator skipped by Auto-Step -- no RL workers (LLM/human/VLM)",
                    extra={"operator_id": op_id, "operator_type": config.operator_type},
                )
                continue
            rl_ids.append(op_id)

        if not rl_ids:
            self.log_constant(
                LOG_AUTO_STEP_NO_RL_OPERATORS,
                message="Auto-Step cancelled -- no RL operators are active",
                extra={"active_operator_count": len(active_operators)},
            )
            self._status_bar.showMessage(
                "Auto-Step requires at least one RL operator", 3000
            )
            # Reset the status label -- it was set to "Auto-Step: caching..." in
            # _on_auto_step_clicked before the signal reached here.
            self._control_panel.operators_tab.set_auto_step_state(False)
            self._control_panel.operators_tab._status_label.setText("Ready")
            return

        self._auto_step_interval_ms = interval_ms
        self.log_constant(
            LOG_AUTO_STEP_COLLECTION_STARTED,
            message="Auto-Step collection started",
            extra={"operator_ids": rl_ids, "seed": seed, "n_steps": 1000},
        )
        self._auto_step_mgr.start_collection(rl_ids, seed, n_steps=1000)
        self._status_bar.showMessage(
            f"Auto-Step: caching {len(rl_ids)} operator(s) from seed {seed}...", 0
        )

    def on_auto_step_stop(self) -> None:
        """Stop Auto-Step replay or collection."""
        self._auto_step_mgr.stop()
        self._control_panel.operators_tab.set_auto_step_state(False)
        self.log_constant(
            LOG_AUTO_STEP_REPLAY_STOPPED,
            message="Auto-Step stopped by user",
        )
        self._status_bar.showMessage("Auto-Step stopped", 3000)

    def on_auto_step_reset_operator(self, operator_id: str, seed: int) -> None:
        """Send reset to one operator as requested by AutoStepManager."""
        handle = self._operator_launcher.get_handle(operator_id)
        if handle is None or not handle.is_running:
            self.log_constant(
                LOG_AUTO_STEP_OPERATOR_NOT_RUNNING,
                message="Auto-Step reset skipped -- operator process not running",
                extra={"operator_id": operator_id},
            )
            return
        handle.send_reset(seed)
        QtCore.QTimer.singleShot(300, lambda: self._poll_auto_step_response(operator_id, handle))

    def on_auto_step_step_operator(self, operator_id: str) -> None:
        """Send one step to an operator as requested by AutoStepManager."""
        handle = self._operator_launcher.get_handle(operator_id)
        if handle is None or not handle.is_running:
            self.log_constant(
                LOG_AUTO_STEP_OPERATOR_NOT_RUNNING,
                message="Auto-Step step skipped -- operator process not running",
                extra={"operator_id": operator_id},
            )
            return
        if handle.send_step():
            QtCore.QTimer.singleShot(50, lambda: self._poll_auto_step_response(operator_id, handle))

    def _poll_auto_step_response(
        self, operator_id: str, handle: Any, attempt: int = 0
    ) -> None:
        """Poll one response for the auto-step path; times out after 30 s."""
        response = handle.try_read_response(timeout=0.1)
        if response is None:
            if not self._auto_step_mgr.is_active_for(operator_id):
                return  # collection stopped or cancelled
            if attempt >= self._MAX_AUTO_STEP_POLL_ATTEMPTS:
                self.log_constant(
                    LOG_AUTO_STEP_STEP_TIMEOUT,
                    message="Auto-Step timed out waiting for operator response",
                    extra={
                        "operator_id": operator_id,
                        "timeout_s": self._MAX_AUTO_STEP_POLL_ATTEMPTS * 0.1,
                    },
                )
                self._auto_step_mgr.stop()
                self._control_panel.operators_tab.set_auto_step_state(False)
                self._status_bar.showMessage(
                    f"Auto-Step: operator '{operator_id}' timed out", 5000
                )
                return
            QtCore.QTimer.singleShot(
                100,
                lambda: self._poll_auto_step_response(operator_id, handle, attempt + 1),
            )
            return
        # Surface operator-reported errors explicitly
        if response.get("type") == "error":
            self.log_constant(
                LOG_AUTO_STEP_OPERATOR_ERROR,
                message="Auto-Step received error response from operator",
                extra={
                    "operator_id": operator_id,
                    "error": response.get("message", "unknown"),
                },
            )
            self._status_bar.showMessage(
                f"Auto-Step error from '{operator_id}': {response.get('message', '?')}",
                5000,
            )
            self._auto_step_mgr.stop()
            self._control_panel.operators_tab.set_auto_step_state(False)
            return
        # Delegate to main_window's _handle_operator_response (still there
        # until step 5 extracts OperatorLifecycleHandler).
        self._handle_operator_response(operator_id, response)

    def on_auto_step_frame_collected(
        self, operator_id: str, step: int, total: int
    ) -> None:
        """Update progress bar in Execution Controls panel."""
        self._control_panel.operators_tab.update_auto_step_progress(step, total)

    def on_auto_step_collection_done(self) -> None:
        """Collection finished -- begin replay."""
        interval_ms = self._auto_step_interval_ms
        total_frames = sum(
            self._auto_step_mgr.frame_count(op_id)
            for op_id in self._multi_operator_service.get_active_operators()
        )
        self.log_constant(
            LOG_AUTO_STEP_COLLECTION_DONE,
            message="Auto-Step collection complete",
            extra={"total_frames": total_frames, "interval_ms": interval_ms},
        )
        self._auto_step_mgr.start_replay(interval_ms)
        self._control_panel.operators_tab.set_auto_step_state(True)
        # Immediately switch to Multi-Operator tab so the replay is visible
        self._render_tabs.switch_to_multi_operator_tab()
        self.log_constant(
            LOG_AUTO_STEP_REPLAY_STARTED,
            message="Auto-Step replay started",
            extra={"total_frames": total_frames, "interval_ms": interval_ms},
        )
        self._status_bar.showMessage(
            f"Auto-Step: replaying {total_frames} frames at {interval_ms}ms interval", 0
        )

    def on_auto_step_display_frame(self, operator_id: str, payload: dict) -> None:
        """Display one cached frame during Auto-Step replay."""
        self._render_tabs.display_operator_payload(operator_id, payload)


__all__ = ["AutoStepHandler"]
