"""Keyboard bridge handler: coordinate the subprocess keyboard workers.

Extracted from `MainWindow` (backup lines 5692-5975) as part of refactor plan
step 4. Wraps the `KeyboardWorkerBridge` and owns the auto-discovery,
per-USB-port keyboard+mouse pairing, multi-cursor setup, and per-agent
subprocess launch.

The `KeyboardWorkerBridge` instance itself stays on MainWindow because it is
created in `__init__` before this handler exists (the bridge needs to be
wired into the session's keyboard_action_source at construction time). This
handler receives the bridge via constructor injection and drives its
`start()` / `stop()` / `request_next_round()` API.

The `_multi_cursor_state` (returned by evdev's `setup_multi_cursor`) moves
into this handler because it is only created inside `auto_launch_keyboard_workers`
and only torn down on shutdown.

Parallel-multi-agent state (`_parallel_multiagent_step_state`,
`_parallel_multiagent_env`, `_execute_parallel_multiagent_step`,
`_render_parallel_multiagent_frame`) is currently still owned by MainWindow;
this handler reaches through the injected parent reference. When
ParallelMultiAgentHandler lands in step 8, those accesses migrate in one place.

Uses `LogConstantMixin` because it emits `LOG_HUMAN_CONTROL_BRIDGE_START`.
Also uses a module-level `_LOGGER` for debug-level informational messages
about assignment changes (matching the original inline `_LOGGER.debug` calls).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gym_gui.core.enums import ControlMode
from gym_gui.logging_config.helpers import LogConstantMixin

if TYPE_CHECKING:
    from qtpy import QtWidgets  # noqa: F401 - kept for status_bar typing

    from gym_gui.controllers.human_input import HumanInputController
    from gym_gui.controllers.keyboard_worker_bridge import KeyboardWorkerBridge
    from gym_gui.controllers.session import SessionController
    from gym_gui.ui.widgets.control_panel import ControlPanelWidget


_LOGGER = logging.getLogger("gym_gui.ui.main_window")


class KeyboardBridgeHandler(LogConstantMixin):
    """Auto-launch and route input from the subprocess keyboard workers.

    Signal connections (owner: MainWindow._connect_signals):
        keyboard_worker_bridge.all_actions_ready       -> on_keyboard_worker_actions_ready
        keyboard_worker_bridge.mouse_delta_received    -> on_keyboard_worker_mouse_delta
        keyboard_worker_bridge.raw_key_received        -> on_keyboard_worker_raw_key
        control_panel._keyboard_widget.assignment_changed         -> on_keyboard_assignment_changed
        control_panel._keyboard_widget.all_assignments_applied    -> on_all_keyboard_assignments_applied
        control_panel.operators_tab.keyboard_assignment_changed   -> on_operator_keyboard_assignment_changed
    """

    def __init__(
        self,
        *,
        parent: Any,  # MainWindow at runtime; typed Any for the same reason
                     # as ScriptModeHandler (reads _episode_finished,
                     # _game_started, _parallel_multiagent_*, and calls
                     # _execute_parallel_multiagent_step,
                     # _render_parallel_multiagent_frame, _update_input_state).
        session: "SessionController",
        control_panel: "ControlPanelWidget",
        status_bar: "QtWidgets.QStatusBar",
        keyboard_worker_bridge: "KeyboardWorkerBridge",
        human_input: "HumanInputController",
    ) -> None:
        self._logger = _LOGGER
        self._parent = parent
        self._session = session
        self._control_panel = control_panel
        self._status_bar = status_bar
        self._keyboard_worker_bridge = keyboard_worker_bridge
        self._human_input = human_input
        # Owned by this handler; set inside auto_launch_keyboard_workers and
        # torn down in shutdown().
        self._multi_cursor_state: Any = None

    def on_keyboard_assignment_changed(self, device_path: str, agent_id: object) -> None:
        """Handle per-keyboard assignment change (informational only).

        All actual keyboard setup is done by ``auto_launch_keyboard_workers``
        which is called from ``_on_start_game`` and uses the subprocess bridge.
        This handler only logs the assignment for debugging.
        """
        agent_str = agent_id if agent_id else "unassigned"
        device_name = device_path.split("/")[-1] if "/" in device_path else device_path
        _LOGGER.debug(
            "Keyboard assignment changed: %s -> %s (handled by bridge on Start Game)",
            device_name, agent_str,
        )

    def on_operator_keyboard_assignment_changed(self, device_path: str, agent_id: object) -> None:
        """Handle per-keyboard assignment change from Operators tab (informational only).

        All actual keyboard setup is done by ``auto_launch_keyboard_workers``
        which uses the subprocess bridge. This handler only logs.
        """
        agent_str = agent_id if agent_id else "unassigned"
        device_name = device_path.split("/")[-1] if "/" in device_path else device_path
        _LOGGER.debug(
            "Operator keyboard assignment changed: %s -> %s (handled by bridge)",
            device_name, agent_str,
        )

    def auto_launch_keyboard_workers(self) -> None:
        """Auto-discover keyboards+mice, pair by USB port, launch workers.

        Called from ``MainWindow._on_start_game()``. For human-controlled modes only.
        Handles both single-agent (1 keyboard, no widget) and multi-agent
        (N keyboards, auto-assigned to agents).
        """
        from gym_gui.logging_config.log_constants import (
            LOG_HUMAN_CONTROL_BRIDGE_START,
        )

        # Only for human-play modes
        mode = self._session._control_mode
        if mode not in {
            ControlMode.HUMAN_ONLY,
            ControlMode.HYBRID_TURN_BASED,
            ControlMode.HYBRID_HUMAN_AGENT,
        }:
            return

        # If bridge is already running (e.g. user restarted), stop it first
        if self._keyboard_worker_bridge.is_active:
            self._keyboard_worker_bridge.stop()

        # Discover devices
        import sys

        from gym_gui.config.paths import HUMAN_WORKER_PKG_DIR
        _hw_path = str(HUMAN_WORKER_PKG_DIR)
        if _hw_path not in sys.path:
            sys.path.insert(0, _hw_path)
        from human_worker.evdev_input import (
            discover_keyboards,
            discover_mice,
            pair_devices_by_usb_port,
            setup_multi_cursor,
            teardown_multi_cursor,
        )

        keyboards = discover_keyboards()
        mice = discover_mice()

        if not keyboards:
            _LOGGER.warning("No keyboards found, cannot launch keyboard workers")
            return

        # Determine agent count
        num_agents = getattr(self._human_input, '_num_agents', 1) or 1
        agent_names = getattr(self._human_input, '_agent_names', None)
        if not agent_names:
            agent_names = [f"agent_{i}" for i in range(num_agents)]

        # Pair keyboards+mice by USB port
        pairs = pair_devices_by_usb_port(keyboards, mice)

        # Limit to available agents (use first N pairs)
        pairs = pairs[:num_agents]

        # Build assignment dicts
        # assignments: {keyboard_device_path: agent_id}
        # mouse_assignments: {agent_id: mouse_device_path}
        assignments = {}
        mouse_assignments = {}
        for i, pair in enumerate(pairs):
            agent_id = agent_names[i] if i < len(agent_names) else f"agent_{i}"
            assignments[pair["keyboard_path"]] = agent_id
            if pair.get("mouse_path"):
                mouse_assignments[agent_id] = pair["mouse_path"]

        if not assignments:
            _LOGGER.warning("No keyboard-agent assignments could be made")
            return

        # Use the environment family for the resolver fallback name.
        family = self._control_panel._selected_family
        env_name = family.value if family is not None else "multigrid"

        # Build the dynamic key-action map from the GUI's ShortcutMappings.
        # This is the exact same mapping the Qt shortcut system uses, converted
        # to Linux keycodes so the subprocess resolver matches perfectly.
        from gym_gui.controllers.human_input import build_key_action_map_for_game
        game_id = self._session.game_id
        key_action_map = build_key_action_map_for_game(
            game_id,
            env_family=family,
            action_space=getattr(self._session, '_action_space', None),
        )

        # Setup multi-cursor if multiple mice
        if len(mice) >= 2 and num_agents >= 2:
            mice_for_agents = mice[:num_agents]
            self._multi_cursor_state = setup_multi_cursor(mice_for_agents, agent_names)
        else:
            self._multi_cursor_state = None

        self.log_constant(
            LOG_HUMAN_CONTROL_BRIDGE_START,
            message=(
                f"Auto-launching {len(assignments)} keyboard worker(s) "
                f"({len(mouse_assignments)} with mouse), env={env_name}"
            ),
            extra={
                "assignments": {k: v for k, v in assignments.items()},
                "mouse_assignments": mouse_assignments,
                "env_name": env_name,
            },
        )

        # Update keyboard widget to show auto-assignments (UI only, no signal).
        kbd_widget = self._control_panel._keyboard_widget
        kbd_widget.set_available_agents(agent_names)
        kbd_widget._detect_keyboards()
        kbd_widget._auto_assign()  # Updates UI rows, does NOT emit signal

        # Tick rate and NOOP action from the session's InteractionController.
        # This matches the environment's own physics/render rate exactly.
        # Single-agent turn-based: blocking (tick_timeout=0), wait for key.
        # Multi-agent or real-time: tick mode, return NOOP after timeout.
        noop_action = 0
        tick_timeout = 0.0
        interaction = getattr(self._session, '_interaction', None)
        if interaction is not None:
            interval_ms = interaction.idle_interval_ms()
            if interval_ms is not None:
                tick_timeout = interval_ms / 1000.0  # ms to seconds
            passive = interaction.maybe_passive_action()
            if passive is not None and isinstance(passive, int):
                noop_action = passive
        # Multi-agent always needs tick mode even if turn-based
        if num_agents > 1 and tick_timeout <= 0:
            tick_timeout = 0.016  # 60Hz fallback for multi-agent turn-based

        # Human-readable action names, when the adapter can supply them, so the
        # runtime log reads "action 12 (shot)" instead of a bare "action 12".
        action_names: list[str] | None = None
        adapter = getattr(self._session, "_adapter", None)
        namer = getattr(adapter, "get_action_name", None)
        num_actions = getattr(adapter, "num_actions", None)
        if callable(namer) and isinstance(num_actions, int) and num_actions > 0:
            try:
                # str() guards against adapters that return non-str objects
                # (getattr on an untyped adapter yields Any; Pyright infers list[object]).
                action_names = [str(namer(i)) for i in range(num_actions)]
            except Exception:  # pragma: no cover - naming must never block input
                action_names = None

        success = self._keyboard_worker_bridge.start(
            assignments=assignments,
            env_name=env_name,
            action_list=action_names,
            mouse_assignments=mouse_assignments if mouse_assignments else None,
            key_action_map=key_action_map,
            noop_action=noop_action,
            tick_timeout=tick_timeout,
        )

        if success:
            device_info = []
            for pair in pairs:
                # `keyboard_name` and `mouse_name` are typed Optional in the
                # pairing helper; guard with `or ""` so `+=` stays valid.
                kb_name = pair["keyboard_name"] or "keyboard"
                mouse_name = pair.get("mouse_name")
                info = f"{kb_name} + {mouse_name}" if mouse_name else kb_name
                device_info.append(info)
            self._status_bar.showMessage(
                f"Keyboard workers: {', '.join(device_info)}",
                5000,
            )
            # Disable old Qt shortcuts now that bridge owns input.
            # _update_input_state stays on MainWindow (touches _keyboard_worker_bridge
            # AND _episode_finished / _game_started / _game_paused / _awaiting_human).
            self._parent._update_input_state()
        else:
            self._status_bar.showMessage(
                "Failed to launch keyboard workers. Check logs.",
                5000,
            )
            # Clean up multi-cursor on failure
            if self._multi_cursor_state is not None:
                teardown_multi_cursor(self._multi_cursor_state)
                self._multi_cursor_state = None

    def on_all_keyboard_assignments_applied(self, assignments: dict) -> None:
        """Handle Apply Assignments from widget. Delegates to auto-launch.

        Args:
            assignments: {device_path: agent_id} mapping from the keyboard widget.
        """
        if not assignments:
            _LOGGER.warning("No keyboard assignments to launch workers for")
            return
        # Delegate to the full auto-launch (discovers mice, pairs by USB port,
        # builds key_action_map, launches bridge workers).
        self.auto_launch_keyboard_workers()

    def on_keyboard_worker_actions_ready(self, actions: list) -> None:
        """Handle completed action collection from all keyboard worker subprocesses.

        Step ownership:
        - Real-time games (idle tick active): do NOT step here. The idle
          tick reads bridge.last_action at the env's native rate. We only
          update last_action (already done in bridge._poll_responses).
        - Turn-based games (no idle tick): step here via perform_human_action.
        - Multi-agent: step here (multi-agent has its own tick logic).
        """
        # Drop actions if episode is done or game stopped
        if self._parent._episode_finished or not self._parent._game_started:
            return

        # For single-agent real-time games, the idle tick owns stepping.
        # last_action is already updated by the bridge poll. Just request
        # the next round and let the idle tick consume the action.
        interaction = getattr(self._session, '_interaction', None)
        idle_active = interaction is not None and interaction.idle_interval_ms() is not None
        if len(actions) == 1 and idle_active:
            self._keyboard_worker_bridge.request_next_round()
            return

        if len(actions) == 1:
            # Single-agent turn-based: bridge triggers step directly
            self._session.perform_human_action(actions[0], key_label="keyboard")
        elif self._parent._parallel_multiagent_handler._parallel_multiagent_step_state is not None:
            # Multi-agent parallel mode: feed each action into step state
            step_state = self._parent._parallel_multiagent_handler._parallel_multiagent_step_state
            agent_order = sorted(step_state.human_agents)
            for i, agent_id in enumerate(agent_order):
                if i < len(actions) and agent_id not in step_state.pending_actions:
                    step_state.add_action(agent_id, actions[i])

            if step_state.is_complete():
                self._parent._parallel_multiagent_handler.execute_parallel_multiagent_step(step_state.get_all_actions())
        elif self._parent._parallel_multiagent_handler._parallel_multiagent_env is not None:
            # Multi-agent but no step state yet (e.g. AEC current turn)
            env = self._parent._parallel_multiagent_handler._parallel_multiagent_env
            current_agent = getattr(env, "agent_selection", None)
            if current_agent is not None and actions:
                env.step(actions[0])
                self._parent._parallel_multiagent_handler.render_parallel_multiagent_frame()
        else:
            # Fallback: treat first action as single-agent
            if actions:
                self._session.perform_human_action(actions[0], key_label="keyboard")

        # Request next round after stepping
        self._keyboard_worker_bridge.request_next_round()

    def on_keyboard_worker_mouse_delta(self, agent_id: str, dx: int, dy: int) -> None:
        """Route mouse delta from keyboard worker subprocess to native mouse handler."""
        self._session.handle_native_mouse(dx, dy)

    def on_keyboard_worker_raw_key(self, agent_id: str, keycode: int, pressed: bool) -> None:
        """Route raw key event from worker subprocess to native key handler.

        Only Malmo environments have a native handler (TCP to Minecraft).
        For all other environments, raw keys are silently ignored.
        """
        interaction = getattr(self._session, '_interaction', None)
        if interaction is not None and hasattr(interaction, 'handle_native_key_evdev'):
            interaction.handle_native_key_evdev(keycode, pressed)

    def shutdown(self) -> None:
        """Stop the bridge and teardown multi-cursor. Called from closeEvent.

        Idempotent: safe to call multiple times or when the bridge was never
        started.
        """
        # Stop keyboard worker subprocesses
        try:
            self._keyboard_worker_bridge.stop()
        except Exception as exc:
            _LOGGER.warning("Failed to stop keyboard worker bridge: %s", exc)

        # Restore multi-cursor (put all mice back on default pointer)
        if self._multi_cursor_state is not None:
            try:
                import sys

                from gym_gui.config.paths import HUMAN_WORKER_PKG_DIR
                _hw_path = str(HUMAN_WORKER_PKG_DIR)
                if _hw_path not in sys.path:
                    sys.path.insert(0, _hw_path)
                from human_worker.evdev_input import teardown_multi_cursor
                teardown_multi_cursor(self._multi_cursor_state)
            except Exception as exc:
                _LOGGER.warning("Failed to teardown multi-cursor: %s", exc)
            self._multi_cursor_state = None


__all__ = ["KeyboardBridgeHandler"]
