"""Training run lifecycle: submit, live-tab creation, telemetry routing, replay tabs.

Extracted from `MainWindow` (backup: `_submit_training_config`,
`_on_training_submitted`, `_on_training_submit_failed`,
`_on_live_telemetry_tab_requested`, `_create_agent_tabs_for`,
`_resolve_run_metadata`, `_on_training_finished`, `_on_run_completed`,
`_build_policy_evaluation_config`) as part of refactor plan step 6.

Owns the `_run_metadata` dict (per plan section 8). TrainingMonitorHandler
and FastLaneTabHandler both read this dict via `resolve_run_metadata(...)`
or direct reference passed at their construction.

This handler complements the existing TrainingFormHandler (which handles
form dialogs) and TrainingMonitorHandler (which polls for new runs) by
owning the actual submit + live-tab + finish/complete flow. It is
instantiated FIRST among the training-related handlers so its methods can
be injected as callbacks into the others.

Uses `LogConstantMixin` because it emits many LOG constants
(`LOG_UI_MAINWINDOW_*`, `LOG_UI_WORKER_TABS_*`, `LOG_LIVE_CONTROLLER_RUN_COMPLETED`).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import grpc
from qtpy import QtCore, QtWidgets

from gym_gui.core.enums import GameId
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_LIVE_CONTROLLER_RUN_COMPLETED,
    LOG_UI_MAINWINDOW_ERROR,
    LOG_UI_MAINWINDOW_INFO,
    LOG_UI_MAINWINDOW_INVALID_CONFIG,
    LOG_UI_MAINWINDOW_TRACE,
    LOG_UI_MAINWINDOW_WARNING,
    LOG_UI_WORKER_TABS_ERROR,
    LOG_UI_WORKER_TABS_INFO,
    LOG_UI_WORKER_TABS_WARNING,
)
from gym_gui.rendering import RendererRegistry
from gym_gui.services.service_locator import get_service_locator
from gym_gui.services.trainer import TrainerClientRunner
from gym_gui.ui.presenters.workers import get_worker_presenter_registry
from gym_gui.ui.widgets.live_telemetry_tab import LiveTelemetryTab

if TYPE_CHECKING:
    from gym_gui.controllers.live_telemetry_controllers import LiveTelemetryController
    from gym_gui.services.trainer.streams import TelemetryAsyncHub
    from gym_gui.ui.handlers.features.fastlane_tab_handler import FastLaneTabHandler
    from gym_gui.ui.panels.analytics_tabs import AnalyticsTabManager
    from gym_gui.ui.widgets.control_panel import ControlPanelWidget
    from gym_gui.ui.widgets.render_tabs import RenderTabs


_LOGGER = logging.getLogger("gym_gui.ui.main_window")

# Reuse the same TRAINER_SUBMIT_DEADLINE_MULTIPLIER logic as MainWindow.
from gym_gui.constants import TRAINER_DEFAULTS

TRAINER_SUBMIT_DEADLINE_MULTIPLIER = 6


def _training_submit_deadline_seconds() -> float:
    """Return the gRPC deadline used for SubmitRun requests."""
    return TRAINER_DEFAULTS.client.deadline_s * TRAINER_SUBMIT_DEADLINE_MULTIPLIER


class TrainingLifecycleHandler(LogConstantMixin):
    """Training run submit, live-tab lifecycle, and completion routing.

    Owns `_run_metadata` (dict keyed by `(run_id, agent_id)`). Callbacks
    exposed for other handlers:
        - submit_training_config: injected into TrainingFormHandler
        - build_policy_evaluation_config: injected into TrainingFormHandler
        - resolve_run_metadata: called by FastLaneTabHandler via lambdas
        - _run_metadata (dict): passed by reference to TrainingMonitorHandler

    Signal connections (owner: MainWindow._connect_signals):
        live_controller.run_tab_requested -> on_live_telemetry_tab_requested
        live_controller.run_completed     -> on_run_completed
        trainer_signals.training_finished -> on_training_finished
    """

    def __init__(
        self,
        *,
        parent: Any,  # MainWindow at runtime; used as QMessageBox parent
        render_tabs: "RenderTabs",
        render_group: "QtWidgets.QGroupBox",
        control_panel: "ControlPanelWidget",
        status_bar: "QtWidgets.QStatusBar",
        live_controller: "LiveTelemetryController",
        analytics_tabs: "AnalyticsTabManager",
        telemetry_hub: "TelemetryAsyncHub",
        fastlane_tab_handler: "FastLaneTabHandler",
    ) -> None:
        self._logger = _LOGGER
        self._parent = parent
        self._render_tabs = render_tabs
        self._render_group = render_group
        self._control_panel = control_panel
        self._status_bar = status_bar
        self._live_controller = live_controller
        self._analytics_tabs = analytics_tabs
        self._telemetry_hub = telemetry_hub
        self._fastlane_tab_handler = fastlane_tab_handler
        # Owned by this handler (per refactor plan section 8).
        # Passed by reference to TrainingMonitorHandler at its construction
        # so both share the same dict.
        self._run_metadata: Dict[tuple[str, str], Dict[str, Any]] = {}

    def build_policy_evaluation_config(
        self, worker_id: str, policy_path: Path
    ) -> Optional[dict[str, object]]:
        """Build training config using the worker presenter registry.

        Delegates configuration composition to the appropriate worker presenter,
        which handles worker-specific logic for config building, metadata composition, etc.
        """
        try:
            registry = get_worker_presenter_registry()
            presenter = registry.get(worker_id)

            if presenter is None:
                raise ValueError(f"Worker presenter '{worker_id}' not found in registry")

            config = presenter.build_train_request(
                policy_path=policy_path,
                current_game=self._control_panel.current_game(),
            )
            return config
        except FileNotFoundError:
            QtWidgets.QMessageBox.warning(
                self._parent,
                "Policy Not Found",
                f"Could not read policy file:\n{policy_path}",
            )
            return None
        except ValueError as e:
            QtWidgets.QMessageBox.warning(
                self._parent,
                "Invalid Configuration",
                f"Configuration error:\n{e}",
            )
            return None
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self._parent,
                "Policy Load Failed",
                f"Could not prepare training request:\n{e}",
            )
            return None

    def submit_training_config(self, config: dict) -> None:
        """Submit a training configuration to the trainer daemon."""
        try:
            self.log_constant(
                LOG_UI_MAINWINDOW_TRACE,
                message="_submit_training_config: START",
            )
            config_json = json.dumps(config)
            self.log_constant(
                LOG_UI_MAINWINDOW_TRACE,
                message=f"_submit_training_config: Config JSON length={len(config_json)}",
                extra={"config_length": len(config_json)},
            )

            locator = get_service_locator()
            runner = locator.resolve(TrainerClientRunner)
            if runner is None:
                raise RuntimeError("TrainerClientRunner not registered")

            self.log_constant(
                LOG_UI_MAINWINDOW_TRACE,
                message="_submit_training_config: TrainerClientRunner resolved",
            )
            self._status_bar.showMessage("Submitting training run...", 3000)

            # Submit returns a Future
            future = runner.submit_run(config_json, deadline=_training_submit_deadline_seconds())
            self.log_constant(
                LOG_UI_MAINWINDOW_TRACE,
                message="_submit_training_config: submit_run() called, future created",
            )

            # Add callback to handle result
            def on_done(fut):
                self.log_constant(
                    LOG_UI_MAINWINDOW_TRACE,
                    message="_submit_training_config: on_done callback called",
                )
                try:
                    response = fut.result()
                    self.log_constant(
                        LOG_UI_MAINWINDOW_TRACE,
                        message=f"_submit_training_config: Got response with run_id={response.run_id}",
                        extra={"run_id": str(response.run_id)},
                    )
                except Exception as error:
                    if isinstance(error, grpc.aio.AioRpcError) and error.code() == grpc.StatusCode.INVALID_ARGUMENT:
                        self.log_constant(
                            LOG_UI_MAINWINDOW_INVALID_CONFIG,
                            message="Trainer rejected training config",
                            extra={
                                "exception": type(error).__name__,
                                "details": error.details() if hasattr(error, "details") else "",
                            },
                            exc_info=error,
                        )
                    else:
                        self.log_constant(
                            LOG_UI_MAINWINDOW_ERROR,
                            message="_submit_training_config: on_done got exception",
                            extra={"exception": type(error).__name__},
                            exc_info=error,
                        )
                    QtCore.QTimer.singleShot(0, lambda e=error: self.on_training_submit_failed(e))
                    return
                run_id = str(response.run_id)
                self.log_constant(
                    LOG_UI_MAINWINDOW_INFO,
                    message=f"_submit_training_config: Scheduling _on_training_submitted with run_id={run_id}",
                    extra={"run_id": run_id},
                )
                QtCore.QTimer.singleShot(0, lambda: self.on_training_submitted(run_id, config))

            future.add_done_callback(on_done)
            self.log_constant(
                LOG_UI_MAINWINDOW_TRACE,
                message="_submit_training_config: Callback added to future",
            )

        except Exception as e:
            self.log_constant(
                LOG_UI_MAINWINDOW_ERROR,
                message="Failed to prepare training submission",
                extra={"exception": type(e).__name__},
                exc_info=e,
            )
            QtWidgets.QMessageBox.critical(
                self._parent,
                "Training Preparation Failed",
                f"Could not prepare training request:\n{e}",
            )

    def on_training_submitted(self, run_id: str, config: dict) -> None:
        """Handle successful training submission (called on main thread)."""
        self._status_bar.showMessage(f"Training run submitted: {run_id[:12]}...", 5000)
        self.log_constant(
            LOG_UI_MAINWINDOW_INFO,
            message="Submitted training run",
            extra={"run_id": run_id, "config": config},
        )

        metadata = config.get("metadata", {})
        environment = config.get("environment", {})
        environment_dict = environment if isinstance(environment, dict) else {}

        # Extract buffer sizes from config and set them in the controller
        try:
            ui_config = metadata.get("ui", {})
            step_buffer_size = ui_config.get("telemetry_buffer_size", 100)
            episode_buffer_size = ui_config.get("episode_buffer_size", 100)
            hub_buffer_size = ui_config.get("hub_buffer_size")  # Hub buffer from training config

            self._live_controller.set_buffer_sizes_for_run(run_id, step_buffer_size, episode_buffer_size)

            # Set hub buffer size if provided
            if hub_buffer_size is not None:
                self._telemetry_hub.set_run_buffer_size(run_id, hub_buffer_size)
                self.log_constant(
                    LOG_UI_MAINWINDOW_TRACE,
                    message="Set hub buffer size for run",
                    extra={
                        "run_id": run_id,
                        "hub_buffer_size": hub_buffer_size,
                    },
                )

            self.log_constant(
                LOG_UI_MAINWINDOW_TRACE,
                message="Set buffer sizes for run",
                extra={
                    "run_id": run_id,
                    "step_buffer_size": step_buffer_size,
                    "episode_buffer_size": episode_buffer_size,
                    "hub_buffer_size": hub_buffer_size,
                },
            )
        except Exception as e:
            self.log_constant(
                LOG_UI_MAINWINDOW_WARNING,
                message=f"Failed to set buffer sizes: {e}",
                extra={"exception": type(e).__name__},
                exc_info=e,
            )

        # Extract game_id from environment and store in controller
        try:
            game_id = environment_dict.get("GYM_ENV_ID", "")
            if game_id:
                self._live_controller.set_game_id_for_run(run_id, game_id)
                self.log_constant(
                    LOG_UI_MAINWINDOW_TRACE,
                    message="Set game_id for run",
                    extra={"run_id": run_id, "game_id": game_id},
                )
        except Exception as e:
            self.log_constant(
                LOG_UI_MAINWINDOW_WARNING,
                message=f"Failed to set game_id: {e}",
                extra={"exception": type(e).__name__},
                exc_info=e,
            )

        self.log_constant(
            LOG_UI_MAINWINDOW_INFO,
            message="Waiting for live telemetry to create dynamic agent tabs",
            extra={
                "run_id": run_id,
                "expected_tabs": [
                    "Agent-{agent_id}-Replay",
                    "Agent-{agent_id}-Online-Grid",
                    "Agent-{agent_id}-Online-Raw",
                    "Agent-{agent_id}-Online-Video",
                ],
            },
        )

        # Extract UI rendering throttle from environment variables and set it on the controller
        if environment_dict:
            try:
                throttle_str = environment_dict.get("TELEMETRY_SAMPLING_INTERVAL", "2")
                throttle_interval = int(throttle_str)
                self._live_controller.set_render_throttle_for_run(run_id, throttle_interval)
                self.log_constant(
                    LOG_UI_MAINWINDOW_TRACE,
                    message="Set render throttle for run",
                    extra={"run_id": run_id, "throttle": throttle_interval},
                )
            except (ValueError, TypeError):
                self.log_constant(
                    LOG_UI_MAINWINDOW_WARNING,
                    message="Failed to parse TELEMETRY_SAMPLING_INTERVAL",
                    extra={
                        "run_id": run_id,
                        "value": environment_dict.get("TELEMETRY_SAMPLING_INTERVAL"),
                    },
                )

        # Apply render delay and enable flag from metadata (defaulting to enabled)
        ui_config = metadata.get("ui", {}) if isinstance(metadata, dict) else {}
        render_delay_ms = int(environment_dict.get("UI_RENDER_DELAY_MS", ui_config.get("render_delay_ms", 100)))
        live_rendering_enabled = ui_config.get("live_rendering_enabled", True)
        self._live_controller.set_render_delay_for_run(run_id, int(render_delay_ms))
        self._live_controller.set_live_render_enabled_for_run(run_id, bool(live_rendering_enabled))

        # Persist metadata keyed by (run_id, agent_id)
        worker_meta = metadata.get("worker", {}) if isinstance(metadata, dict) else {}
        worker_config = worker_meta.get("config", {}) if isinstance(worker_meta, dict) else {}
        agent_id_key = worker_meta.get("agent_id") or worker_config.get("agent_id") or "default"
        self._run_metadata[(run_id, agent_id_key)] = metadata

        # Attempt to provision FastLane tab immediately (fastlane-only runs may never emit telemetry)
        self.log_constant(
            LOG_UI_MAINWINDOW_TRACE,
            message="Calling maybe_open_fastlane_tab",
            extra={
                "run_id": run_id,
                "agent_id_key": agent_id_key,
                "metadata_keys": list(metadata.keys()) if metadata else None,
                "ui_fastlane_only": metadata.get("ui", {}).get("fastlane_only") if metadata else None,
                "worker_module": metadata.get("worker", {}).get("module") if metadata else None,
            },
        )
        self._fastlane_tab_handler.maybe_open_fastlane_tab(run_id, agent_id_key, metadata)

        tb_ready = self._analytics_tabs.ensure_tensorboard_tab(run_id, agent_id_key, metadata)
        wb_ready = self._analytics_tabs.ensure_wandb_tab(run_id, agent_id_key, metadata)

        if not wb_ready:
            # Schedule retries so that WANDB manifest written after initialization triggers the tab.
            self._analytics_tabs.load_and_create_tabs(run_id, agent_id_key)

        if not tb_ready:
            # TensorBoard manifests usually exist up front; if they do not, reuse the same retry path.
            self._analytics_tabs.load_and_create_tabs(run_id, agent_id_key)

        # Subscribe to telemetry
        # NOTE: Do NOT call _create_agent_tabs_for() here!
        # The LiveTelemetryController will create the LiveTelemetryTab dynamically
        # when the first telemetry event arrives. This ensures proper naming and
        # routing of telemetry data to the correct tab.
        self._live_controller.subscribe_to_run(run_id)
        self._render_group.setTitle(f"Live Training - {run_id[:12]}...")

    def on_training_submit_failed(self, error: Exception) -> None:
        """Handle training submission failure (called on main thread)."""
        self.log_constant(
            LOG_UI_MAINWINDOW_ERROR,
            message="Failed to submit training run",
            extra={"exception": type(error).__name__},
            exc_info=error,
        )
        QtWidgets.QMessageBox.critical(
            self._parent,
            "Training Submission Failed",
            f"Could not submit training run:\n{error}\n\n"
            "Make sure the trainer daemon is running:\n"
            "  python -m gym_gui.services.trainer_daemon",
        )

    def on_live_telemetry_tab_requested(self, run_id: str, agent_id: str, tab_title: str) -> None:
        """Create and register a new live telemetry tab dynamically."""

        locator = get_service_locator()
        renderer_registry = locator.resolve(RendererRegistry)

        # Get buffer sizes from controller (set during training submission)
        step_buffer_size, episode_buffer_size = self._live_controller.get_buffer_sizes_for_run(run_id)

        # Get game_id from controller (set during training submission)
        game_id_str = self._live_controller.get_game_id_for_run(run_id)
        game_id = None
        if game_id_str:
            try:
                game_id = GameId(game_id_str)
            except (ValueError, KeyError):
                self.log_constant(
                    LOG_UI_MAINWINDOW_WARNING,
                    message="Invalid game_id for run",
                    extra={"run_id": run_id, "game_id": game_id_str},
                )

        # Get render throttle interval from controller (set during training submission)
        render_throttle_interval = self._live_controller.get_render_throttle_for_run(run_id)
        render_delay_ms = self._live_controller.get_render_delay_for_run(run_id)
        live_render_enabled = self._live_controller.is_live_render_enabled(run_id)

        tab = LiveTelemetryTab(
            run_id,
            agent_id,
            game_id=game_id,
            buffer_size=step_buffer_size,
            episode_buffer_size=episode_buffer_size,
            render_throttle_interval=render_throttle_interval,
            render_delay_ms=render_delay_ms,
            live_render_enabled=live_render_enabled,
            renderer_registry=renderer_registry,
            parent=self._render_tabs,
        )
        self._live_controller.register_tab(run_id, agent_id, tab)

        # Add to render tabs widget using add_dynamic_tab to include close button
        self._render_tabs.add_dynamic_tab(run_id, tab_title, tab)

        self.log_constant(
            LOG_UI_MAINWINDOW_INFO,
            message="Created live telemetry tab",
            extra={"run_id": run_id, "agent_id": agent_id, "title": tab_title, "game_id": game_id_str},
        )

        self._fastlane_tab_handler.maybe_open_fastlane_tab(
            run_id, agent_id, self.resolve_run_metadata(run_id, agent_id)
        )

    def create_agent_tabs_for(self, run_id: str, agent_id: str, first_payload: dict) -> None:
        """Create dynamic agent tabs using the worker presenter registry.

        For ToyText environments (FrozenLake, CliffWalking, Taxi):
        - Online tab shows grid rendering (primary view)
        - Replay tab shows episode browser

        For visual environments (Atari, etc.):
        - Online tab shows video rendering
        - Replay tab shows episode browser

        Delegates tab creation to the appropriate worker presenter based on
        the worker type, which handles environment detection and conditional
        tab instantiation.
        """
        try:
            # Get the presenter registry and resolve the worker presenter
            registry = get_worker_presenter_registry()
            worker_id = "cleanrl_worker"  # TODO: Extract from config/payload if supporting multiple workers
            presenter = registry.get(worker_id)

            if presenter is None:
                self.log_constant(
                    LOG_UI_WORKER_TABS_ERROR,
                    message="Worker presenter not found in registry",
                    extra={"run_id": run_id, "agent_id": agent_id, "worker_id": worker_id},
                )
                return

            tabs = presenter.create_tabs(run_id, agent_id, first_payload, parent=self)

            # Tab names and registration order must match presenter output
            tab_names = [
                f"Agent-{agent_id}-Online",
                f"Agent-{agent_id}-Replay",
                f"Agent-{agent_id}-Live – Grid",
                f"Agent-{agent_id}-Debug",
            ]

            # Determine if video tab was created (check if environment is visual)
            game_id_str = first_payload.get("game_id", "").lower()
            is_toytext = any(name in game_id_str for name in ["frozenlake", "cliffwalking", "taxi", "gridworld"])

            if not is_toytext:
                tab_names.append(f"Agent-{agent_id}-Live – Video")

            # Register tabs with the render container
            if len(tabs) != len(tab_names):
                self.log_constant(
                    LOG_UI_WORKER_TABS_WARNING,
                    message="Tab count mismatch",
                    extra={
                        "run_id": run_id,
                        "agent_id": agent_id,
                        "expected": len(tab_names),
                        "actual": len(tabs),
                    },
                )

            for tab_name, tab_widget in zip(tab_names, tabs):
                self._render_tabs.add_dynamic_tab(run_id, tab_name, tab_widget)

            # Update metadata if available
            metadata = self._run_metadata.get((run_id, agent_id))
            if metadata:
                grid_tab = tabs[2] if len(tabs) > 2 else None
                if grid_tab and hasattr(grid_tab, "update_metadata"):
                    grid_tab.update_metadata(metadata)

            self.log_constant(
                LOG_UI_WORKER_TABS_INFO,
                message="Created dynamic agent tabs via presenter registry",
                extra={
                    "run_id": run_id,
                    "agent_id": agent_id,
                    "worker_id": worker_id,
                    "game_id": game_id_str,
                    "is_toytext": is_toytext,
                    "tabs": tab_names,
                },
            )
        except Exception as e:
            self.log_constant(
                LOG_UI_WORKER_TABS_ERROR,
                message="Failed to create agent tabs",
                extra={"run_id": run_id, "agent_id": agent_id, "error": str(e)},
                exc_info=e,
            )

    def resolve_run_metadata(self, run_id: str, agent_id: str) -> Optional[Dict[str, Any]]:
        meta = self._run_metadata.get((run_id, agent_id))
        if meta is not None:
            return meta
        for (stored_run_id, _stored_agent), stored_meta in self._run_metadata.items():
            if stored_run_id == run_id:
                return stored_meta
        return None

    def on_training_finished(self, run_id: str, outcome: str, failure_reason: str) -> None:
        """Handle training_finished signal - create/refresh replay tabs for all agents in this run."""
        self.log_constant(
            LOG_UI_MAINWINDOW_INFO,
            message="Training finished signal received",
            extra={"run_id": run_id, "outcome": outcome, "failure_reason": failure_reason},
        )

        # Get all agents that participated in this run
        agent_tabs = self._render_tabs._agent_tabs.get(run_id, {})

        self.log_constant(
            LOG_UI_MAINWINDOW_TRACE,
            message="_on_training_finished: agent_tabs for run",
            extra={"run_id": run_id, "tab_count": len(agent_tabs), "tab_names": list(agent_tabs.keys())},
        )

        # Extract unique agent IDs from tab names (e.g., "Agent-1-Online" -> agent_id="1")
        agent_ids_with_tabs = set()
        for tab_name in agent_tabs.keys():
            # Tab names follow pattern: "Agent-{agent_id}-*"
            if tab_name.startswith("Agent-"):
                parts = tab_name.split("-")
                if len(parts) >= 2:
                    agent_id = parts[1]
                    agent_ids_with_tabs.add(agent_id)

        if not agent_ids_with_tabs:
            # Analytics-only runs (Fast Path) may never instantiate live tabs. Fall back to
            # metadata captured at submission time so we can surface analytics tabs.
            for (meta_run_id, meta_agent_id), _metadata in self._run_metadata.items():
                if meta_run_id != run_id:
                    continue
                if not meta_agent_id:
                    continue
                agent_ids_with_tabs.add(meta_agent_id)

            if agent_ids_with_tabs:
                self.log_constant(
                    LOG_UI_MAINWINDOW_TRACE,
                    message="_on_training_finished: using metadata agent ids",
                    extra={"run_id": run_id, "agent_ids": list(agent_ids_with_tabs)},
                )
            else:
                # Guarantee downstream logic executes at least once; analytics tabs will use
                # "default" which matches legacy emitter behaviour.
                agent_ids_with_tabs.add("default")
                self.log_constant(
                    LOG_UI_MAINWINDOW_TRACE,
                    message="_on_training_finished: no agent tabs or metadata; defaulting",
                    extra={"run_id": run_id},
                )

        self.log_constant(
            LOG_UI_MAINWINDOW_TRACE,
            message="_on_training_finished: extracted agent IDs",
            extra={"run_id": run_id, "agent_ids": list(agent_ids_with_tabs)},
        )

        # Create or refresh replay tabs for each agent, and switch to the first one created
        first_replay_tab_index: int | None = None

        for agent_id in agent_ids_with_tabs:
            replay_tab_name = f"Agent-{agent_id}-Replay"

            self.log_constant(
                LOG_UI_MAINWINDOW_TRACE,
                message="_on_training_finished: processing agent",
                extra={"run_id": run_id, "agent_id": agent_id, "replay_tab_name": replay_tab_name},
            )

            # Load analytics.json from disk and create/refresh analytics tabs (TensorBoard, WANDB)
            self._analytics_tabs.load_and_create_tabs(run_id, agent_id)

            # Check if replay tab already exists
            if replay_tab_name in agent_tabs:
                # Refresh existing replay tab
                try:
                    tab_widget = agent_tabs[replay_tab_name]
                    refresh = getattr(tab_widget, "refresh", None)
                    if callable(refresh):
                        self.log_constant(
                            LOG_UI_MAINWINDOW_TRACE,
                            message="_on_training_finished: calling refresh on existing replay tab",
                            extra={"run_id": run_id, "agent_id": agent_id},
                        )
                        refresh()
                        self.log_constant(
                            LOG_UI_MAINWINDOW_TRACE,
                            message="Refreshed replay tab",
                            extra={"run_id": run_id, "tab_name": replay_tab_name},
                        )
                    # Record the tab index for switching later
                    if first_replay_tab_index is None:
                        first_replay_tab_index = self._render_tabs.indexOf(tab_widget)
                        self.log_constant(
                            LOG_UI_MAINWINDOW_TRACE,
                            message="_on_training_finished: recorded existing replay tab index",
                            extra={"run_id": run_id, "agent_id": agent_id, "tab_index": first_replay_tab_index},
                        )
                except Exception as e:
                    self.log_constant(
                        LOG_UI_MAINWINDOW_WARNING,
                        message="Failed to refresh replay tab",
                        exc_info=e,
                        extra={"run_id": run_id, "tab_name": replay_tab_name},
                    )
            else:
                # TODO: Replay tab creation needs to be reimplemented for current workers
                # Workers should implement their own replay tab via their presenter's create_tabs method
                self.log_constant(
                    LOG_UI_MAINWINDOW_TRACE,
                    message="_on_training_finished: replay tab not available for this worker",
                    extra={"run_id": run_id, "agent_id": agent_id, "replay_tab_name": replay_tab_name},
                )

        self.log_constant(
            LOG_UI_MAINWINDOW_TRACE,
            message="_on_training_finished: about to switch to replay tab",
            extra={"run_id": run_id, "first_replay_tab_index": first_replay_tab_index},
        )

        # Switch to the first replay tab created/refreshed so user can see results
        if first_replay_tab_index is not None and first_replay_tab_index >= 0:
            self._render_tabs.setCurrentIndex(first_replay_tab_index)
            self.log_constant(
                LOG_UI_MAINWINDOW_INFO,
                message="Switched to replay tab after training completion",
                extra={"run_id": run_id, "tab_index": first_replay_tab_index},
            )
        else:
            self.log_constant(
                LOG_UI_MAINWINDOW_WARNING,
                message="_on_training_finished: could not find valid replay tab to switch to",
                extra={"run_id": run_id, "first_replay_tab_index": first_replay_tab_index},
            )

    def on_run_completed(self, run_id: str) -> None:
        """Handle run completion - keep Live-Agent tabs open, add Replay tabs."""
        self.log_constant(
            LOG_LIVE_CONTROLLER_RUN_COMPLETED,
            message="Run completed signal received",
            extra={"run_id": run_id},
        )

        # Clear FastLane tab tracking for this run (delegated to handler)
        self._fastlane_tab_handler.clear_tabs_for_run(run_id)

        # Unsubscribe from telemetry (stops new events from arriving)
        if self._live_controller:
            try:
                self._live_controller.unsubscribe_from_run(run_id)
                self.log_constant(
                    LOG_UI_MAINWINDOW_TRACE,
                    message="Unsubscribed from telemetry",
                    extra={"run_id": run_id},
                )
            except Exception as e:
                self.log_constant(
                    LOG_UI_MAINWINDOW_WARNING,
                    message="Failed to unsubscribe from telemetry",
                    exc_info=e,
                    extra={"run_id": run_id},
                )

        # NOTE: Do NOT remove Live-Agent tabs - keep them open so user can review the training
        # The Live-Agent tab will remain visible with the final state
        # Replay tabs will be created by _on_training_finished() signal
        self.log_constant(
            LOG_LIVE_CONTROLLER_RUN_COMPLETED,
            message="Run completed - Live-Agent tabs remain open for review",
            extra={"run_id": run_id},
        )


__all__ = ["TrainingLifecycleHandler"]
