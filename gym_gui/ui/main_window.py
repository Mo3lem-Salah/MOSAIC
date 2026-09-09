from __future__ import annotations

"""Main Qt window for the Gym GUI application."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

if TYPE_CHECKING:
    from gym_gui.core.adapters.base import AdapterStep
import socket

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets  # type: ignore[attr-defined]

try:
    from qtpy.QtGui import QAction
except ImportError:
    from qtpy.QtWidgets import QAction  # type: ignore[attr-defined]

from gym_gui.config import game_configs
from gym_gui.config.deployment import DAEMON_TARGET
from gym_gui.config.paths import VAR_TRAINER_DIR
from gym_gui.config.settings import Settings, get_settings
from gym_gui.constants import TRAINER_DEFAULTS, UI_DEFAULTS
from gym_gui.controllers.human_input import HumanInputController
from gym_gui.controllers.live_telemetry_controllers import LiveTelemetryController
from gym_gui.controllers.session import SessionController
from gym_gui.core.enums import ENVIRONMENT_FAMILY_BY_GAME, ControlMode, EnvironmentFamily, GameId
from gym_gui.core.factories.adapters import available_games
from gym_gui.core.ui.game_config.game_config_builder import GameConfigBuilder
from gym_gui.game_docs import get_game_info
from gym_gui.game_docs.mosaic_welcome import MOSAIC_WELCOME_HTML
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_OPERATOR_ENV_PREVIEW_ERROR,
    LOG_OPERATOR_ENV_PREVIEW_IMPORT_ERROR,
    LOG_OPERATOR_ENV_PREVIEW_STARTED,
    LOG_OPERATOR_ENV_PREVIEW_SUCCESS,
    LOG_UI_MAINWINDOW_ERROR,
    LOG_UI_MAINWINDOW_INFO,
    LOG_UI_MAINWINDOW_TRACE,
    LOG_UI_MAINWINDOW_WARNING,
    LOG_UI_WORKER_TABS_INFO,
)
from gym_gui.logging_config.logger import list_known_components
from gym_gui.services.actor import ActorService
from gym_gui.services.llm import LLM_CHAT_AVAILABLE
from gym_gui.services.operator import (
    MultiAgentStepState,
    MultiOperatorService,
    OperatorConfig,
    OperatorDescriptor,
)
from gym_gui.services.operator_launcher import OperatorLauncher
from gym_gui.services.service_locator import get_service_locator
from gym_gui.services.telemetry import TelemetryService
from gym_gui.services.trainer import (
    RunRegistry,
    TrainerClient,
    TrainerClientRunner,
    TrainingRunManager,
)
from gym_gui.services.trainer.streams import TelemetryAsyncHub
from gym_gui.ui.indicators.busy_indicator import modal_busy_indicator
from gym_gui.ui.logging_bridge import QtLogHandler
from gym_gui.ui.presenters.main_window_presenter import MainWindowPresenter, MainWindowView
from gym_gui.ui.presenters.workers import (
    get_worker_presenter_registry,
)
from gym_gui.ui.themes import DARK_THEME, LIGHT_THEME, apply_theme
from gym_gui.ui.widgets.control_panel import ControlPanelConfig, ControlPanelWidget
from gym_gui.ui.widgets.multi_agent_action_panel import (
    COLOR_PALETTE,
    DEFAULT_AGENT_COLOR_NAMES,
    MultiAgentActionPanel,
)
from gym_gui.ui.widgets.render_tabs import RenderTabs

if LLM_CHAT_AVAILABLE:
    from gym_gui.ui.widgets.chat_panel import ChatPanel
else:
    ChatPanel = None  # type: ignore[misc, assignment]
from gym_gui.constants.optional_deps import (
    OptionalDependencyError,
    get_godot_launcher,
    get_mjpc_launcher,
)
from gym_gui.services.trainer.signals import get_trainer_signals
from gym_gui.ui.forms import ensure_all_forms_registered, get_worker_form_factory
from gym_gui.ui.handlers import (
    AutoStepHandler,
    CheckersEnvLoader,
    CheckersHandler,
    ChessEnvLoader,
    ChessHandler,
    ConnectFourEnvLoader,
    ConnectFourHandler,
    FastLaneTabHandler,
    GameConfigHandler,
    GodotHandler,
    GoEnvLoader,
    GoHandler,
    HumanVsAgentHandler,
    JumanjiGridClickLoader,
    KeyboardBridgeHandler,
    LogHandler,
    MalmoEnvLoader,
    MPCHandler,
    # New composed handlers for extracted functionality
    MultiAgentGameHandler,
    OperatorLifecycleHandler,
    ParallelMultiAgentHandler,
    PettingzooHandler,
    PolicyEvaluationHandler,
    ScriptModeHandler,
    SmacCameraLoader,
    SudokuHandler,
    TicTacToeEnvLoader,
    TrainingFormHandler,
    TrainingLifecycleHandler,
    TrainingMonitorHandler,
    VizdoomEnvLoader,
)
from gym_gui.ui.handlers.env_previewers import (
    CrafterEnvPreview,
    EnvPreview,
    EnvPreviewError,
    EnvPreviewImportError,
    GymnasiumFallbackEnvPreview,
    IniMultigridEnvPreview,
    MeltingpotEnvPreview,
    MinigridEnvPreview,
    MinihackEnvPreview,
    MosaicMultigridEnvPreview,
    NLEEnvPreview,
    OvercookedEnvPreview,
    PettingzooEnvPreview,
    SocialjaxEnvPreview,
    TextworldEnvPreview,
)
from gym_gui.ui.panels.analytics_tabs import AnalyticsTabManager
from gym_gui.ui.widgets.settings import SettingsDialog

TRAINER_SUBMIT_DEADLINE_MULTIPLIER = 6


def _training_submit_deadline_seconds() -> float:
    """Return the gRPC deadline used for SubmitRun requests."""

    return TRAINER_DEFAULTS.client.deadline_s * TRAINER_SUBMIT_DEADLINE_MULTIPLIER


_LOGGER = logging.getLogger(__name__)
# Dedicated operator logger for PettingZoo/multi-agent game operations → operators.log
_OP_LOGGER = logging.getLogger("gym_gui.operators.main_window")


class MainWindow(QtWidgets.QMainWindow, LogConstantMixin):
    """Primary window that orchestrates the Gym session."""

    # Severity-level filters for log viewer
    LOG_SEVERITY_OPTIONS: Dict[str, str | None] = {
        "All": None,
        "DEBUG": "DEBUG",
        "INFO": "INFO",
        "WARNING": "WARNING",
        "ERROR": "ERROR",
    }

    CONTROL_MODE_LABELS: Dict[ControlMode, str] = {
        ControlMode.HUMAN_ONLY: "Human Only",
        ControlMode.AGENT_ONLY: "Agent Only",
        ControlMode.HYBRID_TURN_BASED: "Hybrid (Turn-Based)",
        ControlMode.HYBRID_HUMAN_AGENT: "Hybrid (Human + Agent)",
        ControlMode.MULTI_AGENT_COOP: "Multi-Agent (Cooperation)",
        ControlMode.MULTI_AGENT_COMPETITIVE: "Multi-Agent (Competition)",
    }

    _HUMAN_INPUT_MODES = {
        ControlMode.HUMAN_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
    }

    def __init__(self, settings: Settings, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        ensure_all_forms_registered()
        self._logger = _LOGGER
        self._settings = settings
        self._session = SessionController(settings, self)
        self._log_handler = QtLogHandler(parent=self)
        self._component_filter_options: List[str] = ["All", *list_known_components()]
        self._auto_running = False
        self._episode_finished = False  # Track episode termination state
        self._game_started = False
        self._game_paused = False
        self._awaiting_human = False
        self._latest_fps: float | None = None
        self._human_input = HumanInputController(self, self._session)
        self._session.set_input_controller(self._human_input)

        # Keyboard worker bridge: subprocess-per-agent for human play.
        from gym_gui.controllers.keyboard_worker_bridge import KeyboardWorkerBridge
        self._keyboard_worker_bridge = KeyboardWorkerBridge(parent=self)

        # Wire bridge's last_action to session so the idle tick can read
        # the latest held-key action at the env's native rate.
        self._session.set_keyboard_action_source(
            lambda: self._keyboard_worker_bridge.last_action
        )
        # Bridge signal connections moved to _connect_signals (delegated to
        # KeyboardBridgeHandler after the handler is instantiated in _init_handlers).

        locator = get_service_locator()
        telemetry_service = locator.resolve(TelemetryService)
        actor_service = locator.resolve(ActorService)
        trainer_client = locator.resolve(TrainerClient)
        telemetry_hub = locator.resolve(TelemetryAsyncHub)
        if telemetry_service is None or actor_service is None or trainer_client is None or telemetry_hub is None:
            raise RuntimeError("Required services are not registered in the locator")
        self._telemetry_service: TelemetryService = telemetry_service
        self._actor_service: ActorService = actor_service
        self._trainer_client: TrainerClient = trainer_client
        self._telemetry_hub: TelemetryAsyncHub = telemetry_hub

        # Create TrainingRunManager for the Management tab
        run_registry = locator.resolve(RunRegistry)
        client_runner = locator.resolve(TrainerClientRunner)
        if run_registry is not None:
            self._run_manager: TrainingRunManager | None = TrainingRunManager(
                registry=run_registry,
                client_runner=client_runner,
                telemetry_service=telemetry_service,
            )
        else:
            _LOGGER.warning("RunRegistry not available; Management tab will be disabled")
            self._run_manager = None

        # Multi-operator service for parallel operator execution
        self._multi_operator_service = MultiOperatorService()
        self._operator_launcher = OperatorLauncher()

        # Environment previewer registry. Dispatched by env_name in
        # _on_initialize_operator. See gym_gui/ui/handlers/env_previewers/base.py
        # for the EnvPreview protocol. Steps 1a and 1b of the refactor:
        # every env family is now behind a previewer. Unknown env_names use
        # the generic gymnasium fallback (also a previewer, always present).
        self._env_previewers: Dict[str, EnvPreview] = {
            "minigrid": MinigridEnvPreview(),
            "babyai": MinigridEnvPreview(),  # same handler as minigrid
            "crafter": CrafterEnvPreview(),
            "nle": NLEEnvPreview(),
            "minihack": MinihackEnvPreview(),
            "textworld": TextworldEnvPreview(),
            "pettingzoo": PettingzooEnvPreview(),
            "pettingzoo_classic": PettingzooEnvPreview(),  # same handler
            "mosaic_multigrid": MosaicMultigridEnvPreview(),
            "ini_multigrid": IniMultigridEnvPreview(),
            "meltingpot": MeltingpotEnvPreview(),
            "overcooked": OvercookedEnvPreview(),
            "socialjax": SocialjaxEnvPreview(),
        }
        self._generic_previewer: EnvPreview = GymnasiumFallbackEnvPreview()

        # Auto-Step manager: pre-caches episode frames, replays at user interval.
        # The manager itself stays on MainWindow because _handle_operator_response
        # (still in this file) reads is_active_for(op_id) and dispatches
        # on_step_collected / on_ready_received on it. AutoStepHandler (wrapping
        # the signal-facing methods) is instantiated in _init_handlers() where
        # _control_panel / _render_tabs / _status_bar are already available.
        from gym_gui.services.auto_step_manager import AutoStepManager
        self._auto_step_mgr = AutoStepManager(self)

        # Shared PettingZoo environment for multi-agent games (LLM vs LLM)
        # When env_name == "pettingzoo", the GUI owns ONE shared environment
        # and coordinates turn-based action selection from multiple workers
        # NOTE: PettingZoo shared env + player handles + mode flag moved to
        # PettingzooHandler in step 7. Cross-reads via `self._pettingzoo_handler.is_active()`
        # or direct handler methods.

        # NOTE: Parallel multi-agent state (env, mode, config, step_state, action_panel,
        # player_handles, linkgroup_handles, obs, episode_reward, step_index, episode_index,
        # multigrid_aec_mode) moved to ParallelMultiAgentHandler in step 8.
        # Cross-reads: `self._parallel_multiagent_handler.is_active()`, `.is_aec_mode()`,
        # `.shutdown()`, `.execute_parallel_multiagent_step()`, `.render_parallel_multiagent_frame()`,
        # or direct attr access via `._parallel_multiagent_env` etc.
        # NOTE: `_script_parallel_operators` moved to ScriptModeHandler in step 3
        # of the refactor. Cross-reads use `self._script_mode_handler.owns_operator(op_id)`.

        # Track dynamic agent tabs by (run_id, agent_id)
        self._agent_tab_index: set[tuple[str, str]] = set()
        self._selected_policy_path: Optional[Path] = None
        # NOTE: `_run_metadata` moved to TrainingLifecycleHandler in step 6.
        # Cross-reads use `self._training_lifecycle_handler._run_metadata`
        # (via direct dict reference passed to TrainingMonitorHandler at construction).
        # Note: FastLane tab tracking moved to FastLaneTabHandler
        # Note: Run watch/poll state moved to TrainingMonitorHandler

        # Settings dialog (created on demand, lazy initialization)
        self._settings_dialog: Optional[SettingsDialog] = None

        # Stockfish service (may be set by handlers, cleaned up on close)
        self._stockfish_service: Any = None

        # MuJoCo MPC launcher (optional)
        try:
            self._mjpc_launcher = get_mjpc_launcher()
        except OptionalDependencyError as e:
            _LOGGER.warning(f"MuJoCo MPC launcher not available: {e}")
            self._mjpc_launcher = None

        # Godot game engine launcher (optional)
        try:
            self._godot_launcher = get_godot_launcher()
        except OptionalDependencyError as e:
            _LOGGER.warning(f"Godot launcher not available: {e}")
            self._godot_launcher = None

        # Environment loaders (initialized in _init_handlers after UI components are created)
        self._chess_env_loader: ChessEnvLoader
        self._connect_four_env_loader: ConnectFourEnvLoader
        self._go_env_loader: GoEnvLoader
        self._tictactoe_env_loader: TicTacToeEnvLoader
        self._vizdoom_env_loader: VizdoomEnvLoader
        self._malmo_env_loader: MalmoEnvLoader
        self._jumanji_grid_loader: JumanjiGridClickLoader
        self._smac_camera_loader: SmacCameraLoader

        # Get live telemetry controller from service locator
        live_controller = locator.resolve(LiveTelemetryController)
        if live_controller is None:
            raise RuntimeError("LiveTelemetryController is not registered in the locator")
        self._live_controller: LiveTelemetryController = live_controller

        # Build control panel config
        available_modes = {}
        for game in available_games():
            available_modes[game] = SessionController.supported_control_modes(game)

        actor_descriptors = self._actor_service.describe_actors()
        default_operator_id = self._actor_service.get_active_actor_id()

        # Convert ActorDescriptor to OperatorDescriptor for the UI migration
        operator_descriptors = tuple(
            OperatorDescriptor(
                operator_id=ad.actor_id,
                display_name=ad.display_name,
                description=ad.description,
                category="default",
            )
            for ad in actor_descriptors
        )

        control_config = ControlPanelConfig(
            available_modes=available_modes,
            default_mode=settings.default_control_mode,
            frozen_lake_config=game_configs.FrozenLakeConfig(is_slippery=False),
            taxi_config=game_configs.TaxiConfig(is_raining=False, fickle_passenger=False),
            cliff_walking_config=game_configs.CliffWalkingConfig(is_slippery=False),
            lunar_lander_config=game_configs.LunarLanderConfig(),
            car_racing_config=game_configs.CarRacingConfig.from_env(),
            bipedal_walker_config=game_configs.BipedalWalkerConfig.from_env(),
            minigrid_empty_config=game_configs.DEFAULT_MINIGRID_EMPTY_5x5_CONFIG,
            minigrid_doorkey_5x5_config=game_configs.DEFAULT_MINIGRID_DOORKEY_5x5_CONFIG,
            minigrid_doorkey_6x6_config=game_configs.DEFAULT_MINIGRID_DOORKEY_6x6_CONFIG,
            minigrid_doorkey_8x8_config=game_configs.DEFAULT_MINIGRID_DOORKEY_8x8_CONFIG,
            minigrid_doorkey_16x16_config=game_configs.DEFAULT_MINIGRID_DOORKEY_16x16_CONFIG,
            minigrid_lavagap_config=game_configs.DEFAULT_MINIGRID_LAVAGAP_S7_CONFIG,
            minigrid_redbluedoors_6x6_config=game_configs.DEFAULT_MINIGRID_REDBLUE_DOORS_6x6_CONFIG,
            minigrid_redbluedoors_8x8_config=game_configs.DEFAULT_MINIGRID_REDBLUE_DOORS_8x8_CONFIG,
            default_seed=settings.default_seed,
            allow_seed_reuse=settings.allow_seed_reuse,
            operators=operator_descriptors,
            default_operator_id=default_operator_id,
        )

        self._control_panel = ControlPanelWidget(config=control_config, parent=self)
        if default_operator_id is not None:
            self._control_panel.set_active_operator(default_operator_id)

        # Create presenter to coordinate
        self._presenter = MainWindowPresenter(self._session, self._human_input, parent=self)

        status_bar = self.statusBar()
        if status_bar is None:
            status_bar = QtWidgets.QStatusBar(self)
            self.setStatusBar(status_bar)
        self._status_bar: QtWidgets.QStatusBar = status_bar

        self._configure_logging()
        self._build_ui()
        self._create_view_toolbar()
        self._init_handlers()
        self._connect_signals()
        self._populate_environments()
        self._status_bar.showMessage("Select an environment to begin")
        self._time_refresh_timer = QtCore.QTimer(self)
        self._time_refresh_timer.setInterval(1000)
        self._time_refresh_timer.timeout.connect(self._refresh_time_labels)
        self._time_refresh_timer.start()
        self._refresh_time_labels()
        self._session.set_slow_lane_enabled(not self._control_panel.fastlane_only_enabled())
        self._render_tabs.set_human_replay_enabled(not self._control_panel.fastlane_only_enabled())
        QtCore.QTimer.singleShot(0, self._render_tabs.refresh_replays)

        # Poll for new training runs and auto-subscribe (delegated to handler)
        self._run_poll_timer = QtCore.QTimer(self)
        self._run_poll_timer.setInterval(2000)  # Poll every 2 seconds
        self._run_poll_timer.timeout.connect(self._training_monitor_handler.poll_for_new_runs)
        self._run_poll_timer.start()

        self._training_monitor_handler.start_run_watch()

        # Bind presenter to view
        self._wire_presenter()

    def _wire_presenter(self) -> None:
        """Wire the MainWindowPresenter to coordinate SessionController signals."""
        view = MainWindowView(
            control_panel=self._control_panel,
            status_message_sink=lambda msg, timeout: self._status_bar.showMessage(msg, timeout or 0),
            awaiting_label_setter=lambda waiting: self._on_awaiting_human(waiting, ""),
            turn_label_setter=lambda turn: self._control_panel.set_turn(turn),
            render_adapter=lambda payload: self._render_tabs.display_payload(payload),
            time_refresher=self._refresh_time_labels,
            game_info_setter=self._set_game_info,
        )
        self._presenter.bind_view(view)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------
    def _configure_logging(self) -> None:
        root_logger = logging.getLogger()
        self._log_handler.setLevel(logging.NOTSET)
        # Qt handler emits raw log messages; timestamps and metadata are added when rendering.
        formatter = logging.Formatter("%(message)s")
        self._log_handler.setFormatter(formatter)
        root_logger.addHandler(self._log_handler)

    def _build_ui(self) -> None:
        _client = socket.gethostname()
        self.setWindowTitle(f"MOSAIC - Qt Shell  |  client: {_client}  ·  server: {DAEMON_TARGET}")
        self.resize(800, 600)

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, central)
        splitter.setChildrenCollapsible(True)
        layout.addWidget(splitter)

        layout_defaults = UI_DEFAULTS.layout

        # Use the ControlPanelWidget created in __init__ and wrap it in a scroll area
        self._control_panel_scroll = QtWidgets.QScrollArea(splitter)
        self._control_panel_scroll.setWidgetResizable(True)
        self._control_panel_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._control_panel_scroll.setWidget(self._control_panel)
        self._control_panel_scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        splitter.addWidget(self._control_panel_scroll)
        self._control_panel_scroll.setMinimumWidth(layout_defaults.control_panel_min_width)
        self._control_panel.setMinimumWidth(layout_defaults.control_panel_min_width)

        right_panel = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, central)
        right_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 1)
        right_panel.setChildrenCollapsible(True)
        right_panel.setMinimumWidth(layout_defaults.render_min_width)
        if layout_defaults.render_max_width:
            right_panel.setMaximumWidth(layout_defaults.render_max_width)

        self._render_group = QtWidgets.QGroupBox("Render View", right_panel)
        self._render_group.setMinimumWidth(layout_defaults.render_min_width)
        render_layout = QtWidgets.QVBoxLayout(self._render_group)
        self._render_tabs = RenderTabs(
            self._render_group,
            telemetry_service=self._telemetry_service,
            run_manager=self._run_manager,
        )
        # Note: Chess move signal connected in _connect_signals after handlers init
        self._analytics_tabs = AnalyticsTabManager(self._render_tabs, self)
        render_layout.addWidget(self._render_tabs)
        right_panel.addWidget(self._render_group)

        # Game information panel (right-most column)
        self._info_group = QtWidgets.QGroupBox("Game Info", self)
        info_layout = QtWidgets.QVBoxLayout(self._info_group)
        self._game_info = QtWidgets.QTextBrowser(self._info_group)
        self._game_info.setReadOnly(True)
        self._game_info.setOpenExternalLinks(True)
        # Set MOSAIC welcome message as default
        self._game_info.setHtml(MOSAIC_WELCOME_HTML)
        info_layout.addWidget(self._game_info, 1)

        # Runtime Log panel (far-right column)
        self._log_group = QtWidgets.QGroupBox("Runtime Log", self)
        log_layout = QtWidgets.QVBoxLayout(self._log_group)
        filter_row = QtWidgets.QHBoxLayout()

        # Component filter
        component_label = QtWidgets.QLabel("Component:")
        self._log_filter = QtWidgets.QComboBox()
        self._log_filter.addItems(self._component_filter_options)
        filter_row.addWidget(component_label)
        filter_row.addWidget(self._log_filter, 1)

        # Severity filter
        severity_label = QtWidgets.QLabel("Severity:")
        self._log_severity_filter = QtWidgets.QComboBox()
        self._log_severity_filter.addItems(self.LOG_SEVERITY_OPTIONS.keys())
        filter_row.addWidget(severity_label)
        filter_row.addWidget(self._log_severity_filter, 1)

        log_layout.addLayout(filter_row)

        self._log_console = QtWidgets.QPlainTextEdit()
        self._log_console.setReadOnly(True)
        self._log_console.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        log_layout.addWidget(self._log_console, 1)
        self._log_group.setMinimumWidth(layout_defaults.log_min_width)

        # Chat panel (MOSAIC Assistant) - optional, requires [chat] extra
        self._chat_group: QtWidgets.QGroupBox | None = None
        self._chat_panel = None
        if LLM_CHAT_AVAILABLE and ChatPanel is not None:
            self._chat_group = QtWidgets.QGroupBox("Chat", self)
            chat_layout = QtWidgets.QVBoxLayout(self._chat_group)
            chat_layout.setContentsMargins(0, 0, 0, 0)
            self._chat_panel = ChatPanel(parent=self._chat_group)
            chat_layout.addWidget(self._chat_panel)

        info_log_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, central)
        info_log_splitter.setChildrenCollapsible(True)
        self._info_group.setMinimumWidth(layout_defaults.info_min_width)
        info_log_splitter.addWidget(self._info_group)
        info_log_splitter.addWidget(self._log_group)
        if self._chat_group is not None:
            info_log_splitter.addWidget(self._chat_group)

        # Adjust stretch factors based on whether chat panel is present
        if self._chat_group is not None:
            info_log_splitter.setStretchFactor(0, 2)  # Game Info
            info_log_splitter.setStretchFactor(1, 1)  # Runtime Log
            info_log_splitter.setStretchFactor(2, 2)  # Chat
            # Check if chat should be collapsed by default
            chat_default_size = 0 if get_settings().chat_panel_collapsed else (layout_defaults.info_default_width // 2)
            info_log_splitter.setSizes(
                [
                    layout_defaults.info_default_width // 2,
                    layout_defaults.log_default_width,
                    chat_default_size,
                ]
            )
        else:
            info_log_splitter.setStretchFactor(0, 2)  # Game Info
            info_log_splitter.setStretchFactor(1, 1)  # Runtime Log
            info_log_splitter.setSizes(
                [
                    layout_defaults.info_default_width,
                    layout_defaults.log_default_width,
                ]
            )

        info_log_splitter.setMinimumWidth(layout_defaults.info_min_width + layout_defaults.log_min_width)
        info_log_splitter.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        splitter.addWidget(info_log_splitter)

        splitter.setSizes(
            [
                layout_defaults.control_panel_default_width,
                layout_defaults.render_default_width,
                layout_defaults.info_default_width + layout_defaults.log_default_width,
            ]
        )

        # Configure splitter stretch: control panel (left) small, render view large, info/log column medium
        splitter.setStretchFactor(0, 1)  # Control Panel
        splitter.setStretchFactor(1, 3)  # Render View
        splitter.setStretchFactor(2, 2)  # Game Info + Runtime Log

    def _create_view_toolbar(self) -> None:
        """Create toolbar with quick toggles for key panels."""
        self._view_toolbar = QtWidgets.QToolBar("View", self)
        self._view_toolbar.setObjectName("view-toolbar")
        self._view_toolbar.setMovable(False)
        self._view_toolbar.setFloatable(False)
        self._view_toolbar.setIconSize(QtCore.QSize(16, 16))
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, self._view_toolbar)

        self._view_actions: dict[str, QAction] = {}

        # Add Settings action as first item
        self._settings_action = QAction("Settings...", self)
        self._settings_action.triggered.connect(self._on_settings_clicked)
        self._view_toolbar.addAction(self._settings_action)
        self._view_toolbar.addSeparator()

        # Add panel view toggles
        self._add_view_toggle("Control Panel", self._control_panel_scroll)
        self._add_view_toggle("Render View", self._render_group)
        self._add_view_toggle("Game Info", self._info_group)
        self._add_view_toggle("Runtime Log", self._log_group)
        if self._chat_group is not None:
            self._add_view_toggle("Chat", self._chat_group)

        # Add spacer to push theme toggle to the right
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self._view_toolbar.addWidget(spacer)

        # Add theme toggle
        self._dark_mode = False
        self._theme_action = QAction("Dark Mode", self)
        self._theme_action.setCheckable(True)
        self._theme_action.setChecked(False)
        self._theme_action.toggled.connect(self._on_theme_toggled)
        self._view_toolbar.addAction(self._theme_action)

    def _add_view_toggle(self, label: str, widget: QtWidgets.QWidget) -> None:
        """Insert a checkable action to show/hide a widget."""
        action = QAction(label, self)
        action.setCheckable(True)
        action.setChecked(widget.isVisible())

        def handle_toggle(checked: bool, target: QtWidgets.QWidget = widget, name: str = label) -> None:
            target.setVisible(checked)
            self._status_bar.showMessage(f"{name} {'shown' if checked else 'hidden'}", 2000)

        action.toggled.connect(handle_toggle)
        self._view_toolbar.addAction(action)
        self._view_actions[label] = action

    def _on_theme_toggled(self, checked: bool) -> None:
        """Toggle between light and dark themes."""
        self._dark_mode = checked
        self._theme_action.setText("Light Mode" if checked else "Dark Mode")

        if checked:
            self._apply_dark_theme()
        else:
            self._apply_light_theme()

        self._status_bar.showMessage(
            f"{'Dark' if checked else 'Light'} theme applied", 2000
        )

    def _apply_dark_theme(self) -> None:
        """Apply dark theme from external QSS file."""
        apply_theme(DARK_THEME)

    def _apply_light_theme(self) -> None:
        """Apply light theme (reset to system default)."""
        apply_theme(LIGHT_THEME)

    def _init_handlers(self) -> None:
        """Initialize composed handlers for delegated functionality."""
        # Training lifecycle handler MUST be instantiated first: TrainingFormHandler,
        # TrainingMonitorHandler, and FastLaneTabHandler all inject its methods or
        # its `_run_metadata` dict as callbacks at their own construction time.
        self._training_lifecycle_handler = TrainingLifecycleHandler(
            parent=self,
            render_tabs=self._render_tabs,
            render_group=self._render_group,
            control_panel=self._control_panel,
            status_bar=self._status_bar,
            live_controller=self._live_controller,
            analytics_tabs=self._analytics_tabs,
            telemetry_hub=self._telemetry_hub,
            fastlane_tab_handler=None,  # type: ignore[arg-type]  # bound below after FastLaneTabHandler is created
        )

        # Game configuration handler
        self._game_config_handler = GameConfigHandler(
            control_panel=self._control_panel,
            session=self._session,
            status_bar=self._status_bar,
        )

        # MPC handler
        self._mpc_handler = MPCHandler(
            mjpc_launcher=self._mjpc_launcher,
            render_tabs=self._render_tabs,
            control_panel=self._control_panel,
            status_bar=self._status_bar,
        )

        # Godot handler
        self._godot_handler = GodotHandler(
            godot_launcher=self._godot_launcher,
            render_tabs=self._render_tabs,
            control_panel=self._control_panel,
            status_bar=self._status_bar,
        )

        # Log handler
        self._log_handler_composed = LogHandler(
            log_filter=self._log_filter,
            log_severity_filter=self._log_severity_filter,
            log_console=self._log_console,
            severity_options=self.LOG_SEVERITY_OPTIONS,
            initial_components=self._component_filter_options,
        )

        # Board game handlers for Human Control Mode
        # These handle moves from the BoardGameRendererStrategy in the Grid tab
        self._chess_handler = ChessHandler(
            session=self._session,
            render_tabs=self._render_tabs,
            status_bar=self._status_bar,
        )
        self._connect_four_handler = ConnectFourHandler(
            session=self._session,
            render_tabs=self._render_tabs,
            status_bar=self._status_bar,
        )
        self._go_handler = GoHandler(
            session=self._session,
            render_tabs=self._render_tabs,
            status_bar=self._status_bar,
        )
        self._sudoku_handler = SudokuHandler(
            session=self._session,
            render_tabs=self._render_tabs,
            status_bar=self._status_bar,
        )
        self._checkers_handler = CheckersHandler(
            session=self._session,
            render_tabs=self._render_tabs,
            status_bar=self._status_bar,
        )

        # Environment loaders (for Human vs Agent mode and environment-specific setup)
        self._chess_env_loader = ChessEnvLoader(
            render_tabs=self._render_tabs,
            control_panel=self._control_panel,
            status_bar=self._status_bar,
        )
        self._connect_four_env_loader = ConnectFourEnvLoader(
            render_tabs=self._render_tabs,
            control_panel=self._control_panel,
            status_bar=self._status_bar,
        )
        self._go_env_loader = GoEnvLoader(
            render_tabs=self._render_tabs,
            control_panel=self._control_panel,
            status_bar=self._status_bar,
        )
        self._tictactoe_env_loader = TicTacToeEnvLoader(
            render_tabs=self._render_tabs,
            control_panel=self._control_panel,
            status_bar=self._status_bar,
        )
        self._checkers_env_loader = CheckersEnvLoader(
            render_tabs=self._render_tabs,
            control_panel=self._control_panel,
            status_bar=self._status_bar,
        )
        self._vizdoom_env_loader = VizdoomEnvLoader(
            render_tabs=self._render_tabs,
        )
        self._malmo_env_loader = MalmoEnvLoader(
            render_tabs=self._render_tabs,
        )
        self._jumanji_grid_loader = JumanjiGridClickLoader(
            render_tabs=self._render_tabs,
        )
        self._smac_camera_loader = SmacCameraLoader(
            render_tabs=self._render_tabs,
        )

        # Multi-agent game routing handler
        self._multi_agent_game_handler = MultiAgentGameHandler(
            status_bar=self._status_bar,
            chess_loader=self._chess_env_loader,
            connect_four_loader=self._connect_four_env_loader,
            go_loader=self._go_env_loader,
            tictactoe_loader=self._tictactoe_env_loader,
            checkers_loader=self._checkers_env_loader,
            set_game_info=self._set_game_info,
            get_game_info=get_game_info,
            parent=self,
        )

        # Training form handler (Train/Policy/Resume dialogs)
        self._training_form_handler = TrainingFormHandler(
            parent=self,
            get_form_factory=get_worker_form_factory,
            get_current_game=self._control_panel.current_game,
            get_cleanrl_env_id=self._control_panel.cleanrl_environment_id,
            submit_config=self._training_lifecycle_handler.submit_training_config,
            build_policy_config=self._training_lifecycle_handler.build_policy_evaluation_config,
            log_callback=lambda message=None, extra=None, exc_info=None: self.log_constant(
                LOG_UI_MAINWINDOW_INFO, message=message, extra=extra, exc_info=exc_info
            ),
            status_callback=self._status_bar.showMessage,
        )

        # FastLane tab handler (must be initialized before PolicyEvaluationHandler)
        self._fastlane_tab_handler = FastLaneTabHandler(
            render_tabs=self._render_tabs,
            log_callback=lambda message=None, extra=None, exc_info=None: self.log_constant(
                LOG_UI_WORKER_TABS_INFO, message=message, extra=extra, exc_info=exc_info
            ),
        )
        # Late-bind fastlane_tab_handler on the training lifecycle handler
        # (created earlier so its methods can be injected into TrainingFormHandler,
        # but FastLaneTabHandler didn't exist yet at that point).
        self._training_lifecycle_handler._fastlane_tab_handler = self._fastlane_tab_handler

        # Policy evaluation handler
        self._policy_evaluation_handler = PolicyEvaluationHandler(
            parent=self,
            status_bar=self._status_bar,
            open_ray_fastlane_tabs=self._fastlane_tab_handler.open_ray_fastlane_tabs,
        )

        # Training monitor handler
        # Note: fastlane_callback delegates to FastLaneTabHandler with metadata resolution
        self._training_monitor_handler = TrainingMonitorHandler(
            parent=self,
            live_controller=self._live_controller,
            analytics_tabs=self._analytics_tabs,
            render_tabs=self._render_tabs,
            run_metadata=self._training_lifecycle_handler._run_metadata,
            trainer_dir=VAR_TRAINER_DIR,
            log_callback=lambda message=None, extra=None, exc_info=None: self.log_constant(
                LOG_UI_MAINWINDOW_INFO, message=message, extra=extra, exc_info=exc_info
            ),
            status_callback=self._status_bar.showMessage,
            title_callback=self._render_group.setTitle,
            fastlane_callback=lambda run_id, agent_id: self._fastlane_tab_handler.maybe_open_fastlane_tab(
                run_id, agent_id, self._training_lifecycle_handler.resolve_run_metadata(run_id, agent_id)
            ),
        )

        # Parallel multi-agent handler (owns shared parallel env, per-agent
        # handles, LinkGroup RL subprocesses, action panel, AEC vs Parallel
        # execution, and the GAR ghost-replacement machinery). Must be
        # created BEFORE OperatorLifecycleHandler / ScriptModeHandler /
        # KeyboardBridgeHandler because all three access its state and
        # methods via `self._parent._parallel_multiagent_handler.*`.
        self._parallel_multiagent_handler = ParallelMultiAgentHandler(
            parent=self,
            render_tabs=self._render_tabs,
            control_panel=self._control_panel,
            status_bar=self._status_bar,
            operator_launcher=self._operator_launcher,
            multi_operator_service=self._multi_operator_service,
        )

        # PettingZoo handler (owns shared env + player handles + turn-based
        # coordination for chess/go/connect_four/tictactoe). Must be created
        # BEFORE OperatorLifecycleHandler because OLH's `_on_reset_all_operators`
        # and `_on_step_all_operators` dispatch through `.is_active()` and
        # `.on_reset_pettingzoo_multiagent()` / `.on_step_pettingzoo_multiagent()`.
        self._pettingzoo_handler = PettingzooHandler(
            parent=self,
            render_tabs=self._render_tabs,
            control_panel=self._control_panel,
            status_bar=self._status_bar,
            operator_launcher=self._operator_launcher,
            multi_operator_service=self._multi_operator_service,
        )

        # Operator lifecycle handler (foundation: owns reset_all, step_all,
        # stop_all, poll_operator_responses, handle_operator_response). Must
        # be instantiated BEFORE AutoStepHandler and ScriptModeHandler because
        # they inject its methods as callbacks.
        self._operator_lifecycle_handler = OperatorLifecycleHandler(
            parent=self,
            render_tabs=self._render_tabs,
            control_panel=self._control_panel,
            status_bar=self._status_bar,
            operator_launcher=self._operator_launcher,
            multi_operator_service=self._multi_operator_service,
            auto_step_mgr=self._auto_step_mgr,
        )

        # Auto-Step handler (owns the signal-facing methods for Auto-Step
        # collection and replay). The AutoStepManager instance itself lives on
        # MainWindow so handle_operator_response can still read is_active_for.
        self._auto_step_handler = AutoStepHandler(
            parent=self,
            auto_step_mgr=self._auto_step_mgr,
            operator_launcher=self._operator_launcher,
            multi_operator_service=self._multi_operator_service,
            control_panel=self._control_panel,
            render_tabs=self._render_tabs,
            status_bar=self._status_bar,
            handle_operator_response=self._operator_lifecycle_handler.handle_operator_response,
        )

        # Script Mode handler (owns launch/reset/step/stop lifecycle for
        # scripted batch runs, separate from Manual Mode's multi_operator_service).
        # Parallel-mode reset/step still delegate back to MainWindow methods
        # until step 8 extracts ParallelMultiAgentHandler.
        self._script_mode_handler = ScriptModeHandler(
            parent=self,
            render_tabs=self._render_tabs,
            operator_launcher=self._operator_launcher,
            control_panel=self._control_panel,
            status_bar=self._status_bar,
            reset_parallel_multiagent=self._parallel_multiagent_handler.on_reset_parallel_multiagent,
            step_parallel_multiagent=self._parallel_multiagent_handler.on_step_parallel_multiagent,
            poll_operator_responses=self._operator_lifecycle_handler.poll_operator_responses,
        )

        # Keyboard bridge handler (owns auto-launch of subprocess keyboard
        # workers, per-USB-port device pairing, multi-cursor setup, and
        # routing of bridge signals to session / parallel-mode / raw-key handlers).
        self._keyboard_bridge_handler = KeyboardBridgeHandler(
            parent=self,
            session=self._session,
            control_panel=self._control_panel,
            status_bar=self._status_bar,
            keyboard_worker_bridge=self._keyboard_worker_bridge,
            human_input=self._human_input,
        )

    def _connect_signals(self) -> None:
        # Connect control panel signals to session controller
        self._control_panel.load_requested.connect(self._on_load_requested)
        self._control_panel.reset_requested.connect(self._on_reset_requested)
        # Training form signals (delegated to handler)
        self._control_panel.train_agent_requested.connect(
            self._training_form_handler.on_train_agent_requested
        )
        self._control_panel.trained_agent_requested.connect(
            self._training_form_handler.on_trained_agent_requested
        )
        self._control_panel.resume_training_requested.connect(
            self._training_form_handler.on_resume_training_requested
        )
        self._control_panel.custom_script_requested.connect(
            self._training_form_handler.on_custom_script_requested
        )
        self._control_panel.start_game_requested.connect(self._on_start_game)
        self._control_panel.pause_game_requested.connect(self._on_pause_game)
        self._control_panel.continue_game_requested.connect(self._on_continue_game)
        self._control_panel.terminate_game_requested.connect(self._on_terminate_game)
        self._control_panel.agent_step_requested.connect(self._session.perform_agent_step)
        self._control_panel._fastlane_only_checkbox.toggled.connect(self._on_fastlane_only_toggled)
        self._control_panel.game_changed.connect(self._on_game_changed)
        self._control_panel.control_mode_changed.connect(self._on_mode_changed)
        self._control_panel.operator_changed.connect(self._on_operator_changed)
        # Multi-operator signals - scientific execution for fair comparison
        self._control_panel.operators_changed.connect(self._on_operators_changed)
        self._control_panel.step_all_requested.connect(self._operator_lifecycle_handler.on_step_all_operators)
        self._control_panel.step_player_requested.connect(self._pettingzoo_handler.on_step_player)
        self._control_panel.reset_all_requested.connect(self._operator_lifecycle_handler.on_reset_all_operators)
        self._control_panel.stop_operators_requested.connect(self._operator_lifecycle_handler.on_stop_operators)
        self._control_panel.initialize_operator_requested.connect(self._on_initialize_operator)
        # NOTE: human_action_requested from OperatorsTab is NOT connected here.
        # Human actions come from OperatorRenderContainer via render_tabs.human_action_submitted (line 882)
        # to avoid duplicate signal connections that cause actions to be processed multiple times.

        # Auto-Step manager signals (delegated to AutoStepHandler; see
        # gym_gui/ui/handlers/features/auto_step_handler.py)
        self._control_panel.auto_step_requested.connect(self._auto_step_handler.on_auto_step_requested)
        self._control_panel.auto_step_stop_requested.connect(self._auto_step_handler.on_auto_step_stop)
        self._auto_step_mgr.reset_requested.connect(self._auto_step_handler.on_auto_step_reset_operator)
        self._auto_step_mgr.step_requested.connect(self._auto_step_handler.on_auto_step_step_operator)
        self._auto_step_mgr.frame_collected.connect(self._auto_step_handler.on_auto_step_frame_collected)
        self._auto_step_mgr.collection_done.connect(self._auto_step_handler.on_auto_step_collection_done)
        self._auto_step_mgr.display_frame.connect(self._auto_step_handler.on_auto_step_display_frame)

        # Script execution manager signals (separate from Manual Mode; delegated
        # to ScriptModeHandler; see handlers/features/script_mode_handler.py)
        script_mgr = self._control_panel.operators_tab.script_execution_manager
        script_mgr.launch_operator.connect(self._script_mode_handler.on_launch_operator)
        script_mgr.reset_operator.connect(self._script_mode_handler.on_reset_operator)
        script_mgr.step_operator.connect(self._script_mode_handler.on_step_operator)
        script_mgr.stop_operator.connect(self._script_mode_handler.on_stop_operator)

        # Game configuration handlers (delegated)
        self._control_panel.slippery_toggled.connect(self._game_config_handler.on_slippery_toggled)
        self._control_panel.frozen_v2_config_changed.connect(self._game_config_handler.on_frozen_v2_config_changed)
        self._control_panel.taxi_config_changed.connect(self._game_config_handler.on_taxi_config_changed)
        self._control_panel.cliff_config_changed.connect(self._game_config_handler.on_cliff_config_changed)
        self._control_panel.lunar_config_changed.connect(self._game_config_handler.on_lunar_config_changed)
        self._control_panel.car_config_changed.connect(self._game_config_handler.on_car_config_changed)
        self._control_panel.bipedal_config_changed.connect(self._game_config_handler.on_bipedal_config_changed)
        self._control_panel.vizdoom_config_changed.connect(self._game_config_handler.on_vizdoom_config_changed)

        # MPC handlers (delegated)
        self._control_panel.mpc_launch_requested.connect(self._mpc_handler.on_launch_requested)
        self._control_panel.mpc_stop_all_requested.connect(self._mpc_handler.on_stop_all_requested)

        # Godot handlers (delegated)
        self._control_panel.godot_launch_requested.connect(self._godot_handler.on_launch_requested)
        self._control_panel.godot_editor_requested.connect(self._godot_handler.on_editor_requested)
        self._control_panel.godot_stop_all_requested.connect(self._godot_handler.on_stop_all_requested)

        # Multi-Agent Mode handlers
        # Multi-agent game signals (delegated to handler)
        self._control_panel.multi_agent_load_requested.connect(
            self._multi_agent_game_handler.on_load_requested
        )
        self._control_panel.multi_agent_start_requested.connect(
            self._multi_agent_game_handler.on_start_requested
        )
        self._control_panel.multi_agent_reset_requested.connect(
            self._multi_agent_game_handler.on_reset_requested
        )
        self._control_panel.multi_agent_tab.ai_opponent_changed.connect(
            self._multi_agent_game_handler.on_ai_opponent_changed
        )
        # Policy evaluation (delegated to handler)
        self._control_panel.policy_evaluate_requested.connect(
            self._policy_evaluation_handler.handle_evaluate_request
        )

        # Board game handlers (Human Control Mode)
        # These signals come from BoardGameRendererStrategy in the Grid tab
        self._render_tabs.chess_move_made.connect(self._chess_handler.on_chess_move)
        self._render_tabs.connect_four_column_clicked.connect(self._connect_four_handler.on_column_clicked)
        self._render_tabs.go_intersection_clicked.connect(self._go_handler.on_intersection_clicked)
        self._render_tabs.go_pass_requested.connect(self._go_handler.on_pass_requested)
        # Sudoku handlers (Jumanji environment with mouse selection + keyboard digit entry)
        self._render_tabs.sudoku_cell_selected.connect(self._sudoku_handler.on_cell_selected)
        self._render_tabs.sudoku_digit_entered.connect(self._sudoku_handler.on_digit_entered)
        self._render_tabs.sudoku_cell_cleared.connect(self._sudoku_handler.on_cell_cleared)
        # Checkers handler (OpenSpiel via Shimmy - two-click selection)
        self._render_tabs.checkers_cell_clicked.connect(self._checkers_handler.on_checkers_cell_clicked)

        # Human operator interaction signals (from Multi-Operator view)
        self._render_tabs.human_action_submitted.connect(self._on_human_action_submitted)
        self._render_tabs.board_game_move_made.connect(self._on_human_board_game_move)

        # Keyboard bridge signals (delegated to KeyboardBridgeHandler; see
        # handlers/features/keyboard_bridge_handler.py)
        self._keyboard_worker_bridge.all_actions_ready.connect(
            self._keyboard_bridge_handler.on_keyboard_worker_actions_ready
        )
        self._keyboard_worker_bridge.mouse_delta_received.connect(
            self._keyboard_bridge_handler.on_keyboard_worker_mouse_delta
        )
        self._keyboard_worker_bridge.raw_key_received.connect(
            self._keyboard_bridge_handler.on_keyboard_worker_raw_key
        )

        # Keyboard assignment widget signals (multi-human gameplay)
        self._control_panel._keyboard_widget.assignment_changed.connect(
            self._keyboard_bridge_handler.on_keyboard_assignment_changed
        )
        self._control_panel._keyboard_widget.all_assignments_applied.connect(
            self._keyboard_bridge_handler.on_all_keyboard_assignments_applied
        )

        # Keyboard assignment from Operators tab (multi-human operators with evdev)
        self._control_panel.operators_tab.keyboard_assignment_changed.connect(
            self._keyboard_bridge_handler.on_operator_keyboard_assignment_changed
        )

        # Human Step in parallel multi-agent mode → trigger step cycle
        self._control_panel.operators_tab.human_step_parallel_requested.connect(
            self._parallel_multiagent_handler.on_step_parallel_multiagent
        )

        self._session.seed_applied.connect(self._on_seed_applied)

        # Log filters (delegated)
        self._log_filter.currentTextChanged.connect(self._log_handler_composed.on_filter_changed)
        self._log_severity_filter.currentTextChanged.connect(self._log_handler_composed.on_filter_changed)

        self._session.session_initialized.connect(self._on_session_initialized)
        self._session.step_processed.connect(self._on_step_processed)
        self._session.episode_finished.connect(self._on_episode_finished)
        self._session.status_message.connect(self._on_status_message)
        self._session.fps_updated.connect(self._on_fps_updated)
        # Note: awaiting_human is handled by MainWindowPresenter, not directly here
        self._session.turn_changed.connect(self._on_turn_changed)
        self._session.error_occurred.connect(self._on_error)
        self._session.auto_play_state_changed.connect(self._on_auto_play_state)

        self._log_handler.emitter.record_emitted.connect(self._log_handler_composed.append_log_record)

        # Connect live telemetry controller signals
        # The controller owns tab creation and routing; main window only handles cleanup
        self._live_controller.run_tab_requested.connect(self._training_lifecycle_handler.on_live_telemetry_tab_requested)
        self._live_controller.run_completed.connect(self._training_lifecycle_handler.on_run_completed)

        # Connect trainer lifecycle signals
        try:
            trainer_signals = get_trainer_signals()
            trainer_signals.training_finished.connect(self._training_lifecycle_handler.on_training_finished)
            self.log_constant(
                LOG_UI_MAINWINDOW_TRACE,
                message="Connected to trainer lifecycle signals",
            )
        except Exception as e:
            self.log_constant(
                LOG_UI_MAINWINDOW_WARNING,
                message=f"Failed to connect trainer signals: {e}",
                extra={"exception": type(e).__name__},
                exc_info=e,
            )

        # Ensure status reflects persisted mode on startup
        self._on_mode_changed(self._control_panel.current_mode())

    def _populate_environments(self) -> None:
        """Populate control panel with available games."""
        games = sorted(available_games(), key=lambda g: g.value)
        default = GameId(self._settings.gym_default_env) if games else None
        self._control_panel.populate_games(games, default=default)

    # ------------------------------------------------------------------
    # Slots - Control Panel Signal Handlers
    # ------------------------------------------------------------------
    def _on_game_changed(self, game_id: GameId) -> None:
        """Handle game selection from control panel."""
        self._status_bar.showMessage(f"Selected {game_id.value}. Load to begin.")
        # Update game info panel with a short description from centralized docs
        desc = get_game_info(game_id)
        if desc:
            self._set_game_info(desc)
        # HumanInputController will be configured when environment loads

    def _on_mode_changed(self, mode: ControlMode) -> None:
        """Handle control mode change from control panel."""
        label = self.CONTROL_MODE_LABELS.get(mode, mode.value)
        self._status_bar.showMessage(f"Mode set to {label}")
        self._human_input.update_for_mode(mode)

    def _on_fastlane_only_toggled(self, enabled: bool) -> None:
        self._session.set_slow_lane_enabled(not enabled)
        self._render_tabs.set_human_replay_enabled(not enabled)
        if enabled:
            self._status_bar.showMessage("Fast lane only: telemetry persistence disabled")
        else:
            self._status_bar.showMessage("Slow lane re-enabled")

    def _on_operator_changed(self, operator_id: str) -> None:
        """Handle active operator selection from the control panel."""
        try:
            self._actor_service.set_active_actor(operator_id)
        except KeyError:
            self.log_constant(
                LOG_UI_MAINWINDOW_ERROR,
                message=f"Attempted to activate unknown operator '{operator_id}'",
                extra={"operator_id": operator_id},
            )
            self._status_bar.showMessage(f"Unknown operator '{operator_id}'", 5000)
            return

        descriptor = self._actor_service.get_actor_descriptor(operator_id)
        label = descriptor.display_name if descriptor is not None else operator_id
        self._status_bar.showMessage(f"Active operator set to {label}", 4000)

    # ------------------------------------------------------------------
    # Multi-Operator Signal Handlers (Phase 6)
    # ------------------------------------------------------------------

    def _on_operators_changed(self, configs: list) -> None:
        """Handle operator configuration changes from the control panel.

        Syncs render view containers with operator configurations:
        - Removes containers for deleted operators
        - Adds containers for new operators
        - Updates existing operator configurations

        Args:
            configs: List of OperatorConfig instances representing current operator set.
        """
        current_ids = set(self._multi_operator_service.get_active_operators().keys())
        new_ids = {c.operator_id for c in configs}

        # Remove deleted operators
        for operator_id in current_ids - new_ids:
            self._render_tabs.remove_operator_view(operator_id)
            self._multi_operator_service.remove_operator(operator_id)
            self.log_constant(
                LOG_UI_MAINWINDOW_TRACE,
                message=f"Removed operator from multi-operator view: {operator_id}",
            )

        # Add or update operators
        for config in configs:
            if config.operator_id not in current_ids:
                # New operator
                self._render_tabs.add_operator_view(config)
                self._multi_operator_service.add_operator(config)
                self.log_constant(
                    LOG_UI_MAINWINDOW_TRACE,
                    message=f"Added operator to multi-operator view: {config.operator_id}",
                    extra={"operator_type": config.operator_type, "worker_id": config.worker_id},
                )
            else:
                # Existing operator - update config (including workers) in both service and view
                self._multi_operator_service.add_operator(config)  # Will update if exists
                self._render_tabs.update_operator_view(config)  # Update container's config

        count = len(configs)
        self._status_bar.showMessage(
            f"Multi-operator configuration updated: {count} operator{'s' if count != 1 else ''}",
            3000
        )

    # --- Human Operator Action Handlers ---

    def _on_human_action_submitted(self, operator_id: str, action: int) -> None:
        """Handle action submitted by a human operator via button click or keyboard.

        This sends a step command with the action to the human_worker subprocess,
        which owns the gymnasium environment. The response includes the updated
        render payload which is displayed in the render container.

        In parallel multi-agent mode (GUI owns env), this routes the action
        to the parallel step state instead of a subprocess.

        Args:
            operator_id: The human operator's ID.
            action: The action index selected by the human.
        """
        _OP_LOGGER.info(
            "Human action submitted: operator=%s, action=%d (type=%s)",
            operator_id, action, type(action).__name__,
        )

        # ── Parallel multi-agent mode: route to step state, not subprocess ──
        if self._parallel_multiagent_handler.is_active():
            self._parallel_multiagent_handler.on_human_action_parallel_multiagent(operator_id, action)
            return

        # Get the human operator's configuration
        config = self._multi_operator_service.get_operator(operator_id)
        if config is None:
            _OP_LOGGER.warning(f"Human action for unknown operator: {operator_id}")
            return

        if config.worker_id != "human_worker":
            _OP_LOGGER.warning(f"Action submitted for non-human operator: {operator_id}")
            return

        # Get the subprocess handle for this operator
        handle = self._operator_launcher.get_handle(operator_id)
        if handle is None or not handle.is_running:
            _OP_LOGGER.warning(f"No running subprocess for human operator: {operator_id}")
            self._status_bar.showMessage("Human operator not running", 3000)
            return

        # Send step command with action to the subprocess
        _OP_LOGGER.info(
            "Sending step command: operator=%s, action=%d",
            operator_id, action,
        )
        if not handle.send_step_with_action(action):
            _OP_LOGGER.error(f"Failed to send step command to human operator: {operator_id}")
            self._status_bar.showMessage("Failed to send action", 3000)
            return

        # Wait for step response (blocking with short timeout)
        response = handle.read_response(timeout=5.0)
        if response is None:
            _OP_LOGGER.warning(f"Timeout waiting for step response from human operator: {operator_id}")
            self._status_bar.showMessage("Timeout waiting for response", 3000)
            return

        response_type = response.get("type", "unknown")

        if response_type == "error":
            error_msg = response.get("message", "Unknown error")
            _OP_LOGGER.error(f"Error from human operator: {error_msg}")
            self._status_bar.showMessage(f"Error: {error_msg}", 5000)
            return

        if response_type == "step":
            # Extract step info from response
            step_index = response.get("step_index", 0)
            reward = response.get("reward", 0.0)
            total_reward = response.get("total_reward", 0.0)
            terminated = response.get("terminated", False)
            truncated = response.get("truncated", False)
            render_payload = response.get("render_payload")

            _OP_LOGGER.debug(
                "Human action step result: step=%d, reward=%.2f, terminated=%s",
                step_index, reward, terminated,
            )

            # Update the render container with new state
            wrapped_payload = {
                "render_payload": render_payload if render_payload else {},
                "episode_index": response.get("episode_index", 0),
                "step_index": step_index,
                "reward": reward,
                "episode_reward": total_reward,
                "terminated": terminated,
                "truncated": truncated,
            }
            self._render_tabs.display_operator_payload(operator_id, wrapped_payload)

            # Log if render_payload is empty
            if not render_payload:
                _OP_LOGGER.warning(
                    "Empty render_payload for human operator %s",
                    operator_id,
                )

            done_info = "DONE!" if terminated or truncated else ""
            self._status_bar.showMessage(
                f"Human action: {action} (step={step_index}, reward={reward:.2f}) {done_info}",
                2000
            )

            # If episode ended, log it
            if terminated or truncated:
                _OP_LOGGER.info(
                    f"Human operator {operator_id} episode ended: total_reward={total_reward:.2f}"
                )
        else:
            _OP_LOGGER.warning(f"Unexpected response type from human operator: {response_type}")

        # Notify the Operators tab that the human has completed their step
        # This enables the "Step All" button for AI operators
        self._control_panel.operators_tab.on_human_action_received(operator_id)

        # Hide the "Your Turn" indicator
        self._render_tabs.set_human_turn(operator_id, False)

    def _on_human_board_game_move(self, operator_id: str, from_sq: str, to_sq: str) -> None:
        """Handle board game move submitted by a human operator.

        This is called when a human clicks on a board game (Chess, Go, etc.)
        to make a move in the Multi-Operator view.

        For PettingZoo multi-agent mode:
        - Converts the UCI move (e.g., "e2e4") to action index
        - Submits the action to the shared PettingZoo environment
        - Updates the game state and renders the next frame

        Args:
            operator_id: The human operator's ID.
            from_sq: Source square (e.g., "e2" for chess).
            to_sq: Target square (e.g., "e4" for chess, or "e4q" with promotion piece).
        """
        uci_move = f"{from_sq}{to_sq}"

        # Handle pawn promotion: when a pawn reaches the back rank, must promote
        # Check if this is a chess pawn promotion move
        # Skip if promotion piece already specified (to_sq length > 2, e.g., "b1q")
        promotion_already_specified = len(to_sq) > 2 and to_sq[2] in "qrnb"
        if not promotion_already_specified and self._pettingzoo_handler._shared_pettingzoo_env is not None and hasattr(self._pettingzoo_handler._shared_pettingzoo_env, "board"):
            try:
                import chess
                board = self._pettingzoo_handler._shared_pettingzoo_env.board
                from_square = chess.parse_square(from_sq)
                piece = board.piece_at(from_square)

                # Check if piece is a pawn moving to back rank
                if piece is not None and piece.piece_type == chess.PAWN:
                    to_rank = to_sq[1]  # '1' through '8'
                    # White pawn to rank 8, or Black pawn to rank 1
                    if (piece.color == chess.WHITE and to_rank == '8') or \
                       (piece.color == chess.BLACK and to_rank == '1'):
                        # Auto-promote to Queen (most common choice)
                        uci_move = f"{from_sq}{to_sq}q"
                        _OP_LOGGER.info(f"Pawn promotion detected, using: {uci_move}")
            except Exception as e:
                _OP_LOGGER.debug(f"Promotion check failed: {e}")

        _OP_LOGGER.info(
            "Human board game move: operator=%s, %s -> %s (UCI: %s)",
            operator_id, from_sq, to_sq, uci_move,
        )

        # Check if we're in PettingZoo multi-agent mode
        if self._pettingzoo_handler._shared_pettingzoo_env is not None:
            self._pettingzoo_handler.submit_pettingzoo_human_move(operator_id, uci_move)
            return

        # Get the human operator's configuration (single-agent mode)
        config = self._multi_operator_service.get_operator(operator_id)
        if config is None:
            _OP_LOGGER.warning(f"Board game move for unknown operator: {operator_id}")
            return

        if config.worker_id != "human_worker":
            _OP_LOGGER.warning(f"Board game move for non-human operator: {operator_id}")
            return

        # Single-agent mode: Send move to human_worker subprocess
        self._status_bar.showMessage(f"Human move: {from_sq} -> {to_sq}", 2000)

        # Notify the Operators tab that the human has completed their step
        self._control_panel.operators_tab.on_human_action_received(operator_id)

        # Hide the "Your Turn" indicator
        self._render_tabs.set_human_turn(operator_id, False)

    # -------------------------------------------------------------------------
    # Parallel Multi-Agent Mode (MultiGrid, MeltingPot, Overcooked)
    # -------------------------------------------------------------------------


    def _on_initialize_operator(self, operator_id: str, config: OperatorConfig, seed: int | None) -> None:
        """Initialize environment for operator preview.

        Creates the environment and resets it. When seed is provided (shared
        seed mode), all operators get identical initial layouts for controlled
        scientific comparison. When seed is None, each operator gets a random
        layout.

        Args:
            operator_id: The operator's unique ID
            config: Operator configuration with env_name and task
            seed: Shared seed for reproducible initialization, or None for random
        """
        env_name = config.env_name
        task = config.task

        self._status_bar.showMessage(f"Initializing {task} with seed={seed}...", 2000)
        self.log_constant(
            LOG_OPERATOR_ENV_PREVIEW_STARTED,
            message=f"Loading environment preview for {task}",
            extra={"operator_id": operator_id, "env_name": env_name, "task": task, "seed": seed},
        )

        try:
            rgb_frame = None
            board_game_payload: Dict[str, Any] | None = None

            # Every env family routes through the previewer registry
            # (gym_gui/ui/handlers/env_previewers/). Unknown env_names use
            # the generic gymnasium fallback (also a previewer, always present).
            previewer = self._env_previewers.get(env_name, self._generic_previewer)
            try:
                rgb_frame, board_game_payload, _status_msg = previewer.preview(
                    config, seed, operator_id,
                )
            except EnvPreviewImportError as e:
                self.log_constant(
                    LOG_OPERATOR_ENV_PREVIEW_IMPORT_ERROR,
                    message=str(e),
                    extra={"operator_id": operator_id, "env_name": env_name, "task": task, "error": str(e)},
                )
                self._status_bar.showMessage(str(e), 5000)
                return
            except EnvPreviewError as e:
                self.log_constant(
                    LOG_OPERATOR_ENV_PREVIEW_ERROR,
                    message=str(e),
                    extra={"operator_id": operator_id, "env_name": env_name, "task": task, "error": str(e)},
                )
                self._status_bar.showMessage(str(e), 5000)
                return
            if _status_msg is not None:
                _text, _timeout = _status_msg
                self._status_bar.showMessage(_text, _timeout)

            if rgb_frame is not None:
                # Build payload for the render container
                # Extract dimensions safely for type checker
                if isinstance(rgb_frame, np.ndarray) and rgb_frame.ndim >= 2:
                    # Check if we need to scale image to target resolution
                    image_scale = config.settings.get("image_scale", 0)
                    if image_scale and image_scale > 0:
                        from PIL import Image
                        pil_image = Image.fromarray(rgb_frame)
                        pil_image = pil_image.resize(
                            (image_scale, image_scale),
                            Image.Resampling.LANCZOS
                        )
                        rgb_frame = np.array(pil_image)

                    shape = cast(tuple[int, ...], rgb_frame.shape)
                    frame_height, frame_width = shape[0], shape[1]
                    frame_data = rgb_frame.tolist()
                else:
                    frame_height, frame_width = 0, 0
                    frame_data = rgb_frame
                # Build payload - use board game payload for board games, RGB for others
                if board_game_payload is not None:
                    # Use structured board game payload for BoardGameRendererStrategy
                    payload = {
                        "render_payload": board_game_payload,
                        "episode_index": 0,
                        "step_index": 0,
                        "reward": 0.0,
                    }
                else:
                    # Use RGB payload for generic environments
                    payload = {
                        "render_payload": {
                            "mode": "rgb",
                            "rgb": frame_data,
                            "width": frame_width,
                            "height": frame_height,
                        },
                        "episode_index": 0,
                        "step_index": 0,
                        "reward": 0.0,
                    }

                # Update the operator's config (updates header: name, type badge, env/task)
                self._render_tabs.update_operator_view(config)

                # Set the container display size based on selected container size
                container_size = config.settings.get("container_size", 0)
                if container_size and container_size > 0:
                    self._render_tabs.set_operator_display_size(
                        operator_id, container_size, container_size
                    )

                # Display in the operator's container
                self._render_tabs.display_operator_payload(operator_id, payload)
                # Update status to "loaded" to indicate environment is ready
                self._render_tabs.set_operator_status(operator_id, "loaded")
                self._render_tabs.switch_to_multi_operator_tab()

                # For human operators (single-agent only): launch subprocess to own the environment
                # Multiagent human operators are handled via Reset -> _on_reset_pettingzoo_multiagent
                if config.worker_id == "human_worker" and config.operator_type != "multiagent":
                    # Stop any existing subprocess for this operator
                    self._operator_launcher.stop_operator(operator_id)

                    # Launch human_worker subprocess
                    handle = self._operator_launcher.launch_operator(
                        config,
                        interactive=True,
                    )

                    # Read the "init" message that the worker emits on startup
                    init_response = handle.read_response(timeout=5.0)
                    if init_response is None:
                        raise RuntimeError("Timeout waiting for human_worker init")

                    # Send reset command with seed and env configuration
                    handle.send_command({
                        "cmd": "reset",
                        "seed": seed,
                        "env_name": env_name,
                        "task": task,
                    })

                    # Wait for "ready" response with action labels and render
                    response = handle.read_response(timeout=10.0)
                    if response is None:
                        raise RuntimeError("Timeout waiting for human_worker ready response")

                    if response.get("type") == "error":
                        raise RuntimeError(response.get("message", "Unknown error"))

                    if response.get("type") != "ready":
                        raise RuntimeError(f"Unexpected response: {response.get('type')}")

                    # Extract action labels and render from response
                    action_labels = response.get("action_labels", [])
                    action_space_n = response.get("action_space", len(action_labels))
                    render_payload_human = response.get("render_payload")

                    # Display render from human_worker (overrides preview render)
                    if render_payload_human:
                        wrapped_payload = {
                            "render_payload": render_payload_human,
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
                        game_id = GameId(env_name)
                        self._render_tabs.set_game_id(operator_id, game_id)
                    except ValueError:
                        pass

                    # Set available actions from subprocess response
                    actions = list(range(action_space_n))
                    self._render_tabs.set_available_actions(operator_id, actions, action_labels)

                    # Show "Your Turn" indicator
                    self._render_tabs.set_human_turn(operator_id, True)

                    # Assign run_id and update operator state
                    self._multi_operator_service.assign_run_id(operator_id, handle.run_id)
                    self._multi_operator_service.set_operator_state(operator_id, "running")

                    # Update UI status badge to "running"
                    self._render_tabs.set_operator_status(operator_id, "running")

                    self.log_constant(
                        LOG_UI_MAINWINDOW_INFO,
                        message="Human operator subprocess launched",
                        extra={
                            "operator_id": operator_id,
                            "seed": seed,
                            "env_name": env_name,
                            "task": task,
                            "action_count": action_space_n,
                            "run_id": handle.run_id,
                            "pid": handle.pid,
                        },
                    )

                # Enable interactive mode for multiagent human operators
                # (single-agent human handled above in the subprocess block)
                if config.operator_type == "multiagent" and config.worker_id == "human_worker":
                    self._render_tabs.set_interactive(operator_id, True)
                    _OP_LOGGER.debug(f"Enabled interactive mode for multiagent human preview: {operator_id}")

                # Update the environment size in the operator config widget
                if frame_width > 0 and frame_height > 0:
                    self._control_panel.set_operator_environment_size(
                        operator_id, frame_width, frame_height,
                        container_size if container_size and container_size > 0 else None
                    )

                self._status_bar.showMessage(
                    f"Previewing {task} - ready to start",
                    3000
                )
                self.log_constant(
                    LOG_OPERATOR_ENV_PREVIEW_SUCCESS,
                    message=f"Environment preview loaded for {env_name}/{task}",
                    extra={
                        "operator_id": operator_id,
                        "env_name": env_name,
                        "task": task,
                        "width": frame_width,
                        "height": frame_height,
                    },
                )
            else:
                self._status_bar.showMessage(
                    f"No render available for {task}",
                    3000
                )

        except Exception as e:
            self.log_constant(
                LOG_OPERATOR_ENV_PREVIEW_ERROR,
                message=f"Failed to initialize operator {operator_id}: {e}",
                extra={"operator_id": operator_id, "env_name": env_name, "task": task, "error": str(e)},
            )
            self._status_bar.showMessage(
                f"Failed to initialize: {e}",
                5000
            )

    def _on_load_requested(self, game_id: GameId, mode: ControlMode, seed: int) -> None:
        """Handle load request from control panel."""
        self._episode_finished = False  # Reset episode state on new load
        overrides = self._control_panel.get_overrides(game_id)
        game_config = GameConfigBuilder.build_config(game_id, overrides)
        self._session.load_environment(
            game_id,
            mode,
            seed=seed,
            game_config=game_config,
        )

    def _on_reset_requested(self, seed: int) -> None:
        """Handle reset request from control panel."""
        self._episode_finished = False  # Reset episode state on reset

        self._session.reset_environment(seed=seed)

    # Multi-agent game methods delegated to MultiAgentGameHandler:
    # - on_load_requested
    # - on_start_requested
    # - on_reset_requested
    # - on_ai_opponent_changed

    # Policy evaluation methods delegated to PolicyEvaluationHandler:
    # - handle_evaluate_request
    # - _launch_evaluation (uses EvaluationWorker QThread)

    def _on_start_game(self) -> None:
        """Handle Start Game button."""
        status = "Game started"
        if self._episode_finished:
            seed = self._control_panel.current_seed()
            self._session.reset_environment(seed=seed)
            self._episode_finished = False
            status = f"Loaded new episode with seed {seed}. Game started"

        self._session.set_slow_lane_enabled(not self._control_panel.fastlane_only_enabled())

        self._session.start_game()
        self._game_started = True
        self._game_paused = False
        self._control_panel.set_game_started(True)
        self._control_panel.set_game_paused(False)
        self._update_input_state()
        self._status_bar.showMessage(status, 3000)

        # Auto-launch keyboard (+ mouse) worker subprocesses for human control.
        # start() is non-blocking: spawns processes, sends init commands,
        # and the 60Hz poll timer handles the rest.
        self._keyboard_bridge_handler.auto_launch_keyboard_workers()

    def _on_pause_game(self) -> None:
        """Handle Pause Game button."""
        self._session.pause_game()
        self._game_paused = True
        self._control_panel.set_game_paused(True)
        self._update_input_state()
        self._status_bar.showMessage("Game paused", 3000)

    def _on_continue_game(self) -> None:
        """Handle Continue Game button."""
        self._session.resume_game()
        self._game_paused = False
        self._control_panel.set_game_paused(False)
        self._update_input_state()
        self._status_bar.showMessage("Game continued", 3000)

    def _on_terminate_game(self) -> None:
        """Handle Terminate Game button."""
        with modal_busy_indicator(
            self,
            title="Terminating episode",
            message="Finalizing telemetry and stopping the environment…",
        ):
            self._session.terminate_game()
        self._game_started = False
        self._game_paused = False
        self._control_panel.set_game_started(False)
        self._control_panel.set_game_paused(False)
        self._update_input_state()
        self._episode_finished = True
        self._status_bar.showMessage("Game terminated", 3000)

    def _on_session_initialized(self, game_id: str, mode: str, _step: object) -> None:
        try:
            mode_label = self.CONTROL_MODE_LABELS[ControlMode(mode)]
        except Exception:
            mode_label = mode
        self._status_bar.showMessage(f"Loaded {game_id} in {mode_label} mode - Click 'Start Game' to begin")
        self.log_constant(
            LOG_UI_MAINWINDOW_INFO,
            message=f"Loaded {game_id} ({mode_label})",
            extra={"game_id": getattr(game_id, "value", str(game_id)), "mode": mode_label},
        )
        self._auto_running = False
        self._game_started = False
        self._game_paused = False
        self._awaiting_human = False
        self._latest_fps = None
        # Get overrides for game configuration (passed to human input controller)
        current_game = self._session.game_id
        overrides = None
        if current_game is not None:
            try:
                gid = GameId(current_game) if isinstance(current_game, str) else current_game
                overrides = self._control_panel.get_overrides(gid)
            except Exception:
                pass
        self._human_input.configure(
            self._session.game_id,
            self._session.action_space,
            overrides=overrides,
        )

        # Update keyboard widget for multi-agent environments
        # Cast to AdapterStep for type checking (signal passes object due to PyQt limitations)
        step = cast("AdapterStep[Any]", _step)
        step_info: Dict[str, Any] = dict(getattr(step, 'info', {}) or {})
        if step_info:
            # Debug: Log the raw step info for multi-agent detection
            _LOGGER.info(
                f"_on_session_initialized: step.info keys={list(step_info.keys())}, "
                f"num_agents={step_info.get('num_agents')}, "
                f"agents={step_info.get('agents')}"
            )
            num_agents = step_info.get('num_agents')
            if not num_agents or not isinstance(num_agents, int) or num_agents < 1:
                num_agents = 1

            # Build agent name list from environment or generate defaults
            agent_names = step_info.get('agents')
            if agent_names and isinstance(agent_names, (list, tuple)) and len(agent_names) == num_agents:
                agent_list = list(agent_names)
            else:
                agent_list = [f"agent_{i}" for i in range(num_agents)]

            # Always show the evdev keyboard widget (single and multi-agent).
            # Every human player goes through a worker subprocess.
            self._control_panel._keyboard_widget.set_available_agents(agent_list)
            self._control_panel._keyboard_widget.setVisible(True)

            self._human_input.set_num_agents(num_agents)
            self._human_input.set_agent_names(agent_list)

            self.log_constant(
                LOG_UI_MAINWINDOW_TRACE,
                message=f"Updated keyboard widget for {num_agents} agent(s)",
                extra={"num_agents": num_agents, "agents": agent_list},
            )
        self._configure_mouse_capture()  # Configure FPS-style mouse capture for ViZDoom
        self._control_panel.set_auto_running(False)
        self._control_panel.set_game_started(False)  # Reset game state on new load
        self._control_panel.set_game_paused(False)
        self._control_panel.set_fps(None)
        self._update_input_state()
        self._refresh_time_labels()

        # Notify render view of current game for asset selection
        if self._session.game_id is not None:
            self._render_tabs.set_current_game(self._session.game_id)
            # Update game info with a slightly more detailed description (include mode)
            try:
                gid = GameId(self._session.game_id)
            except Exception:
                gid = None
            if gid is not None:
                desc = get_game_info(gid)
                if desc:
                    self._set_game_info(desc + f"<p><b>Mode:</b> {mode}</p>")

    def _set_game_info(self, html: str) -> None:
        """Set the HTML content of the Game Info panel."""
        if not hasattr(self, "_game_info"):
            return
        if not html:
            html = MOSAIC_WELCOME_HTML
        self._game_info.setHtml(html)

    def _configure_mouse_capture(self) -> None:
        """Configure mouse input for the loaded game.

        Delegates to environment-specific loaders:
        - VizdoomEnvLoader: FPS-style mouse capture for ViZDoom games
        - MalmoEnvLoader: FPS-style mouse capture for MalmoEnv (Minecraft) games
        - JumanjiGridClickLoader: Grid-click for Tetris, Minesweeper, etc.
        - SmacCameraLoader: 3D camera panning for SMAC/SMACv2 environments
        """
        self._vizdoom_env_loader.configure_mouse_capture(self._session)
        self._malmo_env_loader.configure_mouse_capture(self._session)
        self._jumanji_grid_loader.configure_grid_click(self._session)
        self._smac_camera_loader.configure_mouse_capture(self._session)

    def _on_step_processed(self, step: object, index: int) -> None:
        """Handle step processed from session controller."""
        if not hasattr(step, "reward"):
            return
        reward = getattr(step, "reward", 0.0)
        terminated = getattr(step, "terminated", False)
        truncated = getattr(step, "truncated", False)
        render_payload = getattr(step, "render_payload", None)

        # Extract per-team rewards from step info (MOSAIC MultiGrid competitive envs)
        step_info = getattr(step, "info", {}) or {}
        team_rewards = step_info.get("team_episode_rewards")

        # Update control panel status (awaiting_human and turn updated via separate signals)
        self._control_panel.set_status(
            step=index,
            reward=reward,
            total_reward=self._session.current_episode_reward,
            terminated=terminated,
            truncated=truncated,
            turn=self._session._turn,
            awaiting_human=False,  # Will be updated via awaiting_human signal
            session_time=self._session._timers.launch_elapsed_formatted(),
            active_time=self._session._timers.first_move_elapsed_formatted(),
            episode_duration=self._session._timers.episode_duration_formatted(),
            outcome_time=self._session._timers.outcome_elapsed_formatted(),
            outcome_wall_clock=self._session._timers.outcome_wall_clock_formatted(),
            team_rewards=team_rewards,
        )

        self._render_tabs.display_payload(render_payload)

    def _on_fps_updated(self, fps: float) -> None:
        self._latest_fps = fps if fps > 0 else None
        self._control_panel.set_fps(self._latest_fps)

    def _on_episode_finished(self, finished: bool) -> None:
        self._episode_finished = finished
        if finished:
            # Stop keyboard worker bridge so workers stop reading keys
            if self._keyboard_worker_bridge.is_active:
                self._keyboard_worker_bridge.stop()
            # Disable shortcuts when episode terminates
            self._game_started = False
            self._game_paused = False
            self._control_panel.set_game_started(False)  # Reset game state
            self._control_panel.set_game_paused(False)
            self._update_input_state()
            self._render_tabs.on_episode_finished()
            next_seed = self._session.next_seed
            self._control_panel.set_seed_value(next_seed)
            self._status_bar.showMessage(
                f"Episode finished. Next seed prepared: {next_seed}", 4000
            )

    def _on_seed_applied(self, seed: int) -> None:
        self._control_panel.set_seed_value(seed)
        if self._settings.allow_seed_reuse:
            message = (
                f"Seed {seed} applied. Override by adjusting the seed before the next run."
            )
        else:
            message = "Seed applied. Episode will reuse this value until it finishes."
        self._status_bar.showMessage(message, 4000)

    def _on_status_message(self, message: str) -> None:
        self._status_bar.showMessage(message, 5000)

    def _on_awaiting_human(self, waiting: bool, message: str) -> None:
        """
        Handle awaiting_human signal to update UI and keyboard shortcuts.

        Shortcuts are disabled if the episode has finished OR if the game hasn't been started.
        In HUMAN_ONLY mode, shortcuts stay enabled during active started episodes.
        In hybrid modes, shortcuts are only enabled when waiting for human input.
        """
        self._awaiting_human = waiting
        self._control_panel.set_awaiting_human(waiting)
        if message:
            self._status_bar.showMessage(message, 5000)
        self._update_input_state()

    def _on_turn_changed(self, turn: str) -> None:
        self._control_panel.set_turn(turn)

    def _on_error(self, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Session Error", message)
        self._status_bar.showMessage(message, 5000)

    def _on_auto_play_state(self, running: bool) -> None:
        self._auto_running = running
        self._control_panel.set_auto_running(running)
        self._update_input_state()

    # Training form methods delegated to TrainingFormHandler:
    # - on_trained_agent_requested
    # - on_train_agent_requested
    # - on_resume_training_requested

    # NOTE: Removed _on_live_step_received and _on_live_episode_received
    # The LiveTelemetryController now owns all routing (tab creation and step/episode delivery).
    # Main window only handles tab creation via run_tab_requested signal.

    # FastLane tab methods delegated to FastLaneTabHandler:
    # - maybe_open_fastlane_tab
    # - open_ray_fastlane_tabs
    # - open_single_fastlane_tab
    # - get_num_workers, get_canonical_agent_id, get_worker_id, get_env_id
    # - get_run_mode, metadata_supports_fastlane, clear_tabs_for_run

    # Training monitor methods delegated to TrainingMonitorHandler:
    # - poll_for_new_runs, start_run_watch, shutdown_run_watch
    # - backfill_run_metadata_from_disk, auto_subscribe_run

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _update_input_state(self) -> None:
        """Synchronize keyboard input enablement with session state."""
        mode = self._control_panel.current_mode()

        enable_input = False
        if self._episode_finished:
            enable_input = False
        elif not self._game_started:
            enable_input = False
        elif self._game_paused:
            enable_input = False
        elif mode == ControlMode.HUMAN_ONLY:
            enable_input = True
        elif mode in (ControlMode.MULTI_AGENT_COOP, ControlMode.MULTI_AGENT_COMPETITIVE):
            # Simultaneous multi-agent: always enabled when game is running
            enable_input = True
        elif mode in self._HUMAN_INPUT_MODES:
            # Turn-based: only when awaiting human input
            enable_input = self._awaiting_human

        # When bridge workers are active, disable old Qt shortcuts to prevent
        # double-stepping (bridge handles all keyboard input via subprocess).
        if self._keyboard_worker_bridge.is_active:
            enable_input = False

        self._human_input.set_enabled(enable_input)

    @staticmethod
    def _format_bool(value: bool) -> str:
        return "Yes" if value else "No"

    def _refresh_time_labels(self) -> None:
        """Update time labels in control panel."""
        timers = self._session._timers
        self._control_panel.set_time_labels(
            session_time=timers.launch_elapsed_formatted(),
            active_time=timers.first_move_elapsed_formatted(),
            outcome_time=timers.outcome_elapsed_formatted(),
            outcome_timestamp=timers.outcome_wall_clock_formatted(),
        )

    def _on_settings_clicked(self) -> None:
        """Handle Settings toolbar action click.

        Creates dialog on first use (lazy initialization).
        Shows non-modal dialog allowing interaction with main window.
        """
        if self._settings_dialog is None:
            self._settings_dialog = SettingsDialog(parent=self)
            self._settings_dialog.setting_changed.connect(self._on_setting_changed)
            self._settings_dialog.settings_reset.connect(self._on_settings_reset)

        # Show non-modal dialog (user can interact with main window)
        self._settings_dialog.show()
        self._settings_dialog.raise_()
        self._settings_dialog.activateWindow()

    def _on_setting_changed(self, key: str, value: str) -> None:
        """Handle individual setting change from Settings dialog.

        Logs the change but does not reload settings (requires restart).

        Args:
            key: Setting key
            value: New value
        """
        self.log_constant(
            LOG_UI_MAINWINDOW_INFO,
            message=f"Setting changed: {key}",
            extra={"setting_key": key, "setting_value_length": len(value)},
        )
        self._status_bar.showMessage(f"Setting saved: {key}", 3000)

    def _on_settings_reset(self) -> None:
        """Handle reset of all settings to defaults."""
        self.log_constant(
            LOG_UI_MAINWINDOW_INFO,
            message="All settings reset to defaults",
        )
        self._status_bar.showMessage("All settings reset to defaults (restart required)", 5000)

    def closeEvent(self, a0: QtGui.QCloseEvent | None) -> None:
        # Stop keyboard worker subprocesses AND teardown multi-cursor.
        # Both delegated to the handler which owns _multi_cursor_state
        # (see handlers/features/keyboard_bridge_handler.py).
        if hasattr(self, "_keyboard_bridge_handler"):
            self._keyboard_bridge_handler.shutdown()

        logging.getLogger().removeHandler(self._log_handler)

        # Shutdown live telemetry controller
        if hasattr(self, "_live_controller"):
            self._live_controller.shutdown()

        self._training_monitor_handler.shutdown_run_watch()

        # Clean up board games (via env loaders)
        if hasattr(self, "_chess_env_loader"):
            self._chess_env_loader.cleanup()
        if hasattr(self, "_connect_four_env_loader"):
            self._connect_four_env_loader.cleanup()
        if hasattr(self, "_go_env_loader"):
            self._go_env_loader.cleanup()
        if hasattr(self, "_tictactoe_env_loader"):
            self._tictactoe_env_loader.cleanup()

        # Clean up Stockfish service
        if hasattr(self, "_stockfish_service") and self._stockfish_service is not None:
            self._stockfish_service.stop()
            self._stockfish_service = None

        # Clean up chat panel (optional - requires [chat] extra)
        if hasattr(self, "_chat_panel") and self._chat_panel is not None:
            self._chat_panel.cleanup()

        # Close settings dialog if open
        if hasattr(self, "_settings_dialog") and self._settings_dialog is not None:
            self._settings_dialog.close()
            self._settings_dialog = None

        # Stop all operator subprocesses
        if hasattr(self, "_operator_launcher"):
            stopped = self._operator_launcher.stop_all()
            if stopped:
                self.log_constant(
                    LOG_UI_MAINWINDOW_INFO,
                    message=f"Stopped {len(stopped)} operator(s) on shutdown",
                    extra={"operator_ids": stopped},
                )

        # Shutdown session
        self._session.shutdown()

        if hasattr(self, "_time_refresh_timer") and self._time_refresh_timer.isActive():
            self._time_refresh_timer.stop()

        super().closeEvent(a0)


__all__ = ["MainWindow"]
