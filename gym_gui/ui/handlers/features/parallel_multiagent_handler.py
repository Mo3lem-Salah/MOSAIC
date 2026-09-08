"""Parallel multi-agent handler: shared env + simultaneous stepping + GAR.

Extracted from `MainWindow` (17 methods, backup range roughly lines 1132-4152)
as part of refactor plan step 8. This is the largest and most complex
extraction: it owns the shared parallel-multi-agent environment
(mosaic_multigrid, meltingpot, overcooked, ini_multigrid, socialjax, smac),
the per-agent worker handles, the LinkGroup shared RL subprocesses, the
Multi-Agent action panel, the AEC (per-agent physics) vs Parallel
(simultaneous) execution modes, and the GAR (Ghost Agent Replacement)
two-handle logic that submits parallel `send_select_action` calls to both
the shared RL handle (for identity-vector consistency in parameter-sharing
MAPPO) and the replacement handle (whose action is the one submitted to
`env.step()`).

Owned state (per plan section 8):
    _parallel_multiagent_env, _parallel_multiagent_mode,
    _parallel_multiagent_config, _parallel_multiagent_step_state,
    _parallel_action_panel, _parallel_player_handles, _linkgroup_handles,
    _parallel_multiagent_obs, _parallel_episode_reward,
    _parallel_step_index, _parallel_episode_index, _multigrid_aec_mode

Cross-handler public API:
    is_active() -> bool: True if a shared parallel env is running
    is_aec_mode() -> bool: True when using AEC wrapper (per-agent physics)
    shutdown(): close env, stop handles, reset state (idempotent)
    execute_parallel_multiagent_step(actions): called by KeyboardBridgeHandler
    render_parallel_multiagent_frame(): called by KeyboardBridgeHandler
    clear_parallel_action_panel(): called by OperatorLifecycleHandler on stop

Uses `LogConstantMixin` because it emits many LOG constants including
`LOG_OPERATOR_PARALLEL_*`, `LOG_GAR_GHOST_ACTION_DISCARDED`,
`LOG_GAR_REPLACEMENT_LAUNCHED`, `LOG_OPERATOR_ENV_PREVIEW_ERROR`,
`LOG_UI_MAINWINDOW_*`.

Preserves the GAR two-handle timing invariant verbatim
(parallel-send-then-parallel-read pattern; do not reorder or coalesce).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
from qtpy import QtCore, QtWidgets

from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_GAR_GHOST_ACTION_DISCARDED,
    LOG_GAR_REPLACEMENT_LAUNCHED,
    LOG_OPERATOR_ENV_PREVIEW_ERROR,
    LOG_OPERATOR_PARALLEL_RESET_STARTED,
    LOG_OPERATOR_PARALLEL_STEP_COMPLETED,
    LOG_OPERATOR_PARALLEL_STEP_STARTED,
    LOG_UI_MAINWINDOW_ERROR,
    LOG_UI_MAINWINDOW_INFO,
    LOG_UI_MAINWINDOW_TRACE,
    LOG_UI_MAINWINDOW_WARNING,
)
from gym_gui.services.operator import MultiAgentStepState, OperatorConfig
from gym_gui.services.operator_launcher import OperatorLaunchError
from gym_gui.ui.widgets.multi_agent_action_panel import (
    COLOR_PALETTE,
    DEFAULT_AGENT_COLOR_NAMES,
    MultiAgentActionPanel,
)

if TYPE_CHECKING:
    from gym_gui.services.operator import MultiOperatorService
    from gym_gui.services.operator_launcher import OperatorLauncher
    from gym_gui.ui.widgets.control_panel import ControlPanelWidget
    from gym_gui.ui.widgets.render_tabs import RenderTabs


_OP_LOGGER = logging.getLogger("gym_gui.operators.main_window")


class ParallelMultiAgentHandler(LogConstantMixin):
    """Parallel multi-agent env + GAR ghost-replacement machinery."""

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
        # Owned state
        self._parallel_multiagent_env: Any = None
        self._parallel_multiagent_mode: bool = False
        self._parallel_multiagent_config: Optional[OperatorConfig] = None
        self._parallel_multiagent_step_state: Optional[MultiAgentStepState] = None
        self._parallel_action_panel: Optional[MultiAgentActionPanel] = None
        self._parallel_player_handles: Dict[str, Any] = {}
        self._linkgroup_handles: Dict[str, Any] = {}
        self._parallel_multiagent_obs: Dict[Any, Any] = {}
        self._parallel_episode_reward: float = 0.0
        self._parallel_step_index: int = 0
        self._parallel_episode_index: int = 0
        self._multigrid_aec_mode: bool = False

    def is_active(self) -> bool:
        """True if a shared parallel env is running."""
        return self._parallel_multiagent_mode and self._parallel_multiagent_env is not None

    def is_aec_mode(self) -> bool:
        """True when the env is wrapped in the AEC (per-agent physics) wrapper."""
        return self._multigrid_aec_mode

    def shutdown(self) -> None:
        """Close env, stop handles, reset state. Idempotent.

        Called by OperatorLifecycleHandler._on_stop_operators.
        """
        self._control_panel.operators_tab.set_parallel_mode(False)
        self.clear_parallel_action_panel()
        self._parallel_multiagent_mode = False
        self._parallel_multiagent_step_state = None
        if self._parallel_multiagent_env is not None:
            try:
                self._parallel_multiagent_env.close()
            except Exception:
                pass
            self._parallel_multiagent_env = None
        self._parallel_player_handles.clear()
        for _gid, _handle in self._linkgroup_handles.items():
            try:
                _handle.stop()
            except Exception:
                pass
        self._linkgroup_handles.clear()

    def is_parallel_multiagent(self) -> tuple[bool, Optional["OperatorConfig"]]:
        """Check if we're in parallel multi-agent mode (MultiGrid, MeltingPot, Overcooked).

        Parallel multi-agent environments use the Parallel API where all agents
        act simultaneously in each step.

        Returns:
            Tuple of (is_parallel_multiagent, first_config) where first_config
            contains the environment and worker configuration.
        """
        active_operators = self._multi_operator_service.get_active_operators()
        if not active_operators:
            return False, None

        # Get first operator config
        first_id = next(iter(active_operators.keys()))
        first_config = self._multi_operator_service.get_operator(first_id)
        if first_config is None:
            return False, None

        # Check if this is a parallel multi-agent environment
        if first_config.env_name in ("mosaic_multigrid", "ini_multigrid", "socialjax", "meltingpot", "overcooked"):
            # jaxmarl_worker owns its own JAX env + policy internally -- it is NOT
            # an action selector. The GUI must NOT create the env; instead it launches
            # one interactive subprocess and receives render frames via stdin/stdout.
            if any(w.worker_id == "jaxmarl_worker" for w in first_config.workers.values()):
                return False, None
            # Must have multiple workers (agents)
            if len(first_config.workers) > 1:
                return True, first_config

        return False, None

    def on_human_action_parallel_multiagent(
        self, operator_id: str, action: int
    ) -> None:
        """Route a human action in parallel multi-agent mode.

        Handles two sub-modes:
        - **AEC**: ``env.agent_selection`` determines the current agent.
          The action is applied directly and the frame is re-rendered.
        - **Parallel**: Actions are accumulated in ``MultiAgentStepState``.
          Arrow key / button presses are assigned round-robin to pending
          human agents. When all humans have acted the environment steps.

        Args:
            operator_id: The human operator's ID (from button / keyboard).
            action: The action index selected by the human.
        """
        env = self._parallel_multiagent_env
        config = self._parallel_multiagent_config
        if env is None or config is None:
            _OP_LOGGER.warning(
                "_on_human_action_parallel_multiagent: no env/config"
            )
            return

        # ── AEC sub-mode ──────────────────────────────────────────────
        if self._multigrid_aec_mode:
            current_agent = getattr(env, "agent_selection", None)
            if current_agent is None or not env.agents:
                _OP_LOGGER.info("AEC episode done — ignoring human action")
                return

            human_agents = config.get_human_agents()
            if current_agent not in human_agents:
                _OP_LOGGER.debug(
                    "AEC: current agent %s is not human, ignoring action",
                    current_agent,
                )
                return

            _OP_LOGGER.info(
                "AEC human action: %s → action %d", current_agent, action
            )
            env.step(int(action))
            self.render_parallel_multiagent_frame()

            # Clean up panel
            self.clear_parallel_action_panel()

            if not env.agents:
                total_reward = sum(env.rewards.values())
                self._status_bar.showMessage(
                    f"Episode done! Total reward: {total_reward:.2f}", 5000
                )
            else:
                self._status_bar.showMessage(
                    f"AEC: {current_agent} acted → now {env.agent_selection}'s turn",
                    2000,
                )
            return

        # ── Parallel sub-mode ─────────────────────────────────────────
        step_state = self._parallel_multiagent_step_state
        if step_state is None:
            # No step cycle active yet — trigger one (collects AI actions,
            # creates the panel, etc.).
            self.on_step_parallel_multiagent()
            step_state = self._parallel_multiagent_step_state
            if step_state is None:
                _OP_LOGGER.error(
                    "Failed to initialise step state for parallel mode"
                )
                return

        # Assign action to the first pending human agent (round-robin)
        pending = step_state.pending_human_agents()
        if not pending:
            _OP_LOGGER.debug(
                "All human agents already acted — ignoring extra action"
            )
            return

        target_agent = pending[0]
        step_state.add_action(target_agent, action)
        _OP_LOGGER.info(
            "Parallel human action: %s → action %d (%d/%d humans done)",
            target_agent,
            action,
            len(step_state.human_agents) - len(step_state.pending_human_agents()),
            len(step_state.human_agents),
        )

        # Programmatically update the MultiAgentActionPanel row to stay in sync
        if self._parallel_action_panel is not None:
            row = self._parallel_action_panel._agent_rows.get(target_agent)
            if row is not None:
                row.set_action(action)

        # All humans done? Execute the step.
        if step_state.is_complete():
            _OP_LOGGER.info("All actions collected — executing parallel step")
            self.execute_parallel_multiagent_step(step_state.get_all_actions())
        else:
            still_pending = step_state.pending_human_agents()
            self._status_bar.showMessage(
                f"Waiting for: {', '.join(still_pending)}", 10000
            )

    def on_reset_parallel_multiagent(self, seed: int, config: "OperatorConfig") -> None:
        """Reset for parallel multi-agent mode (MultiGrid, MeltingPot, Overcooked).

        In this mode:
        1. GUI creates ONE shared environment (Parallel API)
        2. AI workers are launched in action_selector mode
        3. Human agents are controlled via action panel
        4. All agents step simultaneously

        Args:
            seed: Random seed for environment.
            config: The operator config with task and worker assignments.
        """
        task = config.task
        env_name = config.env_name

        # --- Pre-launch validation (prevents GUI freeze and silent failures) ---

        # V1: AEC mode on simultaneous envs causes GUI freeze. The AEC wrapper
        # blocks on per-agent stepping while the GUI thread waits for responses,
        # creating a deadlock. MultiGrid/MeltingPot/Overcooked are simultaneous
        # environments and must use Parallel execution mode.
        if config.execution_mode == "aec" and env_name in (
            "mosaic_multigrid", "socialjax", "meltingpot", "overcooked", "ini_multigrid",
        ):
            _OP_LOGGER.error(
                "AEC execution mode is incompatible with %s (simultaneous env). "
                "Switching to Parallel mode automatically.",
                env_name,
            )
            QtWidgets.QMessageBox.warning(
                self,
                "Execution Mode Mismatch",
                f"The environment '{task}' uses simultaneous stepping "
                f"(all agents act at the same time).\n\n"
                f"AEC (True Sequential Physics) mode is for turn-based games "
                f"like Chess or Go.\n\n"
                f"Switching to Parallel (Simultaneous) mode automatically.\n"
                f"To avoid this warning, set Execution Mode to "
                f"'Parallel (Simultaneous)' in the operator config.",
            )
            config.execution_mode = "parallel"

        # V2: RL agents without policy_path will fail to launch. Check now so
        # the user gets a clear popup instead of a silent error in the log.
        rl_agents_missing_policy = []
        for agent_id, assignment in config.workers.items():
            if assignment.worker_type == "rl":
                has_policy = bool(assignment.settings.get("policy_path"))
                if not has_policy:
                    # Check LinkGroup as fallback
                    in_group_with_policy = False
                    for group in config.link_groups.values():
                        if group.contains_agent(agent_id) and group.policy_path:
                            in_group_with_policy = True
                            break
                    if not in_group_with_policy:
                        rl_agents_missing_policy.append(agent_id)

        if rl_agents_missing_policy:
            _OP_LOGGER.error(
                "RL agents without policy_path: %s", rl_agents_missing_policy,
            )
            QtWidgets.QMessageBox.critical(
                self,
                "Missing Policy Checkpoint",
                f"The following RL agents have no policy checkpoint:\n\n"
                f"  {', '.join(rl_agents_missing_policy)}\n\n"
                f"Each RL agent needs a trained policy (.pth file).\n"
                f"Either:\n"
                f"  1. Browse to a checkpoint for the primary agent and "
                f"click 'Link Agents'\n"
                f"  2. Set each agent's Policy path individually\n\n"
                f"Trained checkpoints are in var/trainer/runs/",
            )
            return

        # V3: GAR validation -- at least 1 agent must be physically present
        # in the environment (real RL or replacement). If all agents in all
        # LinkGroups are ghosts with no replacements, the episode cannot step.
        ghost_agents = config.get_ghost_agents()
        if ghost_agents:
            replacement_agents = config.get_replacement_agents()
            real_agents = config.get_real_agents()
            standalone_ai = [
                a for a in config.get_ai_agents()
                if a not in ghost_agents and a not in real_agents
            ]
            agents_in_env = len(real_agents) + len(replacement_agents) + len(standalone_ai)
            if agents_in_env == 0:
                _OP_LOGGER.error(
                    "No agents in environment: all %d agents are ghosts with no replacements",
                    len(ghost_agents),
                )
                QtWidgets.QMessageBox.critical(
                    self,
                    "No Agents in Environment",
                    f"All {len(ghost_agents)} agent(s) are configured as ghost agents "
                    f"with no replacement workers.\n\n"
                    f"At least 1 agent must be physically present in the environment "
                    f"for the episode to advance.\n\n"
                    f"Fix: assign at least one agent slot as 'RL' type in the "
                    f"LinkGroup, or assign a replacement worker (LLM, Random, "
                    f"Passive) to at least one ghost slot.",
                )
                return

        _OP_LOGGER.info(
            "Resetting parallel multi-agent: env=%s, task=%s, seed=%d",
            env_name, task, seed,
        )

        # Close existing shared environment if any
        if self._parallel_multiagent_env is not None:
            try:
                self._parallel_multiagent_env.close()
            except Exception:
                pass

        # Create the shared environment
        try:
            raw_env = self._create_parallel_multiagent_env(env_name, task, seed, config)

            # Wrap in AEC env when execution_mode="aec".
            # AEC: env.step([action_i, NOOP]) fires once per agent,
            # so each subsequent agent observes the intermediate state S(t+0.5).
            if config.execution_mode == "aec":
                from gym_gui.services.aec_wrapper import GymnasiumMultiAgentAECWrapper
                self._parallel_multiagent_env = GymnasiumMultiAgentAECWrapper(raw_env)
                self._multigrid_aec_mode = True
                _OP_LOGGER.info(
                    "Wrapped %s/%s as AEC env (per-agent physics)",
                    env_name, task,
                )
            else:
                self._parallel_multiagent_env = raw_env
                self._multigrid_aec_mode = False

            self._parallel_multiagent_mode = True
            self._parallel_multiagent_config = config
            self._parallel_player_handles.clear()
            # Stop any previously launched shared RL subprocesses. Each
            # unique handle lives in _linkgroup_handles; stop them once.
            for _gid, _handle in self._linkgroup_handles.items():
                try:
                    _handle.stop()
                except Exception:
                    pass
            self._linkgroup_handles.clear()

            self.log_constant(
                LOG_OPERATOR_PARALLEL_RESET_STARTED,
                message=f"Created shared parallel environment: {env_name}/{task} (mode={config.execution_mode})",
                extra={"env_name": env_name, "task": task, "seed": seed, "execution_mode": config.execution_mode},
            )
        except Exception as e:
            self.log_constant(
                LOG_OPERATOR_ENV_PREVIEW_ERROR,
                message=f"Failed to create parallel environment: {e}",
                extra={"env_name": env_name, "task": task, "seed": seed},
            )
            self._status_bar.showMessage(f"Failed to create {task}: {e}", 5000)
            return

        # Launch AI workers in action_selector mode
        ai_agents = config.get_ai_agents()
        human_agents = config.get_human_agents()

        _OP_LOGGER.debug(
            "AI agents: %s, Human agents: %s",
            ai_agents, human_agents,
        )

        # --- Step 1: Launch shared RL subprocesses for LinkGroups ---
        #
        # One shared xuance_worker per LinkGroup instead of one per agent.
        # The launch runs on background QThreads so the GUI stays
        # responsive while policies load (30-120s each on CPU).
        # A progress dialog shows the load state in real time.
        #
        # See PLAN_UPDATE_1.md Section 5 and PLAN.md Diagram 1.
        linkgroup_ids = list(config.link_groups.keys())
        linkgroup_handles: dict[str, Any] = {}

        if linkgroup_ids:
            from gym_gui.services.linkgroup_launcher import LinkGroupLauncher
            from gym_gui.ui.widgets.operator_launch_progress_dialog import (
                OperatorLaunchProgressDialog,
            )

            progress_dialog = OperatorLaunchProgressDialog(
                linkgroup_ids, parent=self,
            )

            launchers: list[LinkGroupLauncher] = []
            errors: dict[str, str] = {}

            def _on_ready(group_id: str, handle):
                linkgroup_handles[group_id] = handle

            def _on_error(group_id: str, message: str):
                errors[group_id] = message

            for group_id in linkgroup_ids:
                link_group = config.link_groups[group_id]
                launcher = LinkGroupLauncher(
                    group_id=group_id,
                    link_group=link_group,
                    env_name=env_name,
                    task=task,
                    view_size=config.view_size,
                    launcher=self._operator_launcher,
                    operator_id=config.operator_id,
                    display_name=config.display_name,
                )
                launcher.progress.connect(progress_dialog.on_progress)
                launcher.ready.connect(progress_dialog.on_ready)
                launcher.error.connect(progress_dialog.on_error)
                launcher.ready.connect(_on_ready)
                launcher.error.connect(_on_error)
                launchers.append(launcher)

            def _on_cancel():
                for lg in launchers:
                    lg.cancel()

            progress_dialog.cancelled.connect(_on_cancel)

            # Show the dialog modeless (non-blocking) so run_sync can
            # pump its own event loop. Launchers run sequentially on the
            # main thread but use QApplication.processEvents() between
            # I/O polls to keep the GUI responsive and the progress bars
            # updating in real time.
            progress_dialog.show()
            QtWidgets.QApplication.processEvents()

            for lg in launchers:
                lg.run_sync()

            # All launchers done. Close the dialog if it hasn't auto-closed
            # (it only auto-closes on full success). Any errors stay
            # visible in the dialog for user review.
            if not progress_dialog.has_errors():
                progress_dialog.accept()
            else:
                # Keep dialog open until user clicks Close
                progress_dialog.exec()

            if errors:
                error_text = "\n".join(
                    f"  {gid}: {msg}" for gid, msg in errors.items()
                )
                QtWidgets.QMessageBox.critical(
                    self,
                    "LinkGroup Launch Failed",
                    f"Failed to launch shared RL worker(s):\n\n{error_text}",
                )
                # Clean up any partially launched handles
                for handle in linkgroup_handles.values():
                    try:
                        handle.stop()
                    except Exception:
                        pass
                return

            # Store shared handles for cleanup
            self._linkgroup_handles = linkgroup_handles

            # Map each linked agent to the shared handle for its LinkGroup.
            # Real RL agents keep this mapping. Ghost agents will have it
            # overridden in Step 3 below with their replacement handle.
            for group_id, handle in linkgroup_handles.items():
                link_group = config.link_groups[group_id]
                for agent_id in link_group.all_agents():
                    self._parallel_player_handles[agent_id] = handle
                    _OP_LOGGER.info(
                        "Agent %s -> shared RL process for LinkGroup %s",
                        agent_id, group_id,
                    )

        # --- Step 3: Launch replacement workers for ghost agents ---
        #
        # Ghost agents are LinkGroup members whose worker_type != "rl".
        # Step 1 mapped them to the shared RL handle (for ghost inference);
        # here we launch a separate replacement subprocess and override
        # _parallel_player_handles so the game loop submits the replacement
        # action to env.step() rather than the RL action.
        #
        # After this step the invariant holds:
        #   _parallel_player_handles[real_agent_id]  = shared RL handle
        #   _parallel_player_handles[ghost_agent_id] = replacement handle
        #   _linkgroup_handles[group_id]             = shared RL handle (ghost inference)
        ghost_agents = config.get_ghost_agents()  # agent_id -> group_id
        for agent_id, group_id in ghost_agents.items():
            assignment = config.workers.get(agent_id)
            if assignment is None:
                _OP_LOGGER.warning(
                    "Ghost agent %s has no WorkerAssignment, skipping replacement launch",
                    agent_id,
                )
                continue

            resolved_settings = dict(assignment.settings)
            replacement_config = OperatorConfig.single_agent(
                operator_id=f"{config.operator_id}_gar_repl_{agent_id}",
                display_name=f"{config.display_name} - GAR replacement ({agent_id})",
                worker_id=assignment.worker_id,
                worker_type=assignment.worker_type,
                env_name=env_name,
                task=task,
                settings=resolved_settings,
                view_size=config.view_size,
            )

            try:
                repl_handle = self._operator_launcher.launch_operator(
                    replacement_config, interactive=True,
                )
                startup_msg = repl_handle.read_response(timeout=15.0)
                if startup_msg and startup_msg.get("type") == "init":
                    _OP_LOGGER.debug("Drained startup init from GAR replacement %s", agent_id)

                repl_handle.send_init_agent(game_name=task, player_id=agent_id)
                import time as _time
                _init_deadline = _time.monotonic() + 30.0
                init_resp = None
                while _time.monotonic() < _init_deadline:
                    init_resp = repl_handle.try_read_response(timeout=2.0)
                    if init_resp is not None:
                        break
                if init_resp and init_resp.get("type") in ("agent_ready", "agent_initialized"):
                    _OP_LOGGER.info(
                        "GAR replacement worker ready for ghost agent %s (group %s)",
                        agent_id, group_id,
                    )
                else:
                    _OP_LOGGER.warning(
                        "Unexpected init_agent response for GAR replacement %s: %s",
                        agent_id, init_resp,
                    )
                # Override: ghost slot now points to replacement, not shared RL.
                self._parallel_player_handles[agent_id] = repl_handle
                self.log_constant(
                    LOG_GAR_REPLACEMENT_LAUNCHED,
                    message=f"GAR replacement launched for ghost agent {agent_id}",
                    extra={
                        "agent_id": agent_id,
                        "group_id": group_id,
                        "worker": assignment.worker_id,
                    },
                )
            except OperatorLaunchError as e:
                _OP_LOGGER.error(
                    "Failed to launch GAR replacement for ghost agent %s: %s", agent_id, e
                )
                self.log_constant(
                    LOG_OPERATOR_ENV_PREVIEW_ERROR,
                    message=f"Failed to launch GAR replacement for ghost agent {agent_id}: {e}",
                    extra={"agent_id": agent_id, "task": task},
                )

        # --- Step 2: Launch standalone (non-LinkGroup) workers ---
        #
        # Agents that are NOT in any LinkGroup get their own subprocess
        # as before (one per agent). These are typically lightweight
        # (LLM, random, passive, human) and launch in 1-2s.
        linked_agents = set()
        for link_group in config.link_groups.values():
            linked_agents.update(link_group.all_agents())

        for agent_id in ai_agents:
            if agent_id in linked_agents:
                continue  # already handled via LinkGroup above

            assignment = config.workers.get(agent_id)
            if assignment is None:
                _OP_LOGGER.warning("No worker assignment for agent %s, skipping", agent_id)
                continue

            resolved_settings = dict(assignment.settings)

            player_config = OperatorConfig.single_agent(
                operator_id=f"{config.operator_id}_{agent_id}",
                display_name=f"{config.display_name} - {agent_id}",
                worker_id=assignment.worker_id,
                worker_type=assignment.worker_type,
                env_name=env_name,
                task=task,
                settings=resolved_settings,
                view_size=config.view_size,
            )

            try:
                handle = self._operator_launcher.launch_operator(
                    player_config,
                    interactive=True,
                )
                # Drain startup init + init_agent response synchronously
                # (these workers are fast: random, passive, LLM).
                startup_msg = handle.read_response(timeout=15.0)
                if startup_msg and startup_msg.get("type") == "init":
                    _OP_LOGGER.debug("Drained startup init from %s", agent_id)

                handle.send_init_agent(game_name=task, player_id=agent_id)
                init_resp = None
                import time as _time
                _init_deadline = _time.monotonic() + 30.0
                while _time.monotonic() < _init_deadline:
                    init_resp = handle.try_read_response(timeout=2.0)
                    if init_resp is not None:
                        break
                if init_resp and init_resp.get("type") in ("agent_ready", "agent_initialized"):
                    _OP_LOGGER.info(
                        "Standalone worker ready for agent %s", agent_id,
                    )
                else:
                    _OP_LOGGER.warning(
                        "Unexpected init_agent response for %s: %s", agent_id, init_resp
                    )
                self._parallel_player_handles[agent_id] = handle
            except OperatorLaunchError as e:
                _OP_LOGGER.error("Failed to launch AI worker for %s: %s", agent_id, e)
                self.log_constant(
                    LOG_OPERATOR_ENV_PREVIEW_ERROR,
                    message=f"Failed to launch AI worker for {agent_id}: {e}",
                    extra={"agent_id": agent_id, "task": task},
                )

        # Reset the environment.
        # PettingZoo AEC reset() returns None; initial obs are read via observe().
        # Parallel envs return (obs, info) as usual.
        try:
            reset_result = self._parallel_multiagent_env.reset(seed=seed)
            self._parallel_episode_reward = 0.0
            self._parallel_step_index = 0
            if reset_result is None:
                # PettingZoo AEC env — populate obs by calling observe() per agent
                env = self._parallel_multiagent_env
                self._parallel_multiagent_obs = {
                    a: env.observe(a) for a in env.agents
                }
            else:
                obs, info = reset_result
                self._parallel_multiagent_obs = obs if isinstance(obs, dict) else {}
            _OP_LOGGER.debug(
                "Environment reset, obs keys: %s",
                list(self._parallel_multiagent_obs.keys()),
            )
        except Exception as e:
            self.log_constant(
                LOG_OPERATOR_ENV_PREVIEW_ERROR,
                message=f"Failed to reset environment: {e}",
                extra={"env_name": env_name, "task": task, "seed": seed},
            )
            self._status_bar.showMessage(f"Reset failed: {e}", 5000)
            return

        # Render initial frame
        self.render_parallel_multiagent_frame()

        # Update operator state
        operator_id = config.operator_id
        self._multi_operator_service.set_operator_state(operator_id, "running")
        self._render_tabs.set_operator_status(operator_id, "running")

        # Enable parallel mode on the Operators tab so Step All / Human Step
        # behave correctly for GUI-owned environments.
        self._control_panel.operators_tab.set_parallel_mode(True)

        # Show status
        self._status_bar.showMessage(
            f"Parallel multi-agent {task} ready: {len(human_agents)} human, {len(ai_agents)} AI agents (seed={seed})",
            3000
        )

    def _create_parallel_multiagent_env(
        self, env_name: str, task: str, seed: int, config: "OperatorConfig | None" = None,
    ) -> Any:
        """Create a parallel multi-agent environment.

        Args:
            env_name: Environment family (mosaic_multigrid, ini_multigrid, meltingpot, overcooked).
            task: Specific environment (e.g., MultiGridSports-Soccer-v0).
            seed: Random seed.
            config: Optional operator config (used to extract view_size, etc.).

        Returns:
            The created gymnasium/PettingZoo Parallel environment.
        """
        if env_name == "mosaic_multigrid":
            # All mosaic envs registered via gymnasium.register() in mosaic_multigrid.envs
            import gymnasium
            import mosaic_multigrid.envs  # noqa: F401 - triggers gymnasium.register() calls
            extra_kwargs: dict[str, Any] = {}
            if config is not None and config.view_size is not None:
                extra_kwargs["view_size"] = config.view_size
                _OP_LOGGER.info(
                    "Operator view_size=%d applied to %s",
                    config.view_size, task,
                )
            env = gymnasium.make(task, render_mode='rgb_array', disable_env_checker=True, **extra_kwargs)
            return env

        elif env_name == "ini_multigrid":
            # Import and create INI MultiGrid environment
            import gymnasium as gym
            env = gym.make(task, render_mode="rgb_array", disable_env_checker=True)
            return env

        elif env_name == "meltingpot":
            # MeltingPot support (not yet implemented).
            # MeltingPot uses NOOP=0 (dm_env convention), so when implemented
            # it will support both Parallel and AEC via GymnasiumMultiAgentAECWrapper.
            raise NotImplementedError("MeltingPot environment not yet supported")

        elif env_name == "overcooked":
            # Overcooked support (placeholder)
            raise NotImplementedError("Overcooked environment not yet supported")

        else:
            raise ValueError(f"Unknown parallel multi-agent environment: {env_name}")

    def _get_parallel_action_labels(self, config: "OperatorConfig", env: Any) -> list[str]:
        """Get human-readable action labels for a parallel multi-agent environment.

        Looks up the correct action list from the adapter module based on
        the operator config's env_name. Falls back to generic labels
        derived from the action space size when the env family is unknown.

        Note: This method is ONLY for multi-agent environments. Single-agent
        environments (minigrid, babyai, crafter, procgen) get their action
        labels from the human_worker subprocess.

        Args:
            config: Operator configuration (used for env_name).
            env: The parallel environment instance.

        Returns:
            List of action label strings, one per discrete action.
        """
        env_name = config.env_name

        # Multi-agent environments only
        if env_name == "mosaic_multigrid":
            from gym_gui.core.adapters.mosaic_multigrid import MOSAIC_MULTIGRID_ACTIONS
            return list(MOSAIC_MULTIGRID_ACTIONS)
        elif env_name == "ini_multigrid":
            from gym_gui.core.adapters.ini_multigrid import INI_MULTIGRID_ACTIONS
            return list(INI_MULTIGRID_ACTIONS)
        elif env_name == "overcooked":
            from gym_gui.core.adapters.overcooked import OVERCOOKED_ACTIONS
            return list(OVERCOOKED_ACTIONS)
        elif env_name == "meltingpot":
            from gym_gui.core.adapters.meltingpot import MELTINGPOT_ACTION_NAMES
            return list(MELTINGPOT_ACTION_NAMES)
        elif env_name == "socialjax":
            from gym_gui.core.adapters.socialjax import SOCIALJAX_ACTIONS_BY_ENV
            env_id = config.task.removeprefix("socialjax/") if config.task else ""
            labels = SOCIALJAX_ACTIONS_BY_ENV.get(env_id)
            if labels:
                return list(labels)
        elif env_name == "smac":
            from gym_gui.core.adapters.smac import SMAC_BASE_ACTIONS
            return list(SMAC_BASE_ACTIONS)

        # Generic fallback: discover action count from the environment
        num_actions: int | None = None

        # Single action_space (gymnasium standard)
        if hasattr(env, "action_space") and hasattr(env.action_space, "n"):
            num_actions = env.action_space.n
        # Per-agent action_spaces (PettingZoo parallel)
        elif hasattr(env, "action_spaces"):
            for space in env.action_spaces.values():
                if hasattr(space, "n"):
                    num_actions = space.n
                    break
        # PettingZoo AEC: action_space(agent) is a method
        elif callable(getattr(env, "action_space", None)) and hasattr(env, "agents") and env.agents:
            try:
                space = env.action_space(env.agents[0])
                if hasattr(space, "n"):
                    num_actions = space.n
            except Exception:
                pass

        if num_actions is None:
            _OP_LOGGER.warning(
                "Could not determine action count for env_name=%s, defaulting to 4",
                env_name,
            )
            num_actions = 4

        return [f"Action {i}" for i in range(num_actions)]

    def _resolve_agent_colors(
        self, config: "OperatorConfig",
    ) -> Dict[str, tuple[str, str]]:
        """Build agent_id -> (primary_hex, bg_hex) from operator config.

        Reads agent_color from each worker's settings and maps it
        through COLOR_PALETTE. Falls back to the default
        palette assignment when no custom color is set.
        """
        colors: Dict[str, tuple[str, str]] = {}
        for player_id, worker in config.workers.items():
            color_name = worker.settings.get("agent_color")
            if color_name and color_name != "auto" and color_name in COLOR_PALETTE:
                colors[player_id] = COLOR_PALETTE[color_name]
            else:
                default_name = DEFAULT_AGENT_COLOR_NAMES.get(player_id)
                if default_name and default_name in COLOR_PALETTE:
                    colors[player_id] = COLOR_PALETTE[default_name]
        return colors

    def _get_parallel_agent_obs(self, agent_id: str) -> Optional[Any]:
        """Get the stored observation for an agent, handling int/string key mismatch.

        mosaic_multigrid uses integer agent keys (0, 1) in obs dicts but
        OperatorConfig uses string agent IDs ("agent_0", "agent_1").

        Args:
            agent_id: String agent identifier, e.g. "agent_0".

        Returns:
            The obs array/dict for this agent, or None if not found.
        """
        obs_dict = self._parallel_multiagent_obs
        if not obs_dict:
            return None

        # Direct string lookup first
        if agent_id in obs_dict:
            return obs_dict[agent_id]

        # Extract trailing integer: "agent_0" -> 0, "player_1" -> 1
        try:
            idx = int(str(agent_id).split("_")[-1])
            if idx in obs_dict:
                return obs_dict[idx]
        except (ValueError, AttributeError):
            pass

        # Last resort: direct int cast
        try:
            if int(agent_id) in obs_dict:
                return obs_dict[int(agent_id)]
        except (ValueError, TypeError):
            pass

        return None

    def _embed_parallel_action_panel(self, panel: QtWidgets.QWidget) -> None:
        """Embed a MultiAgentActionPanel inside the operator's render container.

        Places the panel right below the environment render so the human can
        see the game and the action buttons at the same time.
        """
        config = self._parallel_multiagent_config
        if config is None:
            return
        container = self._render_tabs.multi_operator_view.get_container(
            config.operator_id
        )
        if container is not None:
            container.set_parallel_action_panel(panel)
        else:
            _OP_LOGGER.warning(
                "No render container for operator %s — cannot embed action panel",
                config.operator_id,
            )

    def clear_parallel_action_panel(self) -> None:
        """Remove the embedded parallel action panel from the render container."""
        config = self._parallel_multiagent_config
        if config is None:
            return
        container = self._render_tabs.multi_operator_view.get_container(
            config.operator_id
        )
        if container is not None:
            container.clear_parallel_action_panel()

    def render_parallel_multiagent_frame(self) -> None:
        """Render the current state of the parallel multi-agent environment."""
        if self._parallel_multiagent_env is None:
            return

        try:
            env = self._parallel_multiagent_env
            config = self._parallel_multiagent_config
            if config is None:
                return

            # Get RGB frame from environment
            frame = env.render()
            if frame is None:
                _OP_LOGGER.warning("Environment render returned None")
                return

            # Create render payload
            if isinstance(frame, np.ndarray):
                h, w = int(frame.shape[0]), int(frame.shape[1])
                render_payload = {
                    "mode": "rgb",
                    "rgb": frame.tolist(),
                    "width": w,
                    "height": h,
                }
            else:
                _OP_LOGGER.warning(f"Unexpected frame type: {type(frame)}")
                return

            # Display in render container
            wrapped_payload = {
                "render_payload": render_payload,
                "episode_index": self._parallel_episode_index,
                "step_index": self._parallel_step_index,
                "reward": 0.0,
                "episode_reward": self._parallel_episode_reward,
                "terminated": False,
                "truncated": False,
            }
            self._render_tabs.display_operator_payload(config.operator_id, wrapped_payload)

        except Exception as e:
            _OP_LOGGER.warning(f"Failed to render parallel frame: {e}")

    def on_step_parallel_multiagent(self) -> None:
        """Step the parallel multi-agent environment.

        Flow for simultaneous stepping:
        1. Collect actions from AI agents (via workers)
        2. Show action panel for human agents
        3. Wait for all human selections
        4. Call env.step() with all actions
        5. Render updated frame
        """
        env = self._parallel_multiagent_env
        config = self._parallel_multiagent_config
        if env is None or config is None:
            _OP_LOGGER.warning("No parallel multi-agent environment")
            return

        # Get agent lists
        human_agents = config.get_human_agents()
        ai_agents = config.get_ai_agents()

        _OP_LOGGER.debug(
            "Stepping parallel multi-agent: human=%s, ai=%s",
            human_agents, ai_agents,
        )

        # Create step state to track pending actions
        step_state = MultiAgentStepState.from_config(config, step_id=0)
        self._parallel_multiagent_step_state = step_state

        self.log_constant(
            LOG_OPERATOR_PARALLEL_STEP_STARTED,
            message=f"Collecting actions from {len(ai_agents)} AI and {len(human_agents)} human agents",
            extra={"human_agents": human_agents, "ai_agents": ai_agents},
        )

        # Collect AI actions by calling each worker's select_action.
        # Ghost agents (in a LinkGroup with worker_type != "rl") require two
        # sends and two reads per step: one to the shared RL handle for ghost
        # inference (action discarded) and one to the replacement handle for
        # the action submitted to env.step().
        ghost_agents = config.get_ghost_agents()  # agent_id -> group_id

        for agent_id in ai_agents:
            handle = self._parallel_player_handles.get(agent_id)

            if handle is None or not handle.is_running:
                raise RuntimeError(
                    f"No running worker for AI agent {agent_id}. "
                    f"Start the worker before stepping."
                )

            # Get and flatten the agent's observation.
            # mosaic_multigrid IndAgObs: obs is a dict with 'image' (3,3,3 array).
            # XuanCe IPPO/MAPPO was trained on the flattened image (27 floats).
            agent_obs = self._get_parallel_agent_obs(agent_id)
            if agent_obs is None:
                raise RuntimeError(
                    f"No observation for AI agent {agent_id}. "
                    f"Reset the environment before stepping."
                )

            if isinstance(agent_obs, dict):
                image = agent_obs.get("image", next(iter(agent_obs.values())))
                obs_flat = image.flatten().tolist() if hasattr(image, "flatten") else list(image)
            elif hasattr(agent_obs, "flatten"):
                obs_flat = agent_obs.flatten().tolist()
            else:
                obs_flat = list(agent_obs)

            if agent_id in ghost_agents:
                # --- Ghost agent: RL inference (discard) + replacement action (submit) ---
                group_id = ghost_agents[agent_id]
                rl_handle = self._linkgroup_handles.get(group_id)
                repl_handle = handle  # _parallel_player_handles[agent_id] = replacement (Step 3)

                if rl_handle is None:
                    raise RuntimeError(
                        f"Ghost agent {agent_id}: no shared RL handle for group {group_id}. "
                        f"Ensure the LinkGroup was launched successfully."
                    )

                # Drain stale messages from both handles before sending.
                for _ in range(10):
                    if rl_handle.try_read_response(timeout=0.0) is None:
                        break
                for _ in range(10):
                    if repl_handle.try_read_response(timeout=0.0) is None:
                        break

                # Parallel sends: both handles receive the observation before
                # either blocks waiting for a response.
                rl_handle.send_select_action(obs_flat, agent_id)
                repl_handle.send_select_action(obs_flat, agent_id)

                # Read ghost action from shared RL — computed for identity-vector
                # consistency in parameter-sharing MAPPO, then DISCARDED.
                ghost_resp = rl_handle.read_response(timeout=10.0)
                ghost_action = None
                if ghost_resp and ghost_resp.get("type") == "action_selected":
                    ghost_action_val = ghost_resp.get("action", "")
                    if ghost_action_val != "" and ghost_action_val is not None:
                        ghost_action = int(ghost_action_val)

                # Read replacement action — this IS submitted to env.step().
                repl_resp = repl_handle.read_response(timeout=10.0)
                real_action = 0
                if repl_resp and repl_resp.get("type") == "action_selected":
                    repl_action_val = repl_resp.get("action", "")
                    if repl_action_val != "" and repl_action_val is not None:
                        real_action = int(repl_action_val)
                    else:
                        _OP_LOGGER.warning(
                            "GAR replacement for %s returned empty action, using NOOP (0)",
                            agent_id,
                        )
                else:
                    _OP_LOGGER.warning(
                        "GAR replacement for %s: unexpected response %s, using NOOP (0)",
                        agent_id, repl_resp,
                    )

                step_state.add_action(agent_id, real_action)
                self.log_constant(
                    LOG_GAR_GHOST_ACTION_DISCARDED,
                    message=(
                        f"Ghost inference discarded, replacement action submitted "
                        f"for agent {agent_id}"
                    ),
                    extra={
                        "agent_id": agent_id,
                        "group_id": group_id,
                        "ghost_action": ghost_action,
                        "real_action": real_action,
                    },
                )
                continue  # skip standard read path below

            # --- Real RL or standalone agent: standard path ---

            # Drain any pending non-action messages (e.g., agent_ready) before
            # sending the select_action request to avoid reading stale responses.
            for _ in range(10):
                pending = handle.try_read_response(timeout=0.0)
                if pending is None:
                    break
                _OP_LOGGER.debug(
                    "Drained pending message from %s: %s", agent_id, pending.get("type")
                )

            handle.send_select_action(obs_flat, agent_id)

            # Read response, retrying if we get non-action messages (e.g.,
            # a late agent_ready arriving after we drained).
            action: int | None = None
            response: dict | None = None
            max_retries = 5
            for attempt in range(max_retries):
                response = handle.read_response(timeout=10.0)

                if response is None:
                    _OP_LOGGER.warning(
                        "Timeout reading action from AI agent %s (attempt %d/%d)",
                        agent_id, attempt + 1, max_retries,
                    )
                    continue

                if response.get("type") == "action_selected":
                    action_val = response.get("action", "")
                    if action_val == "" or action_val is None:
                        _OP_LOGGER.warning(
                            "AI agent %s returned empty action, using NOOP (0)", agent_id
                        )
                        action = 0
                    else:
                        action = int(action_val)
                    _OP_LOGGER.debug("AI agent %s selected action %d", agent_id, action)
                    break

                if response.get("type") == "agent_ready":
                    _OP_LOGGER.debug(
                        "AI agent %s sent agent_ready, waiting for action_selected "
                        "(attempt %d/%d)", agent_id, attempt + 1, max_retries,
                    )
                    continue

                # Unknown response type — log and retry
                _OP_LOGGER.warning(
                    "AI agent %s sent unexpected response type '%s', retrying "
                    "(attempt %d/%d)", agent_id, response.get("type"), attempt + 1, max_retries,
                )

            if action is None:
                _OP_LOGGER.error(
                    "Failed to get action from AI agent %s after %d attempts. "
                    "Last response: %s", agent_id, max_retries, response,
                )
                raise RuntimeError(
                    f"Worker for agent {agent_id} returned no valid action after "
                    f"{max_retries} attempts. Last response: {response}. "
                    f"Check operator_agent logs for details."
                )

            step_state.add_action(agent_id, action)

        step_state.ai_actions_ready = True

        # If no human agents, step immediately
        if not human_agents:
            self.execute_parallel_multiagent_step(step_state.get_all_actions())
            return

        # Show action panel for human agents
        action_labels = self._get_parallel_action_labels(config, env)

        # Create and show action panel
        if self._parallel_action_panel is not None:
            self._parallel_action_panel.deleteLater()

        self._parallel_action_panel = MultiAgentActionPanel(
            human_agents=human_agents,
            action_labels=action_labels,
            agent_labels={aid: f"Agent {aid.split('_')[-1]}" for aid in human_agents},
            agent_colors=self._resolve_agent_colors(config),
        )
        self._parallel_action_panel.all_actions_submitted.connect(
            self._on_parallel_human_actions_submitted
        )

        # Embed the action panel in the render container (right below the environment)
        self._embed_parallel_action_panel(self._parallel_action_panel)
        _OP_LOGGER.info(
            f"Waiting for human actions from {len(human_agents)} agents"
        )
        input_method = "keyboard (subprocess)" if self._parent._keyboard_worker_bridge.is_active else "action buttons"
        self._status_bar.showMessage(
            f"Select actions for {len(human_agents)} human agent(s) ({input_method})",
            10000
        )

    def on_step_multigrid_aec(self) -> None:
        """Step one turn of the AEC environment (per-agent physics).

        Supports: mosaic_multigrid (v5.0.0+) and MeltingPot (NOOP=0 required).

        In AEC mode, only ONE agent acts per call:
          - GymnasiumMultiAgentAECWrapper.step(action) calls
            env.step([action_i, NOOP, ...]) internally, advancing physics
            immediately for this agent only.
          - The next agent observes the intermediate state S(t+0.5).
          - agent_selection cycles "agent_0" → "agent_1" → ...
          - If AI: query worker for action, then aec_env.step(action)
          - If human: show one-agent action panel, wait for submission
          - Render after each individual step (sequential visual feedback)

        mosaic_multigrid v5 action space: noop=0  left=1  right=2  forward=3
          pickup=4  drop=5  toggle=6  done=7  (8 actions, noop=0 required for AEC)
        """
        env = self._parallel_multiagent_env
        config = self._parallel_multiagent_config
        if env is None or config is None:
            _OP_LOGGER.warning("_on_step_multigrid_aec: no env/config")
            return

        # agent_selection is already a PettingZoo string ID: "agent_0" or "agent_1"
        current_agent = env.agent_selection
        if current_agent is None or not env.agents:
            _OP_LOGGER.info("AEC episode done — agent_selection is None")
            self._status_bar.showMessage("Episode finished. Reset to play again.", 4000)
            return

        human_agents = config.get_human_agents()
        ai_agents = config.get_ai_agents()

        _OP_LOGGER.debug(
            "AEC step: current_agent=%s, human=%s, ai=%s",
            current_agent, human_agents, ai_agents,
        )

        if current_agent in ai_agents:
            # ---------------------------------------------------------------
            # AI agent's turn: query the worker for an action
            # ---------------------------------------------------------------
            # PettingZoo AEC exposes action_space as a method: env.action_space(agent)
            try:
                action_space = env.action_space(current_agent)
            except (TypeError, KeyError):
                action_space = None

            handle = self._parallel_player_handles.get(current_agent)

            if handle is None or not handle.is_running:
                _OP_LOGGER.warning(
                    "AEC: no running worker for AI agent %s — NOOP fallback",
                    current_agent,
                )
                env.step(0)  # NOOP (action 0)
            else:
                # observe() uses the PettingZoo string agent ID
                agent_obs = env.observe(current_agent)
                if agent_obs is None:
                    _OP_LOGGER.warning(
                        "AEC: no observation for %s — NOOP fallback", current_agent
                    )
                    env.step(0)
                else:
                    # Flatten observation for XuanCe worker (IndAgObs: image 3×3×3)
                    if isinstance(agent_obs, dict):
                        image = agent_obs.get("image", next(iter(agent_obs.values())))
                        obs_flat = (
                            image.flatten().tolist()
                            if hasattr(image, "flatten")
                            else list(image)
                        )
                    elif hasattr(agent_obs, "flatten"):
                        obs_flat = agent_obs.flatten().tolist()
                    else:
                        obs_flat = list(agent_obs)

                    handle.send_select_action(obs_flat, current_agent)
                    response = handle.read_response(timeout=10.0)

                    if response and response.get("type") == "action_selected":
                        action = int(response["action"])
                        _OP_LOGGER.debug(
                            "AEC: AI agent %s chose action %d", current_agent, action
                        )
                    else:
                        _OP_LOGGER.warning(
                            "AEC: bad response from %s: %s — fallback to NOOP",
                            current_agent, response,
                        )
                        action = (
                            action_space.sample()
                            if action_space is not None and hasattr(action_space, "sample")
                            else 0
                        )

                    env.step(action)

            # Render after this agent's action
            self.render_parallel_multiagent_frame()

            # Check episode end
            if not env.agents:
                total_reward = sum(env.rewards.values())
                self._status_bar.showMessage(
                    f"Episode done! Total reward: {total_reward:.2f}", 5000
                )
                _OP_LOGGER.info(
                    "AEC episode ended after %s acted, total_reward=%.2f",
                    current_agent, total_reward,
                )
            else:
                self._status_bar.showMessage(
                    f"AEC: {current_agent} acted → now {env.agent_selection}'s turn",
                    1500,
                )

        elif current_agent in human_agents:
            # ---------------------------------------------------------------
            # Human agent's turn: show one-agent action panel
            # ---------------------------------------------------------------
            action_labels = self._get_parallel_action_labels(config, env)

            try:
                act_space = env.action_space(current_agent)
                num_actions = act_space.n if hasattr(act_space, "n") else len(action_labels)
            except (TypeError, KeyError):
                num_actions = len(action_labels)

            # Trim labels to match actual action space size
            action_labels = action_labels[:num_actions]

            if self._parallel_action_panel is not None:
                self._parallel_action_panel.deleteLater()

            self._parallel_action_panel = MultiAgentActionPanel(
                human_agents=[current_agent],
                action_labels=action_labels,
                agent_labels={current_agent: f"Agent {current_agent.split('_')[-1]}"},
                agent_colors=self._resolve_agent_colors(config),
            )
            self._parallel_action_panel.all_actions_submitted.connect(
                self._on_aec_human_action_submitted
            )

            # Embed panel in the render container (right below the environment)
            self._embed_parallel_action_panel(self._parallel_action_panel)

            _OP_LOGGER.info("AEC: waiting for human %s to act", current_agent)
            input_method = "keyboard (subprocess)" if self._parent._keyboard_worker_bridge.is_active else "buttons"
            self._status_bar.showMessage(
                f"AEC: select action for {current_agent} ({input_method})", 10000
            )
        else:
            _OP_LOGGER.warning(
                "AEC: agent %s not in human_agents or ai_agents — NOOP fallback",
                current_agent,
            )
            env.step(0)
            self.render_parallel_multiagent_frame()

    def _on_aec_human_action_submitted(self, actions: Dict[str, int]) -> None:
        """Handle submission of a single human agent's action in AEC mode.

        Args:
            actions: Dict with exactly one entry {agent_str: action_int}.
        """
        env = self._parallel_multiagent_env
        if env is None:
            return

        if not actions:
            _OP_LOGGER.warning("_on_aec_human_action_submitted: empty actions dict")
            return

        action_int = next(iter(actions.values()))
        env.step(int(action_int))

        self.render_parallel_multiagent_frame()

        # Clean up embedded action panel
        self.clear_parallel_action_panel()

        if not env.agents:
            total_reward = sum(env.rewards.values())
            self._status_bar.showMessage(
                f"Episode done! Total reward: {total_reward:.2f}", 5000
            )
        else:
            self._status_bar.showMessage(
                f"AEC: human acted → now {env.agent_selection}'s turn", 2000
            )

    def _on_evdev_aec_agent_action(self, agent_id: str, action: int) -> None:
        """Handle evdev keyboard action during AEC human turn.

        In AEC mode, only the current agent can act. If the evdev action
        comes from the current agent, submit it as if the human clicked
        the action button.

        Args:
            agent_id: The agent ID whose keyboard was pressed.
            action: The resolved action index.
        """
        env = self._parallel_multiagent_env
        if env is None:
            return

        current_agent = getattr(env, "agent_selection", None)
        if current_agent is None:
            return

        # Only accept input from the agent whose turn it is
        if agent_id != current_agent:
            _OP_LOGGER.debug(
                "AEC evdev: ignoring action from %s (current turn: %s)",
                agent_id, current_agent,
            )
            return

        _OP_LOGGER.info(
            "AEC evdev action: %s → action %d", agent_id, action
        )

        # Disconnect evdev bridge before stepping (prevents duplicate handling)
        try:
            self._parent._human_input.agent_action_selected.disconnect(
                self._on_evdev_aec_agent_action
            )
        except (TypeError, RuntimeError):
            pass

        # Step the AEC environment with this action
        env.step(int(action))
        self.render_parallel_multiagent_frame()

        # Clean up embedded action panel
        self.clear_parallel_action_panel()

        if not env.agents:
            total_reward = sum(env.rewards.values())
            self._status_bar.showMessage(
                f"Episode done! Total reward: {total_reward:.2f}", 5000
            )
        else:
            self._status_bar.showMessage(
                f"AEC: {agent_id} acted via keyboard → now {env.agent_selection}'s turn",
                2000,
            )

    def _on_parallel_human_actions_submitted(self, actions: Dict[str, int]) -> None:
        """Handle submission of all human actions.

        Args:
            actions: Dict mapping agent_id to action index.
        """
        _OP_LOGGER.info(f"Human actions submitted: {actions}")

        step_state = self._parallel_multiagent_step_state
        if step_state is None:
            _OP_LOGGER.warning("No step state for human action submission")
            return

        # Add human actions to step state
        for agent_id, action in actions.items():
            step_state.add_action(agent_id, action)

        # Check if all actions are collected
        if step_state.is_complete():
            self.execute_parallel_multiagent_step(step_state.get_all_actions())
        else:
            _OP_LOGGER.warning(
                f"Not all actions collected: {len(step_state.pending_actions)} / "
                f"{len(step_state.human_agents) + len(step_state.ai_agents)}"
            )

    def _on_evdev_agent_action(self, agent_id: str, action: int) -> None:
        """Handle a per-agent action from evdev keyboard input.

        Called when a physical keyboard assigned to an agent presses a key
        combination that resolves to an action. Adds the action to the current
        parallel multi-agent step state and executes the step when all human
        agents have acted.

        Args:
            agent_id: The agent ID (e.g., "agent_0") whose keyboard was pressed.
            action: The resolved action index.
        """
        step_state = self._parallel_multiagent_step_state
        if step_state is None:
            _OP_LOGGER.debug(
                "Evdev agent action ignored (no active step state): agent=%s action=%d",
                agent_id, action,
            )
            return

        # Only accept actions from human agents in this step
        if agent_id not in step_state.human_agents:
            _OP_LOGGER.debug(
                "Evdev action from non-human agent %s ignored", agent_id
            )
            return

        # Skip if this agent already acted in this step
        if agent_id in step_state.pending_actions:
            _OP_LOGGER.debug(
                "Evdev action from %s ignored (already acted this step)", agent_id
            )
            return

        step_state.add_action(agent_id, action)
        _OP_LOGGER.info(
            "Evdev agent action: %s → action %d (%d/%d human actions collected)",
            agent_id, action,
            len([a for a in step_state.human_agents if a in step_state.pending_actions]),
            len(step_state.human_agents),
        )

        # Update status bar with progress
        pending = step_state.pending_human_agents()
        if pending:
            self._status_bar.showMessage(
                f"Waiting for keyboard input from: {', '.join(pending)}",
                10000,
            )

        # Execute step if all actions collected
        if step_state.is_complete():
            _OP_LOGGER.info("All actions collected via evdev + AI — executing step")
            self.execute_parallel_multiagent_step(step_state.get_all_actions())

    def execute_parallel_multiagent_step(self, actions: Dict[str, int]) -> None:
        """Execute a step on the parallel multi-agent environment.

        Args:
            actions: Dict mapping agent_id to action index.
        """
        # Disconnect evdev bridge to prevent stale signals between steps
        try:
            self._parent._human_input.agent_action_selected.disconnect(
                self._on_evdev_agent_action
            )
        except (TypeError, RuntimeError):
            pass

        env = self._parallel_multiagent_env
        config = self._parallel_multiagent_config
        if env is None or config is None:
            return

        _OP_LOGGER.info(f"Executing parallel step with actions: {actions}")

        try:
            # Build action dict with integer keys.
            # config.workers uses string keys ("agent_0", "agent_1") but
            # mosaic_multigrid uses integer keys (0, 1). Always convert.
            int_actions: Dict[Any, int] = {}
            for agent_id, action in actions.items():
                try:
                    idx = int(str(agent_id).split("_")[-1])
                except (ValueError, AttributeError):
                    try:
                        idx = int(agent_id)
                    except (ValueError, TypeError):
                        continue
                int_actions[idx] = action

            if hasattr(env, "agents"):
                action_input = [int_actions.get(agent, 0) for agent in env.agents]
            else:
                action_input = int_actions

            obs, rewards, terminateds, truncateds, infos = env.step(action_input)

            # Track step and accumulate reward
            self._parallel_step_index += 1
            step_reward = sum(rewards.values()) if isinstance(rewards, dict) else sum(rewards)
            self._parallel_episode_reward += step_reward

            # Store updated observations so next step's select_action gets fresh obs
            if isinstance(obs, dict):
                self._parallel_multiagent_obs = obs

            # Render updated frame
            self.render_parallel_multiagent_frame()

            # Check for episode end
            all_done = all(terminateds.values()) if isinstance(terminateds, dict) else all(terminateds)
            all_truncated = all(truncateds.values()) if isinstance(truncateds, dict) else all(truncateds)

            if all_done or all_truncated:
                self._status_bar.showMessage(
                    f"Episode {self._parallel_episode_index} done at step {self._parallel_step_index}! "
                    f"Total reward: {self._parallel_episode_reward:.2f}",
                    5000
                )
                _OP_LOGGER.info(
                    "Episode %d ended at step %d with total reward: %.2f",
                    self._parallel_episode_index, self._parallel_step_index,
                    self._parallel_episode_reward,
                )
                self._parallel_episode_index += 1

                # If this operator is driven by the Script Experiment panel, hand
                # episode-end back to the script manager so it controls the next seed.
                # Otherwise auto-reset for manual continuous play.
                if config is not None and self._script_mode_handler.owns_operator(config.operator_id):
                    self._parallel_episode_reward = 0.0
                    self._parallel_step_index = 0
                    script_mgr = self._control_panel.operators_tab.script_execution_manager
                    script_mgr.on_episode_ended(config.operator_id, all_done, all_truncated)
                else:
                    self._parallel_episode_reward = 0.0
                    self._parallel_step_index = 0
                    reset_result = env.reset()
                    if reset_result is not None:
                        obs, _info = reset_result
                        if isinstance(obs, dict):
                            self._parallel_multiagent_obs = obs
            else:
                self._status_bar.showMessage(
                    f"Step {self._parallel_step_index}. Reward: {step_reward:.2f} "
                    f"(episode total: {self._parallel_episode_reward:.2f})",
                    2000
                )

            # Clear step state and embedded action panel
            self._parallel_multiagent_step_state = None
            self.clear_parallel_action_panel()

            self.log_constant(
                LOG_OPERATOR_PARALLEL_STEP_COMPLETED,
                message="Parallel multi-agent step completed",
                extra={
                    "actions": actions,
                    "terminated": all_done,
                    "truncated": all_truncated,
                },
            )

        except Exception as e:
            _OP_LOGGER.error(f"Failed to step environment: {e}", exc_info=True)
            self._status_bar.showMessage(f"Step failed: {e}", 5000)


__all__ = ["ParallelMultiAgentHandler"]
