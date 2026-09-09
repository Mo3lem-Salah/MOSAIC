"""Google Research Football (GRF) environment adapter.

Bridges the GRF ``gfootball.env.create_environment()`` factory (old
``gym<=0.21.0`` API) to MOSAIC's :class:`EnvironmentAdapter` contract.

GRF does *not* use ``gymnasium.make()``; instead it has its own
``create_environment()`` factory that applies internal wrappers
(``Simple115StateWrapper``, ``SingleAgentObservationWrapper``, etc.).
This adapter bypasses ``super().load()`` entirely, mirroring the
approach used by the SMAC and ViZDoom adapters.

Paper: Kurach et al. (2020). "Google Research Football: A Novel
       Reinforcement Learning Environment"
Source: 3rd_party/environments/football/
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Mapping

import gymnasium as gym  # type: ignore[import]
import numpy as np

from gym_gui.core.adapters.base import (
    AdapterContext,
    AdapterStep,
    AgentSnapshot,
    EnvironmentAdapter,
    StepState,
)
from gym_gui.core.enums import ControlMode, GameId, RenderMode, SteppingParadigm
from gym_gui.core.ui.game_config.game_configs import GFootballConfig
from gym_gui.logging_config.log_constants import (
    LOG_GFOOTBALL_ENV_CLOSED,
    LOG_GFOOTBALL_ENV_CREATED,
    LOG_GFOOTBALL_ENV_RESET,
    LOG_GFOOTBALL_GOAL,
    LOG_GFOOTBALL_INIT_ERROR,
    LOG_GFOOTBALL_RENDER_ERROR,
    LOG_GFOOTBALL_RESET_ERROR,
    LOG_GFOOTBALL_STEP_ERROR,
    LOG_GFOOTBALL_STEP_SUMMARY,
)

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy import guard
# ---------------------------------------------------------------------------

def _ensure_gfootball() -> None:
    """Raise a helpful error if the ``gfootball`` package is not installed."""
    try:
        import gfootball  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Google Research Football is not installed. "
            "Install via: pip install -e 3rd_party/environments/football/  "
            "(requires cmake, libsdl2-dev, and a C++ compiler)."
        ) from exc


# ---------------------------------------------------------------------------
# Default action names (19 actions in the 'default' action set)
# ---------------------------------------------------------------------------

GRF_DEFAULT_ACTIONS: List[str] = [
    "idle",
    "left",
    "top_left",
    "top",
    "top_right",
    "right",
    "bottom_right",
    "bottom",
    "bottom_left",
    "long_pass",
    "high_pass",
    "short_pass",
    "shot",
    "sprint",
    "release_direction",
    "release_sprint",
    "sliding",
    "dribble",
    "release_dribble",
]


# ===================================================================
# Base adapter
# ===================================================================

class GFootballAdapter(EnvironmentAdapter[Any, Any]):
    """Adapter for Google Research Football environments.

    GRF uses the old ``gym<=0.21.0`` 4-tuple step API and its own
    ``create_environment()`` factory.  This adapter translates that to
    MOSAIC's modern 5-tuple contract, constructs equivalent
    ``gymnasium.spaces`` objects, and extracts RGB frames for the
    render view.
    """

    default_render_mode = RenderMode.RGB_ARRAY
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    # HUMAN_ONLY is supported: GRF exposes a flat Discrete(19) action set, so a
    # human can drive the controlled player directly through MOSAIC's keyboard
    # path (see _STANDARD_GRF_ACTIONS in gym_gui/controllers/human_input.py).
    # Football is real-time, so GFootballInteractionController idle-ticks the
    # environment with action 0 (idle) whenever no key is pressed.
    supported_control_modes = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COOP,
    )

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: GFootballConfig | None = None,
    ) -> None:
        super().__init__(context)

        if config is None:
            config = GFootballConfig()
        if not isinstance(config, GFootballConfig):
            config = GFootballConfig()

        self._config: GFootballConfig = config
        self._grf_env: Any = None
        self._n_agents: int = (
            config.number_of_left_players_agent_controls
            + config.number_of_right_players_agent_controls
        )
        self._n_left: int = config.number_of_left_players_agent_controls
        self._n_right: int = config.number_of_right_players_agent_controls
        self._n_actions: int = 0
        self._step_counter: int = 0
        self._episode_step: int = 0
        self._episode_return: float = 0.0
        self._last_frame: np.ndarray | None = None
        self._action_space: gym.Space[Any] | None = None
        self._observation_space: gym.Space[Any] | None = None

    # ------------------------------------------------------------------
    # Stepping paradigm
    # ------------------------------------------------------------------

    @property
    def stepping_paradigm(self) -> SteppingParadigm:  # type: ignore[override]
        if self._n_agents > 1:
            return SteppingParadigm.SIMULTANEOUS
        return SteppingParadigm.SINGLE_AGENT

    # ------------------------------------------------------------------
    # Spaces (synthetic, constructed during load)
    # ------------------------------------------------------------------

    @property
    def action_space(self) -> gym.Space[Any]:
        if self._action_space is not None:
            return self._action_space
        # Fallback before load
        return gym.spaces.Discrete(19)

    @property
    def observation_space(self) -> gym.Space[Any]:
        if self._observation_space is not None:
            return self._observation_space
        # Fallback before load (simple115v2 representation)
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(115,), dtype=np.float32)

    # ------------------------------------------------------------------
    # Lifecycle: load
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Create the GRF environment via ``gfootball.env.create_environment()``."""
        _ensure_gfootball()

        # MOSAIC renders GRF frames inside the Render View, so the engine's own
        # SDL top-level window would be a second, redundant OS window. The
        # patched engine honours GFOOTBALL_HIDDEN_WINDOW by creating the window
        # with SDL_WINDOW_HIDDEN, which keeps the OpenGL context (and therefore
        # rgb_array frames) while leaving the window unmapped.
        #
        # The value is configured in .env (see GFOOTBALL_HIDDEN_WINDOW there);
        # "1" hides the window, "0" restores the upstream behaviour of showing
        # it. It must be exported before the engine starts up.
        hidden_window = os.getenv("GFOOTBALL_HIDDEN_WINDOW", "1")
        os.environ["GFOOTBALL_HIDDEN_WINDOW"] = hidden_window

        import gfootball.env as grf_env

        # Derive the bare scenario name from the GameId value
        # e.g. "gfootball-academy_empty_goal" -> "academy_empty_goal"
        env_name = self.id.removeprefix("gfootball-")

        other_opts: Dict[str, Any] = {}
        if self._config.action_set != "default":
            other_opts["action_set"] = self._config.action_set
        if self._config.seed is not None:
            other_opts["game_engine_random_seed"] = self._config.seed

        try:
            self._grf_env = grf_env.create_environment(
                env_name=env_name,
                stacked=self._config.stacked,
                representation=self._config.representation,
                rewards=self._config.rewards,
                render=self._config.render,
                number_of_left_players_agent_controls=self._n_left,
                number_of_right_players_agent_controls=self._n_right,
                other_config_options=other_opts,
            )
        except Exception as exc:
            self.log_constant(
                LOG_GFOOTBALL_INIT_ERROR,
                exc_info=exc,
                extra={
                    "env_name": env_name,
                    "representation": self._config.representation,
                },
            )
            raise

        # Derive action/observation space info from the old gym env
        old_action_space = self._grf_env.action_space
        if hasattr(old_action_space, "nvec"):
            # MultiDiscrete (multi-agent)
            self._n_actions = int(old_action_space.nvec[0])
            self._action_space = gym.spaces.MultiDiscrete(
                [int(n) for n in old_action_space.nvec]
            )
        elif hasattr(old_action_space, "n"):
            # Discrete (single-agent)
            self._n_actions = int(old_action_space.n)
            self._action_space = gym.spaces.Discrete(self._n_actions)
        else:
            self._n_actions = 19
            self._action_space = gym.spaces.Discrete(19)

        # Build observation space from old gym space
        old_obs_space = self._grf_env.observation_space
        if hasattr(old_obs_space, "shape") and old_obs_space.shape is not None:
            low = old_obs_space.low if hasattr(old_obs_space, "low") else -np.inf
            high = old_obs_space.high if hasattr(old_obs_space, "high") else np.inf
            self._observation_space = gym.spaces.Box(
                low=low, high=high,
                shape=old_obs_space.shape,
                dtype=old_obs_space.dtype if hasattr(old_obs_space, "dtype") else np.float32,
            )
        else:
            self._observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(115,), dtype=np.float32,
            )

        self.log_constant(
            LOG_GFOOTBALL_ENV_CREATED,
            extra={
                "env_name": env_name,
                "representation": self._config.representation,
                "n_agents": self._n_agents,
                "n_actions": self._n_actions,
                "action_set": self._config.action_set,
                "render_enabled": self._config.render,
            },
        )

    # ------------------------------------------------------------------
    # Lifecycle: reset
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> AdapterStep[Any]:
        if self._grf_env is None:
            raise RuntimeError(f"Adapter '{self.id}' has not been loaded.")

        try:
            # Old gym API: reset() returns observation only (no info)
            observation = self._grf_env.reset()
        except Exception as exc:
            self.log_constant(
                LOG_GFOOTBALL_RESET_ERROR,
                exc_info=exc,
                extra={"env_id": self.id},
            )
            raise

        self._step_counter = 0
        self._episode_return = 0.0
        self._episode_step = 0

        info: Dict[str, Any] = {
            "episode_step": 0,
        }

        self.log_constant(
            LOG_GFOOTBALL_ENV_RESET,
            extra={
                "env_id": self.id,
                "seed": seed if seed is not None else "None",
            },
        )

        return self._package_step(observation, 0.0, False, False, info)

    # ------------------------------------------------------------------
    # Lifecycle: step
    # ------------------------------------------------------------------

    def step(self, action: Any) -> AdapterStep[Any]:
        if self._grf_env is None:
            raise RuntimeError(f"Adapter '{self.id}' has not been loaded.")

        try:
            # Old gym API: step returns 4-tuple (obs, reward, done, info)
            observation, reward, done, info = self._grf_env.step(action)
        except Exception as exc:
            self.log_constant(
                LOG_GFOOTBALL_STEP_ERROR,
                exc_info=exc,
                extra={
                    "env_id": self.id,
                    "step": self._step_counter,
                    "action": repr(action),
                },
            )
            raise

        # Convert old gym API to 5-tuple
        terminated = bool(done)
        truncated = False

        # Handle multi-agent reward (numpy array) vs scalar
        if isinstance(reward, np.ndarray):
            scalar_reward = float(np.sum(reward))
            agent_rewards = reward.tolist()
        else:
            scalar_reward = float(reward)
            agent_rewards = [scalar_reward]

        self._step_counter += 1
        self._episode_step += 1
        self._episode_return += scalar_reward

        # Build info dict
        info_dict: Dict[str, Any] = dict(info) if isinstance(info, Mapping) else {}
        info_dict["episode_step"] = self._episode_step
        info_dict["episode_score"] = self._episode_return
        info_dict["agent_rewards"] = agent_rewards

        # Detect goals
        score_reward = info_dict.get("score_reward", 0)
        if score_reward != 0:
            self.log_constant(
                LOG_GFOOTBALL_GOAL,
                extra={
                    "env_id": self.id,
                    "score_reward": score_reward,
                    "step": self._step_counter,
                },
            )

        self.log_constant(
            LOG_GFOOTBALL_STEP_SUMMARY,
            message=(
                f"action {self.get_action_name(action)} "
                f"| reward {scalar_reward:.2f} | step {self._step_counter}"
            ),
            extra={
                "env_id": self.id,
                "action": repr(action),
                "action_name": self.get_action_name(action),
                "reward": scalar_reward,
                "terminated": terminated,
                "step": self._step_counter,
            },
        )

        return self._package_step(observation, scalar_reward, terminated, truncated, info_dict)

    # ------------------------------------------------------------------
    # Lifecycle: render
    # ------------------------------------------------------------------

    def render(self) -> Dict[str, Any] | None:
        if self._grf_env is None:
            return None

        if not self._config.render:
            return {"mode": RenderMode.RGB_ARRAY.value, "rgb": self._last_frame}

        try:
            frame = self._grf_env.render(mode="rgb_array")
            if isinstance(frame, np.ndarray) and frame.ndim == 3:
                # Correct GRF's channel order. Despite the "rgb_array" name the
                # returned frame is BGR: the engine already reads the
                # framebuffer as GL_RGB (opengl_renderer3d.cpp SwapBuffers),
                # then FootballEnvCore.render() applies a second, historical
                # b,g,r -> r,g,b swap on top of it. The two combine to leave R
                # and B transposed, which renders yellow shirts (224, 204, 94)
                # as cyan (94, 204, 224) in the Render View.
                #
                # Flip the last axis back so downstream consumers, which all
                # treat this as QImage.Format_RGB888, get true RGB.
                self._last_frame = np.ascontiguousarray(frame[..., ::-1])
        except Exception as exc:
            self.log_constant(
                LOG_GFOOTBALL_RENDER_ERROR,
                exc_info=exc,
                extra={"env_id": self.id, "step": self._step_counter},
            )

        return {
            "mode": RenderMode.RGB_ARRAY.value,
            "rgb": self._last_frame,
            "game_id": self.id,
            "num_agents": self._n_agents,
            "step": self._step_counter,
        }

    # ------------------------------------------------------------------
    # Lifecycle: close
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._grf_env is not None:
            self.log_constant(
                LOG_GFOOTBALL_ENV_CLOSED,
                extra={"env_id": self.id},
            )
            try:
                self._grf_env.close()
            except Exception:
                pass  # Best-effort cleanup
            self._grf_env = None
            self._last_frame = None

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def build_step_state(
        self, observation: Any, info: Mapping[str, Any],
    ) -> StepState:
        info_dict = dict(info) if isinstance(info, Mapping) else {}
        agent_rewards = info_dict.get("agent_rewards", [0.0] * self._n_agents)

        agent_snapshots: List[AgentSnapshot] = []
        for i in range(self._n_agents):
            side = "left" if i < self._n_left else "right"
            idx = i if i < self._n_left else i - self._n_left
            r = agent_rewards[i] if i < len(agent_rewards) else 0.0
            agent_snapshots.append(
                AgentSnapshot(
                    name=f"{side}_agent_{idx}",
                    role="active",
                    info={"reward": r, "side": side},
                )
            )

        return StepState(
            active_agent=None,  # simultaneous stepping
            agents=tuple(agent_snapshots),
            metrics={
                "step_count": self._step_counter,
                "episode_return": self._episode_return,
                "score_reward": info_dict.get("score_reward", 0),
                "num_agents": self._n_agents,
            },
            environment={
                "scenario": self.id.removeprefix("gfootball-"),
                "family": "gfootball",
                "paradigm": "simultaneous" if self._n_agents > 1 else "single_agent",
                "representation": self._config.representation,
            },
            raw=info_dict,
        )

    # ------------------------------------------------------------------
    # Multi-agent helpers
    # ------------------------------------------------------------------

    @property
    def num_agents(self) -> int:
        return self._n_agents

    @property
    def num_actions(self) -> int:
        return self._n_actions

    def get_action_name(self, action: Any) -> str:
        """Return the human-readable GRF action name for ``action``.

        GRF actions are bare integers in ``Discrete(19)``, so raw logs read
        "action 12" with no indication that it means "shot". This resolves the
        index against :data:`GRF_DEFAULT_ACTIONS` so telemetry and the Runtime
        Log stay readable.

        Multi-agent steps pass a sequence of per-player actions; those are
        resolved element-wise and joined.
        """
        if isinstance(action, (list, tuple, np.ndarray)):
            return ",".join(self.get_action_name(a) for a in action)
        try:
            index = int(action)
        except (TypeError, ValueError):
            return f"unknown_{action!r}"
        if 0 <= index < len(GRF_DEFAULT_ACTIONS):
            return GRF_DEFAULT_ACTIONS[index]
        return f"unknown_{index}"


# ===================================================================
# Per-scenario concrete subclasses
# ===================================================================
# Each subclass sets ``id`` to the corresponding GameId value so
# that the adapter factory can look it up via the registry dict.

class GRF11vs11EasyAdapter(GFootballAdapter):
    id = GameId.GRF_11V11_EASY.value

class GRF11vs11Adapter(GFootballAdapter):
    id = GameId.GRF_11V11.value

class GRF11vs11HardAdapter(GFootballAdapter):
    id = GameId.GRF_11V11_HARD.value

class GRF1vs1EasyAdapter(GFootballAdapter):
    id = GameId.GRF_1V1_EASY.value

class GRF5vs5Adapter(GFootballAdapter):
    id = GameId.GRF_5V5.value

class GRFAcademyEmptyGoalCloseAdapter(GFootballAdapter):
    id = GameId.GRF_ACADEMY_EMPTY_GOAL_CLOSE.value

class GRFAcademyEmptyGoalAdapter(GFootballAdapter):
    id = GameId.GRF_ACADEMY_EMPTY_GOAL.value

class GRFAcademyRunToScoreAdapter(GFootballAdapter):
    id = GameId.GRF_ACADEMY_RUN_TO_SCORE.value

class GRFAcademyRunToScoreWithKeeperAdapter(GFootballAdapter):
    id = GameId.GRF_ACADEMY_RUN_TO_SCORE_WITH_KEEPER.value

class GRFAcademyPassAndShootAdapter(GFootballAdapter):
    id = GameId.GRF_ACADEMY_PASS_AND_SHOOT.value

class GRFAcademyRunPassAndShootAdapter(GFootballAdapter):
    id = GameId.GRF_ACADEMY_RUN_PASS_AND_SHOOT.value

class GRFAcademy3vs1WithKeeperAdapter(GFootballAdapter):
    id = GameId.GRF_ACADEMY_3V1_WITH_KEEPER.value

class GRFAcademyCornerAdapter(GFootballAdapter):
    id = GameId.GRF_ACADEMY_CORNER.value

class GRFAcademyCounterattackEasyAdapter(GFootballAdapter):
    id = GameId.GRF_ACADEMY_COUNTERATTACK_EASY.value

class GRFAcademyCounterattackHardAdapter(GFootballAdapter):
    id = GameId.GRF_ACADEMY_COUNTERATTACK_HARD.value

class GRFAcademySingleGoalVsLazyAdapter(GFootballAdapter):
    id = GameId.GRF_ACADEMY_SINGLE_GOAL_VS_LAZY.value


# ===================================================================
# Adapter registry
# ===================================================================

GFOOTBALL_ADAPTERS: Dict[GameId, type[GFootballAdapter]] = {
    GameId.GRF_11V11_EASY: GRF11vs11EasyAdapter,
    GameId.GRF_11V11: GRF11vs11Adapter,
    GameId.GRF_11V11_HARD: GRF11vs11HardAdapter,
    GameId.GRF_1V1_EASY: GRF1vs1EasyAdapter,
    GameId.GRF_5V5: GRF5vs5Adapter,
    GameId.GRF_ACADEMY_EMPTY_GOAL_CLOSE: GRFAcademyEmptyGoalCloseAdapter,
    GameId.GRF_ACADEMY_EMPTY_GOAL: GRFAcademyEmptyGoalAdapter,
    GameId.GRF_ACADEMY_RUN_TO_SCORE: GRFAcademyRunToScoreAdapter,
    GameId.GRF_ACADEMY_RUN_TO_SCORE_WITH_KEEPER: GRFAcademyRunToScoreWithKeeperAdapter,
    GameId.GRF_ACADEMY_PASS_AND_SHOOT: GRFAcademyPassAndShootAdapter,
    GameId.GRF_ACADEMY_RUN_PASS_AND_SHOOT: GRFAcademyRunPassAndShootAdapter,
    GameId.GRF_ACADEMY_3V1_WITH_KEEPER: GRFAcademy3vs1WithKeeperAdapter,
    GameId.GRF_ACADEMY_CORNER: GRFAcademyCornerAdapter,
    GameId.GRF_ACADEMY_COUNTERATTACK_EASY: GRFAcademyCounterattackEasyAdapter,
    GameId.GRF_ACADEMY_COUNTERATTACK_HARD: GRFAcademyCounterattackHardAdapter,
    GameId.GRF_ACADEMY_SINGLE_GOAL_VS_LAZY: GRFAcademySingleGoalVsLazyAdapter,
}


__all__ = [
    "GFootballAdapter",
    "GFOOTBALL_ADAPTERS",
    "GRF_DEFAULT_ACTIONS",
    "GRF11vs11EasyAdapter",
    "GRF11vs11Adapter",
    "GRF11vs11HardAdapter",
    "GRF1vs1EasyAdapter",
    "GRF5vs5Adapter",
    "GRFAcademyEmptyGoalCloseAdapter",
    "GRFAcademyEmptyGoalAdapter",
    "GRFAcademyRunToScoreAdapter",
    "GRFAcademyRunToScoreWithKeeperAdapter",
    "GRFAcademyPassAndShootAdapter",
    "GRFAcademyRunPassAndShootAdapter",
    "GRFAcademy3vs1WithKeeperAdapter",
    "GRFAcademyCornerAdapter",
    "GRFAcademyCounterattackEasyAdapter",
    "GRFAcademyCounterattackHardAdapter",
    "GRFAcademySingleGoalVsLazyAdapter",
]
