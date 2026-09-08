"""Jidi Olympics Wrestling (sumo) environment adapter.

Bridges the Competition_Olympics-Wrestling environment to MOSAIC's
:class:`EnvironmentAdapter` contract. Two elastic ball agents compete
on a circular arena, attempting to push each other off the platform.

The environment uses a custom multi-agent API with continuous actions
(force + steering angle). This adapter translates the 5-element step
return ``(obs_list, reward_list, done, info_before, info_after)`` to
MOSAIC's standard 5-tuple contract, constructs equivalent
``gymnasium.spaces`` objects, and extracts pygame frames for rendering.

Source: 3rd_party/environments/Competition_Olympics-Wrestling/
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Mapping

import gymnasium as gym
import numpy as np

from gym_gui.core.ui.game_config.game_configs import OlympicsWrestlingConfig
from gym_gui.core.adapters.base import (
    AdapterContext,
    AdapterStep,
    AgentSnapshot,
    EnvironmentAdapter,
    StepState,
)
from gym_gui.core.enums import ControlMode, GameId, RenderMode, SteppingParadigm
from gym_gui.logging_config.log_constants import (
    LOG_OLYMPICS_ENV_CLOSED,
    LOG_OLYMPICS_ENV_CREATED,
    LOG_OLYMPICS_ENV_RESET,
    LOG_OLYMPICS_INIT_ERROR,
    LOG_OLYMPICS_KNOCKOUT,
    LOG_OLYMPICS_RENDER_ERROR,
    LOG_OLYMPICS_RESET_ERROR,
    LOG_OLYMPICS_STEP_ERROR,
    LOG_OLYMPICS_STEP_SUMMARY,
)

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path to the cloned environment
# ---------------------------------------------------------------------------
_OLYMPICS_ROOT = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir, os.pardir, os.pardir,
        "3rd_party", "environments", "Competition_Olympics-Wrestling",
    )
)


def _ensure_olympics() -> None:
    """Add the Olympics-Wrestling package to sys.path if needed.

    The environment's internal imports use bare module names:
    - ``from env.chooseenv import make``
    - ``from olympics_engine.generator import create_scenario``
    - ``from utils.box import Box``

    All resolve relative to the repo root, so that directory must be
    on ``sys.path``.  Additionally, ``run_log.py`` line 9 adds
    ``./olympics_engine`` for the engine's own internal imports.
    """
    if _OLYMPICS_ROOT not in sys.path:
        sys.path.insert(0, _OLYMPICS_ROOT)
    # olympics_engine subpackages also use bare imports among themselves
    engine_path = os.path.join(_OLYMPICS_ROOT, "olympics_engine")
    if engine_path not in sys.path:
        sys.path.insert(0, engine_path)
    try:
        import olympics_engine  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Olympics-Wrestling environment not found. "
            f"Expected at: {_OLYMPICS_ROOT}\n"
            "Clone via: git clone https://github.com/jidiai/Competition_Olympics-Wrestling.git "
            "3rd_party/environments/Competition_Olympics-Wrestling/"
        ) from exc


# ---------------------------------------------------------------------------
# Action presets for keyboard control (discrete -> continuous mapping)
# ---------------------------------------------------------------------------
OLYMPICS_ACTION_PRESETS: Dict[int, List[float]] = {
    0: [0.0, 0.0],       # idle
    1: [150.0, 0.0],     # forward
    2: [-80.0, 0.0],     # backward
    3: [0.0, -20.0],     # turn left
    4: [0.0, 20.0],      # turn right
    5: [150.0, -15.0],   # forward-left
    6: [150.0, 15.0],    # forward-right
    7: [200.0, 0.0],     # sprint forward
    8: [-80.0, -20.0],   # backward-left
    9: [-80.0, 20.0],    # backward-right
}

OLYMPICS_ACTION_NAMES: List[str] = [
    "idle",
    "forward",
    "backward",
    "turn_left",
    "turn_right",
    "forward_left",
    "forward_right",
    "sprint",
    "backward_left",
    "backward_right",
]


# ===================================================================
# Adapter
# ===================================================================

class OlympicsWrestlingAdapter(EnvironmentAdapter[Any, Any]):
    """Adapter for the Jidi Olympics Wrestling (sumo) environment.

    Two elastic balls on a circular arena. Each agent applies continuous
    force [-100, 200] and steering angle [-30, 30] per step. The agent
    that pushes the other off the arena wins.
    """

    default_render_mode = RenderMode.RGB_ARRAY
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    supported_control_modes = (ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP)

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: OlympicsWrestlingConfig | None = None,
    ) -> None:
        super().__init__(context)

        if config is None:
            config = OlympicsWrestlingConfig()
        if not isinstance(config, OlympicsWrestlingConfig):
            config = OlympicsWrestlingConfig()

        self._config = config
        self._env: Any = None
        self._n_agents: int = 2
        self._step_counter: int = 0
        self._episode_step: int = 0
        self._episode_return: float = 0.0
        self._last_frame: np.ndarray | None = None

        # Action space: [force, angle] per agent
        self._action_space: gym.Space[Any] | None = None
        # Observation space: 40x40 color-indexed grid per agent
        self._observation_space: gym.Space[Any] | None = None

    # ------------------------------------------------------------------
    # Stepping paradigm
    # ------------------------------------------------------------------

    @property
    def stepping_paradigm(self) -> SteppingParadigm:  # type: ignore[override]
        return SteppingParadigm.SIMULTANEOUS

    # ------------------------------------------------------------------
    # Spaces
    # ------------------------------------------------------------------

    @property
    def action_space(self) -> gym.Space[Any]:
        if self._action_space is not None:
            return self._action_space
        return gym.spaces.Box(
            low=np.array([-100.0, -30.0], dtype=np.float32),
            high=np.array([200.0, 30.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

    @property
    def observation_space(self) -> gym.Space[Any]:
        if self._observation_space is not None:
            return self._observation_space
        return gym.spaces.Box(low=0, high=10, shape=(40, 40), dtype=np.float64)

    # ------------------------------------------------------------------
    # Lifecycle: load
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Create the Olympics Wrestling environment."""
        _ensure_olympics()

        try:
            from env.chooseenv import make
            self._env = make("olympics-wrestling", seed=self._config.seed)
        except Exception as exc:
            self.log_constant(
                LOG_OLYMPICS_INIT_ERROR,
                exc_info=exc,
                extra={"seed": self._config.seed},
            )
            raise

        # Build gymnasium-compatible spaces
        self._action_space = gym.spaces.Box(
            low=np.array([-100.0, -30.0], dtype=np.float32),
            high=np.array([200.0, 30.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )
        self._observation_space = gym.spaces.Box(
            low=0, high=10, shape=(40, 40), dtype=np.int32,
        )

        self.log_constant(
            LOG_OLYMPICS_ENV_CREATED,
            extra={
                "max_step": self._config.max_step,
                "seed": self._config.seed,
                "n_agents": self._n_agents,
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
        if self._env is None:
            raise RuntimeError("Adapter has not been loaded.")

        if seed is not None:
            self._env.set_seed(seed)

        obs_list = self._env.reset()

        self._step_counter = 0
        self._episode_step = 0
        self._episode_return = 0.0

        # Extract the 40x40 grid from agent 0
        observation = self._extract_obs(obs_list, agent_idx=0)

        info: Dict[str, Any] = {
            "episode_step": 0,
            "all_observes": obs_list,
        }

        self.log_constant(
            LOG_OLYMPICS_ENV_RESET,
            extra={"seed": seed if seed is not None else "None"},
        )

        return self._package_step(observation, 0.0, False, False, info)

    # ------------------------------------------------------------------
    # Lifecycle: step
    # ------------------------------------------------------------------

    def step(self, action: Any) -> AdapterStep[Any]:
        if self._env is None:
            raise RuntimeError("Adapter has not been loaded.")

        # Translate action to the environment's joint action format:
        # [[[force_0], [angle_0]], [[force_1], [angle_1]]]
        joint_action = self._build_joint_action(action)

        try:
            obs_list, reward, done, info_before, info_after = self._env.step(joint_action)
        except Exception as exc:
            self.log_constant(
                LOG_OLYMPICS_STEP_ERROR,
                exc_info=exc,
                extra={
                    "step": self._step_counter,
                    "action": repr(action),
                },
            )
            raise

        terminated = bool(done)
        truncated = False

        # Reward is per-agent: [r0, r1]
        if isinstance(reward, (list, np.ndarray)) and len(reward) >= 2:
            scalar_reward = float(reward[0])
            agent_rewards = [float(r) for r in reward]
        else:
            scalar_reward = float(reward) if reward is not None else 0.0
            agent_rewards = [scalar_reward, 0.0]

        self._step_counter += 1
        self._episode_step += 1
        self._episode_return += scalar_reward

        observation = self._extract_obs(obs_list, agent_idx=0)

        info_dict: Dict[str, Any] = {
            "episode_step": self._episode_step,
            "episode_score": self._episode_return,
            "agent_rewards": agent_rewards,
            "all_observes": obs_list,
        }

        # Detect knockout
        if terminated and any(r > 0 for r in agent_rewards):
            winner = 0 if agent_rewards[0] > agent_rewards[1] else 1
            info_dict["knockout"] = True
            info_dict["winner"] = winner
            self.log_constant(
                LOG_OLYMPICS_KNOCKOUT,
                extra={
                    "winner": winner,
                    "step": self._step_counter,
                    "agent_rewards": agent_rewards,
                },
            )

        self.log_constant(
            LOG_OLYMPICS_STEP_SUMMARY,
            extra={
                "action": repr(action),
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
        """Render the wrestling arena and extract an RGB frame.

        The upstream ``wrestling.render()`` (lines 186-234) and
        ``viewer.py`` pass raw numpy arrays to ``pygame.draw.circle``
        and ``pygame.draw.line``, which worked in ``pygame==2.0.2``
        but breaks in ``pygame-ce>=2.5``.  Since we cannot modify
        upstream code and cannot downgrade pygame (other envs depend
        on pygame-ce), we replicate the drawing logic here with
        pygame-ce-compatible tuple arguments.

        Drawing order follows ``wrestling.render()`` exactly:
        1. ``viewer.draw_background()`` -- white fill (line 198)
        2. ``viewer.draw_map(w)`` -- arcs and walls (lines 203-204)
        3. ``viewer.draw_ball()`` -- agent circles (line 209)
        4. ``viewer.draw_obs()`` -- observation boundaries (lines 210-211)
        5. ``viewer.draw_view()`` -- 40x40 grid side panels (lines 213-216)
        6. ``viewer.draw_direction()`` -- acceleration vectors (line 222)
        """
        if self._env is None:
            return None

        try:
            import pygame
            from olympics_engine.tools.settings import COLORS, IDX_TO_COLOR

            core = self._env.env_core

            # Ensure viewer surface exists (wrestling.render lines 192-194).
            # NOTE: We intentionally avoid ``core.viewer.set_mode()`` here,
            # since it calls ``pygame.display.set_mode(...)`` which opens a
            # real OS window (a popup) outside MOSAIC's own view. We only
            # need the pixels, so we create an off-screen ``pygame.Surface``
            # of the same size and assign it directly as the viewer's
            # drawing target instead.
            if not core.display_mode:
                core.viewer.background = pygame.Surface(core.viewer.WIN_SIZE)
                core.display_mode = True

            # 1. White background (line 198)
            core.viewer.draw_background()

            # 2. Map objects: arcs and walls (lines 203-204)
            for w in core.map['objects']:
                if w.type == 'arc':
                    pygame.draw.arc(
                        core.viewer.background, COLORS[w.color],
                        w.init_pos, w.start_radian, w.end_radian, w.width,
                    )
                else:
                    s, e = w.init_pos
                    pygame.draw.line(
                        surface=core.viewer.background,
                        color=COLORS[w.color],
                        start_pos=(float(s[0]), float(s[1])),
                        end_pos=(float(e[0]), float(e[1])),
                        width=w.width,
                    )

            # 3. Agent circles (viewer.draw_ball lines 36-45)
            for i in range(len(core.agent_pos)):
                t = core.agent_pos[i]
                r = core.agent_list[i].r
                color = core.agent_list[i].color
                center = (int(t[0]), int(t[1]))
                pygame.draw.circle(core.viewer.background, COLORS[color], center, int(r), 0)
                pygame.draw.circle(core.viewer.background, COLORS['black'], center, 2, 2)

            # 4. Observation boundaries (viewer.draw_obs lines 85-93)
            if core.draw_obs and hasattr(core, 'obs_boundary'):
                points = core.obs_boundary
                for b in range(min(len(points), len(core.agent_list))):
                    if points[b] is not None:
                        converted = [(int(p[0]), int(p[1])) for p in points[b]]
                        if len(converted) >= 2:
                            pygame.draw.lines(
                                core.viewer.background,
                                COLORS[core.agent_list[b].color],
                                True, converted, 2,
                            )

            # 5. Side-panel 40x40 grid views (viewer.draw_view lines 142-201)
            if core.draw_obs and hasattr(core, 'obs_list') and len(core.obs_list) > 0:
                grid_w, grid_h = 2, 2
                x_start = 470
                y_start = 10
                gap = 130
                for agent_idx in range(len(core.obs_list)):
                    matrix = core.obs_list[agent_idx]
                    if matrix is None:
                        continue
                    obs_w, obs_h = matrix.shape[0], matrix.shape[1]
                    y = y_start
                    for row in matrix:
                        x = x_start
                        for item in row:
                            pygame.draw.rect(
                                core.viewer.background,
                                COLORS[IDX_TO_COLOR[int(item)]],
                                [x, y, grid_w, grid_h],
                            )
                            x += grid_w
                        y += grid_h
                    # Border
                    pygame.draw.lines(
                        core.viewer.background,
                        color=COLORS[core.agent_list[agent_idx].color],
                        closed=True,
                        points=[
                            (x_start, y_start),
                            (x_start, y_start + obs_h * grid_h),
                            (x_start + obs_w * grid_w, y_start + obs_h * grid_h),
                            (x_start + obs_w * grid_w, y_start),
                        ],
                        width=1,
                    )
                    x_start += gap

            # 6. Direction vectors (viewer.draw_direction lines 48-63)
            if hasattr(core, 'agent_accel'):
                for i in range(len(core.agent_pos)):
                    a_x, a_y = core.agent_accel[i]
                    if a_x != 0 or a_y != 0:
                        t = core.agent_pos[i]
                        start = (float(t[0]), float(t[1]))
                        end = (float(t[0] + a_x / 5), float(t[1] + a_y / 5))
                        pygame.draw.line(
                            core.viewer.background,
                            color=(0, 0, 0),
                            start_pos=start,
                            end_pos=end,
                            width=2,
                        )

            # Extract RGB array (same as _build_minimap line 100)
            frame = pygame.surfarray.array3d(core.viewer.background).swapaxes(0, 1)
            if isinstance(frame, np.ndarray) and frame.ndim == 3:
                self._last_frame = frame

        except Exception as exc:
            self.log_constant(
                LOG_OLYMPICS_RENDER_ERROR,
                exc_info=exc,
                extra={"step": self._step_counter},
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
        if self._env is not None:
            self.log_constant(
                LOG_OLYMPICS_ENV_CLOSED,
                extra={"step": self._step_counter},
            )
            self._env = None
            self._last_frame = None

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def build_step_state(
        self, observation: Any, info: Mapping[str, Any],
    ) -> StepState:
        info_dict = dict(info) if isinstance(info, Mapping) else {}
        agent_rewards = info_dict.get("agent_rewards", [0.0, 0.0])

        agent_snapshots: List[AgentSnapshot] = []
        for i in range(self._n_agents):
            r = agent_rewards[i] if i < len(agent_rewards) else 0.0
            agent_snapshots.append(
                AgentSnapshot(
                    name=f"player_{i}",
                    role="active",
                    info={"reward": r},
                )
            )

        return StepState(
            active_agent=None,
            agents=tuple(agent_snapshots),
            metrics={
                "step_count": self._step_counter,
                "episode_return": self._episode_return,
                "num_agents": self._n_agents,
                "knockout": info_dict.get("knockout", False),
            },
            environment={
                "scenario": "wrestling",
                "family": "olympics",
                "paradigm": "simultaneous",
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
        return len(OLYMPICS_ACTION_PRESETS)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_obs(obs_list: Any, agent_idx: int = 0) -> np.ndarray:
        """Extract the 40x40 grid from the observation list.

        The env returns observations through two layers:
        1. ``wrestling._build_from_raw_obs()`` produces:
           ``[{"agent_obs": np.ndarray(40,40), "id": "team_0"}, ...]``
        2. ``OlympicsWrestling.get_all_observes()`` wraps that into:
           ``[{"obs": <above dict>, "controlled_player_index": 0}, ...]``

        So the actual 40x40 grid lives at ``obs_list[i]["obs"]["agent_obs"]``.
        """
        if isinstance(obs_list, list) and len(obs_list) > agent_idx:
            obs_dict = obs_list[agent_idx]
            if isinstance(obs_dict, dict):
                # Layer 2: get_all_observes wrapping
                inner = obs_dict.get("obs")
                if isinstance(inner, dict):
                    # Layer 1: _build_from_raw_obs dict
                    raw = inner.get("agent_obs")
                    if isinstance(raw, np.ndarray):
                        return raw
                # Fallback: direct agent_obs (if called on inner list)
                raw = obs_dict.get("agent_obs")
                if isinstance(raw, np.ndarray):
                    return raw
        return np.zeros((40, 40), dtype=np.float64)

    def _build_joint_action(self, action: Any) -> list:
        """Convert adapter action to environment's joint action format.

        The environment's ``OlympicsWrestling.decode()`` expects each agent's
        action as ``[force_array, angle_array]`` where each is a numpy array
        of shape ``(1,)``.  This matches what ``submission.py``'s
        ``my_controller`` returns: ``[Box.sample(), Box.sample()]`` producing
        ``[np.array([force]), np.array([angle])]``.

        The full joint_action is:
        ``[[np.array([f0]), np.array([a0])], [np.array([f1]), np.array([a1])]]``

        The adapter accepts:
        - int: discrete action preset index (for keyboard/human control)
        - np.ndarray shape (2,): [force, angle] for agent 0, agent 1 idles
        - np.ndarray shape (4,): [force_0, angle_0, force_1, angle_1]
        - list of 2 arrays: [[force_0, angle_0], [force_1, angle_1]]
        """
        idle = [np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)]

        if isinstance(action, (int, np.integer)):
            preset = OLYMPICS_ACTION_PRESETS.get(int(action), [0.0, 0.0])
            a0 = [np.array([preset[0]], dtype=np.float32),
                  np.array([preset[1]], dtype=np.float32)]
            return [a0, idle]

        action = np.asarray(action, dtype=np.float32)

        if action.ndim == 1 and action.shape[0] == 2:
            a0 = [np.array([action[0]], dtype=np.float32),
                  np.array([action[1]], dtype=np.float32)]
            return [a0, idle]

        if action.ndim == 1 and action.shape[0] == 4:
            a0 = [np.array([action[0]], dtype=np.float32),
                  np.array([action[1]], dtype=np.float32)]
            a1 = [np.array([action[2]], dtype=np.float32),
                  np.array([action[3]], dtype=np.float32)]
            return [a0, a1]

        if action.ndim == 2 and action.shape == (2, 2):
            a0 = [np.array([action[0, 0]], dtype=np.float32),
                  np.array([action[0, 1]], dtype=np.float32)]
            a1 = [np.array([action[1, 0]], dtype=np.float32),
                  np.array([action[1, 1]], dtype=np.float32)]
            return [a0, a1]

        # Fallback: idle
        return [idle, idle]


# ===================================================================
# Concrete subclass (single scenario for now)
# ===================================================================

class OlympicsWrestlingGameAdapter(OlympicsWrestlingAdapter):
    id = GameId.OLYMPICS_WRESTLING.value


# ===================================================================
# Adapter registry
# ===================================================================

OLYMPICS_WRESTLING_ADAPTERS: Dict[GameId, type[OlympicsWrestlingAdapter]] = {
    GameId.OLYMPICS_WRESTLING: OlympicsWrestlingGameAdapter,
}


__all__ = [
    "OlympicsWrestlingAdapter",
    "OlympicsWrestlingGameAdapter",
    "OLYMPICS_WRESTLING_ADAPTERS",
    "OLYMPICS_ACTION_PRESETS",
    "OLYMPICS_ACTION_NAMES",
]
