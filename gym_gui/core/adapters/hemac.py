"""HeMAC (Heterogeneous Multi-Agent Challenge) environment adapter.

HeMAC is a PettingZoo-based benchmark for Heterogeneous Multi-Agent RL (HeMARL)
with three agent types: Quadcopters (drones), Observers, and Provisioners.

Uses PettingZoo's parallel_env API (simultaneous stepping), consistent with
MOSAIC's other multi-agent environments (SMAC, MeltingPot, RWARE).

Published at ECAI 2025 by ThalesGroup.
Repository: https://github.com/ThalesGroup/hemac
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping

import numpy as np

from gym_gui.core.adapters.base import (  # type: ignore[import]
    AdapterContext,
    AdapterStep,
    AgentSnapshot,
    EnvironmentAdapter,
    StepState,
)
from gym_gui.core.enums import ControlMode, GameId, RenderMode, SteppingParadigm
from gym_gui.logging_config.log_constants import (
    LOG_HEMAC_ENV_CLOSED,
    LOG_HEMAC_ENV_CREATED,
    LOG_HEMAC_ENV_RESET,
    LOG_HEMAC_INIT_ERROR,
    LOG_HEMAC_RENDER_ERROR,
    LOG_HEMAC_STEP_SUMMARY,
    LOG_HEMAC_TARGET_REACHED,
)

# ---------------------------------------------------------------------------
# Shared scenario configs (read-only — always .copy() before passing to HeMAC
# because the HeMAC constructor mutates drone_config in place)
# ---------------------------------------------------------------------------

_SIMPLE_FLEET_PATROL: dict = {
    "benchmark": True,
    "area": [(100, 100), (250, 100), (820, 480), (600, 800), (100, 620)],
}

_SIMPLE_FLEET_POI: list = [{"speed": 2.0, "dimension": [8, 8], "spawn_mode": "random"}]

_FLEET_POI: list = [{"speed": 3.0, "dimension": [8, 8], "spawn_mode": "random"}]

_SIMPLE_FLEET_DRONE_CONFIG: dict = {
    "drones_starting_pos": [],
    "drone_ui_dimension": 16,
    "drone_max_speed": 10,
    "drone_max_charge": 100,
    "discrete_action_space": True,
}

_FLEET_DRONE_CONFIG: dict = {
    "drones_starting_pos": [],
    "drone_ui_dimension": 16,
    "drone_max_speed": 12,
    "drone_max_charge": 100,
    "discrete_action_space": True,
}

_SIMPLE_FLEET_DRONE_SENSOR: dict = {
    "model": "RoundCamera",
    "params": {"sensing_range": 50},
}

_SIMPLE_FLEET_OBSERVER_SENSOR: dict = {
    "model": "ForwardFacingCamera",
    "params": {"hfov": math.pi / 6, "sensing_range": 200},
}


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------

class HeMACEnvironmentAdapter(EnvironmentAdapter[Dict[str, Any], Any]):
    """Base adapter for HeMAC heterogeneous multi-agent environments.

    Subclasses set class-level attributes for each scenario's agent counts,
    obstacle range, and max episode length. The base class handles the
    parallel_env lifecycle (load / reset / step / render / close).
    """

    supported_control_modes = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COOP,
    )
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    default_render_mode = RenderMode.RGB_ARRAY

    # Scenario parameters — overridden per subclass
    _n_drones: int = 1
    _n_observers: int = 1
    _n_provisioners: int = 0
    _max_cycles: int = 600
    _min_obstacles: int = 0
    _max_obstacles: int = 0
    _patrol_config: dict | None = None
    _poi_config: list | None = None
    _drone_config: dict | None = None
    _drone_sensor: dict | None = None
    _observer_sensor: dict | None = None

    def __init__(self, context: AdapterContext | None = None) -> None:
        super().__init__(context)
        self._par_env: Any = None
        self._last_frame: np.ndarray | None = None
        self._step_counter: int = 0
        self._episode_step: int = 0
        self._episode_return: float = 0.0
        self._all_agents: list[str] = []

    stepping_paradigm: SteppingParadigm = SteppingParadigm.SIMULTANEOUS

    @property
    def num_agents(self) -> int:
        return self._n_drones + self._n_observers + self._n_provisioners

    # ------------------------------------------------------------------
    # Space properties (override base class — PettingZoo parallel_env
    # exposes action_space(agent) as a callable, not a property)
    # ------------------------------------------------------------------

    @property
    def action_space(self) -> Any:
        if self._par_env is None:
            import gymnasium
            return gymnasium.spaces.Discrete(5)
        agents = self._par_env.possible_agents
        agent = next((a for a in agents if "drone" in a), agents[0] if agents else None)
        if agent is None:
            import gymnasium
            return gymnasium.spaces.Discrete(5)
        return self._par_env.action_space(agent)

    @property
    def observation_space(self) -> Any:
        if self._par_env is None:
            import gymnasium
            return gymnasium.spaces.Box(low=-10000.0, high=10000.0, shape=(29,), dtype=np.float32)
        agents = self._par_env.possible_agents
        agent = next((a for a in agents if "drone" in a), agents[0] if agents else None)
        if agent is None:
            import gymnasium
            return gymnasium.spaces.Box(low=-10000.0, high=10000.0, shape=(29,), dtype=np.float32)
        return self._par_env.observation_space(agent)

    def _require_env(self) -> Any:
        if self._par_env is None:
            from gym_gui.core.adapters.base import AdapterNotReadyError
            raise AdapterNotReadyError(f"Adapter '{self.id}' has not been loaded.")
        return self._par_env

    # ------------------------------------------------------------------
    # Lifecycle: load
    # ------------------------------------------------------------------

    def load(self) -> None:
        try:
            from hemac.environment.HeMAC import parallel_env
        except ImportError as exc:
            self.log_constant(
                LOG_HEMAC_INIT_ERROR,
                exc_info=exc,
                extra={"env_id": self.id},
            )
            raise ImportError(
                "HeMAC is required. Install with: "
                "pip install -e 3rd_party/environments/hemac"
            ) from exc

        # Shallow-copy mutable dicts: HeMAC constructor mutates drone_config
        # when n_drones > len(drones_starting_pos).
        drone_cfg = dict(self._drone_config) if self._drone_config else None

        try:
            self._par_env = parallel_env(
                n_drones=self._n_drones,
                n_observers=self._n_observers,
                n_provisioners=self._n_provisioners,
                max_cycles=self._max_cycles,
                min_obstacles=self._min_obstacles,
                max_obstacles=self._max_obstacles,
                render_mode="rgb_array",
                patrol_config=self._patrol_config,
                poi_config=self._poi_config,
                drone_config=drone_cfg,
                drone_sensor=self._drone_sensor,
                observer_sensor=self._observer_sensor,
            )
        except Exception as exc:
            self.log_constant(
                LOG_HEMAC_INIT_ERROR,
                exc_info=exc,
                extra={"env_id": self.id},
            )
            raise

        self._all_agents = list(self._par_env.possible_agents)

        self.log_constant(
            LOG_HEMAC_ENV_CREATED,
            extra={
                "env_id": self.id,
                "n_drones": self._n_drones,
                "n_observers": self._n_observers,
                "n_provisioners": self._n_provisioners,
                "max_cycles": self._max_cycles,
                "obstacles": f"{self._min_obstacles}-{self._max_obstacles}",
            },
        )

    # ------------------------------------------------------------------
    # Lifecycle: reset
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> AdapterStep[Dict[str, Any]]:
        if self._par_env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")

        obs_dict, info_dict = self._par_env.reset(seed=seed)
        self._step_counter = 0
        self._episode_step = 0
        self._episode_return = 0.0
        self._last_frame = None

        self.log_constant(
            LOG_HEMAC_ENV_RESET,
            extra={
                "env_id": self.id,
                "seed": seed if seed is not None else "None",
                "agents": len(self._par_env.agents),
            },
        )

        return self._package_parallel_step(
            obs_dict,
            {a: 0.0 for a in self._par_env.agents},
            False,
            False,
            info_dict,
        )

    # ------------------------------------------------------------------
    # Lifecycle: step
    # ------------------------------------------------------------------

    def step(self, action: Any) -> AdapterStep[Dict[str, Any]]:
        if self._par_env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")

        self._step_counter += 1
        self._episode_step += 1

        # parallel_env.step() requires {agent_id: int} — convert human scalar / list
        active = self._par_env.agents
        if isinstance(action, dict):
            action_dict: Dict[str, int] = {ag: int(action[ag]) for ag in active if ag in action}
        else:
            raw = action[0] if isinstance(action, (list, tuple)) else action
            action_dict = {ag: int(raw) for ag in active}

        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self._par_env.step(action_dict)

        team_reward = float(sum(rew_dict.values())) if rew_dict else 0.0
        self._episode_return += team_reward

        terminated = bool(all(term_dict.values())) if term_dict else False
        truncated = bool(all(trunc_dict.values())) if trunc_dict else False

        for ag, info in info_dict.items():
            if isinstance(info, dict) and info.get("success"):
                self.log_constant(
                    LOG_HEMAC_TARGET_REACHED,
                    extra={"env_id": self.id, "agent": ag, "step": self._step_counter},
                )

        self.log_constant(
            LOG_HEMAC_STEP_SUMMARY,
            extra={
                "env_id": self.id,
                "episode_step": self._episode_step,
                "team_reward": team_reward,
                "terminated": terminated,
                "truncated": truncated,
                "active_agents": len(self._par_env.agents),
            },
        )

        return self._package_parallel_step(obs_dict, rew_dict, terminated, truncated, info_dict)

    # ------------------------------------------------------------------
    # Lifecycle: render
    # ------------------------------------------------------------------

    def render(self) -> Dict[str, Any] | None:
        if self._par_env is None:
            return None
        try:
            frame = self._par_env.render()
            if isinstance(frame, np.ndarray) and frame.ndim == 3:
                self._last_frame = frame
        except Exception as exc:
            self.log_constant(
                LOG_HEMAC_RENDER_ERROR,
                exc_info=exc,
                extra={"env_id": self.id, "step": self._step_counter},
            )
        return {
            "mode": RenderMode.RGB_ARRAY.value,
            "rgb": self._last_frame,
            "game_id": self.id,
            "step": self._step_counter,
            "n_agents": self.num_agents,
        }

    # ------------------------------------------------------------------
    # Lifecycle: close
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._par_env is not None:
            self.log_constant(
                LOG_HEMAC_ENV_CLOSED,
                extra={"env_id": self.id},
            )
            try:
                self._par_env.close()
            except Exception:
                pass
            self._par_env = None
            self._last_frame = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _package_parallel_step(
        self,
        obs_dict: Dict[str, Any],
        rew_dict: Dict[str, float],
        terminated: bool,
        truncated: bool,
        info_dict: Mapping[str, Any],
    ) -> AdapterStep[Dict[str, Any]]:
        agents_for_snap = self._all_agents or list(obs_dict.keys())
        agent_snapshots: List[AgentSnapshot] = [
            AgentSnapshot(
                name=ag,
                role=_agent_type(ag),
                info={"reward": float(rew_dict.get(ag, 0.0))},
            )
            for ag in agents_for_snap
        ]

        state = StepState(
            active_agent=None,  # simultaneous — no single active agent
            agents=tuple(agent_snapshots),
            metrics={
                "step_count": self._step_counter,
                "episode_return": self._episode_return,
                "n_agents": self.num_agents,
            },
            environment={
                "scenario": self.id,
                "family": "hemac",
                "n_drones": self._n_drones,
                "n_observers": self._n_observers,
                "n_provisioners": self._n_provisioners,
            },
            raw=dict(info_dict) if isinstance(info_dict, Mapping) else {},
        )

        team_reward = float(sum(rew_dict.values())) if rew_dict else 0.0

        render_payload = self.render()
        if render_payload is None:
            render_payload = {"mode": RenderMode.RGB_ARRAY.value, "rgb": None}

        return AdapterStep(
            observation=obs_dict,
            reward=team_reward,
            terminated=terminated,
            truncated=truncated,
            info=dict(info_dict) if isinstance(info_dict, Mapping) else {},
            render_payload=render_payload,
            render_hint={"type": "rgb", "use_rgb": True},
            agent_id=None,
            state=state,
        )


def _agent_type(name: str) -> str:
    name_lower = name.lower()
    if "drone" in name_lower or "quadcopter" in name_lower:
        return "quadcopter"
    if "observer" in name_lower:
        return "observer"
    if "provisioner" in name_lower:
        return "provisioner"
    return "unknown"


# ---------------------------------------------------------------------------
# Simple Fleet scenarios — no obstacles, pentagon patrol area
# ---------------------------------------------------------------------------

class HeMACSimpleFleet1Q1OAdapter(HeMACEnvironmentAdapter):
    id = GameId.HEMAC_SIMPLE_FLEET_1Q1O.value
    _n_drones = 1
    _n_observers = 1
    _n_provisioners = 0
    _max_cycles = 600
    _min_obstacles = 0
    _max_obstacles = 0
    _patrol_config = _SIMPLE_FLEET_PATROL
    _poi_config = _SIMPLE_FLEET_POI
    _drone_config = _SIMPLE_FLEET_DRONE_CONFIG
    _drone_sensor = _SIMPLE_FLEET_DRONE_SENSOR
    _observer_sensor = _SIMPLE_FLEET_OBSERVER_SENSOR


class HeMACSimpleFleet3Q1OAdapter(HeMACEnvironmentAdapter):
    id = GameId.HEMAC_SIMPLE_FLEET_3Q1O.value
    _n_drones = 3
    _n_observers = 1
    _n_provisioners = 0
    _max_cycles = 600
    _min_obstacles = 0
    _max_obstacles = 0
    _patrol_config = _SIMPLE_FLEET_PATROL
    _poi_config = _SIMPLE_FLEET_POI
    _drone_config = _SIMPLE_FLEET_DRONE_CONFIG
    _drone_sensor = _SIMPLE_FLEET_DRONE_SENSOR
    _observer_sensor = _SIMPLE_FLEET_OBSERVER_SENSOR


class HeMACSimpleFleet5Q2OAdapter(HeMACEnvironmentAdapter):
    id = GameId.HEMAC_SIMPLE_FLEET_5Q2O.value
    _n_drones = 5
    _n_observers = 2
    _n_provisioners = 0
    _max_cycles = 600
    _min_obstacles = 0
    _max_obstacles = 0
    _patrol_config = _SIMPLE_FLEET_PATROL
    _poi_config = _SIMPLE_FLEET_POI
    _drone_config = _SIMPLE_FLEET_DRONE_CONFIG
    _drone_sensor = _SIMPLE_FLEET_DRONE_SENSOR
    _observer_sensor = _SIMPLE_FLEET_OBSERVER_SENSOR


# ---------------------------------------------------------------------------
# Fleet scenarios — obstacles, energy constraints, no provisioners
# ---------------------------------------------------------------------------

class HeMACFleet3Q1OAdapter(HeMACEnvironmentAdapter):
    id = GameId.HEMAC_FLEET_3Q1O.value
    _n_drones = 3
    _n_observers = 1
    _n_provisioners = 0
    _max_cycles = 900
    _min_obstacles = 2
    _max_obstacles = 3
    _poi_config = _FLEET_POI
    _drone_config = _FLEET_DRONE_CONFIG
    _drone_sensor = _SIMPLE_FLEET_DRONE_SENSOR
    _observer_sensor = _SIMPLE_FLEET_OBSERVER_SENSOR


class HeMACFleet10Q3OAdapter(HeMACEnvironmentAdapter):
    id = GameId.HEMAC_FLEET_10Q3O.value
    _n_drones = 10
    _n_observers = 3
    _n_provisioners = 0
    _max_cycles = 900
    _min_obstacles = 2
    _max_obstacles = 3
    _poi_config = _FLEET_POI
    _drone_config = _FLEET_DRONE_CONFIG
    _drone_sensor = _SIMPLE_FLEET_DRONE_SENSOR
    _observer_sensor = _SIMPLE_FLEET_OBSERVER_SENSOR


class HeMACFleet20Q5OAdapter(HeMACEnvironmentAdapter):
    id = GameId.HEMAC_FLEET_20Q5O.value
    _n_drones = 20
    _n_observers = 5
    _n_provisioners = 0
    _max_cycles = 900
    _min_obstacles = 2
    _max_obstacles = 5
    _poi_config = _FLEET_POI
    _drone_config = _FLEET_DRONE_CONFIG
    _drone_sensor = _SIMPLE_FLEET_DRONE_SENSOR
    _observer_sensor = _SIMPLE_FLEET_OBSERVER_SENSOR


# ---------------------------------------------------------------------------
# Complex Fleet scenarios — Provisioners + obstacles
# ---------------------------------------------------------------------------

class HeMACComplexFleet3Q1O1PAdapter(HeMACEnvironmentAdapter):
    id = GameId.HEMAC_COMPLEX_FLEET_3Q1O1P.value
    _n_drones = 3
    _n_observers = 1
    _n_provisioners = 1
    _max_cycles = 900
    _min_obstacles = 2
    _max_obstacles = 3
    _poi_config = _FLEET_POI
    _drone_config = _FLEET_DRONE_CONFIG
    _drone_sensor = _SIMPLE_FLEET_DRONE_SENSOR
    _observer_sensor = _SIMPLE_FLEET_OBSERVER_SENSOR


class HeMACComplexFleet5Q2O1PAdapter(HeMACEnvironmentAdapter):
    id = GameId.HEMAC_COMPLEX_FLEET_5Q2O1P.value
    _n_drones = 5
    _n_observers = 2
    _n_provisioners = 1
    _max_cycles = 900
    _min_obstacles = 2
    _max_obstacles = 3
    _poi_config = _FLEET_POI
    _drone_config = _FLEET_DRONE_CONFIG
    _drone_sensor = _SIMPLE_FLEET_DRONE_SENSOR
    _observer_sensor = _SIMPLE_FLEET_OBSERVER_SENSOR


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

HEMAC_ADAPTERS: Dict[GameId, type[HeMACEnvironmentAdapter]] = {
    GameId.HEMAC_SIMPLE_FLEET_1Q1O: HeMACSimpleFleet1Q1OAdapter,
    GameId.HEMAC_SIMPLE_FLEET_3Q1O: HeMACSimpleFleet3Q1OAdapter,
    GameId.HEMAC_SIMPLE_FLEET_5Q2O: HeMACSimpleFleet5Q2OAdapter,
    GameId.HEMAC_FLEET_3Q1O: HeMACFleet3Q1OAdapter,
    GameId.HEMAC_FLEET_10Q3O: HeMACFleet10Q3OAdapter,
    GameId.HEMAC_FLEET_20Q5O: HeMACFleet20Q5OAdapter,
    GameId.HEMAC_COMPLEX_FLEET_3Q1O1P: HeMACComplexFleet3Q1O1PAdapter,
    GameId.HEMAC_COMPLEX_FLEET_5Q2O1P: HeMACComplexFleet5Q2O1PAdapter,
}

__all__ = [
    "HeMACEnvironmentAdapter",
    "HEMAC_ADAPTERS",
    "HeMACSimpleFleet1Q1OAdapter",
    "HeMACSimpleFleet3Q1OAdapter",
    "HeMACSimpleFleet5Q2OAdapter",
    "HeMACFleet3Q1OAdapter",
    "HeMACFleet10Q3OAdapter",
    "HeMACFleet20Q5OAdapter",
    "HeMACComplexFleet3Q1O1PAdapter",
    "HeMACComplexFleet5Q2O1PAdapter",
]
