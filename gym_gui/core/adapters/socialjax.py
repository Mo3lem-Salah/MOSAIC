"""SocialJax environment adapter for the MOSAIC GUI.

SocialJax is a pure-JAX suite of 9 sequential social dilemma environments
Environments use JAX's functional API (no mutable state)
and expose a render() method that produces an RGB image from the environment state.

API:
    env = socialjax.make(env_id)
    obs, state = env.reset(rng_key)         # obs: dict[agent_name -> jnp.ndarray]
    obs, state, rewards, dones, info = env.step(rng_key, state, actions)

Repository: https://github.com/FLAIROx/socialjax
Paper: https://arxiv.org/abs/2503.14576
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

from gym_gui.core.adapters.base import (
    AdapterContext,
    AdapterStep,
    AgentSnapshot,
    EnvironmentAdapter,
    StepState,
)
from gym_gui.core.enums import ControlMode, GameId, RenderMode, SteppingParadigm
from gym_gui.core.ui.game_config.game_configs import SocialJaxConfig
from gym_gui.logging_config.log_constants import (
    LOG_SOCIALJAX_ACTION_TAKEN,
    LOG_SOCIALJAX_ENV_CLOSED,
    LOG_SOCIALJAX_ENV_CREATED,
    LOG_SOCIALJAX_ENV_RESET,
    LOG_SOCIALJAX_PATH_INJECTED,
    LOG_SOCIALJAX_RENDER_ERROR,
)

# Per-environment action labels (indexed by action integer).
# Sourced directly from each env's Actions(IntEnum) in the SocialJax source tree.
# coop_mining is the only env using egocentric movement (step_left/forward/etc);
# all other grid envs use absolute cardinal directions (left/right/up/down).
SOCIALJAX_ACTIONS_BY_ENV: dict[str, tuple[str, ...]] = {
    "coin_game":           ("Turn Left", "Turn Right", "Left", "Right", "Up", "Down", "Stay"),
    "harvest_common_open": ("Turn Left", "Turn Right", "Left", "Right", "Up", "Down", "Stay", "Zap"),
    "clean_up":            ("Turn Left", "Turn Right", "Left", "Right", "Up", "Down", "Stay", "Zap", "Clean"),
    "territory_open":      ("Turn Left", "Turn Right", "Left", "Right", "Up", "Down", "Stay", "Zap", "Claim"),
    "pd_arena":            ("Turn Left", "Turn Right", "Left", "Right", "Up", "Down", "Stay", "Interact"),
    "mushrooms":           ("Turn Left", "Turn Right", "Left", "Right", "Up", "Down", "Stay", "Zap"),
    "gift":                ("Turn Left", "Turn Right", "Left", "Right", "Up", "Down", "Stay", "Zap", "Consume"),
    "coop_mining":         ("Turn Left", "Turn Right", "Step Left", "Step Right", "Forward", "Backward", "Stay", "Mine"),
    "lb_foraging":         ("None", "North", "South", "West", "East", "Load"),
}

_SOCIALJAX_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../3rd_party/environments/SocialJax")
)


def _ensure_socialjax_path() -> None:
    """Inject SocialJax source tree into sys.path if not already importable."""
    import logging
    try:
        import socialjax  # noqa: F401
    except ImportError:
        if _SOCIALJAX_ROOT not in sys.path:
            sys.path.insert(0, _SOCIALJAX_ROOT)
            logging.getLogger(__name__).debug(
                LOG_SOCIALJAX_PATH_INJECTED.message,
                extra={"log_code": LOG_SOCIALJAX_PATH_INJECTED.code, "path": _SOCIALJAX_ROOT},
            )


class SocialJaxAdapter(EnvironmentAdapter):
    """Base adapter for SocialJax pure-JAX social dilemma environments.

    Subclasses override `id` with the concrete GameId value.
    The environment is loaded lazily on the first call to `load()`.
    """

    id: str = "socialjax/base"
    supported_control_modes: tuple[ControlMode, ...] = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    )
    supported_render_modes: tuple[RenderMode, ...] = (RenderMode.RGB_ARRAY,)
    default_render_mode: RenderMode = RenderMode.RGB_ARRAY
    stepping_paradigm: SteppingParadigm = SteppingParadigm.SIMULTANEOUS

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: SocialJaxConfig | None = None,
    ) -> None:
        super().__init__(context)
        self._config: SocialJaxConfig = config or SocialJaxConfig()
        self._sj_env: Any = None
        self._state: Any = None
        self._rng_key: Any = None
        self._agents: list[str] = []
        self._step_count: int = 0
        self._cumulative_rewards: dict[str, float] = {}

    def load(self) -> None:
        """Instantiate the SocialJax environment."""
        _ensure_socialjax_path()
        import jax
        import socialjax

        kwargs: dict[str, Any] = {}
        if self._config.shared_rewards:
            kwargs["shared_rewards"] = True
        if self._config.num_agents is not None:
            kwargs["num_agents"] = self._config.num_agents
        if self._config.num_inner_steps is not None:
            kwargs["num_inner_steps"] = self._config.num_inner_steps

        self._sj_env = socialjax.make(self._config.env_id, **kwargs)
        seed = self._config.seed if self._config.seed is not None else 0
        self._rng_key = jax.random.PRNGKey(seed)
        # env.agents returns [0, 1, ..., n-1]; env.action_spaces is empty for most envs
        self._agents = list(self._sj_env.agents)
        self.log_constant(
            LOG_SOCIALJAX_ENV_CREATED,
            extra={
                "env_id": self._config.env_id,
                "num_agents": len(self._agents),
                "agents": self._agents,
                "shared_rewards": self._config.shared_rewards,
            },
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> AdapterStep:
        import jax

        if self._sj_env is None:
            self.load()

        if seed is not None:
            self._rng_key = jax.random.PRNGKey(seed)

        self._rng_key, subkey = jax.random.split(self._rng_key)
        obs, self._state = self._sj_env.reset(subkey)

        self._step_count = 0
        self._episode_step = 0
        self._episode_return = 0.0
        self._cumulative_rewards = {a: 0.0 for a in self._agents}

        obs_np = {a: np.array(obs[a]) for a in self._agents}
        render_payload = self._safe_render()
        step_state = self._build_step_state(obs_np, {a: 0.0 for a in self._agents})

        self.log_constant(
            LOG_SOCIALJAX_ENV_RESET,
            extra={
                "env_id": self._config.env_id,
                "num_agents": len(self._agents),
                "seed": seed if seed is not None else "None",
            },
        )

        return AdapterStep(
            observation=obs_np,
            reward=0.0,
            terminated=False,
            truncated=False,
            info={},
            render_payload=render_payload,
            render_hint={"mode": "rgb_array"},
            state=step_state,
        )

    def step(self, action: Any) -> AdapterStep:
        import jax

        if self._sj_env is None or self._state is None:
            raise RuntimeError("Call reset() before step()")

        self._rng_key, subkey = jax.random.split(self._rng_key)
        actions_dict = self._normalize_actions(action)
        obs, self._state, rewards, dones, info = self._sj_env.step(
            subkey, self._state, actions_dict
        )

        self._step_count += 1
        self._episode_step += 1
        obs_np = {a: np.array(obs[a]) for a in self._agents}
        rewards_float = {a: float(rewards[a]) for a in self._agents}

        for a in self._agents:
            self._cumulative_rewards[a] = self._cumulative_rewards.get(a, 0.0) + rewards_float[a]

        mean_reward = sum(rewards_float.values()) / max(len(rewards_float), 1)
        self._episode_return += mean_reward
        terminated = bool(dones.get("__all__", False))
        render_payload = self._safe_render()
        step_state = self._build_step_state(obs_np, rewards_float)

        # Build per-agent action labels from the action dict
        _action_labels = SOCIALJAX_ACTIONS_BY_ENV.get(self._config.env_id, ())
        _action_names: dict[str, str] = {}
        if isinstance(actions_dict, dict):
            for a_name, a_val in actions_dict.items():
                idx = int(a_val) if hasattr(a_val, "__int__") else 0
                _action_names[str(a_name)] = _action_labels[idx] if idx < len(_action_labels) else str(idx)
        else:
            for i, a_val in enumerate(np.asarray(actions_dict).tolist()):
                idx = int(a_val)
                _action_names[str(self._agents[i]) if i < len(self._agents) else str(i)] = (
                    _action_labels[idx] if idx < len(_action_labels) else str(idx)
                )

        self.log_constant(
            LOG_SOCIALJAX_ACTION_TAKEN,
            extra={
                "env_id": self._config.env_id,
                "step": self._step_count,
                "actions": _action_names,
                "rewards": rewards_float,
                "terminated": terminated,
            },
        )

        return AdapterStep(
            observation=obs_np,
            reward=mean_reward,
            terminated=terminated,
            truncated=False,
            info={k: v for k, v in (info or {}).items() if isinstance(v, (int, float, str, bool))},
            render_payload=render_payload,
            render_hint={"mode": "rgb_array"},
            state=step_state,
        )

    def render(self) -> dict[str, Any] | None:
        if self._state is None:
            return None
        return self._safe_render()

    def close(self) -> None:
        self.log_constant(
            LOG_SOCIALJAX_ENV_CLOSED,
            extra={"env_id": self._config.env_id, "steps": self._step_count},
        )
        self._sj_env = None
        self._state = None

    def _safe_render(self) -> dict[str, Any] | None:
        """Render the current state, returning a payload dict for RgbRendererStrategy."""
        if self._sj_env is None or self._state is None:
            return None
        try:
            img = self._sj_env.render(self._state)
            arr = np.array(img)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return {
                "mode": "rgb_array",
                "rgb": arr,
                "env_id": self._config.env_id,
                "num_agents": len(self._agents),
                "step": self._step_count,
            }
        except Exception as exc:
            self.log_constant(
                LOG_SOCIALJAX_RENDER_ERROR,
                extra={"env_id": self._config.env_id, "error": str(exc)},
            )
            return None

    def _normalize_actions(self, action: Any) -> Any:
        """Convert MOSAIC's action format to a JAX array of shape (num_agents,).

        SocialJax step_env internally calls jnp.array(actions), so actions must be
        array-like rather than a dict. The jaxmarl_worker wrapper uses the same
        jnp.stack approach.
        """
        import jax.numpy as jnp

        n = len(self._agents)
        if isinstance(action, dict):
            # Dict keyed by agent name or integer index
            vals = []
            for i, a in enumerate(self._agents):
                v = action.get(a, action.get(i, 0))
                vals.append(int(v))
            return jnp.array(vals, dtype=jnp.int32)
        if isinstance(action, (list, tuple)):
            return jnp.array([int(a) for a in action], dtype=jnp.int32)
        # Single scalar broadcast to all agents
        return jnp.array([int(action)] * n, dtype=jnp.int32)

    def _build_step_state(
        self,
        obs_np: dict[str, np.ndarray],
        rewards: dict[str, float],
    ) -> StepState:
        del obs_np
        agents = [
            AgentSnapshot(
                name=name,
                info={
                    "reward": rewards.get(name, 0.0),
                    "cumulative_reward": self._cumulative_rewards.get(name, 0.0),
                },
            )
            for name in self._agents
        ]
        return StepState(
            agents=agents,
            metrics={
                "step": self._step_count,
                "mean_reward": sum(rewards.values()) / max(len(rewards), 1),
            },
            environment={"env_id": self._config.env_id},
        )


class SocialJaxCoinGameAdapter(SocialJaxAdapter):
    """Coin Game: 2-agent coin collection social dilemma."""

    id = "socialjax/coin_game"

    def __init__(self, context: AdapterContext | None = None, *, config: SocialJaxConfig | None = None) -> None:
        super().__init__(context, config=config or SocialJaxConfig(env_id="coin_game"))


class SocialJaxHarvestCommonOpenAdapter(SocialJaxAdapter):
    """Commons Harvest (open): regrowable resources, tragedy of the commons."""

    id = "socialjax/harvest_common_open"

    def __init__(self, context: AdapterContext | None = None, *, config: SocialJaxConfig | None = None) -> None:
        super().__init__(context, config=config or SocialJaxConfig(env_id="harvest_common_open"))


class SocialJaxCleanUpAdapter(SocialJaxAdapter):
    """Clean Up: agents must clean pollution to unlock apple regrowth."""

    id = "socialjax/clean_up"

    def __init__(self, context: AdapterContext | None = None, *, config: SocialJaxConfig | None = None) -> None:
        super().__init__(context, config=config or SocialJaxConfig(env_id="clean_up"))


class SocialJaxCoopMiningAdapter(SocialJaxAdapter):
    """Cooperative Mining: gold extraction requires multi-agent coordination."""

    id = "socialjax/coop_mining"

    def __init__(self, context: AdapterContext | None = None, *, config: SocialJaxConfig | None = None) -> None:
        super().__init__(context, config=config or SocialJaxConfig(env_id="coop_mining"))


class SocialJaxTerritoryOpenAdapter(SocialJaxAdapter):
    """Territory (open): agents claim and defend spatial zones."""

    id = "socialjax/territory_open"

    def __init__(self, context: AdapterContext | None = None, *, config: SocialJaxConfig | None = None) -> None:
        super().__init__(context, config=config or SocialJaxConfig(env_id="territory_open"))


class SocialJaxPdArenaAdapter(SocialJaxAdapter):
    """Prisoner's Dilemma Arena: iterated social dilemma in grid world."""

    id = "socialjax/pd_arena"

    def __init__(self, context: AdapterContext | None = None, *, config: SocialJaxConfig | None = None) -> None:
        super().__init__(context, config=config or SocialJaxConfig(env_id="pd_arena"))


class SocialJaxMushroomsAdapter(SocialJaxAdapter):
    """Mushrooms: foraging with externalities (toxic mushrooms harm neighbours)."""

    id = "socialjax/mushrooms"

    def __init__(self, context: AdapterContext | None = None, *, config: SocialJaxConfig | None = None) -> None:
        super().__init__(context, config=config or SocialJaxConfig(env_id="mushrooms"))


class SocialJaxGiftAdapter(SocialJaxAdapter):
    """Gift Exchange: agents can give resources; tests reciprocity and trust."""

    id = "socialjax/gift"

    def __init__(self, context: AdapterContext | None = None, *, config: SocialJaxConfig | None = None) -> None:
        super().__init__(context, config=config or SocialJaxConfig(env_id="gift"))


class SocialJaxLbForagingAdapter(SocialJaxAdapter):
    """Level-Based Foraging: items require coordinated simultaneous pickup."""

    id = "socialjax/lb_foraging"

    def __init__(self, context: AdapterContext | None = None, *, config: SocialJaxConfig | None = None) -> None:
        super().__init__(context, config=config or SocialJaxConfig(env_id="lb_foraging"))


SOCIALJAX_ADAPTERS: dict[GameId, type[SocialJaxAdapter]] = {
    GameId.SOCIALJAX_COIN_GAME: SocialJaxCoinGameAdapter,
    GameId.SOCIALJAX_HARVEST_COMMON_OPEN: SocialJaxHarvestCommonOpenAdapter,
    GameId.SOCIALJAX_CLEAN_UP: SocialJaxCleanUpAdapter,
    GameId.SOCIALJAX_COOP_MINING: SocialJaxCoopMiningAdapter,
    GameId.SOCIALJAX_TERRITORY_OPEN: SocialJaxTerritoryOpenAdapter,
    GameId.SOCIALJAX_PD_ARENA: SocialJaxPdArenaAdapter,
    GameId.SOCIALJAX_MUSHROOMS: SocialJaxMushroomsAdapter,
    GameId.SOCIALJAX_GIFT: SocialJaxGiftAdapter,
    GameId.SOCIALJAX_LB_FORAGING: SocialJaxLbForagingAdapter,
}


def create_socialjax_adapter(
    game_id: GameId,
    context: AdapterContext | None = None,
    *,
    config: SocialJaxConfig | None = None,
) -> SocialJaxAdapter:
    """Instantiate a SocialJax adapter by GameId, optionally with a config."""
    adapter_cls = SOCIALJAX_ADAPTERS[game_id]
    if config is not None:
        return adapter_cls(context, config=config)
    env_id = game_id.value.removeprefix("socialjax/")
    return adapter_cls(context, config=SocialJaxConfig(env_id=env_id))
