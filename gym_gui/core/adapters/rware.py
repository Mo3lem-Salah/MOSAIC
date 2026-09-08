"""Adapter bridging RWARE's Warehouse to MOSAIC's adapter interface.

RWARE (Robotic Warehouse) is a multi-agent cooperative environment where
robots pick up shelves, deliver them to goal workstations, and return them.

Paper: Papoudakis et al. (2021). "Benchmarking Multi-Agent Deep RL Algorithms"
Source: 3rd_party/robotic-warehouse/
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Mapping

import gymnasium as gym
import numpy as np

from gym_gui.core.adapters.base import (
    AdapterContext,
    AdapterStep,
    EnvironmentAdapter,
    StepState,
)
from gym_gui.core.enums import (
    ControlMode,
    GameId,
    RenderMode,
    SteppingParadigm,
)
from gym_gui.logging_config.log_constants import (
    LOG_RWARE_ENV_CLOSED,
    LOG_RWARE_ENV_CREATED,
    LOG_RWARE_ENV_RESET,
    LOG_RWARE_RENDER_ERROR,
    LOG_RWARE_STEP_SUMMARY,
)

_LOGGER = logging.getLogger(__name__)


_pyglet_patched = False


def _patch_pyglet_window_invisible() -> None:
    """Force pyglet windows to start invisible, before RWARE ever creates one.

    RWARE's ``rware.rendering.Viewer`` unconditionally instantiates
    ``pyglet.window.Window()`` with no ``visible`` argument, which pops a
    real OS window on screen the instant a frame is rendered -- even in
    ``rgb_array`` mode where MOSAIC only wants the raw pixel buffer.

    Monkey-patching ``pyglet.window.Window.__init__`` to default
    ``visible=False`` stops the window from ever appearing, so frames only
    ever show up inside MOSAIC's own Render View. This is strictly
    stronger than hiding the window after it has already flashed on
    screen (the previous approach), since it prevents the pop-up rather
    than reacting to it.

    Controlled by ``RWARE_HIDDEN_WINDOW`` (see ``.env.example``): "1"
    (default) applies the patch; "0" restores the upstream behaviour of
    showing the engine's own window.
    """
    global _pyglet_patched
    if _pyglet_patched:
        return
    if os.getenv("RWARE_HIDDEN_WINDOW", "1") == "0":
        _LOGGER.debug("RWARE_HIDDEN_WINDOW=0: leaving pyglet window visibility untouched")
        return
    try:
        import pyglet

        original_init = pyglet.window.Window.__init__

        def _invisible_init(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("visible", False)
            original_init(self, *args, **kwargs)

        pyglet.window.Window.__init__ = _invisible_init  # type: ignore[method-assign]
        _pyglet_patched = True
        _LOGGER.debug("Patched pyglet.window.Window to default visible=False")
    except Exception as exc:  # pragma: no cover - defensive
        _LOGGER.debug("Could not patch pyglet window visibility: %s", exc)


def _ensure_rware() -> None:
    """Import rware lazily to trigger gymnasium.register() calls.

    Patches pyglet's window visibility *before* importing rware, so that
    when rware's Viewer lazily creates its window on first render, it
    is born invisible instead of popping up on screen.
    """
    _patch_pyglet_window_invisible()
    import rware  # noqa: F401


class RWAREAdapter(EnvironmentAdapter[List[np.ndarray], List[int]]):
    """Base adapter for Robotic Warehouse environments.

    Subclasses set class-level defaults for warehouse size, agent count,
    and difficulty. The config panel can override observation type, reward
    type, sensor range, and communication bits at runtime.
    """

    # Subclasses override these
    _gym_id: str = "rware-tiny-2ag-v2"
    _default_n_agents: int = 2

    id: str = "rware-tiny-2ag-v2"
    default_render_mode = RenderMode.RGB_ARRAY
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    supported_control_modes = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COOP,
    )

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: Any | None = None,
    ) -> None:
        super().__init__(context)

        from gym_gui.core.ui.game_config.game_configs import RWAREConfig

        if config is None:
            config = RWAREConfig()
        if not isinstance(config, RWAREConfig):
            config = RWAREConfig()

        self._config = config
        self._rware_env: gym.Env | None = None
        self._n_agents: int = self._default_n_agents
        self._step_count: int = 0
        self._pyglet_window_hidden: bool = False

    @property
    def stepping_paradigm(self) -> SteppingParadigm:  # type: ignore[override]
        return SteppingParadigm.SIMULTANEOUS

    @property
    def action_space(self) -> gym.Space[Any]:
        if self._rware_env is None:
            # Before load, return a placeholder
            from gymnasium.spaces import Discrete
            from gymnasium.spaces import Tuple as GymTuple
            return GymTuple(tuple(Discrete(5) for _ in range(self._default_n_agents)))
        return self._rware_env.action_space

    @property
    def observation_space(self) -> gym.Space[Any]:
        if self._rware_env is None:
            from gymnasium.spaces import Box
            from gymnasium.spaces import Tuple as GymTuple
            return GymTuple(tuple(
                Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
                for _ in range(self._default_n_agents)
            ))
        return self._rware_env.observation_space

    @property
    def n_agents(self) -> int:
        return self._n_agents

    def load(self) -> None:
        """Create the RWARE Warehouse environment via gym.make()."""
        _ensure_rware()

        gym_id = self._gym_id
        _LOGGER.info("Loading RWARE environment: %s", gym_id)

        self._rware_env = gym.make(gym_id, render_mode=self._config.render_mode)

        # Read actual agent count from the unwrapped env
        env_unwrapped = self._rware_env.unwrapped
        self._n_agents = env_unwrapped.n_agents

        self.log_constant(
            LOG_RWARE_ENV_CREATED,
            extra={
                "env_id": gym_id,
                "n_agents": self._n_agents,
            },
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> AdapterStep[List[np.ndarray]]:
        """Reset the environment and return initial observation."""
        if self._rware_env is None:
            self.load()
        assert self._rware_env is not None

        effective_seed = seed if seed is not None else self._config.seed
        obs_tuple, info = self._rware_env.reset(seed=effective_seed, options=options)
        self._step_count = 0

        self.log_constant(
            LOG_RWARE_ENV_RESET,
            extra={
                "env_id": self._gym_id,
                "seed": effective_seed if effective_seed is not None else "None",
                "n_agents": self._n_agents,
            },
        )

        obs_list = list(obs_tuple)
        return AdapterStep(
            observation=obs_list,
            reward=0.0,
            terminated=False,
            truncated=False,
            info=info if isinstance(info, Mapping) else {},
            render_payload=self._try_render(),
        )

    def step(self, action: List[int]) -> AdapterStep[List[np.ndarray]]:
        """Execute one timestep for all agents simultaneously."""
        if self._rware_env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")

        obs_tuple, rewards, done, truncated, info = self._rware_env.step(tuple(action))
        self._step_count += 1

        # Aggregate reward for logging (sum of per-agent rewards)
        reward_list = list(rewards) if not isinstance(rewards, (int, float)) else [rewards]
        total_reward = float(sum(reward_list))

        self.log_constant(
            LOG_RWARE_STEP_SUMMARY,
            extra={
                "env_id": self._gym_id,
                "step": self._step_count,
                "rewards": [float(r) for r in reward_list],
                "total_reward": total_reward,
                "terminated": done,
                "truncated": truncated,
            },
        )

        obs_list = list(obs_tuple)
        return AdapterStep(
            observation=obs_list,
            reward=total_reward,
            terminated=done,
            truncated=truncated,
            info=info if isinstance(info, Mapping) else {},
            render_payload=self._try_render(),
        )

    def render(self) -> dict[str, Any] | None:
        """Return an RGB frame payload dict for MOSAIC's Render View."""
        return self._try_render()

    def _try_render(self) -> dict[str, Any] | None:
        """Attempt render, returning payload dict or None on error.

        The pyglet window RWARE's Viewer lazily creates on first render is
        born invisible thanks to ``_patch_pyglet_window_invisible()`` in
        ``_ensure_rware()``. As a defensive fallback (in case the patch
        fails to apply, e.g. a differently versioned pyglet), we also
        explicitly hide the window here after the first render. Either
        way, frames only ever surface inside MOSAIC's Render View.

        Both the patch and this fallback respect ``RWARE_HIDDEN_WINDOW``
        (see ``.env.example``): if it is set to "0" the user explicitly
        wants the engine's own window visible, so we leave it alone.
        """
        if self._rware_env is None:
            return None
        try:
            frame = self._rware_env.render()

            # Defensive fallback: hide the window if it somehow became visible
            if not self._pyglet_window_hidden and os.getenv("RWARE_HIDDEN_WINDOW", "1") != "0":
                self._hide_pyglet_window()
                self._pyglet_window_hidden = True

            if isinstance(frame, np.ndarray):
                return {
                    "mode": RenderMode.RGB_ARRAY.value,
                    "rgb": frame,
                    "game_id": self._gym_id,
                    "num_agents": self._n_agents,
                    "step": self._step_count,
                }
            return None
        except Exception as exc:
            self.log_constant(
                LOG_RWARE_RENDER_ERROR,
                extra={"env_id": self._gym_id, "error": str(exc)},
            )
            return None

    def _hide_pyglet_window(self) -> None:
        """Hide the pyglet window that RWARE's Viewer creates.

        RWARE's ``rware.rendering.Viewer`` unconditionally opens a pyglet
        window.  In ``rgb_array`` mode we only need the framebuffer data,
        so we hide the window to prevent it from appearing on screen.
        """
        try:
            env_unwrapped = self._rware_env.unwrapped  # type: ignore[union-attr]
            renderer = getattr(env_unwrapped, "renderer", None)
            if renderer is not None:
                window = getattr(renderer, "window", None)
                if window is not None:
                    window.set_visible(False)
                    _LOGGER.debug("Hidden RWARE pyglet window for in-app rendering")
        except Exception as exc:
            _LOGGER.debug("Could not hide pyglet window: %s", exc)

    def close(self) -> None:
        """Release environment resources."""
        if self._rware_env is not None:
            try:
                self._rware_env.close()
            except Exception:
                pass
            self._rware_env = None
        self.log_constant(
            LOG_RWARE_ENV_CLOSED,
            extra={"env_id": self._gym_id},
        )

    def build_step_state(
        self,
        observation: List[np.ndarray],
        info: Mapping[str, Any],
    ) -> StepState:
        """Build step state with agent info for the UI."""
        from gym_gui.core.adapters.base import AgentSnapshot

        agents = [
            AgentSnapshot(name=f"agent_{i}")
            for i in range(self._n_agents)
        ]
        return StepState(
            agents=agents,
            metrics={"step": self._step_count, "n_agents": self._n_agents},
        )


# ---------------------------------------------------------------------------
# Concrete adapters (one per GameId)
# ---------------------------------------------------------------------------
# Size reference from rware/__init__.py:
#   tiny = (shelf_rows=1, shelf_columns=3)
#   small = (shelf_rows=2, shelf_columns=3)
#   medium = (shelf_rows=2, shelf_columns=5)
#   large = (shelf_rows=3, shelf_columns=5)


class RWARETiny2AgAdapter(RWAREAdapter):
    """Tiny warehouse (1x3 shelves), 2 agents."""
    _gym_id = "rware-tiny-2ag-v2"
    _default_n_agents = 2
    id = "rware-tiny-2ag-v2"


class RWARETiny4AgAdapter(RWAREAdapter):
    """Tiny warehouse (1x3 shelves), 4 agents."""
    _gym_id = "rware-tiny-4ag-v2"
    _default_n_agents = 4
    id = "rware-tiny-4ag-v2"


class RWARESmall2AgAdapter(RWAREAdapter):
    """Small warehouse (2x3 shelves), 2 agents."""
    _gym_id = "rware-small-2ag-v2"
    _default_n_agents = 2
    id = "rware-small-2ag-v2"


class RWARESmall4AgAdapter(RWAREAdapter):
    """Small warehouse (2x3 shelves), 4 agents."""
    _gym_id = "rware-small-4ag-v2"
    _default_n_agents = 4
    id = "rware-small-4ag-v2"


class RWAREMedium2AgAdapter(RWAREAdapter):
    """Medium warehouse (2x5 shelves), 2 agents."""
    _gym_id = "rware-medium-2ag-v2"
    _default_n_agents = 2
    id = "rware-medium-2ag-v2"


class RWAREMedium4AgAdapter(RWAREAdapter):
    """Medium warehouse (2x5 shelves), 4 agents."""
    _gym_id = "rware-medium-4ag-v2"
    _default_n_agents = 4
    id = "rware-medium-4ag-v2"


class RWAREMedium4AgEasyAdapter(RWAREAdapter):
    """Medium warehouse (2x5 shelves), 4 agents, easy difficulty (2x requests)."""
    _gym_id = "rware-medium-4ag-easy-v2"
    _default_n_agents = 4
    id = "rware-medium-4ag-easy-v2"


class RWAREMedium4AgHardAdapter(RWAREAdapter):
    """Medium warehouse (2x5 shelves), 4 agents, hard difficulty (0.5x requests)."""
    _gym_id = "rware-medium-4ag-hard-v2"
    _default_n_agents = 4
    id = "rware-medium-4ag-hard-v2"


class RWARELarge4AgAdapter(RWAREAdapter):
    """Large warehouse (3x5 shelves), 4 agents."""
    _gym_id = "rware-large-4ag-v2"
    _default_n_agents = 4
    id = "rware-large-4ag-v2"


class RWARELarge4AgHardAdapter(RWAREAdapter):
    """Large warehouse (3x5 shelves), 4 agents, hard difficulty."""
    _gym_id = "rware-large-4ag-hard-v2"
    _default_n_agents = 4
    id = "rware-large-4ag-hard-v2"


class RWARELarge8AgAdapter(RWAREAdapter):
    """Large warehouse (3x5 shelves), 8 agents."""
    _gym_id = "rware-large-8ag-v2"
    _default_n_agents = 8
    id = "rware-large-8ag-v2"


class RWARELarge8AgHardAdapter(RWAREAdapter):
    """Large warehouse (3x5 shelves), 8 agents, hard difficulty."""
    _gym_id = "rware-large-8ag-hard-v2"
    _default_n_agents = 8
    id = "rware-large-8ag-hard-v2"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

RWARE_ADAPTERS: Dict[GameId, type[RWAREAdapter]] = {
    GameId.RWARE_TINY_2AG: RWARETiny2AgAdapter,
    GameId.RWARE_TINY_4AG: RWARETiny4AgAdapter,
    GameId.RWARE_SMALL_2AG: RWARESmall2AgAdapter,
    GameId.RWARE_SMALL_4AG: RWARESmall4AgAdapter,
    GameId.RWARE_MEDIUM_2AG: RWAREMedium2AgAdapter,
    GameId.RWARE_MEDIUM_4AG: RWAREMedium4AgAdapter,
    GameId.RWARE_MEDIUM_4AG_EASY: RWAREMedium4AgEasyAdapter,
    GameId.RWARE_MEDIUM_4AG_HARD: RWAREMedium4AgHardAdapter,
    GameId.RWARE_LARGE_4AG: RWARELarge4AgAdapter,
    GameId.RWARE_LARGE_4AG_HARD: RWARELarge4AgHardAdapter,
    GameId.RWARE_LARGE_8AG: RWARELarge8AgAdapter,
    GameId.RWARE_LARGE_8AG_HARD: RWARELarge8AgHardAdapter,
}

ALL_RWARE_GAME_IDS: tuple[GameId, ...] = tuple(RWARE_ADAPTERS.keys())

__all__ = [
    "RWAREAdapter",
    "RWARE_ADAPTERS",
    "ALL_RWARE_GAME_IDS",
]
