"""Craftax environment adapters — Michael Matthews's JAX-accelerated
extension of Crafter.

Registration mirrors alem-env's pattern
(``3rd_party/environments/alem-env/alem/alem_env.py::make_alem_env_from_name``):

    * a top-level ``make_craftax_adapter_from_name(name)`` factory dispatches
      registered Craftax variant names to adapter classes;
    * each variant is a subclass with an overridden ``ENV_NAME`` class
      attribute — the string that Craftax's own
      ``craftax.craftax_env.make_craftax_env_from_name`` accepts;
    * lazy imports inside each factory branch so pixel-mode's pygame deps
      don't get loaded when a caller wants the symbolic variant;
    * unknown names raise ``ValueError`` — same failure mode alem-env uses.

Craftax is JAX-native. This adapter bridges Craftax's
``env.step(rng, state, action, params) -> (obs, next_state, reward, done, info)``
JAX-array API onto MOSAIC's :class:`EnvironmentAdapter` contract, which
expects numpy arrays. The adapter persists a JAX ``PRNGKey`` and Craftax
``EnvState`` across steps within an episode and calls ``jax.device_get()``
on obs/reward/done to move them to host memory before packaging into
:class:`AdapterStep`.

Paper: Matthews, M. et al. (2024).
    Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning.
    arXiv:2402.16801.  https://arxiv.org/abs/2402.16801

Repository: https://github.com/MichaelTMatthews/Craftax
Vendored:   3rd_party/environments/Craftax/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from gym_gui.core.adapters.base import (
    AdapterContext,
    AdapterStep,
    EnvironmentAdapter,
)
from gym_gui.core.enums import ControlMode, GameId, RenderMode
from gym_gui.logging_config.log_constants import (
    LOG_ADAPTER_ENV_RESET,
    LOG_ADAPTER_STEP_SUMMARY,
)

# Top-level Craftax imports are deferred to :meth:`_lazy_import_craftax` so
# that MOSAIC can list this adapter in the registry without pulling in the
# ~200MB JAX toolchain when the adapter isn't actually being used.
try:  # pragma: no cover - import guard exercised in integration tests
    import jax as _jax
except ImportError:  # pragma: no cover - handled gracefully at runtime
    _jax = None  # type: ignore[assignment]


_CRAFTAX_STEP_LOG_FREQUENCY = 100


# ---------------------------------------------------------------------------
# Config — inline for now; can be moved to gym_gui/config/game_configs.py
# later without changing the adapter's public API.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CraftaxConfig:
    """Configuration for a Craftax adapter run.

    All fields have sensible defaults so an adapter can be constructed with
    ``CraftaxConfig()`` and immediately loaded. Advanced Craftax gameplay
    parameters (world size, difficulty knobs) are threaded through Craftax's
    own ``EnvParams`` / ``StaticEnvParams`` dataclasses; ``env_params_kwargs``
    and ``static_env_params_kwargs`` let callers override any of them without
    this adapter having to enumerate every field.
    """

    seed: int | None = 0
    auto_reset: bool = True
    env_params_kwargs: Mapping[str, Any] = field(default_factory=dict)
    static_env_params_kwargs: Mapping[str, Any] = field(default_factory=dict)
    render_mode: RenderMode = RenderMode.RGB_ARRAY


# ---------------------------------------------------------------------------
# Base adapter — shared JAX <-> numpy bridge + rng management + achievement
# passthrough. The four concrete variant adapters below only override the
# ``ENV_NAME`` class attribute (and ``OBS_MODE`` for docstring clarity).
# ---------------------------------------------------------------------------


class BaseCraftaxAdapter(EnvironmentAdapter[np.ndarray, int]):
    """Base class for all Craftax variant adapters.

    Subclasses override :attr:`ENV_NAME` with one of Craftax's registered
    env-name strings. All other logic is shared.
    """

    #: Name string accepted by ``craftax.craftax_env.make_craftax_env_from_name``.
    #: Override in subclasses.
    ENV_NAME: str = ""

    #: ``"symbolic"`` or ``"pixels"`` — documentation only, doesn't affect
    #: dispatch (that's controlled by ``ENV_NAME``).
    OBS_MODE: str = "symbolic"

    #: True for Craftax-Classic variants (Crafter-parity mechanics, 22
    #: achievements); False for full Craftax variants (133 achievements).
    IS_CLASSIC: bool = False

    default_render_mode = RenderMode.RGB_ARRAY
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    supported_control_modes = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    )

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: CraftaxConfig | None = None,
    ) -> None:
        super().__init__(context)
        self._config: CraftaxConfig = config if config is not None else CraftaxConfig()
        self._env: Any = None
        self._env_params: Any = None
        self._state: Any = None
        self._rng: Any = None
        self._last_obs: np.ndarray | None = None
        self._last_reward: float = 0.0
        self._last_done: bool = False
        self._step_counter: int = 0
        self._previous_achievements: dict[str, int] = {}
        assert self.ENV_NAME, (
            f"{type(self).__name__} must override ENV_NAME with a "
            "Craftax-recognised environment string."
        )

    # ------------------------------------------------------------------
    # Load / require
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Construct the underlying Craftax env + params.

        Called lazily on first :meth:`reset` (via :meth:`_require_env`) so
        importing this module doesn't force a JAX toolchain load.
        """
        if _jax is None:  # pragma: no cover
            raise ImportError(
                "Craftax adapter requires JAX; install via "
                "`pip install -r requirements/craftax.txt`"
            )

        # Delegate to Craftax's own factory — mirrors alem-env's practice of
        # trusting the upstream ``make_*_from_name`` dispatcher rather than
        # re-encoding the string -> class map inside MOSAIC.
        from craftax.craftax_env import make_craftax_env_from_name

        self._env = make_craftax_env_from_name(
            self.ENV_NAME, auto_reset=self._config.auto_reset
        )

        # Build the two Craftax params objects with any user overrides.
        env_params = self._env.default_params
        for k, v in self._config.env_params_kwargs.items():
            env_params = env_params.replace(**{k: v})
        self._env_params = env_params

        # Static params are set at __init__ of the env; if the user wants
        # non-default static params they must pass them via env_kwargs to a
        # future factory extension — first-pass adapter uses env defaults.

        # Seed the RNG.
        seed = self._config.seed if self._config.seed is not None else 0
        self._rng = _jax.random.PRNGKey(int(seed))

    def _require_env(self) -> Any:
        if self._env is None:
            self.load()
        return self._env

    # ------------------------------------------------------------------
    # Reset / step
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> AdapterStep[np.ndarray]:
        env = self._require_env()

        # Reseed if the caller asked for one.
        if seed is not None:
            self._rng = _jax.random.PRNGKey(int(seed))

        # Split off a reset key while advancing the persistent rng.
        self._rng, reset_key = _jax.random.split(self._rng)

        obs_jax, state = env.reset(reset_key, self._env_params)
        obs = np.asarray(_jax.device_get(obs_jax))

        self._state = state
        self._last_obs = obs
        self._last_reward = 0.0
        self._last_done = False
        self._step_counter = 0
        self._previous_achievements = {}

        info: dict[str, Any] = {
            "_craftax_env_name": self.ENV_NAME,
            "_craftax_obs_mode": self.OBS_MODE,
            "achievements": {},
        }

        applied_seed = seed if seed is not None else self._config.seed
        self.log_constant(
            LOG_ADAPTER_ENV_RESET,
            extra={
                "env_id": self.ENV_NAME,
                "seed": str(applied_seed) if applied_seed is not None else "None",
            },
        )
        return self._package_step(obs, 0.0, False, False, info)

    def step(self, action: int) -> AdapterStep[np.ndarray]:
        env = self._require_env()
        if self._state is None:
            raise RuntimeError("Craftax adapter.step() called before reset()")

        # Split rng for this step, keeping the persistent key advancing.
        self._rng, step_key = _jax.random.split(self._rng)

        obs_jax, next_state, reward_jax, done_jax, info_jax = env.step(
            step_key, self._state, int(action), self._env_params
        )
        obs = np.asarray(_jax.device_get(obs_jax))
        reward = float(_jax.device_get(reward_jax))
        done = bool(_jax.device_get(done_jax))

        # Move info scalars off-device but keep the raw dict for downstream
        # consumers who might want the JAX state (e.g. jaxmarl_worker).
        info: dict[str, Any] = {}
        for k, v in dict(info_jax).items():
            try:
                info[k] = _jax.device_get(v)
            except Exception:  # pragma: no cover - defensive
                info[k] = v

        self._state = next_state
        self._last_obs = obs
        self._last_reward = reward
        self._last_done = done
        self._step_counter += 1

        # Craftax auto-reset variants set done=True and immediately re-seed
        # internally — MOSAIC still surfaces the terminated/truncated split
        # so downstream episode-level accounting stays sane.
        terminated = bool(done)
        truncated = False  # Craftax has no separate truncation signal today.

        # Periodic structured log — matches Crafter adapter's convention.
        if self._step_counter % _CRAFTAX_STEP_LOG_FREQUENCY == 0:
            self.log_constant(
                LOG_ADAPTER_STEP_SUMMARY,
                extra={
                    "env_id": self.ENV_NAME,
                    "step": self._step_counter,
                    "reward": reward,
                    "done": done,
                },
            )

        return self._package_step(obs, reward, terminated, truncated, info)

    # ------------------------------------------------------------------
    # Render / close
    # ------------------------------------------------------------------

    def render(self, mode: RenderMode | str | None = None) -> dict[str, Any]:
        """Return an RGB frame wrapped in MOSAIC's rendering payload contract.

        Uses Craftax's own render_craftax_pixels() with BLOCK_PIXEL_SIZE_HUMAN
        (the constant Craftax's own play_craftax scripts use), so both symbolic
        and pixels variants surface a consistent human-scale view regardless of
        what the policy observes.
        """
        payload: dict[str, Any] = {
            "mode": RenderMode.RGB_ARRAY.value,
            "rgb": None,
            "game_id": self.ENV_NAME,
        }
        if self._state is None or _jax is None:
            return payload

        if self.IS_CLASSIC:
            from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_HUMAN
            from craftax.craftax_classic.renderer import render_craftax_pixels
        else:
            from craftax.craftax.constants import BLOCK_PIXEL_SIZE_HUMAN
            from craftax.craftax.renderer import render_craftax_pixels

        pixels_jax = render_craftax_pixels(self._state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)
        pixels = np.asarray(_jax.device_get(pixels_jax))
        # Craftax returns float32 in [0, 1]; MOSAIC's Qt image path expects uint8 [0, 255].
        if pixels.dtype != np.uint8:
            pixels = (pixels * 255.0).clip(0, 255).astype(np.uint8)

        payload["rgb"] = pixels
        return payload

    def close(self) -> None:
        """Release JAX state so JIT caches can be GC'd."""
        self._env = None
        self._env_params = None
        self._state = None
        self._rng = None
        self._last_obs = None


# ---------------------------------------------------------------------------
# Concrete variant adapters
# ---------------------------------------------------------------------------


class CraftaxSymbolicAdapter(BaseCraftaxAdapter):
    """Full Craftax with symbolic feature-vector observations.

    Env registered upstream as ``Craftax-Symbolic-v1``. 133 achievements,
    9-level dungeon, elite mobs, magic. Symbolic obs is a flat vector
    encoding the local 7x9 tile grid plus inventory + status stats.
    """

    ENV_NAME = "Craftax-Symbolic-v1"
    OBS_MODE = "symbolic"
    IS_CLASSIC = False


class CraftaxPixelsAdapter(BaseCraftaxAdapter):
    """Full Craftax with 63x63x3 RGB pixel observations.

    Env registered upstream as ``Craftax-Pixels-v1``. Same underlying game
    as :class:`CraftaxSymbolicAdapter` but observations are rendered pixels
    suitable for CNN policies and VLM agents.
    """

    ENV_NAME = "Craftax-Pixels-v1"
    OBS_MODE = "pixels"
    IS_CLASSIC = False


class CraftaxClassicSymbolicAdapter(BaseCraftaxAdapter):
    """Craftax-Classic (Crafter-parity mechanics) with symbolic obs.

    Env registered upstream as ``Craftax-Classic-Symbolic-v1``. Maps 1:1 to
    Crafter's 22-achievement game logic but runs inside JAX for 100-1000x
    faster rollouts. Useful for comparing JAX-based methods against Crafter
    baselines on identical mechanics.
    """

    ENV_NAME = "Craftax-Classic-Symbolic-v1"
    OBS_MODE = "symbolic"
    IS_CLASSIC = True


class CraftaxClassicPixelsAdapter(BaseCraftaxAdapter):
    """Craftax-Classic with 63x63x3 RGB pixel observations.

    Env registered upstream as ``Craftax-Classic-Pixels-v1``. Same game as
    :class:`CraftaxClassicSymbolicAdapter` but rendered pixels for CNN/VLM.
    """

    ENV_NAME = "Craftax-Classic-Pixels-v1"
    OBS_MODE = "pixels"
    IS_CLASSIC = True


# ---------------------------------------------------------------------------
# Alem-style factory — mirrors alem-env's make_alem_env_from_name exactly.
# ---------------------------------------------------------------------------


def make_craftax_adapter_from_name(
    name: str,
    config: CraftaxConfig | None = None,
    context: AdapterContext | None = None,
) -> BaseCraftaxAdapter:
    """Create a Craftax adapter for the given registered variant name.

    Pattern mirrors :func:`alem.alem_env.make_alem_env_from_name`:

        * dispatch on string ``name`` via if/elif;
        * lazy imports would live inside each branch if branches needed
          them — the four Craftax adapter classes are already imported at
          module top since they're pure-Python subclasses (the JAX/pygame
          cost is deferred to :meth:`BaseCraftaxAdapter.load`);
        * unknown ``name`` raises :class:`ValueError` with the same wording
          shape alem-env uses.

    Args:
        name: One of ``"Craftax-Symbolic-v1"``, ``"Craftax-Pixels-v1"``,
            ``"Craftax-Classic-Symbolic-v1"``, ``"Craftax-Classic-Pixels-v1"``.
        config: Optional adapter config. Uses defaults when omitted.
        context: Optional MOSAIC :class:`AdapterContext`. Uses defaults when
            omitted.

    Returns:
        A configured but not-yet-loaded adapter instance. Call
        :meth:`BaseCraftaxAdapter.reset` to load Craftax and start an episode.

    Raises:
        ValueError: When ``name`` is not one of the four registered variants.
    """
    if name == "Craftax-Symbolic-v1":
        return CraftaxSymbolicAdapter(config=config, context=context)
    elif name == "Craftax-Pixels-v1":
        return CraftaxPixelsAdapter(config=config, context=context)
    elif name == "Craftax-Classic-Symbolic-v1":
        return CraftaxClassicSymbolicAdapter(config=config, context=context)
    elif name == "Craftax-Classic-Pixels-v1":
        return CraftaxClassicPixelsAdapter(config=config, context=context)
    raise ValueError(f"Unknown craftax environment: {name}")


# ---------------------------------------------------------------------------
# MOSAIC-convention registry — mirrors CRAFTER_ADAPTERS / ALE_ADAPTERS shape,
# consumed by gym_gui/core/factories/adapters.py to dispatch on GameId.
# ---------------------------------------------------------------------------

CRAFTAX_ADAPTERS: dict[GameId, type[BaseCraftaxAdapter]] = {
    GameId.CRAFTAX_SYMBOLIC: CraftaxSymbolicAdapter,
    GameId.CRAFTAX_PIXELS: CraftaxPixelsAdapter,
    GameId.CRAFTAX_CLASSIC_SYMBOLIC: CraftaxClassicSymbolicAdapter,
    GameId.CRAFTAX_CLASSIC_PIXELS: CraftaxClassicPixelsAdapter,
}


__all__ = [
    "CraftaxConfig",
    "BaseCraftaxAdapter",
    "CraftaxSymbolicAdapter",
    "CraftaxPixelsAdapter",
    "CraftaxClassicSymbolicAdapter",
    "CraftaxClassicPixelsAdapter",
    "make_craftax_adapter_from_name",
    "CRAFTAX_ADAPTERS",
]
