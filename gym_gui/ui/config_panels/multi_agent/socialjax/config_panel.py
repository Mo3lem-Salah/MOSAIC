"""Configuration widgets for SocialJax social dilemma environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from PyQt6 import QtWidgets

from gym_gui.core.ui.game_config.game_configs import SocialJaxConfig
from gym_gui.core.enums import GameId

SOCIALJAX_GAME_IDS: tuple[GameId, ...] = (
    GameId.SOCIALJAX_COIN_GAME,
    GameId.SOCIALJAX_HARVEST_COMMON_OPEN,
    GameId.SOCIALJAX_CLEAN_UP,
    GameId.SOCIALJAX_COOP_MINING,
    GameId.SOCIALJAX_TERRITORY_OPEN,
    GameId.SOCIALJAX_PD_ARENA,
    GameId.SOCIALJAX_MUSHROOMS,
    GameId.SOCIALJAX_GIFT,
    GameId.SOCIALJAX_LB_FORAGING,
)

# Per-env defaults: (default_num_agents, default_num_inner_steps, description)
_ENV_DEFAULTS: dict[str, tuple[int, int, str]] = {
    "coin_game":           (2,  1000, "2 agents. Coin collection: collect own colour (+1), avoid stealing (hurts partner -2)."),
    "harvest_common_open": (7,  1000, "7 agents. Commons harvest: regrowable apples, tragedy of the commons if over-harvested."),
    "clean_up":            (7,  1000, "7 agents. Altruism dilemma: agents must clean river pollution to unlock apple regrowth."),
    "coop_mining":         (4,  1000, "4 agents. Coordination: gold requires simultaneous multi-agent mining at the same tile."),
    "territory_open":      (9,  1000, "9 agents. Territory: claim and defend spatial zones using zap beam."),
    "pd_arena":            (2,  50,   "2 agents. Iterated prisoner's dilemma played out in a grid world arena."),
    "mushrooms":           (6,  1000, "6 agents. Externalities: collecting toxic mushrooms harms neighbouring agents."),
    "gift":                (6,  1000, "6 agents. Reciprocity: agents can gift resources; tests trust and social norms."),
    "lb_foraging":         (4,  100,  "4 agents. Coordination: items require simultaneous pickup by agents with sufficient level sum."),
}


@dataclass(slots=True)
class ControlCallbacks:
    """Bridge callbacks for propagating UI changes to session state."""

    on_change: Callable[[str, Any], None]


def build_socialjax_controls(
    *,
    parent: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
    game_id: GameId,
    overrides: Dict[str, Any],
    defaults: SocialJaxConfig | None = None,
    callbacks: ControlCallbacks | None = None,
) -> None:
    """Populate SocialJax-specific configuration widgets."""

    def emit(key: str, value: Any) -> None:
        overrides[key] = value
        if callbacks:
            callbacks.on_change(key, value)

    cfg = defaults if isinstance(defaults, SocialJaxConfig) else SocialJaxConfig()
    env_id = game_id.value.replace("socialjax/", "")
    default_agents, default_steps, desc = _ENV_DEFAULTS.get(
        env_id, (2, 1000, "SocialJax social dilemma environment.")
    )

    # -------- Num Agents --------
    agents_spin = QtWidgets.QSpinBox(parent)
    agents_spin.setRange(2, 16)
    current_agents = overrides.get("num_agents") or (cfg.num_agents if cfg.num_agents is not None else default_agents)
    agents_spin.setValue(int(current_agents))
    agents_spin.setToolTip(
        f"Number of agents in the environment.\n"
        f"Default for {env_id}: {default_agents}.\n"
        "Changing requires a reload (not just reset)."
    )

    def on_agents_changed(val: int) -> None:
        emit("num_agents", val)

    agents_spin.valueChanged.connect(on_agents_changed)
    layout.addRow("Num Agents", agents_spin)

    # -------- Episode Length (num_inner_steps) --------
    steps_spin = QtWidgets.QSpinBox(parent)
    steps_spin.setRange(10, 5000)
    steps_spin.setSingleStep(50)
    current_steps = overrides.get("num_inner_steps") or (
        cfg.num_inner_steps if cfg.num_inner_steps is not None else default_steps
    )
    steps_spin.setValue(int(current_steps))
    steps_spin.setToolTip(
        f"Steps per episode (num_inner_steps).\n"
        f"Default for {env_id}: {default_steps}.\n"
        "For human play, 100-300 steps is more practical than the MARL default."
    )

    def on_steps_changed(val: int) -> None:
        emit("num_inner_steps", val)

    steps_spin.valueChanged.connect(on_steps_changed)
    layout.addRow("Episode Length", steps_spin)

    # -------- Shared Rewards --------
    shared_rewards_cb = QtWidgets.QCheckBox(parent)
    current_shared = overrides.get("shared_rewards", cfg.shared_rewards)
    shared_rewards_cb.setChecked(bool(current_shared))
    shared_rewards_cb.setToolTip(
        "If checked, all agents share the mean reward (encourages cooperation).\n"
        "If unchecked, each agent receives its own individual reward."
    )

    def on_shared_rewards_changed(state: int) -> None:
        emit("shared_rewards", bool(state))

    shared_rewards_cb.stateChanged.connect(on_shared_rewards_changed)
    layout.addRow("Shared Rewards", shared_rewards_cb)

    # -------- Environment description --------
    info_label = QtWidgets.QLabel(
        f"<i><b>SocialJax:</b> {desc}<br><br>"
        "Pure JAX (CPU/GPU/TPU). Parallel stepping (all agents act simultaneously).<br>"
        "<b>Keys:</b> Q/E=turn, WASD=move, Space=stay, F=special, Z=special2</i>",
        parent,
    )
    info_label.setWordWrap(True)
    layout.addRow("", info_label)
