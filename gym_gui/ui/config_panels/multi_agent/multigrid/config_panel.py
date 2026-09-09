"""Configuration widgets for MultiGrid multi-agent environments.

MultiGrid is a multi-agent extension of MiniGrid. Multiple agents act
simultaneously on a shared grid with walls, doors, keys, goals, etc.

Two types of environments:
- INI multigrid (2023+): Gymnasium-based, configurable agent count (default: 2)
- Legacy gym-multigrid: Fixed agent counts (Soccer=4, Collect=3)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from PyQt6 import QtWidgets

from gym_gui.core.enums import GameId
from gym_gui.core.ui.game_config.game_configs import MultiGridConfig

# All MultiGrid game IDs (v7.0.0: TeamObs removed, 30 asymmetric variants added)
MULTIGRID_GAME_IDS: tuple[GameId, ...] = (
    # MOSAIC multigrid — Soccer (S)
    GameId.MOSAIC_MULTIGRID_S_G_1V0,           # 1 agent (Green solo)
    GameId.MOSAIC_MULTIGRID_S_B_0V1,           # 1 agent (Blue solo)
    GameId.MOSAIC_MULTIGRID_S_1V1_INDAGOBS,    # 2 agents (1v1)
    GameId.MOSAIC_MULTIGRID_S_2V2_INDAGOBS,    # 4 agents (2v2)
    GameId.MOSAIC_MULTIGRID_S_3V3_INDAGOBS,    # 6 agents (3v3)
    GameId.MOSAIC_MULTIGRID_S_4V4_INDAGOBS,    # 8 agents (4v4)
    GameId.MOSAIC_MULTIGRID_S_G_2V0_INDAGOBS,  # 2 agents (Green coop)
    GameId.MOSAIC_MULTIGRID_S_G_3V0_INDAGOBS,  # 3 agents (Green coop)
    GameId.MOSAIC_MULTIGRID_S_G_4V0_INDAGOBS,  # 4 agents (Green coop)
    GameId.MOSAIC_MULTIGRID_S_G_5V0_INDAGOBS,  # 5 agents (Green coop)
    GameId.MOSAIC_MULTIGRID_S_G_6V0_INDAGOBS,  # 6 agents (Green coop)
    GameId.MOSAIC_MULTIGRID_S_B_0V2_INDAGOBS,  # 2 agents (Blue coop)
    GameId.MOSAIC_MULTIGRID_S_B_0V3_INDAGOBS,  # 3 agents (Blue coop)
    GameId.MOSAIC_MULTIGRID_S_B_0V4_INDAGOBS,  # 4 agents (Blue coop)
    GameId.MOSAIC_MULTIGRID_S_B_0V5_INDAGOBS,  # 5 agents (Blue coop)
    GameId.MOSAIC_MULTIGRID_S_B_0V6_INDAGOBS,  # 6 agents (Blue coop)
    # MOSAIC multigrid — Soccer asymmetric competitive (NEW in v7.0.0)
    GameId.MOSAIC_MULTIGRID_S_1V2_INDAGOBS,    # 3 agents (1v2 asymmetric)
    GameId.MOSAIC_MULTIGRID_S_2V1_INDAGOBS,    # 3 agents (2v1 asymmetric)
    GameId.MOSAIC_MULTIGRID_S_1V3_INDAGOBS,    # 4 agents (1v3 asymmetric)
    GameId.MOSAIC_MULTIGRID_S_3V1_INDAGOBS,    # 4 agents (3v1 asymmetric)
    GameId.MOSAIC_MULTIGRID_S_1V4_INDAGOBS,    # 5 agents (1v4 asymmetric)
    GameId.MOSAIC_MULTIGRID_S_4V1_INDAGOBS,    # 5 agents (4v1 asymmetric)
    GameId.MOSAIC_MULTIGRID_S_2V3_INDAGOBS,    # 5 agents (2v3 asymmetric)
    GameId.MOSAIC_MULTIGRID_S_3V2_INDAGOBS,    # 5 agents (3v2 asymmetric)
    GameId.MOSAIC_MULTIGRID_S_2V4_INDAGOBS,    # 6 agents (2v4 asymmetric)
    GameId.MOSAIC_MULTIGRID_S_4V2_INDAGOBS,    # 6 agents (4v2 asymmetric)
    # MOSAIC multigrid — Basketball (BB)
    GameId.MOSAIC_MULTIGRID_BB_G_1V0,          # 1 agent (Green solo)
    GameId.MOSAIC_MULTIGRID_BB_B_0V1,          # 1 agent (Blue solo)
    GameId.MOSAIC_MULTIGRID_BB_1V1_INDAGOBS,   # 2 agents (1v1)
    GameId.MOSAIC_MULTIGRID_BB_2V2_INDAGOBS,   # 4 agents (2v2)
    GameId.MOSAIC_MULTIGRID_BB_3V3_INDAGOBS,   # 6 agents (3v3)
    GameId.MOSAIC_MULTIGRID_BB_4V4_INDAGOBS,   # 8 agents (4v4)
    GameId.MOSAIC_MULTIGRID_BB_G_2V0_INDAGOBS, # 2 agents (Green coop)
    GameId.MOSAIC_MULTIGRID_BB_G_3V0_INDAGOBS, # 3 agents (Green coop)
    GameId.MOSAIC_MULTIGRID_BB_G_4V0_INDAGOBS, # 4 agents (Green coop)
    GameId.MOSAIC_MULTIGRID_BB_G_5V0_INDAGOBS, # 5 agents (Green coop)
    GameId.MOSAIC_MULTIGRID_BB_G_6V0_INDAGOBS, # 6 agents (Green coop)
    GameId.MOSAIC_MULTIGRID_BB_B_0V2_INDAGOBS, # 2 agents (Blue coop)
    GameId.MOSAIC_MULTIGRID_BB_B_0V3_INDAGOBS, # 3 agents (Blue coop)
    GameId.MOSAIC_MULTIGRID_BB_B_0V4_INDAGOBS, # 4 agents (Blue coop)
    GameId.MOSAIC_MULTIGRID_BB_B_0V5_INDAGOBS, # 5 agents (Blue coop)
    GameId.MOSAIC_MULTIGRID_BB_B_0V6_INDAGOBS, # 6 agents (Blue coop)
    # MOSAIC multigrid — Basketball asymmetric competitive (NEW in v7.0.0)
    GameId.MOSAIC_MULTIGRID_BB_1V2_INDAGOBS,   # 3 agents (1v2 asymmetric)
    GameId.MOSAIC_MULTIGRID_BB_2V1_INDAGOBS,   # 3 agents (2v1 asymmetric)
    GameId.MOSAIC_MULTIGRID_BB_1V3_INDAGOBS,   # 4 agents (1v3 asymmetric)
    GameId.MOSAIC_MULTIGRID_BB_3V1_INDAGOBS,   # 4 agents (3v1 asymmetric)
    GameId.MOSAIC_MULTIGRID_BB_1V4_INDAGOBS,   # 5 agents (1v4 asymmetric)
    GameId.MOSAIC_MULTIGRID_BB_4V1_INDAGOBS,   # 5 agents (4v1 asymmetric)
    GameId.MOSAIC_MULTIGRID_BB_2V3_INDAGOBS,   # 5 agents (2v3 asymmetric)
    GameId.MOSAIC_MULTIGRID_BB_3V2_INDAGOBS,   # 5 agents (3v2 asymmetric)
    GameId.MOSAIC_MULTIGRID_BB_2V4_INDAGOBS,   # 6 agents (2v4 asymmetric)
    GameId.MOSAIC_MULTIGRID_BB_4V2_INDAGOBS,   # 6 agents (4v2 asymmetric)
    # MOSAIC multigrid — AmericanFootball (AF)
    GameId.MOSAIC_MULTIGRID_AF_G_1V0,          # 1 agent (Green solo)
    GameId.MOSAIC_MULTIGRID_AF_B_0V1,          # 1 agent (Blue solo)
    GameId.MOSAIC_MULTIGRID_AF_1V1_INDAGOBS,   # 2 agents (1v1)
    GameId.MOSAIC_MULTIGRID_AF_2V2_INDAGOBS,   # 4 agents (2v2)
    GameId.MOSAIC_MULTIGRID_AF_3V3_INDAGOBS,   # 6 agents (3v3)
    GameId.MOSAIC_MULTIGRID_AF_4V4_INDAGOBS,   # 8 agents (4v4)
    GameId.MOSAIC_MULTIGRID_AF_G_2V0_INDAGOBS, # 2 agents (Green coop)
    GameId.MOSAIC_MULTIGRID_AF_G_3V0_INDAGOBS, # 3 agents (Green coop)
    GameId.MOSAIC_MULTIGRID_AF_G_4V0_INDAGOBS, # 4 agents (Green coop)
    GameId.MOSAIC_MULTIGRID_AF_G_5V0_INDAGOBS, # 5 agents (Green coop)
    GameId.MOSAIC_MULTIGRID_AF_G_6V0_INDAGOBS, # 6 agents (Green coop)
    GameId.MOSAIC_MULTIGRID_AF_B_0V2_INDAGOBS, # 2 agents (Blue coop)
    GameId.MOSAIC_MULTIGRID_AF_B_0V3_INDAGOBS, # 3 agents (Blue coop)
    GameId.MOSAIC_MULTIGRID_AF_B_0V4_INDAGOBS, # 4 agents (Blue coop)
    GameId.MOSAIC_MULTIGRID_AF_B_0V5_INDAGOBS, # 5 agents (Blue coop)
    GameId.MOSAIC_MULTIGRID_AF_B_0V6_INDAGOBS, # 6 agents (Blue coop)
    # MOSAIC multigrid — American Football asymmetric competitive (NEW in v7.0.0)
    GameId.MOSAIC_MULTIGRID_AF_1V2_INDAGOBS,   # 3 agents (1v2 asymmetric)
    GameId.MOSAIC_MULTIGRID_AF_2V1_INDAGOBS,   # 3 agents (2v1 asymmetric)
    GameId.MOSAIC_MULTIGRID_AF_1V3_INDAGOBS,   # 4 agents (1v3 asymmetric)
    GameId.MOSAIC_MULTIGRID_AF_3V1_INDAGOBS,   # 4 agents (3v1 asymmetric)
    GameId.MOSAIC_MULTIGRID_AF_1V4_INDAGOBS,   # 5 agents (1v4 asymmetric)
    GameId.MOSAIC_MULTIGRID_AF_4V1_INDAGOBS,   # 5 agents (4v1 asymmetric)
    GameId.MOSAIC_MULTIGRID_AF_2V3_INDAGOBS,   # 5 agents (2v3 asymmetric)
    GameId.MOSAIC_MULTIGRID_AF_3V2_INDAGOBS,   # 5 agents (3v2 asymmetric)
    GameId.MOSAIC_MULTIGRID_AF_2V4_INDAGOBS,   # 6 agents (2v4 asymmetric)
    GameId.MOSAIC_MULTIGRID_AF_4V2_INDAGOBS,   # 6 agents (4v2 asymmetric)
    # MOSAIC multigrid — Collect (C)
    GameId.MOSAIC_MULTIGRID_C_INDAGOBS,        # 3 agents
    GameId.MOSAIC_MULTIGRID_C_1V1_INDAGOBS,    # 2 agents (1v1)
    GameId.MOSAIC_MULTIGRID_C_2V2_INDAGOBS,    # 4 agents (2v2)
    # INI multigrid (configurable agent count, default 2)
    GameId.INI_MULTIGRID_BLOCKED_UNLOCK_PICKUP,
    GameId.INI_MULTIGRID_EMPTY_5X5,
    GameId.INI_MULTIGRID_EMPTY_RANDOM_5X5,
    GameId.INI_MULTIGRID_EMPTY_6X6,
    GameId.INI_MULTIGRID_EMPTY_RANDOM_6X6,
    GameId.INI_MULTIGRID_EMPTY_8X8,
    GameId.INI_MULTIGRID_EMPTY_16X16,
    GameId.INI_MULTIGRID_LOCKED_HALLWAY_2ROOMS,
    GameId.INI_MULTIGRID_LOCKED_HALLWAY_4ROOMS,
    GameId.INI_MULTIGRID_LOCKED_HALLWAY_6ROOMS,
    GameId.INI_MULTIGRID_PLAYGROUND,
    GameId.INI_MULTIGRID_RED_BLUE_DOORS_6X6,
    GameId.INI_MULTIGRID_RED_BLUE_DOORS_8X8,
)

# MOSAIC environments with agent counts (v7.0.0)
MOSAIC_AGENT_COUNTS: dict[GameId, int] = {
    # Soccer
    GameId.MOSAIC_MULTIGRID_S_G_1V0: 1,
    GameId.MOSAIC_MULTIGRID_S_B_0V1: 1,
    GameId.MOSAIC_MULTIGRID_S_1V1_INDAGOBS: 2,
    GameId.MOSAIC_MULTIGRID_S_2V2_INDAGOBS: 4,
    GameId.MOSAIC_MULTIGRID_S_3V3_INDAGOBS: 6,
    GameId.MOSAIC_MULTIGRID_S_4V4_INDAGOBS: 8,
    GameId.MOSAIC_MULTIGRID_S_G_2V0_INDAGOBS: 2,
    GameId.MOSAIC_MULTIGRID_S_G_3V0_INDAGOBS: 3,
    GameId.MOSAIC_MULTIGRID_S_G_4V0_INDAGOBS: 4,
    GameId.MOSAIC_MULTIGRID_S_G_5V0_INDAGOBS: 5,
    GameId.MOSAIC_MULTIGRID_S_G_6V0_INDAGOBS: 6,
    GameId.MOSAIC_MULTIGRID_S_B_0V2_INDAGOBS: 2,
    GameId.MOSAIC_MULTIGRID_S_B_0V3_INDAGOBS: 3,
    GameId.MOSAIC_MULTIGRID_S_B_0V4_INDAGOBS: 4,
    GameId.MOSAIC_MULTIGRID_S_B_0V5_INDAGOBS: 5,
    GameId.MOSAIC_MULTIGRID_S_B_0V6_INDAGOBS: 6,
    # Soccer asymmetric competitive (v7.0.0)
    GameId.MOSAIC_MULTIGRID_S_1V2_INDAGOBS: 3,
    GameId.MOSAIC_MULTIGRID_S_2V1_INDAGOBS: 3,
    GameId.MOSAIC_MULTIGRID_S_1V3_INDAGOBS: 4,
    GameId.MOSAIC_MULTIGRID_S_3V1_INDAGOBS: 4,
    GameId.MOSAIC_MULTIGRID_S_1V4_INDAGOBS: 5,
    GameId.MOSAIC_MULTIGRID_S_4V1_INDAGOBS: 5,
    GameId.MOSAIC_MULTIGRID_S_2V3_INDAGOBS: 5,
    GameId.MOSAIC_MULTIGRID_S_3V2_INDAGOBS: 5,
    GameId.MOSAIC_MULTIGRID_S_2V4_INDAGOBS: 6,
    GameId.MOSAIC_MULTIGRID_S_4V2_INDAGOBS: 6,
    # Basketball
    GameId.MOSAIC_MULTIGRID_BB_G_1V0: 1,
    GameId.MOSAIC_MULTIGRID_BB_B_0V1: 1,
    GameId.MOSAIC_MULTIGRID_BB_1V1_INDAGOBS: 2,
    GameId.MOSAIC_MULTIGRID_BB_2V2_INDAGOBS: 4,
    GameId.MOSAIC_MULTIGRID_BB_3V3_INDAGOBS: 6,
    GameId.MOSAIC_MULTIGRID_BB_4V4_INDAGOBS: 8,
    GameId.MOSAIC_MULTIGRID_BB_G_2V0_INDAGOBS: 2,
    GameId.MOSAIC_MULTIGRID_BB_G_3V0_INDAGOBS: 3,
    GameId.MOSAIC_MULTIGRID_BB_G_4V0_INDAGOBS: 4,
    GameId.MOSAIC_MULTIGRID_BB_G_5V0_INDAGOBS: 5,
    GameId.MOSAIC_MULTIGRID_BB_G_6V0_INDAGOBS: 6,
    GameId.MOSAIC_MULTIGRID_BB_B_0V2_INDAGOBS: 2,
    GameId.MOSAIC_MULTIGRID_BB_B_0V3_INDAGOBS: 3,
    GameId.MOSAIC_MULTIGRID_BB_B_0V4_INDAGOBS: 4,
    GameId.MOSAIC_MULTIGRID_BB_B_0V5_INDAGOBS: 5,
    GameId.MOSAIC_MULTIGRID_BB_B_0V6_INDAGOBS: 6,
    # Basketball asymmetric competitive (v7.0.0)
    GameId.MOSAIC_MULTIGRID_BB_1V2_INDAGOBS: 3,
    GameId.MOSAIC_MULTIGRID_BB_2V1_INDAGOBS: 3,
    GameId.MOSAIC_MULTIGRID_BB_1V3_INDAGOBS: 4,
    GameId.MOSAIC_MULTIGRID_BB_3V1_INDAGOBS: 4,
    GameId.MOSAIC_MULTIGRID_BB_1V4_INDAGOBS: 5,
    GameId.MOSAIC_MULTIGRID_BB_4V1_INDAGOBS: 5,
    GameId.MOSAIC_MULTIGRID_BB_2V3_INDAGOBS: 5,
    GameId.MOSAIC_MULTIGRID_BB_3V2_INDAGOBS: 5,
    GameId.MOSAIC_MULTIGRID_BB_2V4_INDAGOBS: 6,
    GameId.MOSAIC_MULTIGRID_BB_4V2_INDAGOBS: 6,
    # American Football
    GameId.MOSAIC_MULTIGRID_AF_G_1V0: 1,
    GameId.MOSAIC_MULTIGRID_AF_B_0V1: 1,
    GameId.MOSAIC_MULTIGRID_AF_1V1_INDAGOBS: 2,
    GameId.MOSAIC_MULTIGRID_AF_2V2_INDAGOBS: 4,
    GameId.MOSAIC_MULTIGRID_AF_3V3_INDAGOBS: 6,
    GameId.MOSAIC_MULTIGRID_AF_4V4_INDAGOBS: 8,
    GameId.MOSAIC_MULTIGRID_AF_G_2V0_INDAGOBS: 2,
    GameId.MOSAIC_MULTIGRID_AF_G_3V0_INDAGOBS: 3,
    GameId.MOSAIC_MULTIGRID_AF_G_4V0_INDAGOBS: 4,
    GameId.MOSAIC_MULTIGRID_AF_G_5V0_INDAGOBS: 5,
    GameId.MOSAIC_MULTIGRID_AF_G_6V0_INDAGOBS: 6,
    GameId.MOSAIC_MULTIGRID_AF_B_0V2_INDAGOBS: 2,
    GameId.MOSAIC_MULTIGRID_AF_B_0V3_INDAGOBS: 3,
    GameId.MOSAIC_MULTIGRID_AF_B_0V4_INDAGOBS: 4,
    GameId.MOSAIC_MULTIGRID_AF_B_0V5_INDAGOBS: 5,
    GameId.MOSAIC_MULTIGRID_AF_B_0V6_INDAGOBS: 6,
    # American Football asymmetric competitive (v7.0.0)
    GameId.MOSAIC_MULTIGRID_AF_1V2_INDAGOBS: 3,
    GameId.MOSAIC_MULTIGRID_AF_2V1_INDAGOBS: 3,
    GameId.MOSAIC_MULTIGRID_AF_1V3_INDAGOBS: 4,
    GameId.MOSAIC_MULTIGRID_AF_3V1_INDAGOBS: 4,
    GameId.MOSAIC_MULTIGRID_AF_1V4_INDAGOBS: 5,
    GameId.MOSAIC_MULTIGRID_AF_4V1_INDAGOBS: 5,
    GameId.MOSAIC_MULTIGRID_AF_2V3_INDAGOBS: 5,
    GameId.MOSAIC_MULTIGRID_AF_3V2_INDAGOBS: 5,
    GameId.MOSAIC_MULTIGRID_AF_2V4_INDAGOBS: 6,
    GameId.MOSAIC_MULTIGRID_AF_4V2_INDAGOBS: 6,
    # Collect
    GameId.MOSAIC_MULTIGRID_C_INDAGOBS: 3,
    GameId.MOSAIC_MULTIGRID_C_1V1_INDAGOBS: 2,
    GameId.MOSAIC_MULTIGRID_C_2V2_INDAGOBS: 4,
}

# INI environments that support configurable agent count
INI_CONFIGURABLE_GAMES: tuple[GameId, ...] = tuple(
    gid for gid in MULTIGRID_GAME_IDS if gid not in MOSAIC_AGENT_COUNTS
)


@dataclass(slots=True)
class ControlCallbacks:
    """Bridge callbacks for propagating UI changes to session state."""

    on_change: Callable[[str, Any], None]


def build_multigrid_controls(
    *,
    parent: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
    game_id: GameId,
    overrides: Dict[str, Any],
    defaults: MultiGridConfig | None = None,
    callbacks: ControlCallbacks | None = None,
) -> None:
    """Populate MultiGrid-specific configuration widgets.

    Args:
        parent: Parent widget for controls.
        layout: Form layout to add controls to.
        game_id: The selected MultiGrid game ID.
        overrides: Dictionary to store user-modified values.
        defaults: Default configuration values.
        callbacks: Optional callbacks for value changes.
    """
    def emit(key: str, value: Any) -> None:
        overrides[key] = value
        if callbacks:
            callbacks.on_change(key, value)

    cfg = defaults if isinstance(defaults, MultiGridConfig) else MultiGridConfig()
    is_mosaic = game_id in MOSAIC_AGENT_COUNTS

    # -------- Number of Agents --------
    if is_mosaic:
        # MOSAIC environments: Show agent count (read-only)
        agent_count = MOSAIC_AGENT_COUNTS[game_id]
        agents_label = QtWidgets.QLabel(f"{agent_count}", parent)
        agents_label.setToolTip(
            f"This MOSAIC environment has {agent_count} agents.\n"
            "MOSAIC multigrid environments have predefined agent counts."
        )
        layout.addRow("Number of Agents", agents_label)
    else:
        # INI environments: Spinbox for configurable agent count
        agents_spin = QtWidgets.QSpinBox(parent)
        agents_spin.setRange(2, 16)  # Min 2 for multi-agent, max 16 reasonable limit

        # Get current value from overrides or use default of 2
        current_agents = overrides.get("num_agents", cfg.num_agents)
        if current_agents is None:
            current_agents = 2  # Default for INI environments
        agents_spin.setValue(current_agents)

        def on_agents_changed(value: int) -> None:
            emit("num_agents", value)

        agents_spin.valueChanged.connect(on_agents_changed)
        agents_spin.setToolTip(
            "Number of agents in the environment.\n"
            "INI multigrid environments support 2-16 agents.\n"
            "Default: 2 (cooperative pair)"
        )
        layout.addRow("Number of Agents", agents_spin)

        # Set initial value in overrides if not already set
        if "num_agents" not in overrides:
            overrides["num_agents"] = current_agents

    # -------- View Size (MOSAIC only) --------
    if is_mosaic:
        view_spin = QtWidgets.QSpinBox(parent)
        view_spin.setRange(3, 15)
        view_spin.setSingleStep(2)  # Odd values preferred (3, 5, 7, 9, ...)
        view_spin.setSpecialValueText("3 (default)")

        current_view = overrides.get("view_size", cfg.view_size)
        if current_view is None or current_view == 3:
            view_spin.setValue(3)  # Shows "3 (default)"
        else:
            view_spin.setValue(int(current_view))

        def on_view_changed(value: int) -> None:
            # Only emit override if different from default (3)
            emit("view_size", value if value != 3 else None)

        view_spin.valueChanged.connect(on_view_changed)
        view_spin.setToolTip(
            "Agent view size (NxN partial observation window).\n"
            "Default: 3 (9 cells visible, 7.1% of 16x11 grid).\n"
            "Larger values give agents more information:\n"
            "  5 = 25 cells (19.8%)\n"
            "  7 = 49 cells (38.9%)\n"
            "  9 = 81 cells (64.3%)\n"
            "Must be odd for symmetric view. Even values are rounded up."
        )
        layout.addRow("View Size", view_spin)

    # -------- Seed (optional) --------
    seed_spin = QtWidgets.QSpinBox(parent)
    seed_spin.setRange(0, 2147483647)
    seed_spin.setSpecialValueText("Random")

    current_seed = overrides.get("seed", cfg.seed)
    if current_seed is None:
        seed_spin.setValue(0)  # Shows "Random"
    else:
        seed_spin.setValue(current_seed)

    def on_seed_changed(value: int) -> None:
        emit("seed", value if value > 0 else None)

    seed_spin.valueChanged.connect(on_seed_changed)
    seed_spin.setToolTip(
        "Random seed for reproducibility.\n"
        "Set to 0 for random seed, or specify a value for reproducible runs."
    )
    layout.addRow("Seed", seed_spin)

    # -------- Highlight Agent Views --------
    highlight_check = QtWidgets.QCheckBox("Highlight agent view cones", parent)
    current_highlight = overrides.get("highlight", cfg.highlight)
    highlight_check.setChecked(current_highlight)

    def on_highlight_changed(checked: bool) -> None:
        emit("highlight", checked)

    highlight_check.stateChanged.connect(lambda state: on_highlight_changed(state == 2))
    highlight_check.setToolTip(
        "When enabled, renders colored overlays showing each agent's field of view."
    )
    layout.addRow("", highlight_check)

    # -------- Info Label --------
    if is_mosaic:
        env_type = "MOSAIC MultiGrid"
        api_info = "Team-based competitive (Soccer/Collect)"
    else:
        env_type = "INI MultiGrid"
        api_info = "Cooperative exploration"

    info_label = QtWidgets.QLabel(
        f"<i><b>{env_type}</b> ({api_info})<br><br>"
        "<b>Controls:</b> Arrow keys (move), Space (toggle/pickup/drop)<br>"
        "For multi-human play, assign keyboards in Human Control tab.</i>",
        parent,
    )
    info_label.setWordWrap(True)
    layout.addRow("", info_label)
