"""Configuration widgets for HeMAC (Heterogeneous Multi-Agent Challenge) environments."""

from __future__ import annotations

from typing import Any, Callable, Dict

from PyQt6 import QtWidgets

from gym_gui.core.ui.game_config.game_configs import HeMACConfig
from gym_gui.core.enums import GameId

ALL_HEMAC_GAME_IDS: tuple[GameId, ...] = (
    GameId.HEMAC_SIMPLE_FLEET_1Q1O,
    GameId.HEMAC_SIMPLE_FLEET_3Q1O,
    GameId.HEMAC_SIMPLE_FLEET_5Q2O,
    GameId.HEMAC_FLEET_3Q1O,
    GameId.HEMAC_FLEET_10Q3O,
    GameId.HEMAC_FLEET_20Q5O,
    GameId.HEMAC_COMPLEX_FLEET_3Q1O1P,
    GameId.HEMAC_COMPLEX_FLEET_5Q2O1P,
)

# Per-variant metadata: GameId -> (tier, agents_summary, obstacles, default_max_cycles)
# Q=Quadcopter (drone), O=Observer, P=Provisioner (ground vehicle)
_SCENARIO_INFO: dict[GameId, tuple[str, str, str, int]] = {
    GameId.HEMAC_SIMPLE_FLEET_1Q1O:    ("Simple Fleet",  "2 agents (1Q + 1O)",    "None", 600),
    GameId.HEMAC_SIMPLE_FLEET_3Q1O:    ("Simple Fleet",  "4 agents (3Q + 1O)",    "None", 600),
    GameId.HEMAC_SIMPLE_FLEET_5Q2O:    ("Simple Fleet",  "7 agents (5Q + 2O)",    "None", 600),
    GameId.HEMAC_FLEET_3Q1O:           ("Fleet",         "4 agents (3Q + 1O)",    "2-5",  900),
    GameId.HEMAC_FLEET_10Q3O:          ("Fleet",         "13 agents (10Q + 3O)",  "2-5",  900),
    GameId.HEMAC_FLEET_20Q5O:          ("Fleet",         "25 agents (20Q + 5O)",  "2-5",  900),
    GameId.HEMAC_COMPLEX_FLEET_3Q1O1P: ("Complex Fleet", "5 agents (3Q+1O+1P)",   "2-5",  900),
    GameId.HEMAC_COMPLEX_FLEET_5Q2O1P: ("Complex Fleet", "8 agents (5Q+2O+1P)",   "2-5",  900),
}


def build_hemac_controls(
    *,
    parent: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
    game_id: GameId,
    overrides: Dict[str, Any],
    on_change: Callable[[Dict[str, Any]], None] | None = None,
) -> None:
    """Populate HeMAC-specific configuration widgets.

    Args:
        parent: Parent widget for new controls.
        layout: Form layout to add rows to.
        game_id: Currently selected HeMAC GameId.
        overrides: Mutable dict of current config overrides.
        on_change: Callback invoked with updated overrides dict.
    """
    defaults = HeMACConfig()

    def _emit() -> None:
        if on_change is not None:
            on_change(overrides)

    # ── Scenario info (read-only) ────────────────────────────────────
    info = _SCENARIO_INFO.get(game_id)
    default_cycles: int
    if info is not None:
        tier, agents_label, obstacles, default_cycles = info
        info_label = QtWidgets.QLabel(
            f"{tier} | {agents_label} | obstacles: {obstacles}",
            parent,
        )
        info_label.setStyleSheet("color: #aaa; font-style: italic;")
        layout.addRow("Scenario:", info_label)
    else:
        default_cycles = defaults.max_cycles

    # ── Max cycles ───────────────────────────────────────────────────
    cycles_spin = QtWidgets.QSpinBox(parent)
    cycles_spin.setRange(100, 5000)
    cycles_spin.setSingleStep(100)
    cycles_spin.setValue(int(overrides.get("max_cycles", default_cycles)))
    cycles_spin.setToolTip("Maximum steps per episode")

    def _on_cycles_changed(val: int) -> None:
        overrides["max_cycles"] = val
        _emit()

    cycles_spin.valueChanged.connect(_on_cycles_changed)
    layout.addRow("Max Cycles:", cycles_spin)

    # ── Seed ─────────────────────────────────────────────────────────
    seed_spin = QtWidgets.QSpinBox(parent)
    seed_spin.setRange(-1, 99999)
    seed_spin.setSpecialValueText("Random")
    seed_val = overrides.get("seed", defaults.seed)
    seed_spin.setValue(int(seed_val) if seed_val is not None and int(seed_val) >= 0 else -1)
    seed_spin.setToolTip("-1 = random seed each episode")

    def _on_seed_changed(val: int) -> None:
        overrides["seed"] = val if val >= 0 else None
        _emit()

    seed_spin.valueChanged.connect(_on_seed_changed)
    layout.addRow("Seed:", seed_spin)


__all__ = [
    "ALL_HEMAC_GAME_IDS",
    "build_hemac_controls",
]
