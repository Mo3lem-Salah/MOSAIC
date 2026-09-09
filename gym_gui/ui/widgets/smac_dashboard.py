"""SMAC/SMACv2 resource & army-composition dashboard widget.

Shown in the Status area (see ``control_panel.py``) when a SMAC or
SMACv2 environment is loaded and active. Displays:

- Resource bar: minerals, vespene, supply (food used/cap), army count,
  idle worker count -- all pulled from ``observation.player_common``
  (``PlayerCommon``), which SMAC already populates via the ``raw=True``
  interface option (no extra data needs to be requested from the engine).
- Army composition: one portrait icon per distinct unit type present on
  the allied team, with a live count, using unit icons extracted from the
  StarCraft II client's own asset archive (see
  ``gym_gui/assets/SMACv2/ATTRIBUTION.md``).

This widget is intentionally decoupled from any specific renderer mode --
it reads structured ``PlayerCommon``/``raw_data.units`` data, not pixels,
so it stays crisp at any resolution and works even when the render view
itself is showing the heatmap or classic PyGame renderer.

Unit identification uses resolved type **names** (e.g. ``"Marine_RL"``),
not raw ``unit_type`` IDs -- SMAC's custom map units get dynamically
assigned runtime IDs that differ per map/process, so the adapter resolves
names via ``data_raw()`` before this widget ever sees them (see
``gym_gui.core.adapters.smac_unit_icons``).
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from gym_gui.core.adapters.smac_unit_icons import get_unit_icon_path

_ICON_SIZE = 28
_RESOURCE_LABEL_STYLE = "font-weight: bold;"


class SmacDashboardWidget(QtWidgets.QGroupBox):
    """Resource bar + army composition panel for SMAC/SMACv2 sessions."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("SMAC Dashboard", parent)
        self._icon_cache: Dict[str, Optional[QtGui.QPixmap]] = {}
        self._unit_icon_labels: Dict[str, tuple[QtWidgets.QLabel, QtWidgets.QLabel]] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # -------- Resource bar --------
        resource_row = QtWidgets.QHBoxLayout()

        self._minerals_label = self._make_resource_label("Minerals", "#5daeff")
        self._vespene_label = self._make_resource_label("Vespene", "#5dff8f")
        self._supply_label = self._make_resource_label("Supply", "#ffffff")
        self._army_label = self._make_resource_label("Army", "#ffb85d")
        self._idle_label = self._make_resource_label("Idle Workers", "#ff5d5d")

        for lbl in (
            self._minerals_label,
            self._vespene_label,
            self._supply_label,
            self._army_label,
            self._idle_label,
        ):
            resource_row.addWidget(lbl)
        resource_row.addStretch(1)
        layout.addLayout(resource_row)

        # -------- Army composition (unit icons + counts) --------
        self._army_row = QtWidgets.QHBoxLayout()
        self._army_row.setSpacing(6)
        army_container = QtWidgets.QWidget(self)
        army_container.setLayout(self._army_row)
        layout.addWidget(army_container)
        self._army_row.addStretch(1)

        self.setVisible(False)  # Hidden until a SMAC/SMACv2 session is active

    def _make_resource_label(self, name: str, color: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(f"{name}: —", self)
        lbl.setStyleSheet(f"{_RESOURCE_LABEL_STYLE} color: {color};")
        return lbl

    def _get_icon_pixmap(self, unit_name: str) -> Optional[QtGui.QPixmap]:
        if unit_name in self._icon_cache:
            return self._icon_cache[unit_name]
        path = get_unit_icon_path(unit_name)
        if path is None:
            self._icon_cache[unit_name] = None
            return None
        pixmap = QtGui.QPixmap(str(path))
        if pixmap.isNull():
            self._icon_cache[unit_name] = None
            return None
        pixmap = pixmap.scaled(
            _ICON_SIZE, _ICON_SIZE,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self._icon_cache[unit_name] = pixmap
        return pixmap

    def update_from_step_info(self, info: Dict[str, Any]) -> None:
        """Refresh the dashboard from a SMAC/SMACv2 step's info dict.

        Expects (all optional; missing keys leave the corresponding
        display unchanged):
            - ``player_common``: dict with minerals, vespene, food_used,
              food_cap, army_count, idle_worker_count (see
              ``sc2api.proto`` PlayerCommon).
            - ``ally_unit_names``: list of resolved SC2 unit type name
              strings (e.g. ``"Marine_RL"``) for currently-alive allied
              units, used to build the army composition icon row.
        """
        pc = info.get("player_common")
        if pc is not None:
            self._minerals_label.setText(f"Minerals: {int(pc.get('minerals', 0))}")
            self._vespene_label.setText(f"Vespene: {int(pc.get('vespene', 0))}")
            food_used = pc.get("food_used", 0)
            food_cap = pc.get("food_cap", 0)
            self._supply_label.setText(f"Supply: {int(food_used)}/{int(food_cap)}")
            self._army_label.setText(f"Army: {int(pc.get('army_count', 0))}")
            self._idle_label.setText(f"Idle Workers: {int(pc.get('idle_worker_count', 0))}")

        ally_names = info.get("ally_unit_names")
        if ally_names is not None:
            self._update_army_composition(ally_names)

    def _update_army_composition(self, unit_names: list[str]) -> None:
        counts = Counter(unit_names)

        # Remove icons for unit types no longer present
        for unit_name in list(self._unit_icon_labels.keys()):
            if unit_name not in counts:
                icon_lbl, count_lbl = self._unit_icon_labels.pop(unit_name)
                icon_lbl.setParent(None)
                count_lbl.setParent(None)

        for unit_name, count in sorted(counts.items()):
            if unit_name not in self._unit_icon_labels:
                icon_lbl = QtWidgets.QLabel(self)
                pixmap = self._get_icon_pixmap(unit_name)
                if pixmap is not None:
                    icon_lbl.setPixmap(pixmap)
                else:
                    icon_lbl.setText("?")
                    icon_lbl.setFixedSize(_ICON_SIZE, _ICON_SIZE)
                    icon_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    icon_lbl.setStyleSheet("border: 1px solid #666; color: #999;")
                icon_lbl.setToolTip(unit_name)
                count_lbl = QtWidgets.QLabel(self)
                count_lbl.setStyleSheet("font-weight: bold;")
                # Insert before the trailing stretch
                insert_index = self._army_row.count() - 1
                self._army_row.insertWidget(insert_index, icon_lbl)
                self._army_row.insertWidget(insert_index + 1, count_lbl)
                self._unit_icon_labels[unit_name] = (icon_lbl, count_lbl)

            _, count_lbl = self._unit_icon_labels[unit_name]
            count_lbl.setText(f"x{count}")

    def set_smac_active(self, active: bool) -> None:
        """Show or hide the whole dashboard (call when loading/unloading a game)."""
        self.setVisible(active)
        if not active:
            self._minerals_label.setText("Minerals: —")
            self._vespene_label.setText("Vespene: —")
            self._supply_label.setText("Supply: —")
            self._army_label.setText("Army: —")
            self._idle_label.setText("Idle Workers: —")
            for unit_name in list(self._unit_icon_labels.keys()):
                icon_lbl, count_lbl = self._unit_icon_labels.pop(unit_name)
                icon_lbl.setParent(None)
                count_lbl.setParent(None)


__all__ = ["SmacDashboardWidget"]
