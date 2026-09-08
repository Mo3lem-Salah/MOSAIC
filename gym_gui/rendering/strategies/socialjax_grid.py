"""Renderer strategy for SocialJax grid-state payloads.

Draws the coop_mining (or any socialjax) grid directly in Qt without
involving JAX -- the payload carries tile codes, agent positions, and
the colour palette so the renderer is fully env-agnostic.

Payload format emitted by jaxmarl_worker/_state_to_grid_payload():
  {
    "mode":         "socialjax_grid",
    "grid":         [[int, ...], ...],      # (H, W) tile codes
    "agent_locs":   [[row, col, orient],..],# (N, 3)  orient: 0=up 1=right 2=down 3=left
    "n_agents":     int,
    "inner_t":      int,
    "max_inner_t":  int,
    "tile_colors":  {"0": [R,G,B], ...},   # tile code (str key) -> RGB
    "agent_colors": [[R,G,B], ...],        # one entry per agent
  }
"""

from __future__ import annotations

import logging
from typing import Mapping

from qtpy import QtCore, QtGui, QtWidgets

from gym_gui.core.enums import RenderMode
from gym_gui.rendering.interfaces import RendererContext, RendererStrategy

_LOGGER = logging.getLogger(__name__)

_TIME_BAR_H  = 8   # pixels for the progress bar at the bottom
_MIN_CELL    = 4   # minimum cell size before labels are skipped

# Tile codes that are ore items -- drawn as a circle on ore_wait background.
# Matches SocialJax rendering.py: Items.iron_ore=4, gold_ore=5, gold_partial=6
# circle_mask(radius=0.5) -- circle inscribed in the full cell.
_ORE_CODES       = {4, 5, 6}
# ore_wait background colour (tile code 2) -- used behind every ore circle,
# same as SocialJax get_base_item_tile: fill_coords(rect, item_colors[ore_wait])
_ORE_WAIT_RGB    = (200, 200, 170)


class SocialJaxGridStrategy(RendererStrategy):
    """Render SocialJax grid-state payloads as a tile map with agent overlays."""

    mode = RenderMode.SOCIALJAX_GRID

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        self._view = _SocialJaxGridView(parent)

    @property
    def widget(self) -> QtWidgets.QWidget:
        return self._view

    def supports(self, payload: Mapping[str, object]) -> bool:
        return payload.get("mode") == "socialjax_grid"

    def render(self, payload: Mapping[str, object], *, context: RendererContext | None = None) -> None:
        self._view.set_payload(payload)

    def reset(self) -> None:
        self._view.set_payload(None)

    def cleanup(self) -> None:
        try:
            self._view.set_payload(None)
        except Exception:
            pass


class _SocialJaxGridView(QtWidgets.QWidget):
    """Widget that renders a SocialJax grid state via QPainter in paintEvent."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.setMinimumSize(200, 200)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QtGui.QColor(17, 17, 17))
        self.setPalette(palette)

        self._payload: Mapping[str, object] | None = None

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(600, 600)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(200, 200)

    def set_payload(self, payload: Mapping[str, object] | None) -> None:
        self._payload = payload
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        try:
            self._paint(painter)
        finally:
            painter.end()

    def _paint(self, painter: QtGui.QPainter) -> None:
        w = self.width()
        h = self.height()

        if self._payload is None:
            painter.setPen(QtGui.QColor(153, 153, 153))
            painter.drawText(
                QtCore.QRect(0, 0, w, h),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                "Waiting for grid state...",
            )
            return

        grid        = self._payload.get("grid", [])
        agent_locs  = self._payload.get("agent_locs", [])
        n_agents    = int(self._payload.get("n_agents", len(agent_locs)))
        inner_t     = int(self._payload.get("inner_t", 0))
        max_inner_t = int(self._payload.get("max_inner_t", 1000)) or 1000
        tile_colors = self._payload.get("tile_colors", {})
        agent_colors = self._payload.get("agent_colors", [])
        obs_view    = self._payload.get("obs_view")   # {fwd, bwd, left, right} or None

        if not grid:
            painter.setPen(QtGui.QColor(153, 153, 153))
            painter.drawText(
                QtCore.QRect(0, 0, w, h),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                "Empty grid",
            )
            return

        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        if rows == 0 or cols == 0:
            return

        # Reserve bottom strip for time bar
        draw_h = h - _TIME_BAR_H - 2
        cell_w = w / cols
        cell_h = draw_h / rows
        cell = min(cell_w, cell_h)

        # Center the grid in the widget
        grid_px_w = cell * cols
        grid_px_h = cell * rows
        ox = (w - grid_px_w) / 2
        oy = (draw_h - grid_px_h) / 2

        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # --- Tiles ---------------------------------------------------------
        # ore_wait background colour (code 2) for ore circles -- matches SocialJax source
        ore_wait_color = QtGui.QColor(*_ORE_WAIT_RGB)
        for r in range(rows):
            row = grid[r]
            for c in range(cols):
                code = row[c]
                x = ox + c * cell
                y = oy + r * cell

                if code in _ORE_CODES:
                    # SocialJax rendering.py renders ore as:
                    #   fill_coords(rect, ore_wait_color)        <- background
                    #   fill_coords(circle(cx=0.5,cy=0.5,r=0.5), ore_color)  <- circle
                    # radius=0.5 means the circle is inscribed in the full cell.
                    painter.fillRect(QtCore.QRectF(x, y, cell, cell), ore_wait_color)
                    rgb = tile_colors.get(str(code)) or tile_colors.get(code) or [139, 69, 19]
                    ore_color = QtGui.QColor(rgb[0], rgb[1], rgb[2])
                    r_ore = cell * 0.5   # radius=0.5 * cell matches SocialJax circle_mask(radius=0.5)
                    painter.setBrush(QtGui.QBrush(ore_color))
                    painter.setPen(QtCore.Qt.PenStyle.NoPen)
                    painter.drawEllipse(
                        QtCore.QPointF(x + cell * 0.5, y + cell * 0.5),
                        r_ore, r_ore,
                    )
                else:
                    rgb = tile_colors.get(str(code)) or tile_colors.get(code) or [80, 80, 80]
                    color = QtGui.QColor(rgb[0], rgb[1], rgb[2])
                    painter.fillRect(QtCore.QRectF(x, y, cell, cell), color)

        # --- Grid lines (thin, only when cells are large enough) -----------
        if cell >= 6:
            pen = QtGui.QPen(QtGui.QColor(0, 0, 0, 60))
            pen.setWidthF(0.5)
            painter.setPen(pen)
            for r in range(rows + 1):
                y = oy + r * cell
                painter.drawLine(QtCore.QPointF(ox, y), QtCore.QPointF(ox + grid_px_w, y))
            for c in range(cols + 1):
                x = ox + c * cell
                painter.drawLine(QtCore.QPointF(x, oy), QtCore.QPointF(x, oy + grid_px_h))

        # --- Observation windows -------------------------------------------
        # Draw before agents so the circle/arrow renders on top.
        # The egocentric window rotates with orientation: derived from
        # local_to_global() in coop_mining.py (orient 0=up 1=right 2=down 3=left).
        if obs_view is not None:
            fwd   = obs_view.get("fwd",   9)
            bwd   = obs_view.get("bwd",   1)
            oleft = obs_view.get("left",  5)
            oright = obs_view.get("right", 5)
            for i in range(n_agents):
                if i >= len(agent_locs):
                    break
                loc    = agent_locs[i]
                row_i  = int(loc[0])
                col_i  = int(loc[1])
                orient = int(loc[2]) if len(loc) > 2 else 0
                if row_i < 0 or row_i >= rows or col_i < 0 or col_i >= cols:
                    continue

                # Axis-aligned bounding box in world grid coords
                if orient == 0:   # up/north: forward=up (row-), backward=down (row+)
                    rmin, rmax = row_i - fwd,   row_i + bwd
                    cmin, cmax = col_i - oleft, col_i + oright
                elif orient == 1: # right/east: forward=right (col+)
                    rmin, rmax = row_i - oleft, row_i + oright
                    cmin, cmax = col_i - bwd,   col_i + fwd
                elif orient == 2: # down/south: forward=down (row+)
                    rmin, rmax = row_i - bwd,   row_i + fwd
                    cmin, cmax = col_i - oright, col_i + oleft
                else:             # left/west: forward=left (col-)
                    rmin, rmax = row_i - oright, row_i + oleft
                    cmin, cmax = col_i - fwd,    col_i + bwd

                # Clamp to valid grid area
                rmin = max(0, rmin); rmax = min(rows - 1, rmax)
                cmin = max(0, cmin); cmax = min(cols - 1, cmax)

                wx = ox + cmin * cell
                wy = oy + rmin * cell
                ww = (cmax - cmin + 1) * cell
                wh = (rmax - rmin + 1) * cell

                rgb = agent_colors[i] if i < len(agent_colors) else [200, 200, 0]
                outline = QtGui.QColor(rgb[0], rgb[1], rgb[2], 90)
                fill    = QtGui.QColor(rgb[0], rgb[1], rgb[2], 18)
                painter.setBrush(QtGui.QBrush(fill))
                painter.setPen(QtGui.QPen(outline, max(1.0, cell * 0.06)))
                painter.drawRect(QtCore.QRectF(wx, wy, ww, wh))

        # --- Agents --------------------------------------------------------
        for i in range(n_agents):
            if i >= len(agent_locs):
                break
            loc     = agent_locs[i]
            row_i   = int(loc[0])
            col_i   = int(loc[1])
            orient  = int(loc[2]) if len(loc) > 2 else 0

            # Skip agents whose JAX position is outside the valid grid -- this
            # happens when the environment marks an agent as "done" by moving it
            # to a sentinel coordinate (e.g. -1 or >= grid_size).
            if row_i < 0 or row_i >= rows or col_i < 0 or col_i >= cols:
                continue

            rgb = agent_colors[i] if i < len(agent_colors) else [200, 200, 0]
            color = QtGui.QColor(rgb[0], rgb[1], rgb[2])

            # Cell centre in pixels
            cx = ox + (col_i + 0.5) * cell   # col -> x
            cy = oy + (row_i + 0.5) * cell   # row -> y

            r_circle = cell * 0.36

            # Filled circle
            painter.setBrush(QtGui.QBrush(color))
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), max(0.8, cell * 0.05)))
            painter.drawEllipse(
                QtCore.QPointF(cx, cy),
                r_circle,
                r_circle,
            )

            # Direction arrow (small triangle inside the circle)
            if cell >= _MIN_CELL:
                ar = r_circle * 0.55
                tri = _arrow_points(cx, cy, ar, orient)
                painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 210)))
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.drawPolygon(QtGui.QPolygonF(tri))

            # Agent index label -- small, red, shown when cells large enough
            if cell >= 14:
                painter.setPen(QtGui.QColor(255, 60, 60))
                font = painter.font()
                font.setPixelSize(max(6, int(cell * 0.18)))
                font.setBold(True)
                painter.setFont(font)
                label_rect = QtCore.QRectF(cx - r_circle, cy - r_circle, r_circle * 2, r_circle * 2)
                painter.drawText(label_rect, QtCore.Qt.AlignmentFlag.AlignCenter, str(i))

        # --- Time bar ------------------------------------------------------
        bar_y = h - _TIME_BAR_H
        bar_w = int(w * inner_t / max_inner_t)
        painter.fillRect(QtCore.QRect(0, bar_y, w, _TIME_BAR_H), QtGui.QColor(40, 40, 40))
        painter.fillRect(QtCore.QRect(0, bar_y, bar_w, _TIME_BAR_H), QtGui.QColor(60, 160, 80))

        # Step counter text
        painter.setPen(QtGui.QColor(200, 200, 200))
        font = painter.font()
        font.setPixelSize(max(7, _TIME_BAR_H - 1))
        font.setBold(False)
        painter.setFont(font)
        painter.drawText(
            QtCore.QRect(4, bar_y, w - 8, _TIME_BAR_H),
            QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft,
            f"t={inner_t}/{max_inner_t}",
        )


def _arrow_points(cx: float, cy: float, r: float, orient: int) -> list[QtCore.QPointF]:
    """Equilateral-ish triangle pointing in the agent's facing direction.

    orient: 0=up, 1=right, 2=down, 3=left
    """
    if orient == 0:   # up
        return [
            QtCore.QPointF(cx,       cy - r),
            QtCore.QPointF(cx - r * 0.7, cy + r * 0.55),
            QtCore.QPointF(cx + r * 0.7, cy + r * 0.55),
        ]
    if orient == 1:   # right
        return [
            QtCore.QPointF(cx + r,       cy),
            QtCore.QPointF(cx - r * 0.55, cy - r * 0.7),
            QtCore.QPointF(cx - r * 0.55, cy + r * 0.7),
        ]
    if orient == 2:   # down
        return [
            QtCore.QPointF(cx,       cy + r),
            QtCore.QPointF(cx - r * 0.7, cy - r * 0.55),
            QtCore.QPointF(cx + r * 0.7, cy - r * 0.55),
        ]
    # orient == 3  # left
    return [
        QtCore.QPointF(cx - r,       cy),
        QtCore.QPointF(cx + r * 0.55, cy - r * 0.7),
        QtCore.QPointF(cx + r * 0.55, cy + r * 0.7),
    ]


__all__ = ["SocialJaxGridStrategy"]
