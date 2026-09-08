"""Game documentation for GRF multi-agent (Full Match) environments."""

from __future__ import annotations

import importlib

# All directory names start with digits; use importlib for each.
_1v1 = importlib.import_module("gym_gui.game_docs.GRF.multi_agent.1v1_Easy")
GRF_1V1_EASY_HTML: str = _1v1.GRF_1V1_EASY_HTML

_5v5 = importlib.import_module("gym_gui.game_docs.GRF.multi_agent.5v5")
GRF_5V5_HTML: str = _5v5.GRF_5V5_HTML

_11v11_easy = importlib.import_module("gym_gui.game_docs.GRF.multi_agent.11v11_Easy")
GRF_11V11_EASY_HTML: str = _11v11_easy.GRF_11V11_EASY_HTML

_11v11 = importlib.import_module("gym_gui.game_docs.GRF.multi_agent.11v11")
GRF_11V11_HTML: str = _11v11.GRF_11V11_HTML

_11v11_hard = importlib.import_module("gym_gui.game_docs.GRF.multi_agent.11v11_Hard")
GRF_11V11_HARD_HTML: str = _11v11_hard.GRF_11V11_HARD_HTML

__all__ = [
    "GRF_1V1_EASY_HTML",
    "GRF_5V5_HTML",
    "GRF_11V11_EASY_HTML",
    "GRF_11V11_HTML",
    "GRF_11V11_HARD_HTML",
]
