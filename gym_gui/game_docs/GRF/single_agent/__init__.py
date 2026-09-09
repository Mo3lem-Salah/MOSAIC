"""Game documentation for GRF single-agent (Academy) environments."""

from __future__ import annotations

import importlib

from gym_gui.game_docs.GRF.single_agent.Corner import GRF_ACADEMY_CORNER_HTML
from gym_gui.game_docs.GRF.single_agent.Counterattack_Easy import GRF_ACADEMY_COUNTERATTACK_EASY_HTML
from gym_gui.game_docs.GRF.single_agent.Counterattack_Hard import GRF_ACADEMY_COUNTERATTACK_HARD_HTML
from gym_gui.game_docs.GRF.single_agent.Empty_Goal import GRF_ACADEMY_EMPTY_GOAL_HTML
from gym_gui.game_docs.GRF.single_agent.Empty_Goal_Close import GRF_ACADEMY_EMPTY_GOAL_CLOSE_HTML
from gym_gui.game_docs.GRF.single_agent.Pass_And_Shoot import GRF_ACADEMY_PASS_AND_SHOOT_HTML
from gym_gui.game_docs.GRF.single_agent.Run_Pass_And_Shoot import GRF_ACADEMY_RUN_PASS_AND_SHOOT_HTML
from gym_gui.game_docs.GRF.single_agent.Run_To_Score import GRF_ACADEMY_RUN_TO_SCORE_HTML
from gym_gui.game_docs.GRF.single_agent.Run_To_Score_With_Keeper import GRF_ACADEMY_RUN_TO_SCORE_WITH_KEEPER_HTML
from gym_gui.game_docs.GRF.single_agent.Single_Goal_Vs_Lazy import GRF_ACADEMY_SINGLE_GOAL_VS_LAZY_HTML

# Directory name starts with a digit; use importlib.
_3vs1 = importlib.import_module("gym_gui.game_docs.GRF.single_agent.3vs1_With_Keeper")
GRF_ACADEMY_3VS1_WITH_KEEPER_HTML: str = _3vs1.GRF_ACADEMY_3VS1_WITH_KEEPER_HTML

__all__ = [
    "GRF_ACADEMY_EMPTY_GOAL_HTML",
    "GRF_ACADEMY_EMPTY_GOAL_CLOSE_HTML",
    "GRF_ACADEMY_RUN_TO_SCORE_HTML",
    "GRF_ACADEMY_RUN_TO_SCORE_WITH_KEEPER_HTML",
    "GRF_ACADEMY_PASS_AND_SHOOT_HTML",
    "GRF_ACADEMY_RUN_PASS_AND_SHOOT_HTML",
    "GRF_ACADEMY_COUNTERATTACK_EASY_HTML",
    "GRF_ACADEMY_COUNTERATTACK_HARD_HTML",
    "GRF_ACADEMY_CORNER_HTML",
    "GRF_ACADEMY_3VS1_WITH_KEEPER_HTML",
    "GRF_ACADEMY_SINGLE_GOAL_VS_LAZY_HTML",
]
