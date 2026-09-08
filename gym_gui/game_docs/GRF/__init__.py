"""GRF game documentation -- re-exports from single_agent/ and multi_agent/.

Google Research Football (GRF) is a football (soccer) environment for RL
research, developed by the Google Brain team. It provides both focused
single-agent academy drills and full multi-agent match scenarios.

Academy scenarios (single-agent):
    - empty_goal, empty_goal_close, run_to_score,
      run_to_score_with_keeper, pass_and_shoot, run_pass_and_shoot,
      counterattack_easy, counterattack_hard, corner,
      3_vs_1_with_keeper, single_goal_versus_lazy

Full-match scenarios (multi-agent):
    - 1_vs_1_easy, 5_vs_5, 11_vs_11_easy_stochastic,
      11_vs_11_stochastic, 11_vs_11_hard_stochastic

Paper: Kurach et al. (2020). "Google Research Football: A Novel RL Environment"
Source: 3rd_party/environments/football/
"""

from gym_gui.game_docs.GRF.multi_agent import (
    GRF_11V11_EASY_HTML,
    GRF_11V11_HARD_HTML,
    GRF_11V11_HTML,
    GRF_1V1_EASY_HTML,
    GRF_5V5_HTML,
)
from gym_gui.game_docs.GRF.single_agent import (
    GRF_ACADEMY_3VS1_WITH_KEEPER_HTML,
    GRF_ACADEMY_CORNER_HTML,
    GRF_ACADEMY_COUNTERATTACK_EASY_HTML,
    GRF_ACADEMY_COUNTERATTACK_HARD_HTML,
    GRF_ACADEMY_EMPTY_GOAL_CLOSE_HTML,
    GRF_ACADEMY_EMPTY_GOAL_HTML,
    GRF_ACADEMY_PASS_AND_SHOOT_HTML,
    GRF_ACADEMY_RUN_PASS_AND_SHOOT_HTML,
    GRF_ACADEMY_RUN_TO_SCORE_HTML,
    GRF_ACADEMY_RUN_TO_SCORE_WITH_KEEPER_HTML,
    GRF_ACADEMY_SINGLE_GOAL_VS_LAZY_HTML,
)

__all__ = [
    # single-agent (Academy drills)
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
    # multi-agent (Full Match)
    "GRF_1V1_EASY_HTML",
    "GRF_5V5_HTML",
    "GRF_11V11_EASY_HTML",
    "GRF_11V11_HTML",
    "GRF_11V11_HARD_HTML",
]
