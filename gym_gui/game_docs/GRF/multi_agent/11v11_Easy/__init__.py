"""GRF Full Match: 11 vs 11 Easy Stochastic -- documentation for MOSAIC."""

from gym_gui.game_docs.GRF._shared import (
    GRF_ACTIONS_HTML,
    GRF_OBS_HTML,
    GRF_OVERVIEW_HTML,
    GRF_REWARD_HTML,
)

GRF_11V11_EASY_HTML = f"""
<h2>GRF: 11 vs 11 (Easy Stochastic)</h2>
<p style="background-color: #e8f5e9; padding: 8px; border-radius: 4px;">
<strong>Multi-Agent Full Match</strong> -- Full 11-a-side game against easy AI with stochastic elements.
</p>
{GRF_OVERVIEW_HTML}
<h4>Scenario Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Scenario</td><td style="border: 1px solid #ddd; padding: 8px;"><code>11_vs_11_easy_stochastic</code></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Category</td><td style="border: 1px solid #ddd; padding: 8px;">Full Match</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Left Team</td><td style="border: 1px solid #ddd; padding: 8px;">11 players (10 outfield + 1 goalkeeper)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Right Team</td><td style="border: 1px solid #ddd; padding: 8px;">11 AI players (easy)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">AI Difficulty</td><td style="border: 1px solid #ddd; padding: 8px;">Easy</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Stepping</td><td style="border: 1px solid #ddd; padding: 8px;">Simultaneous (all controlled players act in parallel)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Episode Length</td><td style="border: 1px solid #ddd; padding: 8px;">3000 steps (configurable)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Rendering</td><td style="border: 1px solid #ddd; padding: 8px;">RGB frames via GRF engine</td></tr>
</table>
<p>
Full 11-a-side match with standard football rules including offsides, corners,
throw-ins, and penalty kicks. The opponent uses easy difficulty built-in AI,
providing a gentler introduction to the full-scale game. The stochastic engine
variant adds noise to pass trajectories and ball physics for more realistic and
less exploitable play. This is a good entry point for training full-team
coordination before increasing opponent difficulty.
</p>

<h4>Multi-Agent Control</h4>
<p>
The agent controls multiple players simultaneously. Each controlled player
receives its own action at every step, forming a <b>joint action</b>
(MultiDiscrete action space). Uncontrolled players are handled by the
GRF built-in AI. Use <code>number_of_left_players_agent_controls</code> and
<code>number_of_right_players_agent_controls</code> in
<code>GFootballConfig</code> to configure.
</p>

{GRF_OBS_HTML}
{GRF_ACTIONS_HTML}
{GRF_REWARD_HTML}
"""

__all__ = ["GRF_11V11_EASY_HTML"]
