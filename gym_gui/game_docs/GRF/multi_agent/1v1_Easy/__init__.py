"""GRF Full Match: 1 vs 1 Easy -- documentation for MOSAIC."""

from gym_gui.game_docs.GRF._shared import (
    GRF_ACTIONS_HTML,
    GRF_OBS_HTML,
    GRF_OVERVIEW_HTML,
    GRF_REWARD_HTML,
)

GRF_1V1_EASY_HTML = f"""
<h2>GRF: 1 vs 1 (Easy)</h2>
<p style="background-color: #e8f5e9; padding: 8px; border-radius: 4px;">
<strong>Multi-Agent Full Match</strong> -- Simplified 1-on-1 game with easy AI.
</p>
{GRF_OVERVIEW_HTML}
<h4>Scenario Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Scenario</td><td style="border: 1px solid #ddd; padding: 8px;"><code>1_vs_1_easy</code></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Category</td><td style="border: 1px solid #ddd; padding: 8px;">Full Match</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Left Team</td><td style="border: 1px solid #ddd; padding: 8px;">1 agent-controlled player</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Right Team</td><td style="border: 1px solid #ddd; padding: 8px;">1 AI player (easy)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">AI Difficulty</td><td style="border: 1px solid #ddd; padding: 8px;">Easy</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Stepping</td><td style="border: 1px solid #ddd; padding: 8px;">Simultaneous (all controlled players act in parallel)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Episode Length</td><td style="border: 1px solid #ddd; padding: 8px;">3000 steps (configurable)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Rendering</td><td style="border: 1px solid #ddd; padding: 8px;">RGB frames via GRF engine</td></tr>
</table>
<p>
A minimal full-match scenario with just 1 player per team and no goalkeepers.
Strips away team coordination entirely to focus on pure 1v1 dribbling, shooting,
and defending. The opponent uses easy difficulty built-in AI, making this a good
starting point for testing basic football mechanics before scaling up to larger
team sizes.
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

__all__ = ["GRF_1V1_EASY_HTML"]
