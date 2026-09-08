"""GRF Full Match: 5 vs 5 -- documentation for MOSAIC."""

from gym_gui.game_docs.GRF._shared import (
    GRF_ACTIONS_HTML,
    GRF_OBS_HTML,
    GRF_OVERVIEW_HTML,
    GRF_REWARD_HTML,
)

GRF_5V5_HTML = f"""
<h2>GRF: 5 vs 5</h2>
<p style="background-color: #fff3e0; padding: 8px; border-radius: 4px;">
<strong>Multi-Agent Full Match</strong> -- Medium-scale 5-a-side football.
</p>
{GRF_OVERVIEW_HTML}
<h4>Scenario Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Scenario</td><td style="border: 1px solid #ddd; padding: 8px;"><code>5_vs_5</code></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Category</td><td style="border: 1px solid #ddd; padding: 8px;">Full Match</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Left Team</td><td style="border: 1px solid #ddd; padding: 8px;">5 players (4 outfield + 1 goalkeeper)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Right Team</td><td style="border: 1px solid #ddd; padding: 8px;">5 AI players (4 outfield + 1 goalkeeper)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">AI Difficulty</td><td style="border: 1px solid #ddd; padding: 8px;">Medium</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Stepping</td><td style="border: 1px solid #ddd; padding: 8px;">Simultaneous (all controlled players act in parallel)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Episode Length</td><td style="border: 1px solid #ddd; padding: 8px;">3000 steps (configurable)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Rendering</td><td style="border: 1px solid #ddd; padding: 8px;">RGB frames via GRF engine</td></tr>
</table>
<p>
Five players per side (4 outfield + 1 goalkeeper). A good balance between full
11v11 complexity and manageable multi-agent coordination. Tests passing,
positioning, and team strategy without the overhead of controlling a full squad.
Medium complexity makes this well suited for developing and debugging multi-agent
algorithms before scaling to the full 11-a-side game.
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

__all__ = ["GRF_5V5_HTML"]
