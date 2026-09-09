"""GRF Academy: 3 vs 1 With Keeper -- documentation for MOSAIC."""

from gym_gui.game_docs.GRF._shared import (
    GRF_ACTIONS_HTML,
    GRF_OBS_HTML,
    GRF_OVERVIEW_HTML,
    GRF_REWARD_HTML,
)

GRF_ACADEMY_3VS1_WITH_KEEPER_HTML = f"""
<h2>GRF Academy: 3 vs 1 With Keeper</h2>
<p style="background-color: #fff3e0; padding: 8px; border-radius: 4px;">
<strong>Single-Agent Overload Drill</strong> -- 3 attackers (1 controlled, 2 AI) vs 1 defender + goalkeeper.
</p>
{GRF_OVERVIEW_HTML}
<h4>Scenario Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Scenario</td><td style="border: 1px solid #ddd; padding: 8px;"><code>academy_3_vs_1_with_keeper</code></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Category</td><td style="border: 1px solid #ddd; padding: 8px;">Academy (single-agent training drill)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Controlled Players</td><td style="border: 1px solid #ddd; padding: 8px;">1 (left team)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Difficulty</td><td style="border: 1px solid #ddd; padding: 8px;">Medium</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Stepping</td><td style="border: 1px solid #ddd; padding: 8px;">Single-agent</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Rendering</td><td style="border: 1px solid #ddd; padding: 8px;">RGB frames via GRF engine</td></tr>
</table>
<p>
Three attackers (1 controlled by the agent, 2 by built-in AI) face one
defender and a goalkeeper. The agent must use passing to exploit the 3v1
numerical overload and create a clear shot on goal.
</p>
{GRF_OBS_HTML}
{GRF_ACTIONS_HTML}
{GRF_REWARD_HTML}
"""

__all__ = ["GRF_ACADEMY_3VS1_WITH_KEEPER_HTML"]
