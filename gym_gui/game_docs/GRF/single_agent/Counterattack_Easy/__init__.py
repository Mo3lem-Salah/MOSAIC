"""GRF Academy: Counterattack Easy -- documentation for MOSAIC."""

from gym_gui.game_docs.GRF._shared import (
    GRF_ACTIONS_HTML,
    GRF_OBS_HTML,
    GRF_OVERVIEW_HTML,
    GRF_REWARD_HTML,
)

GRF_ACADEMY_COUNTERATTACK_EASY_HTML = f"""
<h2>GRF Academy: Counterattack (Easy)</h2>
<p style="background-color: #fff3e0; padding: 8px; border-radius: 4px;">
<strong>Single-Agent Counterattack Drill</strong> -- Fast break with numerical advantage, fewer defenders.
</p>
{GRF_OVERVIEW_HTML}
<h4>Scenario Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Scenario</td><td style="border: 1px solid #ddd; padding: 8px;"><code>academy_counterattack_easy</code></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Category</td><td style="border: 1px solid #ddd; padding: 8px;">Academy (single-agent training drill)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Controlled Players</td><td style="border: 1px solid #ddd; padding: 8px;">1 (left team)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Difficulty</td><td style="border: 1px solid #ddd; padding: 8px;">Medium</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Stepping</td><td style="border: 1px solid #ddd; padding: 8px;">Single-agent</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Rendering</td><td style="border: 1px solid #ddd; padding: 8px;">RGB frames via GRF engine</td></tr>
</table>
<p>
The attacking team has a numerical advantage with more attackers than
defenders. The agent must exploit the overload to score on a fast break.
This is the easier variant with fewer defenders standing in the way.
</p>
{GRF_OBS_HTML}
{GRF_ACTIONS_HTML}
{GRF_REWARD_HTML}
"""

__all__ = ["GRF_ACADEMY_COUNTERATTACK_EASY_HTML"]
