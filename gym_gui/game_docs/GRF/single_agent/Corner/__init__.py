"""GRF Academy: Corner -- documentation for MOSAIC."""

from gym_gui.game_docs.GRF._shared import (
    GRF_ACTIONS_HTML,
    GRF_OBS_HTML,
    GRF_OVERVIEW_HTML,
    GRF_REWARD_HTML,
)

GRF_ACADEMY_CORNER_HTML = f"""
<h2>GRF Academy: Corner Kick</h2>
<p style="background-color: #fce4ec; padding: 8px; border-radius: 4px;">
<strong>Single-Agent Set Piece Drill</strong> -- Deliver a corner kick to teammates in the box and score.
</p>
{GRF_OVERVIEW_HTML}
<h4>Scenario Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Scenario</td><td style="border: 1px solid #ddd; padding: 8px;"><code>academy_corner</code></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Category</td><td style="border: 1px solid #ddd; padding: 8px;">Academy (single-agent training drill)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Controlled Players</td><td style="border: 1px solid #ddd; padding: 8px;">1 (left team)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Difficulty</td><td style="border: 1px solid #ddd; padding: 8px;">Medium-Hard</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Stepping</td><td style="border: 1px solid #ddd; padding: 8px;">Single-agent</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Rendering</td><td style="border: 1px solid #ddd; padding: 8px;">RGB frames via GRF engine</td></tr>
</table>
<p>
The agent takes a corner kick with teammates positioned in the penalty box.
It must decide where to deliver the ball and coordinate with AI-controlled
teammates to create and convert a scoring chance from the set piece.
</p>
{GRF_OBS_HTML}
{GRF_ACTIONS_HTML}
{GRF_REWARD_HTML}
"""

__all__ = ["GRF_ACADEMY_CORNER_HTML"]
