"""GRF Academy: Run Pass And Shoot -- documentation for MOSAIC."""

from gym_gui.game_docs.GRF._shared import (
    GRF_ACTIONS_HTML,
    GRF_OBS_HTML,
    GRF_OVERVIEW_HTML,
    GRF_REWARD_HTML,
)

GRF_ACADEMY_RUN_PASS_AND_SHOOT_HTML = f"""
<h2>GRF Academy: Run, Pass And Shoot</h2>
<p style="background-color: #fff3e0; padding: 8px; border-radius: 4px;">
<strong>Single-Agent Combination Play Drill</strong> -- Run, pass to a teammate, and score past the keeper.
</p>
{GRF_OVERVIEW_HTML}
<h4>Scenario Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Scenario</td><td style="border: 1px solid #ddd; padding: 8px;"><code>academy_run_pass_and_shoot_with_keeper</code></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Category</td><td style="border: 1px solid #ddd; padding: 8px;">Academy (single-agent training drill)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Controlled Players</td><td style="border: 1px solid #ddd; padding: 8px;">1 (left team)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Difficulty</td><td style="border: 1px solid #ddd; padding: 8px;">Medium</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Stepping</td><td style="border: 1px solid #ddd; padding: 8px;">Single-agent</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Rendering</td><td style="border: 1px solid #ddd; padding: 8px;">RGB frames via GRF engine</td></tr>
</table>
<p>
Extended version of <code>pass_and_shoot</code>: the agent starts further
from goal and must combine dribbling, passing, and shooting in sequence.
Multi-step planning is required to create and convert a scoring opportunity.
</p>
{GRF_OBS_HTML}
{GRF_ACTIONS_HTML}
{GRF_REWARD_HTML}
"""

__all__ = ["GRF_ACADEMY_RUN_PASS_AND_SHOOT_HTML"]
