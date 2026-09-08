"""GRF Full Match: 11 vs 11 Hard Stochastic -- documentation for MOSAIC."""

from gym_gui.game_docs.GRF._shared import (
    GRF_ACTIONS_HTML,
    GRF_OBS_HTML,
    GRF_OVERVIEW_HTML,
    GRF_REWARD_HTML,
)

GRF_11V11_HARD_HTML = f"""
<h2>GRF: 11 vs 11 (Hard Stochastic)</h2>
<p style="background-color: #fce4ec; padding: 8px; border-radius: 4px;">
<strong>Multi-Agent Full Match</strong> -- Full 11-a-side game against hard AI.
</p>
{GRF_OVERVIEW_HTML}
<h4>Scenario Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Scenario</td><td style="border: 1px solid #ddd; padding: 8px;"><code>11_vs_11_hard_stochastic</code></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Category</td><td style="border: 1px solid #ddd; padding: 8px;">Full Match</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Left Team</td><td style="border: 1px solid #ddd; padding: 8px;">11 players (10 outfield + 1 goalkeeper)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Right Team</td><td style="border: 1px solid #ddd; padding: 8px;">11 AI players (hard)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">AI Difficulty</td><td style="border: 1px solid #ddd; padding: 8px;">Hard</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Stepping</td><td style="border: 1px solid #ddd; padding: 8px;">Simultaneous (all controlled players act in parallel)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Episode Length</td><td style="border: 1px solid #ddd; padding: 8px;">3000 steps (configurable)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Rendering</td><td style="border: 1px solid #ddd; padding: 8px;">RGB frames via GRF engine</td></tr>
</table>
<p>
The hardest full-match variant. The opponent uses the strongest built-in AI,
making it extremely challenging to score or maintain possession. Requires
sophisticated team coordination, well-timed passing sequences, disciplined
defensive formations, and intelligent off-the-ball movement across the full
squad. This scenario serves as the ultimate benchmark for multi-agent football
policies and is significantly harder than the medium and easy variants.
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

__all__ = ["GRF_11V11_HARD_HTML"]
