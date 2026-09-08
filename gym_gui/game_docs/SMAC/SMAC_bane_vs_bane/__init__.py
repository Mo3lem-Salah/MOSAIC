"""Documentation for SMAC v1: bane_vs_bane (bane_vs_bane)."""
from __future__ import annotations

from gym_gui.game_docs.SMAC._shared import (
    SMAC_ACTIONS_HTML,
    SMAC_CTDE_HTML,
    SMAC_OBS_HTML,
    SMAC_REWARD_HTML,
)


def get_smac_bane_vs_bane_html() -> str:
    """Generate HTML documentation for the bane_vs_bane map."""
    return f"""
<h2>SMAC v1: bane_vs_bane</h2>

<p style="background-color: #e3f2fd; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>Cooperative Multi-Agent Micromanagement</strong> --
Control 24 Zerglings + Banelings to defeat 24 Zerglings + Banelings controlled by the SC2 built-in AI.
</p>

<h4>Map Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Map Name</td><td style="border: 1px solid #ddd; padding: 8px;"><code>bane_vs_bane</code></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Allies</td><td style="border: 1px solid #ddd; padding: 8px;">24 Zerglings + Banelings (24 agents)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Enemies</td><td style="border: 1px solid #ddd; padding: 8px;">24 Zerglings + Banelings</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Difficulty</td><td style="border: 1px solid #ddd; padding: 8px;">Easy</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Episode Limit</td><td style="border: 1px solid #ddd; padding: 8px;">200 steps</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Stepping</td><td style="border: 1px solid #ddd; padding: 8px;">Simultaneous (all agents act in parallel)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Rendering</td><td style="border: 1px solid #ddd; padding: 8px;">PyGame 2D top-down (terrain, unit circles, health arcs, HUD)</td></tr>
</table>

<p><b>Notes:</b> Symmetric mixed Zerg swarm battle between Zerglings and self-destructing Banelings on both sides.</p>

{SMAC_CTDE_HTML}
{SMAC_OBS_HTML}
{SMAC_ACTIONS_HTML}
{SMAC_REWARD_HTML}

<h4>References</h4>
<ul>
    <li>Samvelyan et al. (2019). "The StarCraft Multi-Agent Challenge"</li>
    <li>Repository: <a href="https://github.com/oxwhirl/smac">github.com/oxwhirl/smac</a></li>
</ul>
"""


SMAC_BANE_VS_BANE_HTML = get_smac_bane_vs_bane_html()

__all__ = ["SMAC_BANE_VS_BANE_HTML", "get_smac_bane_vs_bane_html"]
