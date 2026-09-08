"""Documentation for SMACv2: 10gen_zerg 5v5 (Zerg 5v5 scenario preset)."""
from __future__ import annotations

from gym_gui.game_docs.SMACv2._shared import (
    SMAC_ACTIONS_HTML,
    SMAC_CTDE_HTML,
    SMAC_OBS_HTML,
    SMAC_REWARD_HTML,
)


def get_smacv2_zerg_5v5_html() -> str:
    """Generate HTML documentation for the 10gen_zerg 5v5 scenario preset."""
    return f"""
<h2>SMACv2: 10gen_zerg (5v5)</h2>

<p style="background-color: #fff3e0; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>Procedural Multi-Agent Micromanagement -- 5v5 Preset</strong> --
Team compositions vary every episode with random Zerg units (5 allies vs 5 enemies),
forcing agents to generalise rather than memorise fixed strategies.
</p>

<h4>Map Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Base Map</td><td style="border: 1px solid #ddd; padding: 8px;"><code>10gen_zerg</code></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Scenario Preset</td><td style="border: 1px solid #ddd; padding: 8px;">5v5</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Race</td><td style="border: 1px solid #ddd; padding: 8px;">Zerg</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Unit Pool</td><td style="border: 1px solid #ddd; padding: 8px;">Zerglings, Banelings, Hydralisks</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Allies</td><td style="border: 1px solid #ddd; padding: 8px;">5 agents (varies by procedural generation)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Enemies</td><td style="border: 1px solid #ddd; padding: 8px;">5 units (varies by procedural generation)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Difficulty</td><td style="border: 1px solid #ddd; padding: 8px;">Easy</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Generation</td><td style="border: 1px solid #ddd; padding: 8px;">Procedural (new composition each reset)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Stepping</td><td style="border: 1px solid #ddd; padding: 8px;">Simultaneous</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Rendering</td><td style="border: 1px solid #ddd; padding: 8px;">PyGame 2D top-down (terrain, unit circles, health arcs, HUD)</td></tr>
</table>

<p><b>Key Difference from SMAC v1:</b> Agent count and observation/action shapes may change
between episodes.  The adapter re-queries <code>get_env_info()</code> after each reset.</p>

<p><b>Scenario Table:</b> This preset matches EPyMARL's published SMACv2 scenario table
(<code>5v5</code> = 5 allies vs 5 enemies), one of five standard team-size
configurations (5v5, 10v10, 20v20, 10v11, 20v23) evaluated per race in the SMACv2 paper.</p>

<p><b>Notes:</b> Swarm composition with burst damage. Zerglings and Hydralisks form the main force while Banelings provide explosive area damage on contact.</p>

{SMAC_CTDE_HTML}
{SMAC_OBS_HTML}
{SMAC_ACTIONS_HTML}
{SMAC_REWARD_HTML}

<h4>References</h4>
<ul>
    <li>Ellis et al. (2023). "SMACv2: An Improved Benchmark for Cooperative MARL"</li>
    <li>EPyMARL scenario table: <a href="https://github.com/uoe-agents/epymarl">github.com/uoe-agents/epymarl</a></li>
    <li>Repository: <a href="https://github.com/oxwhirl/smacv2">github.com/oxwhirl/smacv2</a></li>
</ul>
"""


SMACV2_ZERG_5V5_HTML = get_smacv2_zerg_5v5_html()

__all__ = ["SMACV2_ZERG_5V5_HTML", "get_smacv2_zerg_5v5_html"]
