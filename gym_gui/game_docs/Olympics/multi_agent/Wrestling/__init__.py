"""Olympics Wrestling (sumo) environment documentation for MOSAIC."""

from gym_gui.game_docs.Olympics._shared import (
    OLYMPICS_ACTIONS_HTML,
    OLYMPICS_OBS_HTML,
    OLYMPICS_OVERVIEW_HTML,
    OLYMPICS_PHYSICS_HTML,
)

OLYMPICS_WRESTLING_HTML = f"""
<h2>Olympics Wrestling (Sumo)</h2>

<p style="background-color: #fce4ec; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>Multi-Agent Competitive Physics</strong> --
Push your opponent off the circular arena to win.
</p>

{OLYMPICS_OVERVIEW_HTML}

<h4>Scenario Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Scenario</td><td style="border: 1px solid #ddd; padding: 8px;"><code>olympics-wrestling</code></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Players</td><td style="border: 1px solid #ddd; padding: 8px;">2 (competitive, simultaneous)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Arena</td><td style="border: 1px solid #ddd; padding: 8px;">Circular platform with green boundary</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agents</td><td style="border: 1px solid #ddd; padding: 8px;">Elastic balls with equal mass and radius</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Episode Length</td><td style="border: 1px solid #ddd; padding: 8px;">500 steps maximum</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Stepping</td><td style="border: 1px solid #ddd; padding: 8px;">Simultaneous (both agents act in parallel)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Rendering</td><td style="border: 1px solid #ddd; padding: 8px;">Pygame RGB frames</td></tr>
</table>

<p>Two elastic ball agents start on opposite sides of a circular arena.
Each agent can apply continuous force and steering to move around. Agents
can collide with each other (elastic collisions) and must avoid touching
the green boundary, which counts as falling off the platform.</p>

<h4>Win Condition</h4>
<p>An agent wins (+1 reward) when the opponent crosses the green boundary
arc (falls off). If neither agent falls off within 500 steps, the episode
ends in a draw (both get 0 reward). The competitive nature means one
agent's gain is the other's loss.</p>

<h4>Strategy</h4>
<ul>
    <li><b>Aggression vs defense:</b> Charge the opponent to push them off, but risk being deflected toward the edge yourself</li>
    <li><b>Energy management:</b> Sprinting drains energy fast. Fatigue (energy = 0) leaves you unable to apply force</li>
    <li><b>Positioning:</b> Stay near the center, force the opponent toward the edge</li>
    <li><b>Collision angles:</b> Elastic collisions mean approach angle determines who gets pushed where</li>
</ul>

{OLYMPICS_OBS_HTML}
{OLYMPICS_ACTIONS_HTML}
{OLYMPICS_PHYSICS_HTML}
"""

__all__ = ["OLYMPICS_WRESTLING_HTML"]
