"""Documentation for HeMAC Simple Fleet environments."""
from __future__ import annotations

_SIMPLE_FLEET_SCENARIOS = {
    "1q1o": {"quadcopters": 1, "observers": 1, "provisioners": 0},
    "3q1o": {"quadcopters": 3, "observers": 1, "provisioners": 0},
    "5q2o": {"quadcopters": 5, "observers": 2, "provisioners": 0},
}


def get_simple_fleet_html(env_id: str) -> str:
    """Generate Simple Fleet HTML documentation for a specific variant."""
    for key, comp in _SIMPLE_FLEET_SCENARIOS.items():
        if key in env_id:
            composition = comp
            break
    else:
        composition = {"quadcopters": "?", "observers": "?", "provisioners": 0}

    total = composition["quadcopters"] + composition["observers"]

    return f"""
<h2>{env_id}</h2>

<p style="background-color: #fff3e0; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>API:</strong> PettingZoo Parallel — simultaneous stepping.
<a href="https://github.com/ThalesGroup/HeMAC" target="_blank">GitHub</a>
&nbsp;|&nbsp; ECAI 2025 &mdash; ThalesGroup
</p>

<p>Simple Fleet challenge: quadcopter drones and high-altitude observers
cooperate to locate and intercept Points of Interest (POIs) in a 2D
continuous patrol area. No obstacles, no provisioners &mdash; focused on
aerial coordination. {total} agents ({composition['quadcopters']}Q + {composition['observers']}O).</p>

<h4>Available Variants</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Environment ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Quadcopters</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Observers</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Total</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Max Cycles</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>hemac-simple-fleet-1q1o-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">1</td>
        <td style="border: 1px solid #ddd; padding: 8px;">1</td>
        <td style="border: 1px solid #ddd; padding: 8px;">2</td>
        <td style="border: 1px solid #ddd; padding: 8px;">600</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>hemac-simple-fleet-3q1o-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">3</td>
        <td style="border: 1px solid #ddd; padding: 8px;">1</td>
        <td style="border: 1px solid #ddd; padding: 8px;">4</td>
        <td style="border: 1px solid #ddd; padding: 8px;">600</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>hemac-simple-fleet-5q2o-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">5</td>
        <td style="border: 1px solid #ddd; padding: 8px;">2</td>
        <td style="border: 1px solid #ddd; padding: 8px;">7</td>
        <td style="border: 1px solid #ddd; padding: 8px;">600</td>
    </tr>
</table>

<h4>Agent Types &amp; Spaces</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Agent</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Obs Space</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action Space</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Role</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Quadcopter (Q)</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>Box(-10000, 10000, (15,))</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>Discrete(5)</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Low-altitude interceptor. RoundCamera sensor. Max speed 10.</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Observer (O)</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>Box(-10000, 10000, (11,))</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>Discrete(5)</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">High-altitude surveillance. ForwardFacingCamera (hfov=30&deg;, range 200).</td>
    </tr>
</table>

<h4>Action Space: Discrete(5) &mdash; all agent types</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #e3f2fd;">
        <th style="border: 1px solid #ddd; padding: 8px;">ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Drone movement</th>
        <th style="border: 1px solid #ddd; padding: 8px;">MOSAIC key</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>0</strong></td><td style="border: 1px solid #ddd; padding: 8px;">NOOP / recharge in place</td><td style="border: 1px solid #ddd; padding: 8px;"><kbd>Space</kbd></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>1</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Fly SE &nbsp;(+vx, +vy)</td><td style="border: 1px solid #ddd; padding: 8px;"><kbd>D</kbd> / <kbd>&rarr;</kbd></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>2</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Fly NE &nbsp;(+vx, &minus;vy)</td><td style="border: 1px solid #ddd; padding: 8px;"><kbd>W</kbd> / <kbd>&uarr;</kbd></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>3</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Fly SW &nbsp;(&minus;vx, +vy)</td><td style="border: 1px solid #ddd; padding: 8px;"><kbd>S</kbd> / <kbd>&darr;</kbd></td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>4</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Fly NW &nbsp;(&minus;vx, &minus;vy)</td><td style="border: 1px solid #ddd; padding: 8px;"><kbd>A</kbd> / <kbd>&larr;</kbd></td></tr>
</table>
<p style="color: #888; font-size: 0.9em;">All movements are diagonal (no pure cardinal directions). In Human Only mode
the same action is broadcast to every agent each tick.</p>

<h4>Scenario Parameters</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Parameter</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Area</td><td style="border: 1px solid #ddd; padding: 8px;">Pentagon patrol (100,100)&rarr;(250,100)&rarr;(820,480)&rarr;(600,800)&rarr;(100,620)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Obstacles</td><td style="border: 1px solid #ddd; padding: 8px;">None</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">POI speed</td><td style="border: 1px solid #ddd; padding: 8px;">2.0, random spawn</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Drone max charge</td><td style="border: 1px solid #ddd; padding: 8px;">100</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max cycles</td><td style="border: 1px solid #ddd; padding: 8px;">600 (configurable)</td></tr>
</table>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/ThalesGroup/HeMAC" target="_blank">HeMAC GitHub (ThalesGroup)</a></li>
    <li>Dansereau et al. (2025). "The Heterogeneous Multi-Agent Challenge." ECAI 2025. arXiv:2509.19512</li>
</ul>
"""


HEMAC_SIMPLE_FLEET_HTML = get_simple_fleet_html("hemac-simple-fleet-1q1o-v0")

__all__ = ["HEMAC_SIMPLE_FLEET_HTML", "get_simple_fleet_html"]
